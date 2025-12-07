import os
import base64
from flask import Flask, render_template, request, jsonify, session
import openai
from dotenv import load_dotenv
import json
from flask_cors import CORS
import uuid
from PIL import Image
import io
import logging
import requests
import datetime
from flask import Flask, request, send_file, jsonify, render_template
from google import genai
from google.genai import types
import pathlib
from werkzeug.utils import secure_filename  # Add this import at the top
# Configure logging
logging.basicConfig(level=logging.DEBUG)

# ---------------- Image / Memory Optimization Constants ----------------
# These can be tuned via environment variables at deploy time.
MAX_SOURCE_SIDE = int(os.getenv("MAX_SOURCE_SIDE", "2048"))          # Hard cap for any uploaded image side length
WORKING_THUMB_SIDE = int(os.getenv("WORKING_THUMB_SIDE", "1024"))      # Size we downscale to for compositing
MAX_COMBINE_MEMORY_BYTES = int(os.getenv("MAX_COMBINE_MEMORY_BYTES", str(150 * 1024 * 1024)))  # ~150MB safety cap
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))  # 5MB per file default
ENABLE_MEM_LOG = os.getenv("ENABLE_MEM_LOG", "0") == "1"

def optimize_saved_image(path: str):
    """Shrink and recompress an already-saved upload in-place to save RAM + disk.
    - Limits max side to MAX_SOURCE_SIDE.
    - Converts to RGB JPEG (quality configurable) to drastically reduce footprint.
    Silently ignores failures so it never breaks the request path.
    """
    try:
        if not os.path.exists(path):
            return
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width > MAX_SOURCE_SIDE or im.height > MAX_SOURCE_SIDE:
                im.thumbnail((MAX_SOURCE_SIDE, MAX_SOURCE_SIDE), Image.LANCZOS)
            # Re-save as JPEG (even if originally PNG) to cut size.
            im.save(path, format='JPEG', quality=JPEG_QUALITY, optimize=True)
    except Exception as e:
        logging.warning(f"optimize_saved_image failed for {path}: {e}")

def log_memory(prefix: str):
    if not ENABLE_MEM_LOG:
        return
    try:
        import psutil, os as _os
        proc = psutil.Process(_os.getpid())
        rss = proc.memory_info().rss / (1024 * 1024)
        logging.debug(f"{prefix} RSS: {rss:.1f} MB")
    except Exception:
        pass

def save_generated_image(b64_data: str, upload_dir: str, base_name: str):
    """Decode a base64 PNG returned by a model and optionally transcode to JPEG to save space.
    OUTPUT_IMAGE_FORMAT env var controls final format (png or jpeg). Defaults to png.
    Returns absolute file path of saved image.
    """
    target_format = os.getenv("OUTPUT_IMAGE_FORMAT", "png").lower()
    tmp_png_path = os.path.join(upload_dir, f"{base_name}.png")
    try:
        with open(tmp_png_path, 'wb') as f:
            f.write(base64.b64decode(b64_data))
    except Exception as e:
        logging.error(f"Failed writing raw generated image: {e}")
        raise

    if target_format in ("jpg", "jpeg"):
        try:
            with Image.open(tmp_png_path) as im:
                im = im.convert("RGB")
                jpeg_path = os.path.join(upload_dir, f"{base_name}.jpg")
                im.save(jpeg_path, format='JPEG', quality=JPEG_QUALITY, optimize=True)
            # Remove larger png if jpeg smaller
            try:
                if os.path.getsize(jpeg_path) < os.path.getsize(tmp_png_path):
                    os.remove(tmp_png_path)
                    return jpeg_path
            except OSError:
                return jpeg_path
            return jpeg_path
        except Exception as e:
            logging.warning(f"Transcode to JPEG failed, keeping PNG: {e}")
            return tmp_png_path
    return tmp_png_path


# Load FAQ data (unchanged)
with open('faq.json', 'r', encoding='utf-8') as file:
    faq_data = json.load(file)

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key")  # Required for sessions
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Load OpenAI API key
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Helper function to encode images (unchanged)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

def classify_prompt_type(prompt):
    classification_messages = [
        {
            "role": "system",
            "content": "You are a classification assistant. Based on the user's input, decide if they are asking for a sculpture to be generated, edit an image, or just get a text response. Reply with only one word: 'generate', 'edit', or 'text'."
        },
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=classification_messages
    )
    return response.choices[0].message.content.strip().lower()



import hashlib
from PIL import Image
import logging
Image.MAX_IMAGE_PIXELS = 999999999 
logging.basicConfig(level=logging.DEBUG)

def combine_images(image_paths, output_path, max_size=1024):
    """
    Combines multiple images into one with a white background.
    - Resizes each image to fit within max_size x max_size while keeping aspect ratio.
    - Avoids duplicate images based on content hash.
    - Aligns all images by height for a clean horizontal combination.
    """

    unique_images = []
    seen_hashes = set()
    est_total_bytes = 0

    for path in image_paths:
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")

                # Hard limit on original size
                if img.width > MAX_SOURCE_SIDE or img.height > MAX_SOURCE_SIDE:
                    img.thumbnail((MAX_SOURCE_SIDE, MAX_SOURCE_SIDE), Image.LANCZOS)

                # Resize for working composite
                img.thumbnail((max_size, max_size), Image.LANCZOS)

                # Hash for duplicates (dimension + md5 of raw bytes)
                img_bytes = img.tobytes()
                img_hash = f"{img.width}x{img.height}-" + hashlib.md5(img_bytes).hexdigest()

                if img_hash in seen_hashes:
                    continue
                seen_hashes.add(img_hash)

                est_total_bytes += len(img_bytes)
                if est_total_bytes > MAX_COMBINE_MEMORY_BYTES:
                    logging.warning("combine_images aborted: memory estimate exceeded limit")
                    break

                # Keep a copy and immediately free original reference
                unique_images.append(img.copy())
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")
            continue

    if not unique_images:
        raise ValueError("No valid images to combine")

    # Normalize heights
    max_height = max(im.height for im in unique_images)
    normalized = []
    for im in unique_images:
        if im.height < max_height:
            ratio = max_height / im.height
            new_w = int(im.width * ratio)
            im = im.resize((new_w, max_height), Image.LANCZOS)
        normalized.append(im)

    total_width = sum(im.width for im in normalized)
    new_im = Image.new('RGB', (total_width, max_height), 'white')

    x_off = 0
    for im in normalized:
        new_im.paste(im, (x_off, 0))
        x_off += im.width

    new_im.save(output_path, format='JPEG', quality=JPEG_QUALITY, optimize=True)
    logging.info(f"Combined image saved at {output_path}")

    # Explicit cleanup
    del unique_images
    del normalized
    del new_im
from rapidfuzz import fuzz

# Function to check similarity
def is_similar_to_ludge(input_text, threshold=80):
    words = input_text.lower().split()
    return any(fuzz.ratio(word, "lobster") >= threshold for word in words)

# Dictionary mapping keywords to image filenames in static folder
SCULPTURE_BASES = {

    #Bases
    "champagne": os.path.join("bases", "champagne_base.png"),
    "crystal base": os.path.join("bases", "crystal_base.png"),
    "frame base": os.path.join("bases", "frame_base.png"),
    "golf base": os.path.join("bases", "golf_base.png"),
    "heart base": os.path.join("bases", "heartBase.png"),  # Modified name
    "plynth base w logo": os.path.join("bases", "Plynth base w logo.png"), # Exact name
    "rings base": os.path.join("bases", "rings base.png"), # Exact name
    "star base": os.path.join("bases", "Star base.png"), # Exact name
    "swirl base": os.path.join("bases", "swirl base.png"), # Exact name
    "tilt plynth base": os.path.join("bases", "tilt plynth base.png") ,# Exact name
   
    "plynth base": os.path.join("bases", "plynth_base.png"),
    "ring base": os.path.join("bases", "ring_base.png"),
    "tee base": os.path.join("bases", "tee_base.png"),
    "waves base": os.path.join("bases", "waves_base.png"),

    ##ludges

    "double ludge": os.path.join("sculptures", "double_ludge.png"),
    "martini": os.path.join("sculptures", "martini.png"),
    "tube": os.path.join("sculptures", "tube.png"),

    #Sculptures
        "alligator head": os.path.join("sculptures", "alligator_head.png"),
        "alligator": os.path.join("sculptures", "alligator.png"),
        "anchor": os.path.join("sculptures", "anchor.png"),
        "crab claw": os.path.join("sculptures", "crab claw.png"),
        "dolphin": os.path.join("sculptures", "dolphin.png"),
        "dragon head": os.path.join("sculptures", "dragon_head.png"),
        "griffen": os.path.join("sculptures", "griffen.png"),
        "guitar": os.path.join("sculptures", "guitar.png"),
        "heel": os.path.join("sculptures", "heel.png"),
        "horse head": os.path.join("sculptures", "horse_head.png"),
        "indian head": os.path.join("sculptures", "indian head.png"),
        "leopard": os.path.join("sculptures", "leopard.png"),
        "lion": os.path.join("sculptures", "lion.png"),
        "lobster": os.path.join("sculptures", "lobster.png"),
       
        "mahi mahi": os.path.join("sculptures", "mahi mahi.png"),
        "mask": os.path.join("sculptures", "mask.png"),
        "mermaid": os.path.join("sculptures", "mermaid.png"),
        "palm_trees": os.path.join("sculptures", "palm_trees.png"),
        "panther": os.path.join("sculptures", "panther.png"),
        "penguin": os.path.join("sculptures", "penguin.png"),
        "selfish": os.path.join("sculptures", "selfish.png"),
        "shark": os.path.join("sculptures", "shark.png"),
        "shrimp": os.path.join("sculptures", "shrimp.png"),
        "turkey": os.path.join("sculptures", "turkey.png"),
        "turle": os.path.join("sculptures", "turle.png"),
        "turtle cartoon": os.path.join("sculptures", "turtle_cartoon.png"),
        "unicorn": os.path.join("sculptures", "unicorn.png"),
        "vase": os.path.join("sculptures", "vase.png"),
        "whale": os.path.join("sculptures", "whale.png"),
        "women butt": os.path.join("sculptures", "women_butt.png"),
        "women torso": os.path.join("sculptures", "women_torso.png"),

        #Wedding Sculptures
        "interlocking rings": os.path.join("wedding_Showpieces", "interlocking_rings_showpiece_wedding.png"),
        "wedding frame": os.path.join("wedding_Showpieces", "picture_frame_wedding.png"),

        #Topper
        "banana single luge": os.path.join("Toppers", "banana single luge.jpg"),
        "ice bar mini single luge": os.path.join("Toppers", "ice bar mini single luge.jpg"),
        "ice bowl": os.path.join("Toppers", "ice bowl.png"),
        "crown logo as topper": os.path.join("Toppers", "crown logo as topper.jpg"),
        "crown logo as topper": os.path.join("Toppers", "crown logo as topper.jpg"),
        "crown logo as topper": os.path.join("Toppers", "crown logo as topper.jpg"),
            
         #Ice Bars
        "6ft ice bar": os.path.join("Ice bars", "6ft ice bar.jpg"),
        "8ft ice bar":os.path.join("Ice bars", "8ft ice bar.jpg"),   
        "12ft ice bar": os.path.join("Ice bars", "12ft ice bar.jpg"),   
        
       
        
        

        #LOGOS
    # "logo base": os.path.join("bases", "logo_base.png"),
    # "logo plynth base": os.path.join("bases", "logo_plynth_base.png"),

    # Add more mappings as needed
}
LUDGE_TYPES = {
    "martini": os.path.join("sculptures", "martini.png"),
    "tube": os.path.join("sculptures", "tube.png"),
    "double": os.path.join("sculptures", "double_ludge.png")
}

REFERENCE_IMAGES = {
    "double luge": [
        "static/double luge/BOOMBOX Double LUGE w J LOGO  Render.jpg",  # Replace with the actual path to your first reference image
        "static/double luge/HAKU VODKA Logo on KANPAI Double Luge.jpg"  # Replace with the actual path to your second reference image
        # "/static/uploads/butcher_3.jpg",  # Replace with the actual path to your third reference image
        # "/static/uploads/butcher_4.jpg",  # Replace with the actual path to your fourth reference image
    ]
    # Add more mappings for other sculptures later, e.g., "martini luge": [...]
}

# Then modify your ludge detection logic:
def detect_ludge_type(input_text):
    input_text = input_text.lower()
    for ludge in LUDGE_TYPES:
        if ludge in input_text:
            return LUDGE_TYPES[ludge]
    return None

def detect_sculpture_bases(input_text, threshold=80):
    """Detects which sculpture bases are mentioned in the input text."""
    input_text = input_text.lower()
    detected_bases = []
    
    # First remove "base" from input to avoid matching all bases
    input_without_base = input_text.replace("base", "").strip()
    if not input_without_base:  # If only "base" was specified
        input_without_base = input_text  # Use original input
    
    for keyword, image_path in SCULPTURE_BASES.items():
        # Get the main descriptor (remove "base" from keyword)
        main_keyword = keyword.replace("base", "").strip()
        
        # Check if main descriptor is in input
        if main_keyword and main_keyword in input_without_base:
            detected_bases.append(image_path)
        # Original exact match check (for full phrases)
        elif keyword in input_text:
            detected_bases.append(image_path)
        # Fuzzy matching only on main descriptors
        else:
            words = input_without_base.split()
            if any(fuzz.ratio(word, main_keyword) >= threshold for word in words):
                detected_bases.append(image_path)
    
    return detected_bases

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        image_url = data.get('image_url')  # e.g., /static/uploads/sculpture_61be3d96.png
        rating = data.get('rating')
        comment = data.get('comment')

        # Validate inputs
        if not image_url or not rating:
            return jsonify({'error': 'Missing image_url or rating'}), 400

        # Convert relative URL to absolute file path
        base_dir = os.path.abspath(os.path.dirname(__file__))
        image_path = os.path.join(base_dir, 'static/uploads', os.path.basename(image_url))
        print(f"Attempting to read image from: {image_path}")  # Debug log

        if not os.path.exists(image_path):
            return jsonify({'error': f'Image file not found at {image_path}'}), 404

        # Read the image file and convert to Base64
        with open(image_path, 'rb') as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            print(f"Base64 length: {len(base64_image)}")  # Debug log

        # Prepare feedback data for Google Apps Script
        feedback_data = {
            'imageData': base64_image,
            'rating': rating,
            'comment': comment or 'No comment provided'
        }

        # Send to Google Apps Script web app
        script_url = 'https://script.google.com/macros/s/AKfycbzKOxVD9ju-bewiQuCZS8hHByJ2JfU0mhhDbQFxWYIaQemRPgeJLtrvbOC8G-yf3vmg/exec'  # Replace with your web app URL
        response = requests.post(script_url, json=feedback_data, timeout=30)
        print(f"Google Apps Script response status: {response.status_code}, text: {response.text}")  # Debug log

        if response.status_code != 200:
            return jsonify({'error': f'Failed to save feedback, status: {response.status_code}, response: {response.text}'}), 500

        return jsonify({'message': 'Feedback submitted successfully'}), 200

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {str(e)}")
        return jsonify({'error': f'Image file not found: {str(e)}'}), 404
    except PermissionError as e:
        print(f"PermissionError: {str(e)}")
        return jsonify({'error': f'Permission denied accessing image: {str(e)}'}), 403
    except Exception as e:
        print(f"Unexpected error in submit_feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
ICE_CUBE_PROMPTS = {
    "Snofilled": """
    "task": "add the logo image into the center of the icecube",
    "instructions": 
        "effect": "Create a carved snow-filled appearance inside the ice sculpture, the image should not be colored and it should be engraved into the icecube with some depth",
        "ice":"ice should be crystal clear, transparent, and completely free of any cloudiness, bubbles, or impurities. The ice must appear as pure, pristine, see-through ice with perfect clarity",
        "Strict": "the logo should be engraved into the ice few centimeters with some depth",
        "Extra":"remove any background of the image before adding it to the icecube"
    """,
    "Colored": """
    "task": "add the logo image into the center of the icecube ", 
    "instructions": 
        "effect": "it should look like the ice is colored and not etched",
        "Strict": "the image should be engraved into the ice few centimeters with some depth",
        "Extra":"remove any background of the image before adding it to the icecube",
        "ice":"ice should be crystal clear, transparent, and completely free of any cloudiness, bubbles, or impurities. The ice must appear as pure, pristine, see-through ice with perfect clarity"
    """,
    "Paper": """
    "task": "add the image inside the icecube, ",
    "instructions": 
        "effect": "it should look like a colored printed paper is frozen into the icecube, the Logo should be colored with some white outline and transparent background and should be in center of the cube",
        "Strict": "the image should be placed into the ice few centimeters in some depth",
        "Extra":"remove any background of the image before adding it to the icecube",
        "ice":"ice should be crystal clear, transparent, and completely free of any cloudiness, bubbles, or impurities. The ice must appear as pure, pristine, see-through ice with perfect clarity, increase the size of the cube if the logo doesnot fit"
    """,
    "Snofilled+paper": """
    "task": "add the image into the center of the icecube ",
    "instructions": 
        "effect": "it should look like a colored printed paper is frozen into the icecube, the paper should be colored and should be in center of the cube, and the ice should be etched a little bit on the outlines of the image logo",
        "Strict": "the logo should be engraved into the ice few centimeters with some depth",
        "Extra":"remove any background of the image before adding it to the icecube",
        "ice":"ice should be crystal clear, transparent, and completely free of any cloudiness, bubbles, or impurities. The ice must appear as pure, pristine, see-through ice with perfect clarity"
    """
}

@app.route('/template_selected', methods=['POST'])
def handle_template_selection():
    data = request.get_json()
    template_type = data.get('template')
    template_name = data.get('templateName', '')
    
    # Print the template information
    print(f"Selected template type: {template_type}")
    if template_name:
        print(f"Selected template name: {template_name}")
    
    # Store the template type in session if it's an ice cube
    if template_type in ICE_CUBE_PROMPTS:
        session['selected_ice_cube'] = template_type
        print(f"Ice cube selected: {template_type}")  # This will now print for ice cubes
        print(f"Using prompt: {ICE_CUBE_PROMPTS[template_type]}")  # Print the prompt being used
    
    # For non-ice cube/ice bar templates, store the lighting message
    elif "ice bar" not in template_type.lower() and "ice cube" not in template_type.lower():
        session['template_selected_message'] = "Add a silver plastic rectangular led with decent blue lighting at the bottom of the sculpture"
        print("Lighting message set for non-ice template")
    
    return jsonify({"status": "success", "message": "Template selection received"})



client2 = genai.Client(api_key="AIzaSyCPw79xvCt4ZNOXJh4ORZ0OBZ4S7bZka7U")
MODEL_ID = "gemini-2.0-flash-exp"


@app.route('/extract_logo', methods=['POST'])
def extract_logo():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        file.stream.seek(0, os.SEEK_END)
        size = file.stream.tell()
        file.stream.seek(0)
        if size > MAX_UPLOAD_BYTES:
            return jsonify({"error": f"File too large. Max {MAX_UPLOAD_BYTES//1024//1024}MB"}), 400
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Securely save the uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(upload_path)
            optimize_saved_image(upload_path)
            logging.debug(f"Upload saved & optimized: {upload_path}")
        except Exception as e:
            return jsonify({"error": f"Error saving file: {e}"}), 500

        logging.debug(f"Received file: {file.filename}")

        # Open & prepare a working copy (downscaled) for the model
        try:
            with Image.open(upload_path) as img:
                img = img.convert('RGB')
                if img.width > WORKING_THUMB_SIDE or img.height > WORKING_THUMB_SIDE:
                    img.thumbnail((WORKING_THUMB_SIDE, WORKING_THUMB_SIDE), Image.LANCZOS)
                model_image = img.copy()
        except Exception as e:
            return jsonify({"error": f"Error opening image with PIL: {e}"}), 400

        prompt = "Extract the logo from this image and display it on a pure white background, tightly cropped."  # Clarified prompt
        response = client2.models.generate_content(
            model=MODEL_ID,
            contents=[prompt, model_image],
            config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                data = part.inline_data.data
                logo_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted_logo.png')
                try:
                    pathlib.Path(logo_path).write_bytes(data)
                    optimize_saved_image(logo_path)
                    return send_file(logo_path, mimetype='image/png')
                except Exception as e:
                    return jsonify({"error": f"Error writing image to file: {e}"}), 500

        return jsonify({"error": "No image returned from model"}), 500


    except Exception as e:
        print("Error extracting logo:", str(e))
        return jsonify({"error": str(e)}), 500

def get_logo_instructions(effect_type):
    shared = {
        "task": "add the image into the sculpture",
        "instructions": {
            "strict": "The logo must be embedded a few centimeters into the ice and should not form the sculpture itself.",
            "processing": "Remove any background of the image before embedding it into the ice cube.",
            "clarification": "If a blue image is provided, it is always ice and should be used as the ice sculpture, not as a logo. Blue images are never logos or image overlays—they are the ice.",
           "ice_structure": "The ice sculpture must precisely match the input image, with 100% accuracy. Do not add, remove, or modify any elements. Avoid including any extra components made of ice or anything not explicitly requested in the input."
        }
    }

    effects = {
        "Snofilled": "Create a carved snow appearance inside the ice sculpture. The image should not be colored and should be engraved with visible depth inside the ice cube.",
        "Colored": "It should look like the ice is colored from the logo, not etched. The image appears as colored pigmentation embedded inside the ice.",
        "Paper": "It should look like a colored printed paper is frozen inside the ice cube. The logo should be colored, have a slight white outline, and a transparent background, and should be centered within the cube."
    }

    shared["instructions"]["effect"] = effects[effect_type]
    return shared


@app.route('/log_button_press', methods=['POST'])
def log_button_press():
    data = request.get_json()
    button = data.get('button')
    image_url = data.get('image_url')
    timestamp = data.get('timestamp')
    
    print(f"Expand button pressed for image: {image_url} at {timestamp}")
    
    # You could also log this to a file or database
    # with open('button_logs.txt', 'a') as f:
    #     f.write(f"{timestamp} - {button} button pressed for {image_url}\n")
    
    return jsonify({'status': 'success'})

@app.route("/expand_chatbot", methods=["POST"])
def expand_chatbot():
    user_input = request.form.get("user_input", "").strip()
    uploaded_files = request.files.getlist("images")

    print("expand_chatbot route was hit!")

    try:
        #  Verify there is an image
        if not uploaded_files:
            return jsonify({"response": "No image provided for expansion"}), 400

        # Process just the first image
        uploaded_file = uploaded_files[0]
        uploaded_file.stream.seek(0, os.SEEK_END)
        size = uploaded_file.stream.tell(); uploaded_file.stream.seek(0)
        if size > MAX_UPLOAD_BYTES:
            return jsonify({"response": f"File too large. Max {MAX_UPLOAD_BYTES//1024//1024}MB"}), 400
        combined_id = uuid.uuid4().hex[:8]
        combined_path = os.path.join(app.config["UPLOAD_FOLDER"], f"combined_{combined_id}.jpg")
        uploaded_file.save(combined_path)
        optimize_saved_image(combined_path)

        # Generate using ONLY the user's prompt
        print("Generating image using ONLY user input:", user_input)
        with open(combined_path, "rb") as image_file:
            result = client.images.edit(
                model="gpt-image-1",
                image=image_file,
                prompt=user_input,
            )

        # Save and return the result (with optional JPEG conversion)
        output_id = uuid.uuid4().hex[:8]
        base_name = f"sculpture_{output_id}"
        output_path = save_generated_image(result.data[0].b64_json, app.config["UPLOAD_FOLDER"], base_name)
        output_filename = os.path.basename(output_path)

        print("Image generated successfully. Returning:", {"image_url": f"/static/uploads/{output_filename}"})
        return jsonify({"image_url": f"/static/uploads/{output_filename}"})

    except Exception as e:
        logging.exception("expand_chatbot failed")
        return jsonify({"response": f"Error expanding image: {str(e)}"}), 500
    
@app.route("/chatbot", methods=["POST"])
def chatbot():
    # Initialize conversation history if it doesn't exist
    if "conversation" not in session or not isinstance(session["conversation"], list):
        session["conversation"] = []

    # Limit conversation history size
    if len(session["conversation"]) > 5:
        session["conversation"] = session["conversation"][-5:]

    user_input = request.form.get("user_input", "").strip()
    uploaded_files = request.files.getlist("images")
    logging.debug(f"Received request.files: {request.files}")

    # Check for ice cube selection first - PROCESS AND RETURN EARLY
    selected_ice_cube = session.get('selected_ice_cube')
    
    if selected_ice_cube and uploaded_files:
        ice_prompt = ICE_CUBE_PROMPTS.get(selected_ice_cube, "")
        image_generation_prompt = f"{ice_prompt}\nUSER INPUT:\n{user_input}"
        print(f"Using {selected_ice_cube} ice cube prompt with user input:\n{image_generation_prompt}")
        session.pop('selected_ice_cube', None)
        
        # Save uploaded files and process ONLY with ice cube prompt
        uploaded_paths = []
        if uploaded_files:
            uploaded_ids = [uuid.uuid4().hex[:8] for _ in uploaded_files]
            uploaded_paths = [os.path.join(app.config["UPLOAD_FOLDER"], f"uploaded_{id}.jpg") for id in uploaded_ids]
            for i, file in enumerate(uploaded_files):
                file.stream.seek(0, os.SEEK_END)
                size = file.stream.tell(); file.stream.seek(0)
                if size > MAX_UPLOAD_BYTES:
                    return jsonify({"response": f"One of the files is too large (>{MAX_UPLOAD_BYTES//1024//1024}MB)"})
                file.save(uploaded_paths[i])
                optimize_saved_image(uploaded_paths[i])
                logging.debug(f"Saved & optimized uploaded file: {uploaded_paths[i]}")
        
        combined_id = uuid.uuid4().hex[:8]
        combined_path = os.path.join(app.config["UPLOAD_FOLDER"], f"combined_{combined_id}.jpg")

        if uploaded_paths:
            try:
                combine_images(uploaded_paths, combined_path)
                logging.debug(f"Combined image saved to: {combined_path}")
            except ValueError as e:
                return jsonify({"response": str(e)})
            except Exception as e:
                return jsonify({"response": f"Error combining images: {str(e)}"})
        else:
            return jsonify({"response": "No valid images provided"})
        
        # Generate image with ONLY ice cube prompt
        try:
            with open(combined_path, "rb") as image_file:
                result = client.images.edit(
                    model="gpt-image-1",
                    image=image_file,
                    prompt=image_generation_prompt,
                )
            output_id = uuid.uuid4().hex[:8]
            base_name = f"sculpture_{output_id}"
            output_path = save_generated_image(result.data[0].b64_json, app.config["UPLOAD_FOLDER"], base_name)
            output_filename = os.path.basename(output_path)
            print("Ice Cube Image Created")

            session["conversation"].append({"role": "user", "content": user_input})
            session["conversation"].append({
                "role": "assistant", 
                "content": "Here is your ice sculpture:",
                "image": f"/static/uploads/{output_filename}"
            })
            session.modified = True

            return jsonify({
                "image_url": f"/static/uploads/{output_filename}"
            })
        except Exception as e:
            return jsonify({"response": f"Error generating ice cube image: {str(e)}"})

    # Convert to lowercase after checking prefixes
    user_input_lower = user_input.lower()

    # Check for ludge in user input
    detected_ludge = detect_ludge_type(user_input_lower)

    if "ludge" in user_input_lower and not detected_ludge:
        session["conversation"].append({"role": "user", "content": user_input})
        response = "Can you please specify which ludge? We have martini ludge, tube ludge, and double ludge."
        session["conversation"].append({"role": "assistant", "content": response})
        session.modified = True
        return jsonify({"response": response})

    # Detect base images from user input
    base_images = detect_sculpture_bases(user_input_lower)

    if detected_ludge:
        base_images.append(detected_ludge)
        
    # If user uploaded images or we detected base images
    if uploaded_files or base_images:
        # Save any uploaded files
        uploaded_paths = []
        if uploaded_files:
            uploaded_ids = [uuid.uuid4().hex[:8] for _ in uploaded_files]
            uploaded_paths = [os.path.join(app.config["UPLOAD_FOLDER"], f"uploaded_{id}.jpg") for id in uploaded_ids]
            for i, file in enumerate(uploaded_files):
                file.save(uploaded_paths[i])
                optimize_saved_image(uploaded_paths[i])
                logging.debug(f"Saved & optimized uploaded file: {uploaded_paths[i]}")
        
        # Get full paths to any detected base images
        base_image_paths = [os.path.join("static", img) for img in base_images] if base_images else []
        
        combined_id = uuid.uuid4().hex[:8]
        combined_path = os.path.join(app.config["UPLOAD_FOLDER"], f"combined_{combined_id}.jpg")

        all_images = uploaded_paths + base_image_paths

        if all_images:
            try:
                combine_images(all_images, combined_path)
                logging.debug(f"Combined image saved to: {combined_path}")
            except ValueError as e:
                return jsonify({"response": str(e)})
            except Exception as e:
                return jsonify({"response": f"Error combining images: {str(e)}"})
        else:
            return jsonify({"response": "No valid images provided"})
        


        # Use the appropriate prompt based on prefix
        user_input_lower = user_input.lower()
       
        image_generation_dict = {
            "user_input": user_input,
            "Sculpture_instructions": {
                "sculpture_preservation": {
                    "shape": "Maintain EXACT shape, proportions, and details from the reference image",
                    "alterations": "Do NOT alter, add, or remove any elements of the sculpture",
                    "Extra_ice": "Do NOT ADD EXTRA ICE TO THE SCULPTURE, ONLY THE ORIGINAL IMAGE SHOULD BE USED",
                    "contours": "Preserve all original contours and features precisely",
                    "size": "Sculpture should be large, around 6 to 7 feet tall or wide accordingly",
                    "color_coding": "blue color in the image always means ice so make it of ice, where the light blue color means there is recess in the ice, and any other color means it is made of paper and not ice"
                },
                "material_properties": {
                    "rendering": "Render as crystal-clear, see-through ice with perfect transparency and zero cloudiness, no scratches or imperfections, just clear wet ice",
                    "lighting": "Include realistic light refraction and subtle imperfections",
                    "surface": "Surface should appear smooth and polished",
                    "ice_clarity": "Ice must be completely transparent, free of any bubbles, cloudiness, or impurities. The ice should appear as pure, pristine, see-through ice with perfect clarity"
                },
                "background_environment": {
                    "placement": "Place the sculpture on a wooden table",
                    "setting": "Environment should be realistic, preferably a country club"
                },
                "prohibited_modifications": [
                    "DO NOT ADD ANYTHING THAT IS NOT REQUESTED BY USER",
                    "NO changes to the sculpture structure",
                    "NO additional decorative elements",
                    "NO human figures or living creatures",
                    "NO elements detached from the sculpture even if requested",
                    "NO small details",
                    "NO foggy ice",
                    "NO cloudy ice - ice must be crystal clear and transparent",
                    "NO bubbles or impurities in the ice",
                    "NO extra ice base for the sculpture",
                    "NO changes in the sculpture itself allowed",
                    "DO NOT change the sculpture design",
                    "NO extra ice pieces on the sculpture",
                    "Place the sculpture directly on the table without extra ice base",
                    "DO NOT add any company logos (e.g., 'ice butcher, purveyors of perfect ice')",
                    "DO NOT add any text, labels, or words to the sculpture unless explicitly requested by user",
                    "DO NOT add any logos, brand names, or company names unless explicitly provided by user",
                    "DO NOT interpret visual elements as text or add text based on visual patterns",
                    "DO NOT add placeholder text, sample text, or any written content",
                    "ONLY add text or logos if the user explicitly uploads them or requests them in their input"
                ],
                "image_quality": "Always create an HD high-resolution image captured by a high-resolution camera",
                "sculpture_image_rules": {
                    "stickers": {
                        "condition": "If any detail in the sculpture is other than blue",
                        "appearance": "It should look like a colored paper sticker pasted on the ice, not made of ice"
                    }
                },
                "modular_components": {
                    "topper": {
                        "condition": "ONLY if the user explicitly requests a topper in their text input",
                        "instruction": "Place it on top of the sculpture, do NOT add the text 'TOPPER'",
                        "restriction": "Do NOT modify the base sculpture in any way. Do NOT add toppers unless explicitly requested by user."
                    },
                    "topper_with_logo": {
                        "condition": "ONLY if the user explicitly requests a topper with logo in their text input",
                        "instruction": "Place it on top of the sculpture with a centered placeholder logo, but do NOT add a logo unless provided",
                        "restriction": "Do NOT modify the base sculpture in any way. Do NOT add toppers or logos unless explicitly requested by user."
                    },
                    "base": {
                        "condition": "ONLY if the user explicitly requests a base in their text input",
                        "instruction": "Place it directly at the bottom of the sculpture, do NOT add the text 'BASE'",
                        "restriction": "Do NOT modify the base sculpture in any way. Do NOT add bases unless explicitly requested by user."
                    }
                },
                "reminders": [
                    "DO NOT add any text labels such as 'TOPPER', 'BASE', or 'TOPPER(WITH LOGO)' in the image",
                    "THE SCULPTURE MUST MATCH EXACTLY WITH THE ORIGINAL IMAGE WITHOUT ANY MODIFICATIONS",
                    "DO NOT add any text, words, or written content to the sculpture",
                    "DO NOT add any logos, brand names, or company names unless explicitly provided by user",
                    "DO NOT interpret any visual elements as text or add text based on what you think you see",
                    "ONLY add text or logos if the user explicitly uploads them or mentions them in their request"
                ]
            }          
}

        # Conditionally add logo instructions ONLY if user mentions specific terms
        if any(term in user_input_lower for term in ['snofilled', 'paper', 'colored']):
            logo_instructions = {
                "Snofilled": {
                    "effect": "Create a carved snow appearance inside the ice sculpture. The image should not be colored and should be engraved with visible depth inside the ice cube.",
                    "strict": "The logo must be embedded a few centimeters into the ice",
                    "processing": "Remove any background of the image before embedding it into the ice cube"
                },
                "Colored": {
                    "effect": "It should look like the ice is colored from the logo, not etched. The image appears as colored pigmentation embedded inside the ice.",
                    "strict": "The logo must be embedded a few centimeters into the ice",
                    "processing": "Remove any background of the image before embedding it into the ice cube"
                },
                "Paper": {
                    "effect": "It should look like a colored printed paper is frozen inside the ice cube. The logo should be colored, have a slight white outline, and a transparent background, and should be centered within the cube.",
                    "strict": "The logo must be embedded a few centimeters into the ice",
                    "processing": "Remove any background of the image before embedding it into the ice cube"
                }
            }
            
            # Determine which specific effect to use based on user input
            effect_type = None
            if 'snofilled' in user_input_lower:
                effect_type = "Snofilled"
            elif 'paper' in user_input_lower:
                effect_type = "Paper"
            elif 'colored' in user_input_lower:
                effect_type = "Colored"
            
            if effect_type:
                image_generation_dict["logo_instructions"] = {
                    "task": "add the image into the sculpture",
                    "effect_type": effect_type,
                    "instructions": logo_instructions[effect_type],
                    "clarification": "If a blue image is provided, it is always ice and should be used as the ice sculpture, not as a logo. Blue images are never logos or image overlays—they are the ice.",
                    "ice_structure": "The ice sculpture must precisely match the input image, with 100% accuracy. Do not add, remove, or modify any elements."
                }

        # Convert to string for the prompt
        image_generation_prompt = json.dumps(image_generation_dict, indent=2)
          
        if not image_generation_prompt:
            image_generation_prompt = json.dumps(image_generation_dict, indent=4)

        #Convert dictionary to a nicely formatted string prompt
        

        if 'template_selected_message' in session:
            image_generation_prompt += f"\n\nNOTE: {session['template_selected_message']}"
            # Clear the message after using it to avoid repetition
            session.pop('template_selected_message', None)
        
        try:
            with open(combined_path, "rb") as image_file:
                result = client.images.edit(
                    model="gpt-image-1",
                    image=image_file,
                    prompt=image_generation_prompt,
                )
            output_id = uuid.uuid4().hex[:8]
            base_name = f"sculpture_{output_id}"
            output_path = save_generated_image(result.data[0].b64_json, app.config["UPLOAD_FOLDER"], base_name)
            output_filename = os.path.basename(output_path)
            print("Image Created")

            session["conversation"].append({"role": "user", "content": user_input})
            session["conversation"].append({
                "role": "assistant", 
                "content": "Here is your ice sculpture:",
                "image": f"/static/uploads/{output_filename}"
            })
            session.modified = True

            return jsonify({
                # "response": "Here's your custom ice sculpture:",
                "image_url": f"/static/uploads/{output_filename}"
            })
        except Exception as e:
            return jsonify({"response": f"Error generating sculpture image: {str(e)}"})

    # If no images were uploaded or detected, proceed with text/classification flow
    try:
        classification = classify_prompt_type(user_input)
        print(f"Prompt classified as: {classification}")

        # Normal text conversation (with memory)
        if classification == "text":
            system_prompt = f"""You are an AI assistant for an ice sculpture company. You can create ice sculpture images 
            from text prompts using the latest GPT image generation model. You also accept image inputs for editing 
            based on user prompts. Never say you can't generate or edit images — just use the input prompt to create 
            a realistic ice sculpture.
            """
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            gpt_response = completion.choices[0].message.content
            session["conversation"].append({"role": "user", "content": user_input})
            session["conversation"].append({"role": "assistant", "content": gpt_response})
            return jsonify({"response": gpt_response})

        # Image generation flow
        elif classification == "generate":
            generation_id = uuid.uuid4().hex[:8]
            
            # Use the appropriate prompt based on prefix if not already set
            if not image_generation_prompt:
                image_generation_prompt = f"""
        "task": "Generate realistic images of ice engravings based solely on user text input.",
        "instructions": 
            "design": "Accurately follow the user's text description with no creative additions or modifications.",
            "ice_quality": "Use natural, crystal-clear ice with perfect transparency and zero cloudiness. No bubbles, rough edges, or imperfections. The ice must appear as pure, pristine, see-through ice with perfect clarity.",
            "surface": "Ice must appear smooth, clean, and polished.",
            "detail_level": "Keep details minimal; emphasize the natural beauty of ice.",
            "creativity": "This is a technical execution. No creative interpretation.",
            "environment": "Place the sculpture the requested theme, but do not change the look of the sculpture until requested explicitly",
            "small items":"do not add any items items that cannot be made by ice or that can be easily broken",
            "human_presence": "Do not include any humans in the image." 
            "user_input": {user_input}
    """
            result = client.images.generate(
                model="gpt-image-1",
                
                prompt=image_generation_prompt,
                
                
            )

            base_name = f"generated_{generation_id}"
            output_path = save_generated_image(result.data[0].b64_json, app.config["UPLOAD_FOLDER"], base_name)
            output_filename = os.path.basename(output_path)
            print("Image Created")

            session["conversation"].append({"role": "user", "content": user_input})
            session["conversation"].append({
                "role": "assistant", 
                "content": "Here is your ice sculpture:",
                "image": f"/static/uploads/{output_filename}"
            })
            session.modified = True

            return jsonify({
                "response": "Here is your ice sculpture:",
                "image_url": f"/static/uploads/{output_filename}"
            })

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

if __name__ == "__main__":
    # Development server ONLY.
    # In production, run with Gunicorn (example Procfile line):
    # web: gunicorn app:app --workers=1 --threads=4 --timeout=180
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug)
