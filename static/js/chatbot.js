
    // Template data organized by rows
    const templateCategories = [
        {
            name: "Ludges",
            items: [
                { type: "double ludge", name: "Double Ludge", image: "static/templates/ludge.png" },
                { type: "martini ludge", name: "Martini Ludge", image: "static/templates/martini.png" },
                { type: "tube ludge", name: "tube Ludge", image: "static/templates/tube.png" }

            ]
        },
        {
            name: "Topper Logos",
            items: [
                { type: "crown logo", name: "crown logo", image: "static/Topper/crown logo as topper.jpg" },
                { type: "Oval logo", name: "Oval logo", image: "static/Topper/Oval logo for as topper.jpg" },
                { type: "Swoosh logo", name: "Swoosh logo", image: "static/Topper/Swoosh logo.jpg" }

            ]
        },
       {
            name: "Toppers",
            items: [
                { type: "banana single luge", name: "banana single luge", image: "static/Topper/banana single luge.jpg" },
                { type: "ice bar mini single luge", name: "ice bar mini single luge", image: "static/Topper/ice bar mini single luge.jpg" },
                { type: "ice bowl", name: "ice bowl", image: "static/Topper/ice bowl.png" },
              

            ]
       },
       {
        name: "Ice Bars",
        items:[
            {type: "6ft Ice Bar", name: "6ft Ice Bar", image: "static/Ice bars/6ft ice bar.jpg"},
            {type: "8ft Ice Bar", name: "8ft Ice Bar", image: "static/Ice bars/8ft ice bar.jpg"}, 
            {type: "12ft Ice Bar ", name: "12ft Ice Bar", image: "static/Ice bars/12ft ice bar.jpg"}
        ]
       },
       {
        name: "Ice Cubes",
        items:[
            {type: "Snofilled", name: "Snofilled", image: "static/ice cubes/empty.png"},
            {type: "Colored", name: "Colored", image: "static/ice cubes/empty.png"}, 
            {type: "Paper", name: "Paper", image: "static/ice cubes/empty.png"}, 
            {type: "Snofilled+paper", name: "Snofilled+Paper", image: "static/ice cubes/empty.png"}
        ]
       }
    // ,
    //     {
    //         name: "Showpieces",
    //         items: [
    //             { type: "vase", name: "Vase", image: "static/templates/alligator_head.png" },
    //             { type: "logo", name: "Logo", image: "static/templates/alligator_head.png" },
    //             { type: "abstract", name: "Abstract", image: "static/templates/alligator_head.png" }
    //         ]
    //     },
    //     {
    //         name: "Seafood Displays",
    //         items: [
    //             { type: "wedding", name: "Wedding", image: "static/templates/alligator_head.png" },
    //             { type: "custom", name: "Custom", image: "static/templates/alligator_head.png" }
    //         ]
    //     }
    ];

    // Function to render template categories
    function renderTemplateCategories() {
        const dropupContent = document.getElementById('template-dropup');
        dropupContent.innerHTML = ''; // Clear existing content

        // Create columns first
        const columns = {};
// In your renderTemplateCategories function, modify how the category title is created
            templateCategories.forEach(category => {
                columns[category.name] = document.createElement('div');
                columns[category.name].className = 'dropup-column';

                // Create container div for the title
                const titleContainer = document.createElement('div');
                titleContainer.className = 'dropup-category-title-container';

                // Add Category Title to its container
                const categoryTitle = document.createElement('div');
                categoryTitle.className = 'dropup-category-title';
                categoryTitle.textContent = category.name;
                
                // Add title to container
                titleContainer.appendChild(categoryTitle);
                
                // Add container to column
                columns[category.name].appendChild(titleContainer);
            });

        // Add items to their respective columns
        templateCategories.forEach(category => {
            category.items.forEach(item => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'dropup-item';
                itemDiv.onclick = () => selectTemplate(item.type);

                itemDiv.innerHTML = `
                    <img src="${item.image}" alt="${item.name}">
                    <span>${item.name}</span>
                `;

                columns[category.name].appendChild(itemDiv);
            });
        });

        // Append all columns to the dropup content
        for (const categoryName in columns) {
            dropupContent.appendChild(columns[categoryName]);
        }
    }

    let dropupOpen = false; // Track if the dropup is currently open.

// Add this variable at the top of your script to track the current ice cube selection
let currentIceCubeSelection = null;


// Modify the selectTemplate function (only adding ice cube logic)
// Modify the selectTemplate function
function selectTemplate(templateType) {
    // Find the selected template
    let selectedTemplate = null;
    for (const category of templateCategories) {
        const found = category.items.find(item => item.type === templateType);
        if (found) {
            selectedTemplate = found;
            break;
        }
    }

    if (selectedTemplate) {
        const isIceCube = templateCategories.find(cat => cat.name === "Ice Cubes")?.items.some(item => item.type === templateType);
        const inputField = document.getElementById('user-input');
        
        // Clear ice cube tag if selecting non-ice-cube template
        if (!isIceCube && currentIceCubeSelection) {
            const oldTag = `#${currentIceCubeSelection.toUpperCase()}`;
            inputField.value = inputField.value.replace(oldTag, '').trim();
            currentIceCubeSelection = null;
        }
        
        // Handle ice cube selection
        if (isIceCube) {
            // If clicking same ice cube again, deselect it
            if (currentIceCubeSelection === selectedTemplate.type) {
                currentIceCubeSelection = null;
                inputField.value = inputField.value.replace(`#${selectedTemplate.type.toUpperCase()}`, '').trim();
            } 
            // If selecting different ice cube
            else {
                // Remove any existing ice cube tag
                if (currentIceCubeSelection) {
                    const oldTag = `#${currentIceCubeSelection.toUpperCase()}`;
                    inputField.value = inputField.value.replace(oldTag, '').trim();
                }
                
                // Add new ice cube tag
                currentIceCubeSelection = selectedTemplate.type;
                const newTag = `#${selectedTemplate.type.toUpperCase()}`;
                
                // Add tag if not already present
                if (!inputField.value.includes(newTag)) {
                    inputField.value = (inputField.value.trim() + ' ' + newTag).trim();
                }
            }
        }

        // ORIGINAL TEMPLATE SELECTION LOGIC (unchanged)
        fetch(selectedTemplate.image)
            .then(res => res.blob())
            .then(blob => {
                const file = new File([blob], `${selectedTemplate.type}_template.png`, {
                    type: 'image/png',
                    lastModified: Date.now()
                });

                selectedFiles = [file];
                updateInputFiles();
                previewImage();
                addChatMessage(`You selected the ${selectedTemplate.name} template`, 'bot');
                document.querySelector('.dropup-content').style.display = 'none';
                dropupOpen = false;
            })
            .catch(error => {
                console.error('Error loading template:', error);
                addChatMessage('Failed to load the selected template', 'bot');
            });
    }
}
// Add this to prevent deleting the ice cube tag
document.getElementById('user-input').addEventListener('keydown', function(e) {
    if (!currentIceCubeSelection) return;
    
    const tag = `#${currentIceCubeSelection.toUpperCase()}`;
    const currentValue = this.value;
    const tagStart = currentValue.indexOf(tag);
    
    // If tag doesn't exist or user isn't deleting, do nothing
    if (tagStart === -1 || !(e.key === 'Backspace' || e.key === 'Delete')) return;
    
    // Check if cursor is inside the tag
    const cursorPos = this.selectionStart;
    if (cursorPos > tagStart && cursorPos <= tagStart + tag.length) {
        e.preventDefault();
    }
});
// New function to handle ice cube selections
function handleIceCubeSelection(selectedTemplate) {
    const inputField = document.getElementById('user-input');
    
    // If clicking the same ice cube again, deselect it
    if (currentIceCubeSelection === selectedTemplate.type) {
        currentIceCubeSelection = null;
        inputField.value = inputField.value.replace(`#${selectedTemplate.type.toUpperCase()}`, '').trim();
        addChatMessage(`Deselected ${selectedTemplate.name} ice cube`, 'bot');
    } 
    // If selecting a different ice cube
    else {
        // Remove any existing ice cube tag
        if (currentIceCubeSelection) {
            const oldTag = `#${currentIceCubeSelection.toUpperCase()}`;
            inputField.value = inputField.value.replace(oldTag, '').trim();
        }
        
        // Add the new ice cube tag
        currentIceCubeSelection = selectedTemplate.type;
        const newTag = `#${selectedTemplate.type.toUpperCase()}`;
        
        // Add the tag to the input field if it's not already there
        if (!inputField.value.includes(newTag)) {
            inputField.value = (inputField.value.trim() + ' ' + newTag).trim();
        }
        
        addChatMessage(`Selected ${selectedTemplate.name} ice cube`, 'bot');
    }
    
    // Close the dropup menu
    document.querySelector('.dropup-content').style.display = 'none';
    dropupOpen = false;
}

// Add event listener to prevent deleting the ice cube tag
document.getElementById('user-input').addEventListener('keydown', function(e) {
    if (!currentIceCubeSelection) return;
    
    const tag = `#${currentIceCubeSelection.toUpperCase()}`;
    const currentValue = this.value;
    const tagStart = currentValue.indexOf(tag);
    const tagEnd = tagStart + tag.length;
    
    // If tag doesn't exist in input, do nothing
    if (tagStart === -1) return;
    
    // Check cursor/selection position relative to tag
    const cursorPos = this.selectionStart;
    const selectionEnd = this.selectionEnd;
    
    // Check if user is trying to delete part of the tag
    const isDeleting = (e.key === 'Backspace' || e.key === 'Delete');
    const isInsideTag = (
        (cursorPos > tagStart && cursorPos < tagEnd) || // Cursor inside tag
        (selectionEnd > tagStart && cursorPos < tagEnd) // Selection overlaps tag
    );
    
    if (isDeleting && isInsideTag) {
        e.preventDefault();
    }
});



    // Initialize the template menu when the page loads
    document.addEventListener('DOMContentLoaded', renderTemplateCategories);

    // JavaScript function to toggle the chatbot visibility
    function toggleChatbot() {
        var chatbotContainer = document.getElementById('chatbot-container');
        if (chatbotContainer.style.display === 'none' || chatbotContainer.style.display === '') {
            chatbotContainer.style.display = 'flex';
            renderTemplateCategories(); // Make sure to render the templates when the chatbot opens
        } else {
            chatbotContainer.style.display = 'none';
        }
    }

    document.addEventListener('click', function(event) {
    const dropupContent = document.querySelector('.dropup-content');
    const dropupButton = document.getElementById('start-prompt-button');
    
    if (dropupOpen && 
        !dropupContent.contains(event.target) && 
        !dropupButton.contains(event.target)) {
        dropupContent.style.display = 'none';
        dropupOpen = false;
    }
});
