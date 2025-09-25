path = '../frames/final/'
current_page = 0

function placeDialogs(page) {
    var gridItems = document.querySelectorAll('.grid-item');
    var MAX_PANELS = 4; // Force 2x2 grid
    var panels = (page.panels || []).slice(0, MAX_PANELS);

    // First, clear all items and hide by default
    for (var j = 0; j < gridItems.length; j++) {
        var gi = gridItems[j];
        gi.style.display = 'none';
        gi.style.gridRow = '';
        gi.style.gridColumn = '';
        gi.style.backgroundImage = '';
        gi.innerHTML = '';
    }

    // Render only first 4 panels to guarantee 2x2 with zero gap
    panels.forEach(function (panel, index) {
        var gridItem = gridItems[index];
        if (!gridItem) return;

        gridItem.style.display = 'flex';
        gridItem.style.backgroundImage = `url("${path}${panel.image}.png")`;

        // Reset content
        gridItem.innerHTML = "";

        // Add speech bubble if present and not an action scene
        var bubbleData = (page['bubbles'] || [])[index];
        if (!bubbleData) return;

        const dialog_temp = bubbleData['dialog'];
        if (dialog_temp == "((action-scene))") return;

        const wrapper = document.createElement('div');
        wrapper.style.position = 'relative';
        wrapper.style.width = '100%';
        wrapper.style.height = '100%';

        const bubble_temp = document.createElement('div');
        bubble_temp.classList.add('bubble');
        bubble_temp.innerHTML = dialog_temp;

        const emotion = bubbleData['emotion'];
        if (emotion == 'jagged') {
            bubble_temp.style.backgroundImage = `url("assets/jagged.png")`;
            bubble_temp.style.backgroundPosition = 'center center';
            bubble_temp.style.backgroundRepeat = 'no-repeat';
            bubble_temp.style.backgroundSize = 'cover';
            bubble_temp.style.backgroundColor = 'transparent';
            bubble_temp.style.width = '200px';
            bubble_temp.style.height = '94px';
            bubble_temp.style.padding = '70px';
        }

        // Keep existing offset logic
        bubble_temp.style.fontSize = dialog_temp.length;
        bubble_temp.style.transform = `translate(${bubbleData['bubble_offset_x']}px, ${bubbleData['bubble_offset_y']}px)`;

        const tail = document.createElement('div');
        tail.classList.add('tail');
        if (bubbleData['tail_offset_x'] == null || emotion == 'jagged') {
            tail.style.display = 'none';
        } else {
            tail.style.transform = `translate(${bubbleData['tail_offset_x']}px, ${bubbleData['tail_offset_y']}px) rotate(${bubbleData['tail_deg']}deg)`;
        }

        bubble_temp.appendChild(tail);
        wrapper.appendChild(bubble_temp);
        gridItem.appendChild(wrapper);
    });
}

document.addEventListener('DOMContentLoaded', function() {
    placeDialogs(pages[current_page]);
});

function prevPage(){
    current_page = (current_page - 1);
    if(current_page < 0){
        current_page = pages.length - 1;
    }
    placeDialogs(pages[current_page]);
}

function nextPage(){
    current_page = (current_page + 1) % pages.length;
    placeDialogs(pages[current_page]);
}

