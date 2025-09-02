// Simple Comic Editor for text editing and bubble dragging
let draggedBubble = null;
let offsetX = 0;
let offsetY = 0;

function enableComicEditing() {
    // Make all bubbles editable and draggable
    document.querySelectorAll('.bubble').forEach(bubble => {
        // Make bubble draggable
        bubble.style.cursor = 'move';
        bubble.draggable = false; // Use custom drag
        
        // Double-click to edit text
        bubble.addEventListener('dblclick', function(e) {
            e.stopPropagation();
            editBubbleText(this);
        });
        
        // Mouse down to start dragging
        bubble.addEventListener('mousedown', startDrag);
    });
    
    // Global mouse events for dragging
    document.addEventListener('mousemove', drag);
    document.addEventListener('mouseup', stopDrag);
    
    // Add editing instructions
    addEditingInstructions();
}

function editBubbleText(bubble) {
    const currentText = bubble.innerText;
    const input = document.createElement('input');
    input.type = 'text';
    input.value = currentText;
    input.style.cssText = bubble.style.cssText;
    input.style.width = '100%';
    input.style.background = 'white';
    input.style.border = '2px solid #4CAF50';
    
    bubble.innerHTML = '';
    bubble.appendChild(input);
    input.focus();
    input.select();
    
    // Save on Enter
    input.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            bubble.innerText = this.value;
            saveComicState();
        }
    });
    
    // Save on blur
    input.addEventListener('blur', function() {
        bubble.innerText = this.value;
        saveComicState();
    });
}

function startDrag(e) {
    if (e.target.tagName === 'INPUT') return;
    
    draggedBubble = e.target.closest('.bubble');
    if (!draggedBubble) return;
    
    const rect = draggedBubble.getBoundingClientRect();
    offsetX = e.clientX - rect.left;
    offsetY = e.clientY - rect.top;
    
    draggedBubble.style.opacity = '0.8';
    draggedBubble.style.zIndex = '1000';
    e.preventDefault();
}

function drag(e) {
    if (!draggedBubble) return;
    
    const parent = draggedBubble.parentElement;
    const parentRect = parent.getBoundingClientRect();
    
    let x = e.clientX - parentRect.left - offsetX;
    let y = e.clientY - parentRect.top - offsetY;
    
    // Keep within bounds
    x = Math.max(0, Math.min(x, parentRect.width - draggedBubble.offsetWidth));
    y = Math.max(0, Math.min(y, parentRect.height - draggedBubble.offsetHeight));
    
    draggedBubble.style.position = 'absolute';
    draggedBubble.style.left = x + 'px';
    draggedBubble.style.top = y + 'px';
}

function stopDrag() {
    if (draggedBubble) {
        draggedBubble.style.opacity = '';
        draggedBubble.style.zIndex = '';
        saveComicState();
        draggedBubble = null;
    }
}

function addEditingInstructions() {
    const instructions = document.createElement('div');
    instructions.style.cssText = `
        position: fixed; bottom: 20px; right: 20px;
        background: rgba(0,0,0,0.8); color: white;
        padding: 15px; border-radius: 10px;
        font-size: 14px; z-index: 999;
    `;
    instructions.innerHTML = `
        <strong>✏️ Edit Mode</strong><br>
        • Drag bubbles to move<br>
        • Double-click to edit text
    `;
    document.body.appendChild(instructions);
}

function saveComicState() {
    // Save state to localStorage
    const bubbles = [];
    document.querySelectorAll('.bubble').forEach(bubble => {
        bubbles.push({
            text: bubble.innerText,
            left: bubble.style.left,
            top: bubble.style.top
        });
    });
    localStorage.setItem('comicBubbles', JSON.stringify(bubbles));
}

// Initialize when page loads
window.addEventListener('load', () => {
    setTimeout(enableComicEditing, 500);
});