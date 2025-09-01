/**
 * Interactive Comic Editor
 * Allows dragging speech bubbles and editing text
 */

class ComicEditor {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.bubbles = [];
        this.selectedBubble = null;
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };
        this.isEditing = false;
        
        this.init();
    }
    
    init() {
        // Add editor styles
        this.addStyles();
        
        // Load comic data
        this.loadComicData();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Add toolbar
        this.createToolbar();
    }
    
    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .comic-editor-container {
                position: relative;
                user-select: none;
                background: #f0f0f0;
                padding: 20px;
                border-radius: 10px;
            }
            
            .comic-page {
                position: relative;
                background: white;
                margin: 20px auto;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            }
            
            .comic-panel {
                position: absolute;
                border: 2px solid #333;
                overflow: hidden;
                background: white;
            }
            
            .comic-panel img {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
            
            .speech-bubble {
                position: absolute;
                background: white;
                border: 3px solid #333;
                border-radius: 20px;
                padding: 15px;
                cursor: move;
                min-width: 100px;
                min-height: 50px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                transition: transform 0.1s;
                z-index: 10;
            }
            
            .speech-bubble:hover {
                transform: scale(1.02);
                box-shadow: 4px 4px 10px rgba(0,0,0,0.2);
            }
            
            .speech-bubble.selected {
                border-color: #007bff;
                box-shadow: 0 0 0 3px rgba(0,123,255,0.3);
                z-index: 100;
            }
            
            .speech-bubble.dragging {
                opacity: 0.8;
                z-index: 1000;
            }
            
            .bubble-text {
                font-family: 'Comic Sans MS', cursive;
                font-size: 14px;
                font-weight: bold;
                text-align: center;
                line-height: 1.4;
                color: #000;
                word-wrap: break-word;
                cursor: text;
            }
            
            .bubble-text.editing {
                background: rgba(255,255,255,0.9);
                border: 1px dashed #007bff;
                padding: 5px;
                outline: none;
            }
            
            .bubble-tail {
                position: absolute;
                bottom: -15px;
                left: 20px;
                width: 0;
                height: 0;
                border-left: 15px solid transparent;
                border-right: 5px solid transparent;
                border-top: 20px solid #333;
                transform: rotate(-20deg);
            }
            
            .bubble-tail::after {
                content: '';
                position: absolute;
                bottom: 3px;
                left: -12px;
                width: 0;
                height: 0;
                border-left: 12px solid transparent;
                border-right: 4px solid transparent;
                border-top: 16px solid white;
            }
            
            .editor-toolbar {
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                border: 2px solid #333;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                z-index: 1000;
            }
            
            .toolbar-btn {
                display: block;
                width: 100%;
                padding: 10px 15px;
                margin: 5px 0;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                transition: background 0.2s;
            }
            
            .toolbar-btn:hover {
                background: #0056b3;
            }
            
            .toolbar-btn.danger {
                background: #dc3545;
            }
            
            .toolbar-btn.danger:hover {
                background: #c82333;
            }
            
            .toolbar-btn.success {
                background: #28a745;
            }
            
            .toolbar-btn.success:hover {
                background: #218838;
            }
            
            .resize-handle {
                position: absolute;
                width: 10px;
                height: 10px;
                background: #007bff;
                border: 1px solid white;
                border-radius: 50%;
                cursor: nwse-resize;
            }
            
            .resize-handle.se {
                bottom: -5px;
                right: -5px;
            }
            
            .coordinates {
                position: absolute;
                bottom: -25px;
                left: 0;
                font-size: 10px;
                color: #666;
                background: white;
                padding: 2px 5px;
                border-radius: 3px;
                display: none;
            }
            
            .selected .coordinates {
                display: block;
            }
            
            .edit-hint {
                position: fixed;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: #333;
                color: white;
                padding: 10px 20px;
                border-radius: 20px;
                font-size: 14px;
                z-index: 1000;
                opacity: 0;
                transition: opacity 0.3s;
            }
            
            .edit-hint.show {
                opacity: 1;
            }
        `;
        document.head.appendChild(style);
    }
    
    loadComicData() {
        // Load existing comic data or create new
        const savedData = localStorage.getItem('comicEditorData');
        if (savedData) {
            const data = JSON.parse(savedData);
            this.renderComic(data);
        } else {
            // Load from server or create default
            this.loadFromServer();
        }
    }
    
    loadFromServer() {
        // Load from server
        fetch('/load_comic')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error('Error loading comic:', data.error);
                    this.createDefaultComic();
                } else {
                    this.renderComic(data);
                }
            })
            .catch(error => {
                console.error('Failed to load comic:', error);
                this.createDefaultComic();
            });
    }
    
    createDefaultComic() {
        // Create a default comic if loading fails
        const sampleData = {
            pages: [{
                width: 800,
                height: 600,
                panels: [
                    {
                        x: 10, y: 10, width: 380, height: 280,
                        image: '/frames/frame000.png'
                    },
                    {
                        x: 410, y: 10, width: 380, height: 280,
                        image: '/frames/frame001.png'
                    }
                ],
                bubbles: [
                    {
                        id: 'bubble1',
                        x: 50, y: 50, width: 150, height: 60,
                        text: 'Add your text here!',
                        panelIndex: 0
                    }
                ]
            }]
        };
        
        this.renderComic(sampleData);
    }
    
    renderComic(data) {
        this.container.innerHTML = '';
        this.container.className = 'comic-editor-container';
        
        data.pages.forEach((page, pageIndex) => {
            const pageDiv = document.createElement('div');
            pageDiv.className = 'comic-page';
            pageDiv.style.width = page.width + 'px';
            pageDiv.style.height = page.height + 'px';
            pageDiv.dataset.pageIndex = pageIndex;
            
            // Render panels
            page.panels.forEach((panel, panelIndex) => {
                const panelDiv = document.createElement('div');
                panelDiv.className = 'comic-panel';
                panelDiv.style.left = panel.x + 'px';
                panelDiv.style.top = panel.y + 'px';
                panelDiv.style.width = panel.width + 'px';
                panelDiv.style.height = panel.height + 'px';
                panelDiv.dataset.panelIndex = panelIndex;
                
                const img = document.createElement('img');
                img.src = panel.image;
                panelDiv.appendChild(img);
                
                pageDiv.appendChild(panelDiv);
            });
            
            // Render bubbles
            page.bubbles.forEach(bubble => {
                this.createBubble(bubble, pageDiv);
            });
            
            this.container.appendChild(pageDiv);
        });
    }
    
    createBubble(bubbleData, pageDiv) {
        const bubble = document.createElement('div');
        bubble.className = 'speech-bubble';
        bubble.id = bubbleData.id || 'bubble_' + Date.now();
        bubble.style.left = bubbleData.x + 'px';
        bubble.style.top = bubbleData.y + 'px';
        bubble.style.width = bubbleData.width + 'px';
        bubble.style.height = bubbleData.height + 'px';
        
        // Add text
        const text = document.createElement('div');
        text.className = 'bubble-text';
        text.textContent = bubbleData.text || 'Click to edit';
        text.contentEditable = false;
        bubble.appendChild(text);
        
        // Add tail
        const tail = document.createElement('div');
        tail.className = 'bubble-tail';
        bubble.appendChild(tail);
        
        // Add resize handle
        const resizeHandle = document.createElement('div');
        resizeHandle.className = 'resize-handle se';
        bubble.appendChild(resizeHandle);
        
        // Add coordinates display
        const coords = document.createElement('div');
        coords.className = 'coordinates';
        bubble.appendChild(coords);
        
        // Store data
        bubble.dataset.bubbleData = JSON.stringify(bubbleData);
        
        pageDiv.appendChild(bubble);
        this.bubbles.push(bubble);
        
        // Setup bubble events
        this.setupBubbleEvents(bubble);
    }
    
    setupEventListeners() {
        // Document-wide mouse events
        document.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        document.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Delete' && this.selectedBubble && !this.isEditing) {
                this.deleteBubble(this.selectedBubble);
            }
            if (e.key === 'Escape') {
                this.deselectBubble();
            }
        });
        
        // Click outside to deselect
        this.container.addEventListener('click', (e) => {
            if (e.target === this.container || e.target.classList.contains('comic-page')) {
                this.deselectBubble();
            }
        });
    }
    
    setupBubbleEvents(bubble) {
        const text = bubble.querySelector('.bubble-text');
        const resizeHandle = bubble.querySelector('.resize-handle');
        
        // Drag start
        bubble.addEventListener('mousedown', (e) => {
            if (e.target === text && this.isEditing) return;
            if (e.target === resizeHandle) return;
            
            this.startDragging(bubble, e);
        });
        
        // Click to select
        bubble.addEventListener('click', (e) => {
            e.stopPropagation();
            this.selectBubble(bubble);
        });
        
        // Double-click to edit text
        text.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            this.startEditingText(bubble, text);
        });
        
        // Handle text editing
        text.addEventListener('blur', () => {
            if (this.isEditing) {
                this.stopEditingText(bubble, text);
            }
        });
        
        text.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                text.blur();
            }
        });
        
        // Resize handle
        resizeHandle.addEventListener('mousedown', (e) => {
            e.stopPropagation();
            this.startResizing(bubble, e);
        });
    }
    
    startDragging(bubble, e) {
        this.isDragging = true;
        this.selectedBubble = bubble;
        bubble.classList.add('dragging');
        
        const rect = bubble.getBoundingClientRect();
        const containerRect = this.container.getBoundingClientRect();
        
        this.dragOffset = {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
        
        this.selectBubble(bubble);
    }
    
    handleMouseMove(e) {
        if (!this.isDragging || !this.selectedBubble) return;
        
        const containerRect = this.container.getBoundingClientRect();
        const pageRect = this.selectedBubble.parentElement.getBoundingClientRect();
        
        let newX = e.clientX - pageRect.left - this.dragOffset.x;
        let newY = e.clientY - pageRect.top - this.dragOffset.y;
        
        // Constrain to page bounds
        const maxX = pageRect.width - this.selectedBubble.offsetWidth;
        const maxY = pageRect.height - this.selectedBubble.offsetHeight;
        
        newX = Math.max(0, Math.min(newX, maxX));
        newY = Math.max(0, Math.min(newY, maxY));
        
        this.selectedBubble.style.left = newX + 'px';
        this.selectedBubble.style.top = newY + 'px';
        
        this.updateCoordinates(this.selectedBubble);
    }
    
    handleMouseUp(e) {
        if (this.isDragging && this.selectedBubble) {
            this.selectedBubble.classList.remove('dragging');
            this.isDragging = false;
            this.saveBubblePosition(this.selectedBubble);
        }
    }
    
    selectBubble(bubble) {
        // Deselect previous
        this.deselectBubble();
        
        // Select new
        this.selectedBubble = bubble;
        bubble.classList.add('selected');
        
        this.updateCoordinates(bubble);
        this.showHint('Double-click to edit text • Drag to move • Delete key to remove');
    }
    
    deselectBubble() {
        if (this.selectedBubble) {
            this.selectedBubble.classList.remove('selected');
            this.selectedBubble = null;
        }
        this.hideHint();
    }
    
    startEditingText(bubble, textElement) {
        this.isEditing = true;
        textElement.contentEditable = true;
        textElement.classList.add('editing');
        textElement.focus();
        
        // Select all text
        const range = document.createRange();
        range.selectNodeContents(textElement);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        
        this.showHint('Press Enter to save • Shift+Enter for new line');
    }
    
    stopEditingText(bubble, textElement) {
        this.isEditing = false;
        textElement.contentEditable = false;
        textElement.classList.remove('editing');
        
        // Save the text
        this.saveBubbleText(bubble, textElement.textContent);
        this.hideHint();
    }
    
    deleteBubble(bubble) {
        if (confirm('Delete this speech bubble?')) {
            bubble.remove();
            const index = this.bubbles.indexOf(bubble);
            if (index > -1) {
                this.bubbles.splice(index, 1);
            }
            this.selectedBubble = null;
            this.saveComicData();
        }
    }
    
    updateCoordinates(bubble) {
        const coords = bubble.querySelector('.coordinates');
        coords.textContent = `x: ${parseInt(bubble.style.left)}, y: ${parseInt(bubble.style.top)}`;
    }
    
    createToolbar() {
        const toolbar = document.createElement('div');
        toolbar.className = 'editor-toolbar';
        
        // Add bubble button
        const addBtn = document.createElement('button');
        addBtn.className = 'toolbar-btn';
        addBtn.textContent = '➕ Add Bubble';
        addBtn.onclick = () => this.addNewBubble();
        toolbar.appendChild(addBtn);
        
        // Save button
        const saveBtn = document.createElement('button');
        saveBtn.className = 'toolbar-btn success';
        saveBtn.textContent = '💾 Save Comic';
        saveBtn.onclick = () => this.saveComic();
        toolbar.appendChild(saveBtn);
        
        // Export button
        const exportBtn = document.createElement('button');
        exportBtn.className = 'toolbar-btn';
        exportBtn.textContent = '📥 Export';
        exportBtn.onclick = () => this.exportComic();
        toolbar.appendChild(exportBtn);
        
        // Reset button
        const resetBtn = document.createElement('button');
        resetBtn.className = 'toolbar-btn danger';
        resetBtn.textContent = '🔄 Reset';
        resetBtn.onclick = () => this.resetComic();
        toolbar.appendChild(resetBtn);
        
        document.body.appendChild(toolbar);
    }
    
    addNewBubble() {
        const page = this.container.querySelector('.comic-page');
        if (!page) return;
        
        const newBubble = {
            id: 'bubble_' + Date.now(),
            x: 100,
            y: 100,
            width: 150,
            height: 60,
            text: 'New bubble!'
        };
        
        this.createBubble(newBubble, page);
        this.saveComicData();
    }
    
    saveBubblePosition(bubble) {
        this.saveComicData();
    }
    
    saveBubbleText(bubble, text) {
        const data = JSON.parse(bubble.dataset.bubbleData || '{}');
        data.text = text;
        bubble.dataset.bubbleData = JSON.stringify(data);
        this.saveComicData();
    }
    
    saveComicData() {
        const data = {
            pages: []
        };
        
        this.container.querySelectorAll('.comic-page').forEach(page => {
            const pageData = {
                width: parseInt(page.style.width),
                height: parseInt(page.style.height),
                panels: [],
                bubbles: []
            };
            
            // Save panel data
            page.querySelectorAll('.comic-panel').forEach(panel => {
                pageData.panels.push({
                    x: parseInt(panel.style.left),
                    y: parseInt(panel.style.top),
                    width: parseInt(panel.style.width),
                    height: parseInt(panel.style.height),
                    image: panel.querySelector('img').src
                });
            });
            
            // Save bubble data
            page.querySelectorAll('.speech-bubble').forEach(bubble => {
                pageData.bubbles.push({
                    id: bubble.id,
                    x: parseInt(bubble.style.left),
                    y: parseInt(bubble.style.top),
                    width: parseInt(bubble.style.width),
                    height: parseInt(bubble.style.height),
                    text: bubble.querySelector('.bubble-text').textContent
                });
            });
            
            data.pages.push(pageData);
        });
        
        localStorage.setItem('comicEditorData', JSON.stringify(data));
        this.showHint('Comic saved!');
    }
    
    saveComic() {
        this.saveComicData();
        
        // Send to server
        fetch('/save_comic', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(this.getComicData())
        })
        .then(response => response.json())
        .then(data => {
            this.showHint('Comic saved to server!');
        })
        .catch(error => {
            console.error('Error:', error);
            this.showHint('Error saving to server!');
        });
    }
    
    exportComic() {
        const data = this.getComicData();
        const json = JSON.stringify(data, null, 2);
        
        // Create download
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'comic_data.json';
        a.click();
        URL.revokeObjectURL(url);
        
        this.showHint('Comic exported!');
    }
    
    resetComic() {
        if (confirm('Reset all changes? This cannot be undone!')) {
            localStorage.removeItem('comicEditorData');
            this.loadFromServer();
            this.showHint('Comic reset!');
        }
    }
    
    getComicData() {
        return JSON.parse(localStorage.getItem('comicEditorData') || '{}');
    }
    
    showHint(message) {
        let hint = document.querySelector('.edit-hint');
        if (!hint) {
            hint = document.createElement('div');
            hint.className = 'edit-hint';
            document.body.appendChild(hint);
        }
        
        hint.textContent = message;
        hint.classList.add('show');
        
        clearTimeout(this.hintTimeout);
        this.hintTimeout = setTimeout(() => {
            hint.classList.remove('show');
        }, 3000);
    }
    
    hideHint() {
        const hint = document.querySelector('.edit-hint');
        if (hint) {
            hint.classList.remove('show');
        }
    }
    
    startResizing(bubble, e) {
        e.preventDefault();
        
        const startX = e.clientX;
        const startY = e.clientY;
        const startWidth = parseInt(bubble.style.width);
        const startHeight = parseInt(bubble.style.height);
        
        const handleResize = (e) => {
            const newWidth = startWidth + (e.clientX - startX);
            const newHeight = startHeight + (e.clientY - startY);
            
            bubble.style.width = Math.max(100, newWidth) + 'px';
            bubble.style.height = Math.max(50, newHeight) + 'px';
            
            this.updateCoordinates(bubble);
        };
        
        const stopResize = () => {
            document.removeEventListener('mousemove', handleResize);
            document.removeEventListener('mouseup', stopResize);
            this.saveComicData();
        };
        
        document.addEventListener('mousemove', handleResize);
        document.addEventListener('mouseup', stopResize);
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('comic-editor')) {
        window.comicEditor = new ComicEditor('comic-editor');
    }
});