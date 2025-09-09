# 🎮 Unity Comic Generator

A comprehensive comic generation system that creates 48 pages with 2x2 grid layout, optimized for Unity integration with interactive speech bubbles.

## ✨ Features

### 📄 **48 Pages with 2x2 Grid Layout**
- **4 panels per page** in a 2x2 grid arrangement
- **800x540 pixels per panel** for high quality
- **1600x1080 pixels per page** total resolution
- **192 total panels** across all pages

### 🖼️ **High-Quality PNG Output**
- **No compression** for maximum quality
- **Perfect for Unity import** as sprites
- **Consistent dimensions** across all panels
- **No cropping** - images are resized with padding

### 💬 **Interactive Speech Bubbles**
- **Drag to move** bubbles anywhere on the page
- **Double-click to edit** bubble text
- **Real-time updates** to page data
- **Print-ready** bubble positioning

### 🖨️ **Print & Export Features**
- **Print individual pages** with proper formatting
- **Export all pages as PNG** for Unity
- **High-resolution output** for professional use
- **Unity-optimized file structure**

## 🚀 Quick Start

### 1. **Web Interface**
1. Upload a video file or paste a YouTube link
2. Click **"🎮 Generate Unity Comic (48 Pages)"**
3. Wait for processing to complete
4. Open the interactive viewer to edit bubbles
5. Print or export pages as needed

### 2. **Command Line**
```bash
# Test mode (creates sample data)
python3 run_unity_comic.py --test

# Generate from video
python3 run_unity_comic.py --video path/to/video.mp4

# Custom number of pages
python3 run_unity_comic.py --pages 24 --video path/to/video.mp4
```

### 3. **Direct Integration**
```python
from unity_comic_generator import UnityComicGenerator

generator = UnityComicGenerator()
success = generator.generate_48_pages_comic('video/uploaded.mp4')
```

## 📁 Output Structure

```
output/unity_pages/
├── interactive_viewer.html      # Interactive web viewer
├── pages_data.json             # Page and bubble data
├── UNITY_INTEGRATION.md        # Unity setup guide
├── page_001.png               # Individual page PNGs
├── page_002.png
├── ...
└── page_048.png
```

## 🎮 Unity Integration

### **Import Settings**
1. Import PNG files into Unity as **Sprites**
2. Set **Texture Type**: Sprite (2D and UI)
3. Set **Sprite Mode**: Single
4. Set **Pixels Per Unit**: 100
5. Set **Filter Mode**: Point (no filter)
6. Set **Compression**: None (for best quality)

### **UI Setup**
1. Create **UI Canvas** for comic display
2. Use **Image components** to display pages
3. Implement **page navigation system**
4. Add **speech bubble UI** using saved position data

### **Panel Layout**
- **Page size**: 1600x1080 pixels
- **Panel size**: 800x540 pixels each
- **Grid**: 2x2 (4 panels per page)
- **Total**: 48 pages = 192 panels

## 🎨 Interactive Features

### **Speech Bubble Editing**
- **Drag**: Click and drag bubbles to reposition
- **Edit**: Double-click to edit text content
- **Save**: Changes are automatically saved to JSON
- **Print**: Bubbles appear in printed pages

### **Page Navigation**
- **Previous/Next**: Navigate between pages
- **Zoom**: Adjust zoom level (25% to 100%)
- **Print**: Print current page with bubbles
- **Export**: Save individual pages as PNG

### **Keyboard Shortcuts**
- **Arrow Keys**: Navigate between pages
- **Enter**: Confirm bubble text edits
- **Escape**: Cancel bubble text edits

## 📊 Technical Specifications

### **Image Processing**
- **Input**: Any video format supported by OpenCV
- **Output**: PNG format with no compression
- **Resizing**: Aspect ratio preserved with padding
- **Quality**: High-quality LANCZOS interpolation

### **Layout System**
- **Grid**: CSS Grid for precise positioning
- **Responsive**: Scales properly at different zoom levels
- **Print-friendly**: Optimized for both screen and print

### **Data Format**
```json
{
  "page_number": 1,
  "panels": [
    {
      "panel_number": 1,
      "frame_path": "frame_001.png",
      "x": 0, "y": 0,
      "width": 800, "height": 540
    }
  ],
  "bubbles": [
    {
      "panel_number": 1,
      "text": "Sample dialogue",
      "x": 50, "y": 50,
      "width": 200, "height": 80,
      "font_size": 16,
      "color": "#000000",
      "background": "#FFFFFF",
      "border": "#000000"
    }
  ]
}
```

## 🔧 Customization

### **Page Count**
```python
generator = UnityComicGenerator()
generator.total_pages = 24  # Custom page count
```

### **Panel Dimensions**
```python
generator.panel_width = 800
generator.panel_height = 540
generator.page_width = 1600
generator.page_height = 1080
```

### **Bubble Styling**
```python
bubble_data = {
    'font_size': 18,           # Larger text
    'color': '#FF0000',        # Red text
    'background': '#FFFF00',   # Yellow background
    'border': '#000000'        # Black border
}
```

## 🐛 Troubleshooting

### **Common Issues**

1. **"No video found"**
   - Ensure video file exists at specified path
   - Check video format compatibility

2. **"OpenCV not available"**
   - Install OpenCV: `pip install opencv-python`
   - Use test mode for basic functionality

3. **"PIL not available"**
   - Install Pillow: `pip install Pillow`
   - Required for image processing

4. **"Memory issues with large videos"**
   - Reduce total pages: `--pages 24`
   - Use shorter video clips

### **Performance Tips**

1. **Use shorter videos** for faster processing
2. **Reduce page count** for testing
3. **Close other applications** during generation
4. **Use SSD storage** for better I/O performance

## 📝 Examples

### **Basic Usage**
```python
from unity_comic_generator import UnityComicGenerator

# Create generator
generator = UnityComicGenerator()

# Generate 48-page comic
success = generator.generate_48_pages_comic('video/uploaded.mp4')

if success:
    print("Comic generated successfully!")
    print("Open output/unity_pages/interactive_viewer.html")
```

### **Custom Configuration**
```python
# Custom page count
generator = UnityComicGenerator()
generator.total_pages = 24
generator.panels_per_page = 4

# Generate with custom settings
success = generator.generate_48_pages_comic('video/uploaded.mp4')
```

## 🎯 Use Cases

### **Game Development**
- **Visual novels** with comic-style storytelling
- **Interactive comics** with clickable elements
- **Cutscenes** with professional comic layout
- **Tutorial sequences** with step-by-step visuals

### **Content Creation**
- **Web comics** with interactive elements
- **Educational materials** with visual storytelling
- **Marketing materials** with engaging visuals
- **Social media content** with comic format

### **Print Production**
- **Physical comics** with high-resolution output
- **Posters and prints** with professional quality
- **Books and magazines** with consistent layout
- **Merchandise** with custom designs

## 📞 Support

For issues, questions, or feature requests:
1. Check the troubleshooting section above
2. Review the Unity integration guide
3. Test with the sample data first
4. Check console output for error messages

---

**🎮 Unity Comic Generator** - Transform any video into a professional comic with interactive elements, perfect for Unity integration and high-quality output!