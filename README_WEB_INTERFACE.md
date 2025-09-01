# 🎬 Enhanced Comic Generator - Web Interface

A modern web interface for generating high-quality comics from videos using AI-enhanced processing.

## 🚀 Quick Start

### Method 1: Simple Web Interface (Recommended)
```bash
python3 run_web_interface.py
```

### Method 2: Direct Flask App
```bash
python3 app_enhanced.py
```

### Method 3: Manual Comic Generation
```bash
python3 generate_comic_manual.py
```

## 🌐 Web Interface Features

### **Upload Methods**
- **📁 File Upload**: Upload MP4 videos directly from your computer
- **🔗 YouTube Links**: Paste YouTube URLs to download and process videos

### **AI-Enhanced Processing**
- **🎯 High-Quality Keyframes**: Intelligent frame extraction using PyTorch
- **✨ Image Enhancement**: Multi-stage quality improvement
- **🎨 Comic Styling**: Modern comic art transformation
- **👤 Face Detection**: Advanced face and lip detection
- **💬 Smart Bubbles**: AI-powered speech bubble placement
- **📐 Optimized Layout**: 2x2 grid layout with content analysis

## 📋 How to Use the Web Interface

### **Step 1: Start the Server**
```bash
python3 run_web_interface.py
```

### **Step 2: Access the Interface**
- Open your browser and go to: `http://localhost:5000`
- The interface will automatically open in your default browser

### **Step 3: Upload Your Video**
1. **For Local Files**: Click the "Upload Video" button and select an MP4 file
2. **For YouTube**: Click "Enter Link" and paste a YouTube URL

### **Step 4: Generate Comic**
- Click the "Submit" button
- Watch the progress in the terminal
- The comic will automatically open in your browser when complete

## 🎯 What You Get

### **Output Files**
- **📄 Comic HTML**: `/output/page.html` - Viewable comic with speech bubbles
- **🖼️ Enhanced Frames**: `/frames/final/` - High-quality processed images
- **📊 JSON Data**: `/output/pages.json` - Comic structure data

### **Features**
- **2x2 Grid Layout**: 4 panels per page in a clean grid
- **Speech Bubbles**: AI-placed dialogue bubbles avoiding faces
- **High Quality**: Enhanced images with comic styling
- **Responsive Design**: Works on desktop and mobile

## 🔧 Technical Details

### **AI Models Used**
- **Face Detection**: MediaPipe (with OpenCV fallback)
- **Image Enhancement**: Multi-stage processing pipeline
- **Layout Optimization**: Content-aware panel arrangement
- **Bubble Placement**: Salient region analysis

### **Processing Pipeline**
1. **Video Processing**: Extract keyframes using PyTorch
2. **Black Bar Removal**: Automatic detection and cropping
3. **Image Enhancement**: Quality improvement and noise reduction
4. **Comic Styling**: Artistic transformation
5. **Face Detection**: Locate faces and lips
6. **Bubble Placement**: Smart positioning avoiding faces
7. **Layout Generation**: 2x2 grid with content analysis
8. **Output Creation**: HTML comic with embedded images

## 🛠️ Installation & Dependencies

### **Required Packages**
```bash
pip install flask yt-dlp opencv-python pillow numpy torch torchvision transformers mediapipe scikit-image scipy matplotlib nltk textblob imageio imageio-ffmpeg tqdm requests urllib3 srt --break-system-packages
```

### **System Requirements**
- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space
- **GPU**: Optional (CUDA support for faster processing)

## 📁 File Structure

```
comic-generator/
├── app_enhanced.py              # Main Flask application
├── run_web_interface.py         # Web interface runner
├── generate_comic_manual.py     # Direct comic generation
├── templates/
│   └── index.html              # Web interface template
├── static/
│   ├── styles.css              # CSS styles
│   ├── script.js               # JavaScript functionality
│   └── images/                 # Interface images
├── backend/
│   ├── ai_enhanced_core.py     # AI processing core
│   ├── ai_bubble_placement.py  # Smart bubble placement
│   ├── subtitles/
│   ├── keyframes/
│   └── speech_bubble/
├── video/                      # Uploaded videos
├── frames/final/               # Processed frames
└── output/                     # Generated comics
```

## 🎨 Customization

### **Environment Variables**
```bash
export HIGH_QUALITY=1          # Enable high-quality processing
export AI_ENHANCED=1           # Enable AI features
export HIGH_ACCURACY=1         # Enable high-accuracy mode
export GRID_LAYOUT=1           # Force 2x2 grid layout
```

### **Quality Settings**
- **Standard**: Faster processing, good quality
- **High Quality**: Slower processing, excellent quality
- **AI Enhanced**: Advanced features, best results

## 🐛 Troubleshooting

### **Common Issues**

#### **"Module not found" errors**
```bash
pip install [package-name] --break-system-packages
```

#### **Flask server won't start**
```bash
# Check if port 5000 is in use
lsof -i :5000
# Kill existing process if needed
kill -9 [PID]
```

#### **Video upload fails**
- Ensure video is MP4 format
- Check file size (max 100MB recommended)
- Verify video file is not corrupted

#### **Comic generation fails**
- Check terminal for error messages
- Ensure sufficient disk space
- Verify all dependencies are installed

### **Performance Tips**
- **GPU Usage**: Install CUDA for faster processing
- **Memory**: Close other applications during processing
- **Storage**: Ensure adequate free space
- **Network**: Stable connection for YouTube downloads

## 📊 Performance Metrics

### **Processing Times** (approximate)
- **Short Video (30s)**: 2-3 minutes
- **Medium Video (2min)**: 5-8 minutes
- **Long Video (5min)**: 10-15 minutes

### **Quality Levels**
- **Standard**: 720p output, basic enhancement
- **High Quality**: 1080p output, advanced enhancement
- **AI Enhanced**: Best quality, smart features

## 🔄 API Endpoints

### **Web Interface**
- `GET /` - Main interface
- `POST /uploader` - File upload
- `POST /handle_link` - YouTube link processing
- `GET /status` - System status
- `GET /output/<file>` - Serve output files
- `GET /frames/final/<file>` - Serve frame files

## 📝 Examples

### **Upload Local Video**
1. Start server: `python3 run_web_interface.py`
2. Open browser: `http://localhost:5000`
3. Click "Upload Video"
4. Select MP4 file
5. Click "Submit"
6. Wait for processing
7. Comic opens automatically

### **Process YouTube Video**
1. Start server: `python3 run_web_interface.py`
2. Open browser: `http://localhost:5000`
3. Click "Enter Link"
4. Paste YouTube URL
5. Click "Submit"
6. Wait for download and processing
7. Comic opens automatically

## 🎉 Success!

Your enhanced comic generator web interface is now ready! 

**Key Features:**
- ✅ Modern web interface
- ✅ AI-enhanced processing
- ✅ Smart bubble placement
- ✅ High-quality output
- ✅ YouTube support
- ✅ Automatic browser opening

**Next Steps:**
1. Run `python3 run_web_interface.py`
2. Open `http://localhost:5000`
3. Upload a video or paste a YouTube link
4. Generate your comic!

Happy comic creating! 🎬✨