# 📁 Flask Upload Functionality for Unity Comic Generator

Complete Flask-based upload system for Unity Comic Generator with real-time progress tracking, file management, and download capabilities.

## 🚀 Quick Start

### **Option 1: Integrated with Main App**
```bash
# Start the main Flask app
python3 app_enhanced.py

# Access Unity upload interface
# http://localhost:5000/unity-upload
```

### **Option 2: Standalone Flask App**
```bash
# Start dedicated Unity comic Flask app
python3 unity_comic_flask.py

# Access upload interface
# http://localhost:5001/
```

## ✨ Features

### 📤 **File Upload**
- **Drag & Drop** interface for easy file uploads
- **File validation** (type, size limits)
- **Progress tracking** in real-time
- **Multiple format support** (MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V)
- **500MB file size limit**

### ⚙️ **Generation Options**
- **Custom page count** (1-100 pages)
- **Quality settings** (High, Medium, Fast)
- **Background processing** with threading
- **Job management** with unique IDs

### 📊 **Progress Tracking**
- **Real-time updates** via AJAX
- **Detailed progress messages**
- **Error handling** with user feedback
- **Status persistence** across sessions

### 📥 **Download & Export**
- **Individual file downloads**
- **Complete ZIP export**
- **Interactive viewer** access
- **Unity integration files**

## 🌐 Web Interface

### **Upload Page Features**
- **Modern, responsive design**
- **Drag & drop file upload**
- **Real-time progress bar**
- **Generation options panel**
- **Results management**

### **Interactive Elements**
- **File validation feedback**
- **Progress animations**
- **Error message display**
- **Download buttons**
- **Viewer links**

## 📡 API Endpoints

### **Upload & Generation**
```http
POST /unity-upload-file
Content-Type: multipart/form-data

Parameters:
- file: Video file
- pages: Number of pages (1-100)
- quality: Generation quality

Response:
{
  "success": true,
  "job_id": "uuid-string",
  "message": "File uploaded successfully...",
  "status_url": "/unity-status/{job_id}",
  "pages": 48
}
```

### **Progress Tracking**
```http
GET /unity-status/{job_id}

Response:
{
  "status": "processing|completed|error",
  "progress": 75,
  "message": "Creating PNG files...",
  "start_time": "2024-01-01T12:00:00",
  "end_time": "2024-01-01T12:05:00",
  "pages_generated": 48,
  "png_files": 48,
  "viewer_url": "/unity-viewer/{job_id}",
  "download_url": "/unity-download-all/{job_id}"
}
```

### **File Downloads**
```http
# Download individual file
GET /unity-download/{job_id}/{filename}

# Download all files as ZIP
GET /unity-download-all/{job_id}

# Access interactive viewer
GET /unity-viewer/{job_id}
```

### **Job Management**
```http
# List all jobs
GET /list-jobs

# Delete job and files
DELETE /delete/{job_id}
```

## 🎯 Usage Examples

### **1. Basic Upload**
```javascript
const formData = new FormData();
formData.append('file', videoFile);
formData.append('pages', '48');
formData.append('quality', 'high');

fetch('/unity-upload-file', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        startProgressTracking(data.job_id);
    }
});
```

### **2. Progress Tracking**
```javascript
function startProgressTracking(jobId) {
    const interval = setInterval(() => {
        fetch(`/unity-status/${jobId}`)
        .then(response => response.json())
        .then(data => {
            updateProgressBar(data.progress);
            updateStatusMessage(data.message);
            
            if (data.status === 'completed') {
                clearInterval(interval);
                showResults(data);
            } else if (data.status === 'error') {
                clearInterval(interval);
                showError(data.message);
            }
        });
    }, 1000);
}
```

### **3. File Download**
```javascript
// Download individual file
function downloadFile(jobId, filename) {
    window.open(`/unity-download/${jobId}/${filename}`);
}

// Download all files as ZIP
function downloadAll(jobId) {
    window.open(`/unity-download-all/${jobId}`);
}
```

## 🔧 Configuration

### **File Upload Settings**
```python
# Maximum file size (500MB)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Upload directory
app.config['UPLOAD_FOLDER'] = 'uploads'

# Output directory
app.config['OUTPUT_FOLDER'] = 'output/unity_pages'

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'mp4', 'avi', 'mov', 'mkv', 
    'wmv', 'flv', 'webm', 'm4v'
}
```

### **Generation Options**
```python
# Page count limits
pages = max(1, min(pages, 100))

# Quality settings
quality_options = {
    'high': 'No Compression',
    'medium': 'Balanced',
    'fast': 'Quick Generation'
}
```

## 📁 File Structure

```
uploads/                          # Uploaded video files
├── 20240101_120000_uuid_video.mp4
└── ...

output/unity_pages/               # Generated comics
├── job-uuid-1/
│   ├── interactive_viewer.html
│   ├── pages_data.json
│   ├── UNITY_INTEGRATION.md
│   ├── page_001.png
│   ├── page_002.png
│   └── ...
└── job-uuid-2/
    └── ...
```

## 🎮 Unity Integration

### **Generated Files**
- **PNG Pages**: High-quality images (1600x1080 each)
- **Interactive Viewer**: Web-based comic viewer
- **JSON Data**: Page and bubble positioning data
- **Integration Guide**: Unity setup instructions

### **Unity Import Settings**
1. **Texture Type**: Sprite (2D and UI)
2. **Sprite Mode**: Single
3. **Pixels Per Unit**: 100
4. **Filter Mode**: Point (no filter)
5. **Compression**: None (for best quality)

## 🐛 Troubleshooting

### **Common Issues**

1. **"File too large"**
   - Reduce video file size
   - Increase upload limit in Flask config

2. **"Invalid file type"**
   - Check file extension is supported
   - Convert to supported format

3. **"Generation failed"**
   - Check video file integrity
   - Verify OpenCV/PIL dependencies
   - Check server logs for errors

4. **"Progress not updating"**
   - Check JavaScript console for errors
   - Verify AJAX requests are working
   - Check network connectivity

### **Performance Tips**

1. **Use shorter videos** for faster processing
2. **Reduce page count** for testing
3. **Close other applications** during generation
4. **Use SSD storage** for better I/O performance
5. **Monitor server resources** during processing

## 📊 Monitoring

### **Server Logs**
```bash
# Monitor Flask app logs
tail -f flask_app.log

# Check generation progress
grep "Generation" logs/
```

### **System Resources**
```bash
# Monitor CPU usage
top -p $(pgrep -f "python.*flask")

# Monitor disk usage
df -h uploads/ output/unity_pages/
```

## 🔒 Security Considerations

### **File Upload Security**
- **File type validation**
- **File size limits**
- **Secure filename handling**
- **Path traversal protection**

### **Access Control**
- **Rate limiting** for uploads
- **Session management**
- **Job isolation** by user
- **Automatic cleanup** of old files

## 📈 Scaling

### **Production Deployment**
- **Use production WSGI server** (Gunicorn, uWSGI)
- **Implement Redis** for job tracking
- **Add load balancing** for multiple workers
- **Use CDN** for file serving

### **Database Integration**
- **Store job metadata** in database
- **User authentication** and authorization
- **File management** with database
- **Audit logging** for compliance

## 🎯 Use Cases

### **Content Creators**
- **Web comics** with interactive elements
- **Educational materials** with visual storytelling
- **Marketing content** with engaging visuals
- **Social media** comic posts

### **Game Developers**
- **Visual novels** with comic-style storytelling
- **Interactive comics** with clickable elements
- **Cutscenes** with professional layout
- **Tutorial sequences** with step-by-step visuals

### **Print Production**
- **Physical comics** with high-resolution output
- **Posters and prints** with professional quality
- **Books and magazines** with consistent layout
- **Merchandise** with custom designs

---

**📁 Flask Upload System** - Complete file upload and processing solution for Unity Comic Generator with real-time progress tracking and professional output!