"""
Flask App for Unity Comic Generator
Handles file uploads and comic generation with progress tracking
"""

import os
import json
import threading
import time
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

# Import the Unity Comic Generator
try:
    from unity_comic_generator import UnityComicGenerator
    UNITY_GENERATOR_AVAILABLE = True
except ImportError:
    UNITY_GENERATOR_AVAILABLE = False
    print("⚠️ Unity Comic Generator not available")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output/unity_pages'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v'}

# Global storage for generation progress
generation_progress = {}
generation_results = {}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_comic_async(job_id, video_path, pages=48):
    """Generate comic in background thread"""
    try:
        generation_progress[job_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting comic generation...',
            'start_time': datetime.now().isoformat()
        }
        
        if not UNITY_GENERATOR_AVAILABLE:
            generation_progress[job_id] = {
                'status': 'error',
                'progress': 0,
                'message': 'Unity Comic Generator not available',
                'error': 'Missing dependencies'
            }
            return
        
        # Initialize generator
        generator = UnityComicGenerator()
        generator.total_pages = pages
        
        # Update progress
        generation_progress[job_id].update({
            'progress': 10,
            'message': 'Extracting frames from video...'
        })
        
        # Extract frames
        frames = generator._extract_frames(video_path)
        if not frames:
            generation_progress[job_id] = {
                'status': 'error',
                'progress': 0,
                'message': 'Failed to extract frames from video',
                'error': 'No frames extracted'
            }
            return
        
        generation_progress[job_id].update({
            'progress': 30,
            'message': f'Extracted {len(frames)} frames, resizing for Unity...'
        })
        
        # Resize frames
        resized_frames = generator._resize_frames_for_unity(frames)
        if not resized_frames:
            generation_progress[job_id] = {
                'status': 'error',
                'progress': 0,
                'message': 'Failed to resize frames',
                'error': 'Frame resizing failed'
            }
            return
        
        generation_progress[job_id].update({
            'progress': 50,
            'message': f'Resized {len(resized_frames)} frames, extracting subtitles...'
        })
        
        # Extract subtitles
        subtitles = generator._extract_subtitles()
        
        generation_progress[job_id].update({
            'progress': 60,
            'message': f'Extracted {len(subtitles)} subtitles, generating {pages} pages...'
        })
        
        # Generate pages
        pages_data = generator._generate_48_pages(resized_frames, subtitles)
        
        generation_progress[job_id].update({
            'progress': 80,
            'message': f'Generated {len(pages_data)} pages, creating PNG files...'
        })
        
        # Create PNG pages
        png_pages = generator._create_png_pages(pages_data)
        
        generation_progress[job_id].update({
            'progress': 90,
            'message': f'Created {len(png_pages)} PNG pages, generating interactive viewer...'
        })
        
        # Create interactive viewer
        generator._create_interactive_viewer(pages_data)
        
        # Save Unity data
        generator._save_unity_data(pages_data)
        
        # Complete
        generation_progress[job_id] = {
            'status': 'completed',
            'progress': 100,
            'message': f'Successfully generated {pages} pages with {len(png_pages)} PNG files',
            'end_time': datetime.now().isoformat(),
            'pages_generated': len(pages_data),
            'png_files': len(png_pages),
            'viewer_url': f'/unity-viewer/{job_id}',
            'download_url': f'/download-all/{job_id}'
        }
        
        # Store results
        generation_results[job_id] = {
            'pages_data': pages_data,
            'png_pages': png_pages,
            'video_path': video_path,
            'pages': pages,
            'created_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        generation_progress[job_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Generation failed: {str(e)}',
            'error': str(e),
            'end_time': datetime.now().isoformat()
        }

@app.route('/')
def index():
    """Main upload page"""
    return render_template('unity_upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start comic generation"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, wmv, flv, webm, m4v'}), 400
        
        # Get optional parameters
        pages = int(request.form.get('pages', 48))
        pages = max(1, min(pages, 100))  # Limit between 1-100 pages
        
        # Create upload directory
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{job_id}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Start background generation
        thread = threading.Thread(
            target=generate_comic_async,
            args=(job_id, file_path, pages)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': f'File uploaded successfully. Generating {pages} pages...',
            'status_url': f'/status/{job_id}',
            'pages': pages
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status/<job_id>')
def get_status(job_id):
    """Get generation status"""
    if job_id not in generation_progress:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(generation_progress[job_id])

@app.route('/unity-viewer/<job_id>')
def unity_viewer(job_id):
    """Serve the Unity comic interactive viewer"""
    if job_id not in generation_results:
        return jsonify({'error': 'Comic not found'}), 404
    
    # Create job-specific output directory
    job_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    viewer_path = os.path.join(job_output_dir, 'interactive_viewer.html')
    
    if os.path.exists(viewer_path):
        return send_from_directory(job_output_dir, 'interactive_viewer.html')
    else:
        return jsonify({'error': 'Viewer not found'}), 404

@app.route('/download/<job_id>/<filename>')
def download_file(job_id, filename):
    """Download individual files"""
    if job_id not in generation_results:
        return jsonify({'error': 'Job not found'}), 404
    
    job_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    file_path = os.path.join(job_output_dir, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/download-all/<job_id>')
def download_all(job_id):
    """Download all files as ZIP"""
    if job_id not in generation_results:
        return jsonify({'error': 'Job not found'}), 404
    
    try:
        import zipfile
        import tempfile
        
        job_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        
        # Create temporary ZIP file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(job_output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, job_output_dir)
                    zipf.write(file_path, arcname)
        
        return send_file(
            temp_zip.name,
            as_attachment=True,
            download_name=f'unity_comic_{job_id}.zip',
            mimetype='application/zip'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/list-jobs')
def list_jobs():
    """List all generation jobs"""
    jobs = []
    for job_id, progress in generation_progress.items():
        job_info = {
            'job_id': job_id,
            'status': progress.get('status', 'unknown'),
            'progress': progress.get('progress', 0),
            'message': progress.get('message', ''),
            'start_time': progress.get('start_time', ''),
            'end_time': progress.get('end_time', ''),
            'pages': progress.get('pages_generated', 0)
        }
        jobs.append(job_info)
    
    return jsonify({'jobs': jobs})

@app.route('/delete/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a generation job and its files"""
    try:
        # Remove from progress tracking
        if job_id in generation_progress:
            del generation_progress[job_id]
        
        # Remove from results
        if job_id in generation_results:
            del generation_results[job_id]
        
        # Remove files
        job_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        if os.path.exists(job_output_dir):
            import shutil
            shutil.rmtree(job_output_dir)
        
        return jsonify({'success': True, 'message': 'Job deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve static files
@app.route('/output/<path:filename>')
def serve_output(filename):
    """Serve output files"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    print("🎮 Unity Comic Generator Flask App")
    print("=" * 50)
    print("✨ Features:")
    print("   - File upload with progress tracking")
    print("   - 48 pages with 2x2 grid layout")
    print("   - Interactive speech bubbles")
    print("   - High-quality PNG output")
    print("   - Unity-optimized files")
    print("   - Download as ZIP")
    print("")
    print("🌐 Web interface: http://localhost:5001")
    print("📁 Upload videos to generate Unity comics!")
    print("")
    
    app.run(debug=True, host='0.0.0.0', port=5001)