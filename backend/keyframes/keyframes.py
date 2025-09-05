# Cell 1
# Import torch-related modules conditionally
try:
    import torch
    from torchvision import transforms
    from backend.keyframes.model import DSN
    import torch.nn as nn
    TORCH_AVAILABLE = True
    print("✅ PyTorch available for AI-based keyframe selection")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - using fallback frame selection")

from PIL import Image
import numpy as np
import cv2
import time
import os
import srt
from backend.keyframes.extract_frames import extract_frames
from backend.utils import copy_and_rename_file, get_black_bar_coordinates, crop_image

# Cell 2
# Global model cache to avoid reloading
_googlenet_model = None
_preprocess_pipeline = None

def _get_features(frames, gpu=True, batch_size=1):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available for AI feature extraction")
        
    global _googlenet_model, _preprocess_pipeline
    
    # Load pre-trained GoogLeNet model only once
    if _googlenet_model is None:
        print("🔄 Loading GoogLeNet model (this happens only once)...")
        _googlenet_model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT')
        # Remove the classification layer (last layer) to obtain features
        _googlenet_model = torch.nn.Sequential(*(list(_googlenet_model.children())[:-1]))
        _googlenet_model.eval()
        
        # Initialize preprocessing pipeline
        _preprocess_pipeline = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Move to GPU if available
        if gpu:
            _googlenet_model.to('cuda')
        print("✅ GoogLeNet model loaded successfully")

    # Initialize a list to store the features
    features = []

    # Iterate through frames
    for frame_path in frames:
        # Load and preprocess the frame
        input_image = Image.open(frame_path)
        input_tensor = _preprocess_pipeline(input_image)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

        # Move the input to GPU if available
        if gpu:
            input_batch = input_batch.to('cuda')

        # Perform feature extraction
        with torch.no_grad():
            output = _googlenet_model(input_batch)

        # Append the features to the list
        features.append(output.squeeze().cpu().numpy())

    # Convert the list of features to a NumPy array
    features = np.array(features)

    return features.astype(np.float32)

# Global DSN model cache
_dsn_models = {}

def _get_probs(features, gpu=True, mode=0):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available for AI probability calculation")
        
    global _dsn_models
    
    # Create cache key
    cache_key = f"dsn_model_{mode}_{gpu}"
    
    # Load model only if not already cached
    if cache_key not in _dsn_models:
        print(f"🔄 Loading DSN model {mode} (this happens only once)...")
        
        if mode == 1:
            model_path = "backend/keyframes/pretrained_model/model_1.pth.tar"
        else:
            model_path = "backend/keyframes/pretrained_model/model_0.pth.tar"
        
        model = DSN(in_dim=1024, hid_dim=256, num_layers=1, cell="lstm")
        
        if gpu:
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        
        model.load_state_dict(checkpoint)
        
        if gpu:
            model = nn.DataParallel(model).cuda()
        
        model.eval()
        _dsn_models[cache_key] = model
        print(f"✅ DSN model {mode} loaded successfully")
    
    model = _dsn_models[cache_key]
    seq = torch.from_numpy(features).unsqueeze(0)
    if gpu: seq = seq.cuda()
    probs = model(seq)
    probs = probs.data.cpu().squeeze().numpy()
    return probs


   
def generate_keyframes(video):
    # Check if video file exists
    if not os.path.exists(video):
        print(f"❌ Video file not found: {video}")
        return
        
    print(f"🎬 Starting keyframe generation for: {video}")
    
    data=""
    with open("test1.srt") as f:
        data = f.read()

    subs = srt.parse(data)
    
    # Only clear GPU cache if torch is available
    if TORCH_AVAILABLE:
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ GPU cache clear failed: {e}")
    else:
        print("⚠️ PyTorch not available - proceeding without GPU optimization")
    
    # Add timeout protection using time-based checks (thread-safe alternative to signal)
    # Track start time for timeout
    start_time = time.time()
    timeout_seconds = 600  # 10 minutes timeout

    # Create final directory if it doesn't exist
    final_dir = os.path.join("frames", "final")
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
        print(f"Created directory: {final_dir}")

    total_subs = len(list(subs))
    subs = list(subs)  # Convert to list to avoid exhaustion
    frames_generated = 0
    
    print(f"🎯 Processing {total_subs} subtitle segments...")
    
    try:
        # Enhanced story-aware keyframe extraction
        for i, sub in enumerate(subs, 1):
            # Check for timeout
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError("Keyframe generation timed out")
                
            print(f"📝 Processing segment {i}/{total_subs} (sub {sub.index}): {sub.content[:30]}...")
            frames = []
            if not os.path.exists(f"frames/sub{sub.index}"):
                os.makedirs(f"frames/sub{sub.index}")
            
            # Extract more frames per segment for better story selection
            sub_dir = os.path.join("frames", f"sub{sub.index}")
            frames = extract_frames(video, sub_dir, 
                                  sub.start.total_seconds(), sub.end.total_seconds(), 10)  # Increased from 3 to 10
            print(f"🔍 Extracted {len(frames)} frames for segment {sub.index} (time: {sub.start.total_seconds():.2f}s - {sub.end.total_seconds():.2f}s)")
            
            if len(frames) > 0:
                # Check for timeout before AI processing
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError("Keyframe generation timed out")
                
                # Use subtitle index for frame naming (this is what the rest of the system expects)
                frame_filename = f"frame{sub.index:03}.png"
                
                # Check if AI processing is available
                if TORCH_AVAILABLE:
                    try:
                        # Try AI-based selection first
                        features = _get_features(frames, gpu=False)
                        highlight_scores = _get_probs(features, gpu=False)
                        
                        # Enhanced story-aware selection
                        story_frames = _select_story_relevant_frames(frames, highlight_scores, sub)
                        
                        # Save the best story frame (use first one)
                        if len(story_frames) > 0:
                            try:
                                copy_and_rename_file(frames[story_frames[0]], final_dir, frame_filename)
                                print(f"📖 Story frame {sub.index}: {sub.content} (score: {highlight_scores[story_frames[0]]:.3f})")
                                frames_generated += 1
                            except Exception as e:
                                print(f"⚠️ Failed to copy frame for sub {sub.index}: {e}")
                                    
                    except Exception as e:
                        print(f"⚠️ AI processing failed for segment {sub.index}: {e}")
                        # Fallback: use simple frame selection without AI
                        try:
                            # Use first frame as fallback
                            if len(frames) > 0:
                                copy_and_rename_file(frames[0], final_dir, frame_filename)
                                print(f"📖 Fallback frame {sub.index}: {sub.content}")
                                frames_generated += 1
                        except Exception as fallback_error:
                            print(f"⚠️ Fallback frame copy also failed: {fallback_error}")
                else:
                    # No AI available, use simple selection
                    try:
                        # Use first frame as fallback
                        if len(frames) > 0:
                            copy_and_rename_file(frames[0], final_dir, frame_filename)
                            print(f"📖 Simple frame {sub.index}: {sub.content}")
                            frames_generated += 1
                    except Exception as fallback_error:
                        print(f"⚠️ Simple frame copy failed: {fallback_error}")
            else:
                # Fallback if no frames extracted
                print(f"⚠️ No frames extracted for subtitle {sub.index}")
        
        print(f"✅ Generated {frames_generated} story-relevant frames")
        
    except TimeoutError:
        print("⏰ Keyframe generation timed out, using fallback method...")
        frames_generated += _generate_fallback_frames(video, subs, final_dir)
        
    except Exception as e:
        print(f"❌ Error during keyframe generation: {e}")
        frames_generated += _generate_fallback_frames(video, subs, final_dir)
    
    # Ensure we have at least some frames
    if frames_generated == 0:
        print("⚠️ No frames generated, attempting emergency fallback...")
        frames_generated = _emergency_frame_generation(video, final_dir, subs)
        
    print(f"✅ Final result: Generated {frames_generated} frames")

def _generate_fallback_frames(video, subs, final_dir):
    """Generate fallback frames when AI processing fails"""
    print("🔄 Generating fallback frames without AI...")
    
    fallback_count = 0
    # Use first few subtitle segments
    for i, sub in enumerate(subs[:8], 1):  # Use first 8 segments for better coverage
        try:
            # Create subtitle directory if it doesn't exist
            sub_dir = os.path.join("frames", f"sub{sub.index}")
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            
            # Simple frame extraction without AI
            frames = extract_frames(video, sub_dir, 
                                  sub.start.total_seconds(), sub.end.total_seconds(), 1)
            if frames and os.path.exists(frames[0]):
                frame_filename = f"frame{sub.index:03}.png"
                copy_and_rename_file(frames[0], final_dir, frame_filename)
                print(f"📖 Fallback frame {sub.index}: {sub.content[:50]}...")
                fallback_count += 1
            else:
                print(f"⚠️ No frames extracted for subtitle {sub.index}")
        except Exception as e:
            print(f"⚠️ Failed to extract fallback frame for subtitle {sub.index}: {e}")
                
    print(f"✅ Generated {fallback_count} fallback frames")
    return fallback_count

def _emergency_frame_generation(video, final_dir, subs):
    """Emergency frame generation when everything else fails"""
    print("🚨 Emergency frame generation - extracting frames from video directly...")
    
    emergency_count = 0
    try:
        import cv2
        cap = cv2.VideoCapture(video)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video file: {video}")
            return emergency_count
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📊 Video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s")
        
        # If we have subtitles, try to extract frames at subtitle times
        if subs and len(subs) > 0:
            for sub in subs[:8]:  # Use first 8 subtitles
                try:
                    # Extract frame at the middle of the subtitle timing
                    mid_time = (sub.start.total_seconds() + sub.end.total_seconds()) / 2
                    frame_pos = int(mid_time * fps)
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        frame_filename = f"frame{sub.index:03}.png"
                        frame_path = os.path.join(final_dir, frame_filename)
                        cv2.imwrite(frame_path, frame)
                        print(f"📖 Emergency frame {sub.index} extracted at {mid_time:.2f}s: {sub.content[:30]}...")
                        emergency_count += 1
                    else:
                        print(f"⚠️ Failed to read frame at position {frame_pos} for sub {sub.index}")
                except Exception as e:
                    print(f"⚠️ Failed to extract emergency frame for sub {sub.index}: {e}")
        else:
            # No subtitles, extract frames at regular intervals with sequential naming
            intervals = min(8, max(4, int(duration / 30)))  # Extract 4-8 frames depending on duration
            frame_step = max(1, total_frames // intervals)
            
            for i in range(intervals):
                frame_pos = i * frame_step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    frame_filename = f"frame{i+1:03}.png"
                    frame_path = os.path.join(final_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    print(f"📖 Emergency frame {i+1} extracted at position {frame_pos}")
                    emergency_count += 1
                else:
                    print(f"⚠️ Failed to read frame at position {frame_pos}")
                
        cap.release()
        
    except Exception as e:
        print(f"❌ Emergency frame generation failed: {e}")
        
    print(f"✅ Generated {emergency_count} emergency frames")
    return emergency_count

def _select_story_relevant_frames(frames, highlight_scores, subtitle):
    """Enhanced story-aware frame selection"""
    try:
        highlight_scores = list(highlight_scores)
        
        # 1. Get top AI-scored frames
        sorted_indices = [i[0] for i in sorted(enumerate(highlight_scores), key=lambda x: x[1], reverse=True)]
        
        # 2. Analyze frames for story relevance
        story_scores = []
        for i, frame_path in enumerate(frames):
            story_score = _analyze_story_relevance(frame_path, highlight_scores[i], subtitle)
            story_scores.append(story_score)
        
        # 3. Combine AI scores with story relevance
        combined_scores = []
        for i in range(len(frames)):
            combined_score = (highlight_scores[i] * 0.6) + (story_scores[i] * 0.4)  # 60% AI, 40% story
            combined_scores.append(combined_score)
        
        # 4. Select top frames based on combined scores
        sorted_combined = [i[0] for i in sorted(enumerate(combined_scores), key=lambda x: x[1], reverse=True)]
        
        # Return top 2-3 frames per segment for better story coverage
        num_frames_to_select = min(3, len(frames))
        return sorted_combined[:num_frames_to_select]
        
    except Exception as e:
        print(f"Story selection failed: {e}")
        # Fallback to original method
        try:
            highlight_scores = list(highlight_scores)
            sorted_indices = [i[0] for i in sorted(enumerate(highlight_scores), key=lambda x: x[1], reverse=True)]
            return [sorted_indices[0]] if sorted_indices else [0]
        except:
            return [0]  # Ultimate fallback

def _analyze_story_relevance(frame_path, ai_score, subtitle):
    """Analyze frame for story relevance"""
    try:
        img = cv2.imread(frame_path)
        if img is None:
            return ai_score
        
        # 1. Face detection (dialogue scenes are important)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_score = len(faces) * 0.2  # Bonus for faces
        
        # 2. Motion/action detection
        motion_score = _detect_motion(img) * 0.15
        
        # 3. Scene complexity (more complex scenes might be more important)
        complexity_score = _analyze_scene_complexity(img) * 0.1
        
        # 4. Subtitle content analysis
        content_score = _analyze_subtitle_relevance(subtitle.content) * 0.15
        
        # Combine scores
        story_score = ai_score + face_score + motion_score + complexity_score + content_score
        
        return min(story_score, 1.0)  # Cap at 1.0
        
    except Exception as e:
        return ai_score  # Fallback to AI score

def _detect_motion(img):
    """Detect motion/action in frame"""
    try:
        # Simple edge density as motion indicator
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        return min(edge_density * 10, 1.0)  # Normalize to 0-1
    except:
        return 0.0

def _analyze_scene_complexity(img):
    """Analyze scene complexity"""
    try:
        # Use color variance as complexity indicator
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        complexity = np.std(l_channel) / 255.0
        return min(complexity * 2, 1.0)  # Normalize to 0-1
    except:
        return 0.0

def _analyze_subtitle_relevance(subtitle_text):
    """Analyze subtitle content for story relevance"""
    # Keywords that indicate important story moments
    important_keywords = [
        'hello', 'goodbye', 'thank', 'please', 'sorry', 'yes', 'no',
        'love', 'hate', 'help', 'danger', 'important', 'secret',
        'action', 'fight', 'run', 'stop', 'go', 'come', 'leave'
    ]
    
    text_lower = subtitle_text.lower()
    relevance_score = 0.0
    
    for keyword in important_keywords:
        if keyword in text_lower:
            relevance_score += 0.1
    
    return min(relevance_score, 1.0)  # Cap at 1.0
    

def black_bar_crop():
    ref_img_path = "frames/final/frame001.png"
    
    # Check if reference image exists
    if not os.path.exists(ref_img_path):
        print(f"❌ Reference image not found: {ref_img_path}")
        return 0, 0, 0, 0
    
    x, y, w, h = get_black_bar_coordinates(ref_img_path)
    
    # Loop through each keyframe
    folder_dir = "frames/final"
    if not os.path.exists(folder_dir):
        print(f"❌ Frames directory not found: {folder_dir}")
        return x, y, w, h
    
    for image in os.listdir(folder_dir): 
        img_path = os.path.join("frames",'final',image)
        if os.path.exists(img_path):
            image_data = cv2.imread(img_path)
            if image_data is not None:
                # Crop the image
                crop = image_data[y:y+h, x:x+w]
                # Save the cropped image
                cv2.imwrite(img_path, crop)
    
    return x, y, w, h