# Cell 1
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from backend.keyframes.model import DSN
import torch.nn as nn
import cv2
import time
import os
import srt
from backend.keyframes.extract_frames import extract_frames
from backend.utils import copy_and_rename_file, get_black_bar_coordinates, crop_image

# Cell 2
def _get_features(frames, gpu=True, batch_size=1):
    # Load pre-trained GoogLeNet model
    googlenet = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT')

    # Remove the classification layer (last layer) to obtain features
    googlenet = torch.nn.Sequential(*(list(googlenet.children())[:-1]))

    # Set the model to evaluation mode
    googlenet.eval()

    # Initialize a list to store the features
    features = []

    # Image preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Iterate through frames
    for frame_path in frames:
        # Load and preprocess the frame
        input_image = Image.open(frame_path)
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

        # Move the input and model to GPU if available
        if gpu:
            input_batch = input_batch.to('cuda')
            googlenet.to('cuda')

        # Perform feature extraction
        with torch.no_grad():
            output = googlenet(input_batch)

        # Append the features to the list
        features.append(output.squeeze().cpu().numpy())

    # Convert the list of features to a NumPy array
    features = np.array(features)

    return features.astype(np.float32)

# Cell 3
def _get_probs(features, gpu=True, mode=0):
    # model_cache_key = "keyframes_rl_model_cache_" + str(mode)
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

    seq = torch.from_numpy(features).unsqueeze(0)
    if gpu: seq = seq.cuda()
    probs = model(seq)
    probs = probs.data.cpu().squeeze().numpy()
    return probs


   
def generate_keyframes(video):
    data=""
    with open("test1.srt") as f:
        data = f.read()

    subs = srt.parse(data)
    torch.cuda.empty_cache()

    # Create final directory if it doesn't exist
    final_dir = os.path.join("frames", "final")
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
        print(f"Created directory: {final_dir}")

    frame_counter = 1
    
    # Enhanced story-aware keyframe extraction
    for sub in subs:
        frames = []
        if not os.path.exists(f"frames/sub{sub.index}"):
            os.makedirs(f"frames/sub{sub.index}")
        
        # Extract more frames per segment for better story selection
        frames = extract_frames(video, os.path.join("frames", f"sub{sub.index}"), 
                              sub.start.total_seconds(), sub.end.total_seconds(), 10)  # Increased from 3 to 10
        
        if len(frames) > 0:
            # Get AI highlight scores
            features = _get_features(frames, gpu=False)
            highlight_scores = _get_probs(features, gpu=False)
            
            # Enhanced story-aware selection
            story_frames = _select_story_relevant_frames(frames, highlight_scores, sub)
            
            # Save the best story frames
            for i, frame_idx in enumerate(story_frames):
                if frame_counter <= 16:  # Limit to 16 frames total
                    try:
                        copy_and_rename_file(frames[frame_idx], final_dir, f"frame{frame_counter:03}.png")
                        print(f"📖 Story frame {frame_counter}: {sub.content} (score: {highlight_scores[frame_idx]:.3f})")
                        frame_counter += 1
                    except:
                        pass
        else:
            # Fallback if no frames extracted
            print(f"⚠️ No frames extracted for subtitle {sub.index}")
    
    print(f"✅ Generated {frame_counter-1} story-relevant frames")

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
        return [sorted_indices[0]] if sorted_indices else [0]

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