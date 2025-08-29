#!/usr/bin/env python3
"""
Modern Face Detection for Accurate Bubble Placement
Uses state-of-the-art models for better face and lip detection
"""

import cv2
import numpy as np
import os
from typing import Tuple, List, Optional

class ModernFaceDetector:
    def __init__(self):
        """Initialize modern face detection models"""
        
        # Option 1: MediaPipe (Google's modern face detection)
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=4,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            self.use_mediapipe = True
            print("Using MediaPipe face detection")
        except ImportError:
            self.use_mediapipe = False
            print("MediaPipe not available, using OpenCV")
        
        # Option 2: OpenCV DNN face detector (more modern than dlib)
        if not self.use_mediapipe:
            # Load OpenCV's DNN face detector
            model_path = "backend/speech_bubble/face_detection_yunet_2023mar.onnx"
            if not os.path.exists(model_path):
                # Download if not available
                self._download_face_model()
            
            self.face_detector = cv2.FaceDetectorYN_create(
                model_path,
                "",
                (320, 320),
                0.9,
                0.3,
                5000
            )
    
    def _download_face_model(self):
        """Download OpenCV face detection model if not available"""
        import urllib.request
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        print(f"Downloading face detection model from {url}")
        urllib.request.urlretrieve(url, "backend/speech_bubble/face_detection_yunet_2023mar.onnx")
    
    def detect_faces_mediapipe(self, image) -> List[Tuple[int, int]]:
        """Detect faces using MediaPipe (most accurate)"""
        # Handle both file paths and image objects
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image
            
        if img is None:
            return [(-1, -1)]
        
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        lip_positions = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # MediaPipe lip landmarks (more accurate than dlib)
                # Upper lip center
                upper_lip = face_landmarks.landmark[13]  # Upper lip center
                # Lower lip center  
                lower_lip = face_landmarks.landmark[14]  # Lower lip center
                
                # Calculate lip center
                lip_x = int((upper_lip.x + lower_lip.x) / 2 * image.shape[1])
                lip_y = int((upper_lip.y + lower_lip.y) / 2 * image.shape[0])
                
                lip_positions.append((lip_x, lip_y))
        
        return lip_positions if lip_positions else [(-1, -1)]
    
    def detect_faces_opencv(self, image) -> List[Tuple[int, int]]:
        """Detect faces using OpenCV DNN (fallback)"""
        # Handle both file paths and image objects
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image
            
        if img is None:
            return [(-1, -1)]
        
        height, width = img.shape[:2]
        self.face_detector.setInputSize((width, height))
        
        _, faces = self.face_detector.detect(img)
        lip_positions = []
        
        if faces is not None:
            for face in faces:
                # Extract face bounding box
                x, y, w, h = face[:4].astype(int)
                
                # Estimate lip position (center of lower face area)
                lip_x = x + w // 2
                lip_y = y + int(h * 0.7)  # 70% down the face (lip area)
                
                lip_positions.append((lip_x, lip_y))
        
        return lip_positions if lip_positions else [(-1, -1)]
    
    def detect_faces(self, image) -> List[Tuple[int, int]]:
        """Main face detection method"""
        if self.use_mediapipe:
            return self.detect_faces_mediapipe(image)
        else:
            return self.detect_faces_opencv(image)

def get_modern_lip_positions(video_path: str, frame_paths: List[str]) -> dict:
    """
    Get lip positions using modern face detection
    Returns: {frame_index: (lip_x, lip_y)}
    """
    detector = ModernFaceDetector()
    lip_positions = {}
    
    for i, frame_path in enumerate(frame_paths, 1):
        if os.path.exists(frame_path):
            positions = detector.detect_faces(frame_path)
            # Use the first detected face (most prominent)
            lip_positions[i] = positions[0] if positions else (-1, -1)
        else:
            lip_positions[i] = (-1, -1)
    
    return lip_positions

if __name__ == "__main__":
    # Test the modern face detector
    test_image = "frames/final/frame001.png"
    if os.path.exists(test_image):
        detector = ModernFaceDetector()
        positions = detector.detect_faces(test_image)
        print(f"Detected lip positions: {positions}")
    else:
        print("Test image not found")