"""
Frame-Dialogue Synchronization Module
Ensures perfect matching between frames and their corresponding dialogue
"""

import os
import json
import srt
from typing import List, Dict, Tuple
import cv2
import math

class FrameDialogueSync:
    """Synchronize frames with their corresponding dialogue"""
    
    def __init__(self):
        self.frame_sub_map = {}
        self.dialogue_frames = []
        
    def create_frame_dialogue_mapping(self, subtitles: List, frame_files: List[str], video_path: str = None) -> Dict:
        """
        Create precise mapping between frames and dialogue
        
        Args:
            subtitles: List of SRT subtitle objects
            frame_files: List of frame file paths
            video_path: Path to source video for timing info
            
        Returns:
            Dictionary with frame-dialogue mappings
        """
        
        print("🔗 Creating frame-dialogue synchronization...")
        
        # Calculate video timing info if available
        frame_rate = 30.0  # Default
        if video_path and os.path.exists(video_path):
            try:
                cap = cv2.VideoCapture(video_path)
                frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / frame_rate
                cap.release()
                print(f"   📹 Video: {frame_rate:.1f} FPS, {total_frames} frames, {duration:.1f}s")
            except Exception as e:
                print(f"   ⚠️  Could not read video info: {e}")
        
        # Create mapping
        mapping = {
            "frame_count": len(frame_files),
            "subtitle_count": len(subtitles),
            "frame_rate": frame_rate,
            "mappings": []
        }
        
        # For each subtitle, find the best matching frame(s)
        for i, sub in enumerate(subtitles):
            # Calculate time range for this subtitle
            start_seconds = sub.start.total_seconds()
            end_seconds = sub.end.total_seconds()
            mid_seconds = (start_seconds + end_seconds) / 2
            
            # Calculate corresponding frame indices
            start_frame = int(start_seconds * frame_rate)
            end_frame = int(end_seconds * frame_rate)
            mid_frame = int(mid_seconds * frame_rate)
            
            # Find available frames in this range
            available_frames = []
            for frame_idx, frame_file in enumerate(frame_files):
                if start_frame <= frame_idx <= end_frame:
                    available_frames.append({
                        "frame_index": frame_idx,
                        "frame_file": frame_file,
                        "distance_from_mid": abs(frame_idx - mid_frame)
                    })
            
            # If no frames in exact range, find closest frame
            if not available_frames:
                closest_frame = min(range(len(frame_files)), 
                                  key=lambda x: abs(x - mid_frame))
                available_frames = [{
                    "frame_index": closest_frame,
                    "frame_file": frame_files[closest_frame],
                    "distance_from_mid": abs(closest_frame - mid_frame)
                }]
            
            # Sort by distance from middle and pick best
            available_frames.sort(key=lambda x: x["distance_from_mid"])
            best_frame = available_frames[0]
            
            # Create mapping entry
            mapping_entry = {
                "subtitle_index": i + 1,  # 1-based for SRT
                "subtitle_text": sub.content,
                "start_time": start_seconds,
                "end_time": end_seconds,
                "best_frame_index": best_frame["frame_index"],
                "best_frame_file": best_frame["frame_file"],
                "frame_options": [f["frame_index"] for f in available_frames[:3]]  # Top 3 options
            }
            
            mapping["mappings"].append(mapping_entry)
            
            print(f"   📝 Sub {i+1}: '{sub.content[:30]}...' → Frame {best_frame['frame_index']}")
        
        # Save mapping
        mapping_path = os.path.join("frames", "final", "frame_dialogue_map.json")
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)
        
        print(f"✅ Frame-dialogue mapping saved: {mapping_path}")
        print(f"   📊 {len(mapping['mappings'])} subtitle-frame pairs created")
        
        return mapping
    
    def get_optimized_frame_selection(self, target_panels: int = 12) -> List[str]:
        """
        Get optimized frame selection ensuring dialogue coverage
        
        Args:
            target_panels: Number of panels to select
            
        Returns:
            List of selected frame files with guaranteed dialogue coverage
        """
        
        mapping_path = os.path.join("frames", "final", "frame_dialogue_map.json")
        if not os.path.exists(mapping_path):
            print("⚠️  No frame-dialogue mapping found, using default selection")
            return self._fallback_frame_selection(target_panels)
        
        try:
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load mapping: {e}")
            return self._fallback_frame_selection(target_panels)
        
        print(f"🎯 Selecting {target_panels} panels with optimal dialogue coverage...")
        
        mappings = mapping.get("mappings", [])
        if len(mappings) <= target_panels:
            # Use all available dialogue frames
            selected_frames = [m["best_frame_file"] for m in mappings]
            print(f"   ✅ Using all {len(selected_frames)} dialogue frames")
            return selected_frames
        
        # Select most important dialogues using story importance
        scored_mappings = []
        for m in mappings:
            text = m["subtitle_text"].lower()
            
            # Score based on dialogue importance
            score = 0
            
            # Longer dialogue = more important
            score += len(text) * 0.1
            
            # Question or exclamation = important
            if "?" in text or "!" in text:
                score += 10
            
            # Contains keywords indicating story importance
            story_keywords = ["but", "however", "because", "so", "then", "now", "finally", 
                            "suddenly", "meanwhile", "therefore", "actually", "really"]
            for keyword in story_keywords:
                if keyword in text:
                    score += 5
            
            # Emotional words = important
            emotion_words = ["love", "hate", "angry", "happy", "sad", "afraid", "surprised", 
                           "excited", "worried", "sorry", "thank", "please", "help"]
            for word in emotion_words:
                if word in text:
                    score += 3
            
            # Dialogue with names = important
            if any(c.isupper() for c in text):
                score += 2
            
            scored_mappings.append((score, m))
        
        # Sort by score (highest first) and select top frames
        scored_mappings.sort(key=lambda x: x[0], reverse=True)
        selected_frames = [m[1]["best_frame_file"] for m in scored_mappings[:target_panels]]
        
        print(f"   ✅ Selected {len(selected_frames)} frames with highest dialogue importance")
        return selected_frames
    
    def _fallback_frame_selection(self, target_panels: int) -> List[str]:
        """Fallback frame selection when mapping is not available"""
        
        frames_dir = os.path.join("frames", "final")
        if not os.path.exists(frames_dir):
            return []
        
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
        frame_files.sort()
        
        if len(frame_files) <= target_panels:
            return frame_files
        
        # Select evenly distributed frames
        step = len(frame_files) / target_panels
        selected = []
        for i in range(target_panels):
            frame_idx = int(i * step)
            if frame_idx < len(frame_files):
                selected.append(frame_files[frame_idx])
        
        return selected
    
    def verify_sync_quality(self) -> Dict:
        """Verify the quality of frame-dialogue synchronization"""
        
        mapping_path = os.path.join("frames", "final", "frame_dialogue_map.json")
        if not os.path.exists(mapping_path):
            return {"error": "No mapping file found"}
        
        try:
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
        except Exception as e:
            return {"error": f"Could not load mapping: {e}"}
        
        mappings = mapping.get("mappings", [])
        
        # Calculate sync quality metrics
        total_mappings = len(mappings)
        perfect_sync = 0  # Frame exactly in dialogue time range
        good_sync = 0     # Frame within 1 second of dialogue
        poor_sync = 0     # Frame more than 2 seconds away
        
        for m in mappings:
            start_time = m["start_time"]
            end_time = m["end_time"]
            frame_time = m["best_frame_index"] / mapping.get("frame_rate", 30.0)
            
            if start_time <= frame_time <= end_time:
                perfect_sync += 1
            elif abs(frame_time - (start_time + end_time) / 2) <= 1.0:
                good_sync += 1
            else:
                poor_sync += 1
        
        quality_score = (perfect_sync * 1.0 + good_sync * 0.7) / total_mappings if total_mappings > 0 else 0
        
        return {
            "total_mappings": total_mappings,
            "perfect_sync": perfect_sync,
            "good_sync": good_sync,
            "poor_sync": poor_sync,
            "quality_score": quality_score,
            "quality_rating": "Excellent" if quality_score > 0.8 else "Good" if quality_score > 0.6 else "Fair" if quality_score > 0.4 else "Poor"
        }

def enhance_bubble_sync(video_path: str, frame_files: List[str], subtitles: List) -> Dict:
    """
    Main function to enhance frame-dialogue synchronization
    
    Args:
        video_path: Path to source video
        frame_files: List of available frame files
        subtitles: List of SRT subtitle objects
        
    Returns:
        Synchronization mapping dictionary
    """
    
    sync = FrameDialogueSync()
    mapping = sync.create_frame_dialogue_mapping(subtitles, frame_files, video_path)
    
    # Verify quality
    quality = sync.verify_sync_quality()
    print(f"🎯 Sync Quality: {quality.get('quality_rating', 'Unknown')} ({quality.get('quality_score', 0):.1%})")
    
    return mapping

if __name__ == "__main__":
    # Test the synchronization
    video_path = "video/uploaded.mp4"
    
    # Load subtitles
    if os.path.exists("test1.srt"):
        with open("test1.srt", "r") as f:
            subs = list(srt.parse(f.read()))
        
        # Get frame files
        frames_dir = "frames/final"
        if os.path.exists(frames_dir):
            frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
            frame_files.sort()
            
            # Create mapping
            mapping = enhance_bubble_sync(video_path, frame_files, subs)
            print(f"✅ Created mapping for {len(subs)} subtitles and {len(frame_files)} frames")
        else:
            print("❌ No frames directory found")
    else:
        print("❌ No subtitle file found")