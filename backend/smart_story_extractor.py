"""
Smart Story Extractor - Extracts meaningful story moments for full comic generation
"""

import json
import os
import re
from typing import List, Dict, Tuple
import numpy as np

class SmartStoryExtractor:
    def __init__(self):
        """Initialize the smart story extractor"""
        self.story_keywords = {
            'introduction': ['hello', 'hi', 'name', 'meet', 'introduce', 'welcome', 'start', 'begin', 'once upon'],
            'conflict': ['but', 'however', 'problem', 'issue', 'challenge', 'difficult', 'trouble', 'wrong', 'bad'],
            'action': ['run', 'fight', 'jump', 'attack', 'defend', 'escape', 'chase', 'battle', 'move', 'quick'],
            'emotion': ['happy', 'sad', 'angry', 'scared', 'love', 'hate', 'fear', 'joy', 'cry', 'laugh', 'smile'],
            'climax': ['finally', 'suddenly', 'then', 'biggest', 'most', 'intense', 'peak', 'critical', 'important'],
            'resolution': ['end', 'finally', 'resolve', 'solve', 'peace', 'happy', 'conclude', 'finish', 'done']
        }
        
    def extract_meaningful_story(self, subtitles_file: str, target_panels: int = 15) -> List[Dict]:
        """Extract meaningful story moments for comic panels
        
        Args:
            subtitles_file: Path to subtitles JSON file
            target_panels: Target number of panels (default 12, range 10-15)
            
        Returns:
            List of selected subtitle entries for comic panels
        """
        # Load subtitles
        try:
            with open(subtitles_file, 'r') as f:
                subtitles = json.load(f)
        except:
            print(f"❌ Failed to load subtitles from {subtitles_file}")
            return []
            
        if not subtitles:
            return []
            
        print(f"📖 Analyzing {len(subtitles)} subtitles for meaningful story moments...")
        
        # Score each subtitle
        scored_subtitles = []
        for i, sub in enumerate(subtitles):
            score = self._score_subtitle(sub, i, len(subtitles))
            scored_subtitles.append((score, i, sub))
            
        # Sort by score
        scored_subtitles.sort(key=lambda x: x[0], reverse=True)
        
        # Select panels ensuring story flow
        selected_indices = self._select_story_panels(scored_subtitles, target_panels, len(subtitles))
        
        # Get selected subtitles in chronological order
        selected_indices.sort()
        selected_subtitles = [subtitles[i] for i in selected_indices]
        
        print(f"✅ Selected {len(selected_subtitles)} meaningful story moments")
        
        return selected_subtitles
    
    def _score_subtitle(self, subtitle: Dict, index: int, total: int) -> float:
        """Score a subtitle based on story importance"""
        text = subtitle.get('text', '').lower()
        score = 0.0
        
        # 1. Length score (longer = more important)
        words = text.split()
        if len(words) > 5:
            score += 2.0
        elif len(words) > 3:
            score += 1.0
            
        # 2. Story phase score
        position = index / total
        if position < 0.1:  # Introduction
            score += 3.0
            for keyword in self.story_keywords['introduction']:
                if keyword in text:
                    score += 2.0
                    
        elif position > 0.85:  # Resolution
            score += 3.0
            for keyword in self.story_keywords['resolution']:
                if keyword in text:
                    score += 2.0
                    
        elif 0.4 < position < 0.6:  # Climax area
            score += 2.0
            for keyword in self.story_keywords['climax']:
                if keyword in text:
                    score += 3.0
                    
        # 3. Conflict/Action score
        for keyword in self.story_keywords['conflict'] + self.story_keywords['action']:
            if keyword in text:
                score += 2.5
                
        # 4. Emotion score
        for keyword in self.story_keywords['emotion']:
            if keyword in text:
                score += 2.0
                
        # 5. Punctuation score (questions, exclamations = important)
        if '?' in text:
            score += 1.5
        if '!' in text:
            score += 2.0
            
        # 6. Character names (assuming capitalized words mid-sentence)
        for word in words:
            if len(word) > 2 and word[0].isupper() and word not in ['I', 'The', 'A', 'An']:
                score += 1.0
                break
                
        # 7. Dialogue indicators
        if '"' in text or "'" in text:
            score += 1.0
            
        return score
    
    def _select_story_panels(self, scored_subtitles: List[Tuple], target: int, total: int) -> List[int]:
        """Select panels ensuring good story coverage"""
        selected = []
        
        # Ensure we get introduction (first 10%)
        intro_candidates = [(s, i, sub) for s, i, sub in scored_subtitles if i < total * 0.1]
        if intro_candidates:
            selected.append(intro_candidates[0][1])
            
        # Ensure we get conclusion (last 10%)
        conclusion_candidates = [(s, i, sub) for s, i, sub in scored_subtitles if i > total * 0.9]
        if conclusion_candidates:
            selected.append(conclusion_candidates[0][1])
            
        # Get high-scoring middle parts
        middle_candidates = [(s, i, sub) for s, i, sub in scored_subtitles 
                           if i not in selected and total * 0.1 <= i <= total * 0.9]
        
        # Add panels with minimum spacing
        min_spacing = max(1, total // (target * 2))  # Avoid too close panels
        
        for score, idx, sub in middle_candidates:
            # Check spacing from already selected
            too_close = False
            for selected_idx in selected:
                if abs(idx - selected_idx) < min_spacing:
                    too_close = True
                    break
                    
            if not too_close:
                selected.append(idx)
                
            if len(selected) >= target:
                break
                
        # If we still need more, relax spacing constraint
        if len(selected) < target:
            remaining = [(s, i, sub) for s, i, sub in scored_subtitles if i not in selected]
            for score, idx, sub in remaining[:target - len(selected)]:
                selected.append(idx)
                
        return selected[:target]

    def get_adaptive_layout(self, num_panels: int) -> List[Dict]:
        """Get adaptive page layout based on number of panels
        
        Returns layout configuration for pages
        """
        layouts = []
        
        if num_panels <= 4:
            # Single page, 2x2 grid
            layouts.append({
                'panels_per_page': 4,
                'rows': 2,
                'cols': 2
            })
        elif num_panels <= 6:
            # Single page, 2x3 grid
            layouts.append({
                'panels_per_page': 6,
                'rows': 2,
                'cols': 3
            })
        elif num_panels <= 9:
            # Single page, 3x3 grid
            layouts.append({
                'panels_per_page': 9,
                'rows': 3,
                'cols': 3
            })
        elif num_panels <= 12:
            # Two pages, 2x3 grid each
            layouts.extend([
                {'panels_per_page': 6, 'rows': 2, 'cols': 3},
                {'panels_per_page': 6, 'rows': 2, 'cols': 3}
            ])
        else:
            # Multiple pages with varied layouts
            remaining = num_panels
            while remaining > 0:
                if remaining >= 6:
                    layouts.append({
                        'panels_per_page': 6,
                        'rows': 2,
                        'cols': 3
                    })
                    remaining -= 6
                elif remaining >= 4:
                    layouts.append({
                        'panels_per_page': 4,
                        'rows': 2,
                        'cols': 2
                    })
                    remaining -= 4
                else:
                    layouts.append({
                        'panels_per_page': remaining,
                        'rows': 1,
                        'cols': remaining
                    })
                    remaining = 0
                    
        return layouts
    
    def create_story_timeline(self, selected_subtitles: List[Dict]) -> Dict:
        """Create a story timeline with phases"""
        total = len(selected_subtitles)
        
        timeline = {
            'introduction': selected_subtitles[:int(total * 0.2)],
            'development': selected_subtitles[int(total * 0.2):int(total * 0.5)],
            'climax': selected_subtitles[int(total * 0.5):int(total * 0.8)],
            'resolution': selected_subtitles[int(total * 0.8):]
        }
        
        # Ensure each phase has at least one panel
        for phase, subs in timeline.items():
            if not subs and selected_subtitles:
                # Take from nearest phase
                if phase == 'introduction':
                    timeline[phase] = [selected_subtitles[0]]
                elif phase == 'resolution':
                    timeline[phase] = [selected_subtitles[-1]]
                else:
                    mid = len(selected_subtitles) // 2
                    timeline[phase] = [selected_subtitles[mid]]
                    
        return timeline