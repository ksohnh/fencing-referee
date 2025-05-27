"""
Enhanced Annotation Pipeline for Sabre Fencing AI Referee
Handles hierarchical temporal annotations with right-of-way analysis
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

@dataclass
class TemporalEvent:
    """Represents a single temporal event in the bout"""
    timestamp: float  # seconds from video start
    event_type: str   # 'attack_start', 'parry', 'hit', 'priority_change'
    fencer: str      # 'red' or 'white'
    confidence: float = 1.0
    metadata: Dict = None

@dataclass
class SpatialAnnotation:
    """Frame-level spatial annotations"""
    frame_number: int
    timestamp: float
    red_pose_keypoints: List[Tuple[float, float]]  # 17 keypoints
    white_pose_keypoints: List[Tuple[float, float]]
    red_blade_tip: Optional[Tuple[float, float]]
    white_blade_tip: Optional[Tuple[float, float]]
    red_blade_guard: Optional[Tuple[float, float]]
    white_blade_guard: Optional[Tuple[float, float]]
    valid_target_masks: Dict[str, List[Tuple[int, int]]]  # polygon points

@dataclass
class ActionSequence:
    """Represents a complete fencing action sequence"""
    start_time: float
    end_time: float
    action_type: str  # 'simple_attack', 'compound_attack', 'parry_riposte', etc.
    initiating_fencer: str
    priority_fencer: str  # who has right-of-way
    outcome: str  # 'hit', 'miss', 'parried', 'simultaneous'
    sub_actions: List[TemporalEvent]
    complexity_score: int  # 1-5, how complex the exchange is

class SabreAnnotationPipeline:
    """Main annotation pipeline for sabre fencing videos"""
    
    def __init__(self, config_path: str = "annotation_config.json"):
        self.config = self.load_config(config_path)
        self.current_video_path = None
        self.video_capture = None
        self.annotations = {
            'temporal_events': [],
            'spatial_annotations': [],
            'action_sequences': [],
            'video_metadata': {},
            'annotation_metadata': {}
        }
        
    def load_config(self, config_path: str) -> Dict:
        """Load annotation configuration"""
        default_config = {
            'keypoint_model': 'mediaipipe',  # or 'openpose'
            'target_fps': 30,  # fps for annotation (can downsample from source)
            'action_types': [
                'preparation', 'simple_attack', 'compound_attack',
                'parry_riposte', 'counter_attack', 'simultaneous'
            ],
            'event_types': [
                'attack_start', 'attack_end', 'parry', 'hit_valid', 
                'hit_off_target', 'miss', 'priority_establish', 'priority_transfer'
            ],
            'annotation_window': 5.0,  # seconds around events to annotate
            'min_action_duration': 0.5,  # minimum seconds for valid action
            'quality_thresholds': {
                'pose_confidence': 0.7,
                'blade_tracking_confidence': 0.8
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except FileNotFoundError:
            print(f"Config file {config_path} not found, using defaults")
            
        return default_config
    
    def load_video(self, video_path: str) -> bool:
        """Load video and extract metadata"""
        self.current_video_path = Path(video_path)
        self.video_capture = cv2.VideoCapture(str(video_path))
        
        if not self.video_capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Extract video metadata
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.annotations['video_metadata'] = {
            'path': str(video_path),
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'resolution': (width, height),
            'loaded_at': datetime.now().isoformat()
        }
        
        print(f"Loaded video: {duration:.1f}s, {fps}fps, {width}x{height}")
        return True
    
    def extract_action_proposals(self, method: str = 'motion_based') -> List[Tuple[float, float]]:
        """
        Extract candidate time intervals where actions might occur
        Returns list of (start_time, end_time) tuples
        """
        if method == 'motion_based':
            return self._extract_motion_based_proposals()
        elif method == 'audio_based':
            return self._extract_audio_based_proposals()
        else:
            return self._extract_uniform_proposals()
    
    def _extract_motion_based_proposals(self) -> List[Tuple[float, float]]:
        """Use optical flow to detect high-motion periods"""
        proposals = []
        fps = self.annotations['video_metadata']['fps']
        
        # Read frames and compute optical flow
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, prev_frame = self.video_capture.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        motion_scores = []
        frame_idx = 1
        
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowPyrLK(prev_gray, gray, None, None)
            
            # Calculate motion magnitude
            if flow[0] is not None:
                motion_mag = np.mean(np.sqrt(flow[0][:, :, 0]**2 + flow[0][:, :, 1]**2))
            else:
                motion_mag = 0
                
            motion_scores.append(motion_mag)
            prev_gray = gray
            frame_idx += 1
        
        # Find peaks in motion (potential action start points)
        motion_scores = np.array(motion_scores)
        threshold = np.percentile(motion_scores, 75)  # Top 25% of motion
        
        in_action = False
        start_frame = 0
        
        for i, score in enumerate(motion_scores):
            if score > threshold and not in_action:
                start_frame = i
                in_action = True
            elif score <= threshold and in_action:
                end_frame = i
                start_time = start_frame / fps
                end_time = end_frame / fps
                
                # Only keep actions longer than minimum duration
                if end_time - start_time >= self.config['min_action_duration']:
                    proposals.append((start_time, end_time))
                in_action = False
        
        print(f"Found {len(proposals)} motion-based action proposals")
        return proposals
    
    def _extract_uniform_proposals(self) -> List[Tuple[float, float]]:
        """Create uniform time windows for manual annotation"""
        duration = self.annotations['video_metadata']['duration']
        window_size = self.config['annotation_window']
        overlap = window_size * 0.5  # 50% overlap
        
        proposals = []
        start = 0
        while start < duration:
            end = min(start + window_size, duration)
            proposals.append((start, end))
            start += (window_size - overlap)
            
        return proposals
    
    def annotate_temporal_events(self, action_proposals: List[Tuple[float, float]]) -> None:
        """
        Interactive annotation of temporal events within action proposals
        """
        print(f"\nStarting temporal event annotation for {len(action_proposals)} proposals")
        print("Controls:")
        print("  SPACE: Mark event at current time")
        print("  'q': Next action proposal")
        print("  'r': Mark as red fencer action")
        print("  'w': Mark as white fencer action")
        print("  'p': Mark priority change")
        print("  ESC: Exit annotation")
        
        for i, (start_time, end_time) in enumerate(action_proposals):
            print(f"\nAnnotating proposal {i+1}/{len(action_proposals)}: {start_time:.1f}s - {end_time:.1f}s")
            
            # Set video to start of action
            start_frame = int(start_time * self.annotations['video_metadata']['fps'])
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            current_events = []
            
            while True:
                ret, frame = self.video_capture.read()
                if not ret:
                    break
                    
                current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                current_time = current_frame / self.annotations['video_metadata']['fps']
                
                if current_time > end_time:
                    break
                
                # Display frame with annotations
                display_frame = self._draw_temporal_annotations(frame, current_time, current_events)
                cv2.imshow('Temporal Annotation', display_frame)
                
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord(' '):  # Space - mark event
                    event_type = self._get_event_type_input()
                    fencer = self._get_fencer_input()
                    
                    event = TemporalEvent(
                        timestamp=current_time,
                        event_type=event_type,
                        fencer=fencer,
                        confidence=1.0
                    )
                    current_events.append(event)
                    print(f"Added event: {event_type} by {fencer} at {current_time:.2f}s")
                    
                elif key == ord('q'):  # Next proposal
                    break
                elif key == 27:  # ESC - exit
                    return
            
            # Add events to main annotation store
            self.annotations['temporal_events'].extend(current_events)
            
            # Create action sequence summary
            if current_events:
                action_seq = self._create_action_sequence(start_time, end_time, current_events)
                self.annotations['action_sequences'].append(action_seq)
        
        cv2.destroyAllWindows()
        print(f"Completed temporal annotation. Added {len(self.annotations['temporal_events'])} events")
    
    def _draw_temporal_annotations(self, frame: np.ndarray, current_time: float, 
                                 events: List[TemporalEvent]) -> np.ndarray:
        """Draw current annotations on frame"""
        display_frame = frame.copy()
        
        # Draw timeline
        h, w = frame.shape[:2]
        timeline_y = h - 50
        cv2.line(display_frame, (50, timeline_y), (w-50, timeline_y), (255, 255, 255), 2)
        
        # Draw current time indicator
        time_text = f"Time: {current_time:.2f}s"
        cv2.putText(display_frame, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw events
        for i, event in enumerate(events):
            color = (0, 0, 255) if event.fencer == 'red' else (255, 255, 255)
            event_text = f"{event.event_type} ({event.fencer})"
            cv2.putText(display_frame, event_text, (10, 70 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return display_frame
    
    def _get_event_type_input(self) -> str:
        """Get event type from user input"""
        print("\nSelect event type:")
        for i, event_type in enumerate(self.config['event_types']):
            print(f"  {i+1}: {event_type}")
        
        while True:
            try:
                choice = int(input("Enter choice (1-{}): ".format(len(self.config['event_types']))))
                if 1 <= choice <= len(self.config['event_types']):
                    return self.config['event_types'][choice-1]
            except ValueError:
                pass
            print("Invalid choice, try again")
    
    def _get_fencer_input(self) -> str:
        """Get fencer selection from user"""
        while True:
            fencer = input("Fencer (r/w): ").lower()
            if fencer in ['r', 'red']:
                return 'red'
            elif fencer in ['w', 'white']:
                return 'white'
            print("Invalid input, use 'r' or 'w'")
    
    def _create_action_sequence(self, start_time: float, end_time: float, 
                              events: List[TemporalEvent]) -> ActionSequence:
        """Create action sequence from temporal events"""
        
        # Determine action type based on events
        action_types = [e.event_type for e in events]
        if 'attack_start' in action_types and 'parry' in action_types:
            action_type = 'parry_riposte'
        elif 'attack_start' in action_types:
            action_type = 'simple_attack'
        else:
            action_type = 'preparation'
        
        # Determine initiating fencer (first to act)
        initiating_fencer = events[0].fencer if events else 'unknown'
        
        # Determine priority (simplified logic)
        priority_fencer = initiating_fencer
        for event in events:
            if event.event_type == 'priority_transfer':
                priority_fencer = event.fencer
                break
        
        # Determine outcome
        hit_events = [e for e in events if 'hit' in e.event_type]
        if hit_events:
            outcome = 'hit'
        elif any('parry' in e.event_type for e in events):
            outcome = 'parried'
        else:
            outcome = 'miss'
        
        return ActionSequence(
            start_time=start_time,
            end_time=end_time,
            action_type=action_type,
            initiating_fencer=initiating_fencer,
            priority_fencer=priority_fencer,
            outcome=outcome,
            sub_actions=events,
            complexity_score=min(len(events), 5)
        )
    
    def export_annotations(self, output_path: str) -> None:
        """Export annotations to JSON format"""
        
        # Convert dataclasses to dictionaries
        export_data = {
            'temporal_events': [asdict(event) for event in self.annotations['temporal_events']],
            'action_sequences': [asdict(seq) for seq in self.annotations['action_sequences']],
            'video_metadata': self.annotations['video_metadata'],
            'annotation_metadata': {
                'annotator': 'human',  # Could be expanded
                'annotation_date': datetime.now().isoformat(),
                'config_used': self.config,
                'total_events': len(self.annotations['temporal_events']),
                'total_sequences': len(self.annotations['action_sequences'])
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Annotations exported to {output_path}")
        print(f"Total events: {len(self.annotations['temporal_events'])}")
        print(f"Total sequences: {len(self.annotations['action_sequences'])}")
    
    def load_annotations(self, annotation_path: str) -> None:
        """Load existing annotations"""
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        # Convert back to dataclasses
        self.annotations['temporal_events'] = [
            TemporalEvent(**event) for event in data['temporal_events']
        ]
        self.annotations['action_sequences'] = [
            ActionSequence(**seq) for seq in data['action_sequences']
        ]
        self.annotations['video_metadata'] = data['video_metadata']
        
        print(f"Loaded {len(self.annotations['temporal_events'])} events and "
              f"{len(self.annotations['action_sequences'])} sequences")
    
    def generate_training_dataset(self, output_dir: str) -> None:
        """Generate training dataset from annotations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create CSV for easy analysis
        events_df = pd.DataFrame([asdict(event) for event in self.annotations['temporal_events']])
        sequences_df = pd.DataFrame([asdict(seq) for seq in self.annotations['action_sequences']])
        
        events_df.to_csv(output_path / 'temporal_events.csv', index=False)
        sequences_df.to_csv(output_path / 'action_sequences.csv', index=False)
        
        # Create training clips (extract video segments around each action)
        for i, sequence in enumerate(self.annotations['action_sequences']):
            clip_path = output_path / f'clip_{i:04d}_{sequence.action_type}.mp4'
            self._extract_video_clip(sequence.start_time, sequence.end_time, str(clip_path))
        
        print(f"Training dataset generated in {output_dir}")

    def _extract_video_clip(self, start_time: float, end_time: float, output_path: str):
        """Extract video clip for training"""
        import subprocess
        
        cmd = [
            'ffmpeg', '-i', str(self.current_video_path),
            '-ss', str(start_time),
            '-t', str(end_time - start_time),
            '-c:v', 'libx264', '-c:a', 'aac',
            '-y', output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting clip: {e}")

def main():
    parser = argparse.ArgumentParser(description='Sabre Fencing Annotation Pipeline')
    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--config', default='annotation_config.json', help='Config file path')
    parser.add_argument('--output', default='annotations.json', help='Output annotations file')
    parser.add_argument('--load', help='Load existing annotations file')
    parser.add_argument('--export-training', help='Export training dataset to directory')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SabreAnnotationPipeline(args.config)
    
    # Load video
    pipeline.load_video(args.video_path)
    
    # Load existing annotations if specified
    if args.load:
        pipeline.load_annotations(args.load)
    
    # Extract action proposals
    proposals = pipeline.extract_action_proposals(method='motion_based')
    
    # Annotate temporal events
    pipeline.annotate_temporal_events(proposals)
    
    # Export annotations
    pipeline.export_annotations(args.output)
    
    # Export training dataset if requested
    if args.export_training:
        pipeline.generate_training_dataset(args.export_training)

if __name__ == "__main__":
    main()
