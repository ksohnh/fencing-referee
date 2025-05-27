# Sabre Fencing Annotation Pipeline - Setup & Usage Guide

## Overview

This enhanced annotation pipeline addresses the specific challenges of sabre fencing AI referee development, focusing on:

- **Hierarchical temporal annotations** (events → sequences → actions)
- **Right-of-way rule validation** specific to sabre
- **Quality assurance and validation tools**
- **Training dataset generation** with proper class balancing

## Quick Start

### 1. Installation & Setup

```bash
# Clone your repository
git clone https://github.com/ksohnh/fencing-referee
cd fencing-referee

# Install dependencies
pip install -r requirements.txt

# Additional dependencies for annotation pipeline
pip install opencv-python pandas matplotlib seaborn mediapipe
pip install ffmpeg-python  # for video clip extraction
```

### 2. Configuration

Create `annotation_config.json` in your project root:

```json
{
  "keypoint_model": "mediapipe",
  "target_fps": 30,
  "action_types": [
    "preparation", "simple_attack", "compound_attack",
    "parry_riposte", "counter_attack", "simultaneous"
  ],
  "sabre_specific": {
    "right_of_way_rules": {
      "simultaneous_threshold_ms": 40,
      "attack_in_preparation": "attack beats preparation"
    }
  },
  "annotation_window": 5.0,
  "min_action_duration": 0.3
}
```

### 3. Basic Usage

```bash
# Annotate a video
python sabre_annotation_pipeline.py video.mp4 --output annotations.json

# Validate annotations
python annotation_validation.py annotations.json --validate --report validation_report.md

# Generate training dataset
python sabre_annotation_pipeline.py video.mp4 --load annotations.json --export-training ./training_data/
```

## Detailed Workflow

### Phase 1: Video Preparation

**Recommended Video Specifications:**
- **Frame Rate**: 120fps minimum (sabre actions are extremely fast)
- **Resolution**: 1080p minimum 
- **Duration**: 2-5 minute clips (easier to manage than full bouts)
- **Camera Angles**: Side view preferred for distance/timing analysis
- **Lighting**: Consistent, avoid shadows on fencers

**Video Selection Strategy:**
```python
# Prioritize videos with:
priority_criteria = {
    'clear_right_of_way_actions': 40,  # Simple, unambiguous exchanges
    'complex_exchanges': 30,           # Multi-tempo actions
    'edge_cases': 20,                  # Simultaneous, preparation attacks
    'variety_of_actions': 10           # Different attack types
}
```

### Phase 2: Annotation Workflow

#### 2.1 Automated Action Proposals
The pipeline automatically identifies potential action sequences using optical flow:

```python
# Extract action proposals
pipeline = SabreAnnotationPipeline('annotation_config.json')
pipeline.load_video('bout_video.mp4')
proposals = pipeline.extract_action_proposals(method='motion_based')
```

#### 2.2 Interactive Temporal Annotation
For each proposed action sequence:

1. **Watch the sequence** in slow motion
2. **Mark key events** using keyboard shortcuts:
   - `SPACE`: Mark event at current timestamp
   - `R`: Red fencer action
   - `W`: White fencer action  
   - `P`: Priority change
3. **Classify the overall action** (simple attack, parry-riposte, etc.)
4. **Determine right-of-way** based on sabre rules

#### 2.3 Event Types to Annotate

**Essential Events (must annotate):**
- `attack_start`: When attack motion begins
- `hit_valid`: Valid target area contact
- `hit_off_target`: Invalid target contact
- `parry`: Successful blade deflection
- `priority_establish`: Right-of-way establishment
- `priority_transfer`: Right-of-way changes hands

**Optional Events (for advanced analysis):**
- `preparation_start`: Preparatory movements
- `blade_contact`: Non-parry blade interactions
- `distance_close`: Fencers come into measure

### Phase 3: Quality Validation

#### 3.1 Automated Validation
```python
# Run comprehensive validation
validator = AnnotationValidator()
results = validator.validate_annotations('annotations.json')

# Check key metrics
print(f"Overall Quality: {results['quality_metrics']['overall_quality']:.1%}")
print(f"Right-of-Way Accuracy: {results['right_of_way_logic']['right_of_way_accuracy']:.1%}")
```

#### 3.2 Manual Review Checklist

**Temporal Consistency:**
- [ ] Events are in logical chronological order
- [ ] No impossible event sequences (hit before attack)
- [ ] Timing precision within 33ms (1 frame at 30fps)

**Right-of-Way Logic:**
- [ ] Attack in preparation correctly prioritized
- [ ] Simultaneous actions within 40ms threshold
- [ ] Parry-riposte sequences properly sequenced
- [ ] Priority transfers are justified

**Coverage Quality:**
- [ ] At least 70% of video annotated
- [ ] Balanced representation of both fencers
- [ ] Mix of simple and complex actions
- [ ] Edge cases included (simultaneous, preparation attacks)

### Phase 4: Training Dataset Generation

```python
# Generate training dataset
pipeline.generate_training_dataset('./training_data/')

# This creates:
# - Individual video clips for each action sequence
# - CSV files with temporal annotations
# - JSON metadata for model training
# - Class distribution analysis
```

## Advanced Features

### Inter-Annotator Agreement

For critical competitions or research, use multiple annotators:

```python
# Compare annotations from different annotators
def calculate_agreement(annotations1, annotations2):
    # Implement Cohen's Kappa for temporal events
    # Compare right-of-way decisions
    # Calculate timing precision agreement
```

### Active Learning Integration

```python
# Use model predictions to suggest annotations
class ActiveAnnotationPipeline(SabreAnnotationPipeline):
    def __init__(self, model_path=None):
        super().__init__()
        if model_path:
            self.prediction_model = load_model(model_path)
    
    def suggest_events(self, video_clip):
        # Use existing model to suggest event locations
        # Human annotator confirms/corrects suggestions
        pass
```

### Specialized Annotation Modes

**Competition Mode**: Focus on referee decisions
```python
config['competition_mode'] = {
    'focus_on_calls': True,
    'include_referee_signals': True,
    'track_score_changes': True
}
```

**Training Mode**: Focus on technique analysis
```python
config['training_mode'] = {
    'detailed_blade_work': True,
    'footwork_analysis': True,
    'tactical_intentions': True
}
```

## File Structure

```
fencing-referee/
├── annotation/
│   ├── sabre_annotation_pipeline.py
│   ├── annotation_validation.py
│   ├── annotation_config.json
│   └── requirements.txt
├── data/
│   ├── raw_videos/
│   ├── annotations/
│   └── training_clips/
├── models/
│   └── pose_estimation/
└── analysis/
    ├── validation_reports/
    └── statistics/
```

## Best Practices

### Annotation Consistency
1. **Start simple**: Begin with clear, unambiguous actions
2. **Use reference standards**: Establish ground truth with expert referees
3. **Regular validation**: Run validation after every 10-20 annotations
4. **Document edge cases**: Keep notes on difficult decisions

### Quality Control
1. **Multiple passes**: Review annotations multiple times
2. **Cross-validation**: Have different annotators check subsets
3. **Rule book reference**: Keep FIE sabre rules handy
4. **Video quality**: Don't annotate poor quality footage

### Efficiency Tips
1. **Batch similar actions**: Annotate similar action types together
2. **Use keyboard shortcuts**: Learn all hotkeys for speed
3. **Focus sessions**: Annotate for 45-60 minutes max per session
4. **Save frequently**: Export annotations after each video

## Troubleshooting

### Common Issues

**Slow annotation speed:**
- Reduce video resolution for annotation (can train on full resolution)
- Use SSD storage for video files
- Close other applications during annotation

**Inconsistent event timing:**
- Use frame-by-frame navigation (`←`/`→` keys)
- Zoom in on timeline for precision
- Mark events at action initiation, not completion

**Right-of-way confusion:**
- Review sabre priority rules before each session
- When in doubt, mark as "simultaneous" and review later
- Keep referee handbook accessible

**Validation failures:**
- Check temporal ordering of events
- Verify fencer labels (red/white consistency)
- Ensure realistic action durations

### Getting Help

1. **Check validation report** for specific issues
2. **Review sabre rulebook** for right-of-way questions  
3. **Consult expert referees** for ambiguous situations
4. **Use visualization tools** to spot patterns in annotations

## Next Steps

After completing annotation:

1. **Train initial models** on your annotated dataset
2. **Evaluate model performance** on held-out validation set
3. **Identify failure cases** and add targeted annotations
4. **Iterate and improve** based on model feedback
5. **Scale up** with more videos and automated assistance

The goal is to create a high-quality, consistent dataset that captures the complexity and nuance of sabre fencing for effective AI referee training.