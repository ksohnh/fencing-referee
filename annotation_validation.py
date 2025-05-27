"""
Annotation Quality Validation and Analysis Tools
Ensures consistency and quality of sabre fencing annotations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import warnings

class AnnotationValidator:
    """Validates annotation quality and consistency"""
    
    def __init__(self, config_path: str = "annotation_config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.validation_results = {}
    
    def validate_annotations(self, annotation_path: str) -> Dict:
        """Run complete validation suite"""
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        
        results = {
            'temporal_consistency': self._validate_temporal_consistency(annotations),
            'action_sequences': self._validate_action_sequences(annotations),
            'right_of_way_logic': self._validate_right_of_way(annotations),
            'coverage_analysis': self._analyze_coverage(annotations),
            'quality_metrics': self._calculate_quality_metrics(annotations)
        }
        
        self.validation_results = results
        return results
    
    def _validate_temporal_consistency(self, annotations: Dict) -> Dict:
        """Check temporal ordering and consistency"""
        events = annotations['temporal_events']
        sequences = annotations['action_sequences']
        
        issues = []
        
        # Check event ordering within sequences
        for seq in sequences:
            seq_events = [e for e in events if seq['start_time'] <= e['timestamp'] <= seq['end_time']]
            seq_events.sort(key=lambda x: x['timestamp'])
            
            # Validate temporal logic
            prev_event = None
            for event in seq_events:
                if prev_event:
                    # Check for impossible sequences
                    if (prev_event['event_type'] == 'hit_valid' and 
                        event['event_type'] == 'attack_start'):
                        issues.append({
                            'type': 'temporal_logic_error',
                            'description': 'Attack after valid hit in same sequence',
                            'sequence': seq,
                            'events': [prev_event, event]
                        })
                prev_event = event
        
        # Check for overlapping sequences
        sequences.sort(key=lambda x: x['start_time'])
        for i in range(len(sequences) - 1):
            if sequences[i]['end_time'] > sequences[i+1]['start_time']:
                issues.append({
                    'type': 'overlapping_sequences',
                    'sequences': [sequences[i], sequences[i+1]]
                })
        
        return {
            'issues': issues,
            'total_sequences': len(sequences),
            'total_events': len(events),
            'consistency_score': max(0, 1 - len(issues) / max(1, len(sequences)))
        }
    
    def _validate_action_sequences(self, annotations: Dict) -> Dict:
        """Validate action sequence logic and completeness"""
        sequences = annotations['action_sequences']
        issues = []
        
        for seq in sequences:
            # Check duration reasonableness
            duration = seq['end_time'] - seq['start_time']
            if duration < self.config['min_action_duration']:
                issues.append({
                    'type': 'sequence_too_short',
                    'sequence': seq,
                    'duration': duration
                })
            elif duration > self.config['max_action_duration']:
                issues.append({
                    'type': 'sequence_too_long', 
                    'sequence': seq,
                    'duration': duration
                })
            
            # Check action type consistency
            if seq['action_type'] not in self.config['action_types']:
                issues.append({
                    'type': 'invalid_action_type',
                    'sequence': seq
                })
            
            # Check right-of-way consistency
            if seq['initiating_fencer'] not in ['red', 'white']:
                issues.append({
                    'type': 'invalid_fencer',
                    'field': 'initiating_fencer',
                    'sequence': seq
                })
        
        return {
            'issues': issues,
            'sequence_distribution': Counter(s['action_type'] for s in sequences),
            'outcome_distribution': Counter(s['outcome'] for s in sequences),
            'avg_duration': np.mean([s['end_time'] - s['start_time'] for s in sequences])
        }
    
    def _validate_right_of_way(self, annotations: Dict) -> Dict:
        """Validate sabre right-of-way logic"""
        sequences = annotations['action_sequences']
        events = annotations['temporal_events']
        issues = []
        
        for seq in sequences:
            seq_events = [e for e in events if seq['start_time'] <= e['timestamp'] <= seq['end_time']]
            
            # Check simultaneous action timing
            if seq['outcome'] == 'simultaneous':
                hit_events = [e for e in seq_events if 'hit' in e['event_type']]
                if len(hit_events) >= 2:
                    time_diff = abs(hit_events[0]['timestamp'] - hit_events[1]['timestamp'])
                    max_simultaneous = self.config['sabre_specific']['right_of_way_rules']['simultaneous_threshold_ms'] / 1000
                    
                    if time_diff > max_simultaneous:
                        issues.append({
                            'type': 'invalid_simultaneous',
                            'sequence': seq,
                            'time_difference': time_diff,
                            'threshold': max_simultaneous
                        })
            
            # Check attack in preparation logic
            attack_events = [e for e in seq_events if e['event_type'] == 'attack_start']
            prep_events = [e for e in seq_events if e['event_type'] == 'preparation_start']
            
            if attack_events and prep_events:
                # Attack should beat preparation if attack comes during prep
                for attack in attack_events:
                    for prep in prep_events:
                        if prep['timestamp'] < attack['timestamp'] and attack['fencer'] != prep['fencer']:
                            if seq['priority_fencer'] != attack['fencer']:
                                issues.append({
                                    'type': 'attack_in_preparation_error',
                                    'sequence': seq,
                                    'attack_event': attack,
                                    'prep_event': prep
                                })
        
        return {
            'issues': issues,
            'right_of_way_accuracy': max(0, 1 - len(issues) / max(1, len(sequences))),
            'complex_exchanges': len([s for s in sequences if s['complexity_score'] > 3])
        }
    
    def _analyze_coverage(self, annotations: Dict) -> Dict:
        """Analyze annotation coverage and distribution"""
        sequences = annotations['action_sequences']
        events = annotations['temporal_events']
        video_duration = annotations['video_metadata']['duration']
        
        # Calculate coverage percentage
        total_annotated_time = sum(s['end_time'] - s['start_time'] for s in sequences)
        coverage_percentage = (total_annotated_time / video_duration) * 100
        
        # Analyze action type distribution
        action_distribution = Counter(s['action_type'] for s in sequences)
        
        # Analyze fencer distribution
        red_actions = len([s for s in sequences if s['initiating_fencer'] == 'red'])
        white_actions = len([s for s in sequences if s['initiating_fencer'] == 'white'])
        
        # Analyze outcome distribution
        outcome_distribution = Counter(s['outcome'] for s in sequences)
        
        # Check for annotation gaps
        sequences_sorted = sorted(sequences, key=lambda x: x['start_time'])
        gaps = []
        for i in range(len(sequences_sorted) - 1):
            gap_start = sequences_sorted[i]['end_time']
            gap_end = sequences_sorted[i+1]['start_time']
            gap_duration = gap_end - gap_start
            if gap_duration > 2.0:  # Gaps longer than 2 seconds
                gaps.append({
                    'start': gap_start,
                    'end': gap_end,
                    'duration': gap_duration
                })
        
        return {
            'coverage_percentage': coverage_percentage,
            'total_annotated_time': total_annotated_time,
            'action_distribution': dict(action_distribution),
            'fencer_balance': {
                'red': red_actions,
                'white': white_actions,
                'balance_ratio': min(red_actions, white_actions) / max(red_actions, white_actions, 1)
            },
            'outcome_distribution': dict(outcome_distribution),
            'annotation_gaps': gaps,
            'avg_sequence_duration': total_annotated_time / len(sequences) if sequences else 0
        }
    
    def _calculate_quality_metrics(self, annotations: Dict) -> Dict:
        """Calculate overall annotation quality metrics"""
        sequences = annotations['action_sequences']
        events = annotations['temporal_events']
        
        # Completeness score
        required_event_types = ['attack_start', 'hit_valid', 'hit_off_target', 'miss']
        event_types_present = set(e['event_type'] for e in events)
        completeness = len(event_types_present.intersection(required_event_types)) / len(required_event_types)
        
        # Consistency score (no conflicting events)
        consistency_issues = len(self.validation_results.get('temporal_consistency', {}).get('issues', []))
        consistency = max(0, 1 - consistency_issues / max(1, len(sequences)))
        
        # Complexity distribution
        complexity_scores = [s['complexity_score'] for s in sequences]
        avg_complexity = np.mean(complexity_scores) if complexity_scores else 0
        
        # Temporal precision (events per second)
        video_duration = annotations['video_metadata']['duration']
        event_density = len(events) / video_duration if video_duration > 0 else 0
        
        return {
            'completeness_score': completeness,
            'consistency_score': consistency,
            'overall_quality': (completeness + consistency) / 2,
            'avg_complexity': avg_complexity,
            'event_density': event_density,
            'annotation_richness': len(events) / len(sequences) if sequences else 0
        }
    
    def generate_validation_report(self, output_path: str = None) -> str:
        """Generate comprehensive validation report"""
        if not self.validation_results:
            raise ValueError("Run validate_annotations() first")
        
        report = []
        report.append("# Sabre Fencing Annotation Validation Report\n")
        report.append(f"Generated: {pd.Timestamp.now()}\n")
        
        # Overall Quality Summary
        quality = self.validation_results['quality_metrics']
        report.append("## Overall Quality Metrics")
        report.append(f"- **Overall Quality Score**: {quality['overall_quality']:.2%}")
        report.append(f"- **Completeness**: {quality['completeness_score']:.2%}")
        report.append(f"- **Consistency**: {quality['consistency_score']:.2%}")
        report.append(f"- **Average Complexity**: {quality['avg_complexity']:.1f}/5")
        report.append(f"- **Event Density**: {quality['event_density']:.1f} events/second\n")
        
        # Coverage Analysis
        coverage = self.validation_results['coverage_analysis']
        report.append("## Coverage Analysis")
        report.append(f"- **Video Coverage**: {coverage['coverage_percentage']:.1f}% of video annotated")
        report.append(f"- **Total Annotated Time**: {coverage['total_annotated_time']:.1f} seconds")
        report.append(f"- **Fencer Balance**: Red {coverage['fencer_balance']['red']} vs White {coverage['fencer_balance']['white']} (ratio: {coverage['fencer_balance']['balance_ratio']:.2f})")
        
        # Action Distribution
        report.append("\n### Action Type Distribution:")
        for action, count in coverage['action_distribution'].items():
            percentage = (count / sum(coverage['action_distribution'].values())) * 100
            report.append(f"- {action}: {count} ({percentage:.1f}%)")
        
        # Issues Summary
        temporal_issues = self.validation_results['temporal_consistency']['issues']
        sequence_issues = self.validation_results['action_sequences']['issues']
        row_issues = self.validation_results['right_of_way_logic']['issues']
        
        total_issues = len(temporal_issues) + len(sequence_issues) + len(row_issues)
        
        report.append(f"\n## Issues Found ({total_issues} total)")
        if temporal_issues:
            report.append(f"- **Temporal Issues**: {len(temporal_issues)}")
            for issue in temporal_issues[:3]:  # Show first 3
                report.append(f"  - {issue['type']}: {issue.get('description', 'No description')}")
        
        if sequence_issues:
            report.append(f"- **Sequence Issues**: {len(sequence_issues)}")
            for issue in sequence_issues[:3]:
                report.append(f"  - {issue['type']}")
        
        if row_issues:
            report.append(f"- **Right-of-Way Issues**: {len(row_issues)}")
            for issue in row_issues[:3]:
                report.append(f"  - {issue['type']}")
        
        # Recommendations
        report.append("\n## Recommendations")
        if quality['overall_quality'] < 0.8:
            report.append("- Overall quality is below 80%. Consider re-annotating problematic sequences.")
        
        if coverage['coverage_percentage'] < 70:
            report.append("- Video coverage is low. Consider annotating more sequences.")
        
        if coverage['fencer_balance']['balance_ratio'] < 0.7:
            report.append("- Unbalanced fencer representation. Add more sequences from underrepresented fencer.")
        
        if len(coverage['annotation_gaps']) > 5:
            report.append("- Many large gaps in annotation. Consider adding transition sequences.")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Validation report saved to {output_path}")
        
        return report_text

class AnnotationAnalyzer:
    """Advanced analysis tools for annotation patterns and statistics"""
    
    def __init__(self):
        self.data = None
    
    def load_annotations(self, annotation_path: str):
        """Load annotations for analysis"""
        with open(annotation_path, 'r') as f:
            self.data = json.load(f)
    
    def analyze_temporal_patterns(self) -> Dict:
        """Analyze temporal patterns in the annotations"""
        events = self.data['temporal_events']
        sequences = self.data['action_sequences']
        
        # Convert to DataFrame for easier analysis
        events_df = pd.DataFrame(events)
        sequences_df = pd.DataFrame(sequences)
        
        # Analyze action timing patterns
        sequence_durations = sequences_df['end_time'] - sequences_df['start_time']
        
        # Analyze event clustering
        event_timestamps = events_df['timestamp'].values
        inter_event_intervals = np.diff(sorted(event_timestamps))
        
        # Analyze action type transitions
        sequences_sorted = sequences_df.sort_values('start_time')
        action_transitions = []
        for i in range(len(sequences_sorted) - 1):
            current = sequences_sorted.iloc[i]['action_type']
            next_action = sequences_sorted.iloc[i+1]['action_type']
            action_transitions.append((current, next_action))
        
        transition_counts = Counter(action_transitions)
        
        return {
            'sequence_duration_stats': {
                'mean': sequence_durations.mean(),
                'std': sequence_durations.std(),
                'median': sequence_durations.median(),
                'min': sequence_durations.min(),
                'max': sequence_durations.max()
            },
            'inter_event_stats': {
                'mean_interval': np.mean(inter_event_intervals),
                'median_interval': np.median(inter_event_intervals),
                'std_interval': np.std(inter_event_intervals)
            },
            'action_transitions': dict(transition_counts.most_common(10)),
            'events_per_sequence': len(events) / len(sequences) if sequences else 0
        }
    
    def plot_annotation_timeline(self, output_path: str = None):
        """Create visualization of annotation timeline"""
        sequences = self.data['action_sequences']
        events = self.data['temporal_events']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot sequences
        colors = {'simple_attack': 'red', 'compound_attack': 'orange', 
                 'parry_riposte': 'blue', 'counter_attack': 'green',
                 'preparation': 'gray', 'simultaneous': 'purple'}
        
        for i, seq in enumerate(sequences):
            color = colors.get(seq['action_type'], 'black')
            ax1.barh(i, seq['end_time'] - seq['start_time'], 
                    left=seq['start_time'], color=color, alpha=0.7)
            ax1.text(seq['start_time'], i, seq['action_type'], 
                    fontsize=8, va='center')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Sequence Index')
        ax1.set_title('Action Sequences Timeline')
        ax1.grid(True, alpha=0.3)
        
        # Plot events
        events_df = pd.DataFrame(events)
        for event_type in events_df['event_type'].unique():
            event_subset = events_df[events_df['event_type'] == event_type]
            ax2.scatter(event_subset['timestamp'], 
                       [event_type] * len(event_subset),
                       alpha=0.7, s=50)
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Event Type')
        ax2.set_title('Temporal Events')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Timeline plot saved to {output_path}")
        
        plt.show()
    
    def export_training_statistics(self, output_path: str):
        """Export detailed statistics for training preparation"""
        stats = {
            'dataset_overview': {
                'total_sequences': len(self.data['action_sequences']),
                'total_events': len(self.data['temporal_events']),
                'video_duration': self.data['video_metadata']['duration'],
                'annotation_coverage': sum(s['end_time'] - s['start_time'] 
                                         for s in self.data['action_sequences'])
            },
            'temporal_patterns': self.analyze_temporal_patterns(),
            'class_distribution': self._calculate_class_distributions(),
            'complexity_analysis': self._analyze_sequence_complexity()
        }
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"Training statistics exported to {output_path}")
    
    def _calculate_class_distributions(self) -> Dict:
        """Calculate class distributions for training balance"""
        sequences = self.data['action_sequences']
        events = self.data['temporal_events']
        
        return {
            'action_types': dict(Counter(s['action_type'] for s in sequences)),
            'outcomes': dict(Counter(s['outcome'] for s in sequences)),
            'event_types': dict(Counter(e['event_type'] for e in events)),
            'fencer_distribution': dict(Counter(s['initiating_fencer'] for s in sequences)),
            'complexity_distribution': dict(Counter(s['complexity_score'] for s in sequences))
        }
    
    def _analyze_sequence_complexity(self) -> Dict:
        """Analyze complexity patterns in sequences"""
        sequences = self.data['action_sequences']
        
        complexity_by_type = defaultdict(list)
        for seq in sequences:
            complexity_by_type[seq['action_type']].append(seq['complexity_score'])
        
        complexity_stats = {}
        for action_type, scores in complexity_by_type.items():
            complexity_stats[action_type] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'max': max(scores),
                'count': len(scores)
            }
        
        return complexity_stats


def main():
    """Example usage of validation and analysis tools"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate and analyze fencing annotations')
    parser.add_argument('annotation_file', help='Path to annotation JSON file')
    parser.add_argument('--validate', action='store_true', help='Run validation')
    parser.add_argument('--analyze', action='store_true', help='Run analysis')
    parser.add_argument('--plot', help='Generate timeline plot (output path)')
    parser.add_argument('--report', help='Generate validation report (output path)')
    parser.add_argument('--stats', help='Export training statistics (output path)')
    
    args = parser.parse_args()
    
    if args.validate or args.report:
        validator = AnnotationValidator()
        validation_results = validator.validate_annotations(args.annotation_file)
        
        if args.report:
            validator.generate_validation_report(args.report)
        else:
            print("Validation Results:")
            print(f"Overall Quality: {validation_results['quality_metrics']['overall_quality']:.2%}")
            print(f"Total Issues: {len(validation_results['temporal_consistency']['issues']) + len(validation_results['action_sequences']['issues']) + len(validation_results['right_of_way_logic']['issues'])}")
    
    if args.analyze or args.plot or args.stats:
        analyzer = AnnotationAnalyzer()
        analyzer.load_annotations(args.annotation_file)
        
        if args.plot:
            analyzer.plot_annotation_timeline(args.plot)
        
        if args.stats:
            analyzer.export_training_statistics(args.stats)
        
        if args.analyze:
            patterns = analyzer.analyze_temporal_patterns()
            print("Temporal Analysis:")
            print(f"Average sequence duration: {patterns['sequence_duration_stats']['mean']:.2f}s")
            print(f"Events per sequence: {patterns['events_per_sequence']:.1f}")
            print("Top action transitions:", list(patterns['action_transitions'].items())[:3])

if __name__ == "__main__":
    main()