"""
Auto-Evaluation System for Online Learning
Automatically tracks and evaluates model performance.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np

from src.config import ROOT_DIR


class AutoEvaluator:
    """Automatically evaluate model performance over time."""
    
    def __init__(self, save_path: Path = None):
        """Initialize auto evaluator.
        
        Args:
            save_path: Path to save evaluation results
        """
        if save_path is None:
            save_path = ROOT_DIR / "data" / "auto_evaluation.json"
        
        self.save_path = save_path
        self.evaluations = []
        
        # Load existing evaluations
        if self.save_path.exists():
            with open(self.save_path, 'r', encoding='utf-8') as f:
                self.evaluations = json.load(f)
            print(f"[AutoEval] Loaded {len(self.evaluations)} evaluations")
        else:
            print(f"[AutoEval] Starting fresh")
    
    def evaluate_retrain(self, retrain_id: int, loss: float, accuracy: float, 
                        corrections_count: int, buffer_size: int):
        """Evaluate a retrain session.
        
        Args:
            retrain_id: Retrain number
            loss: Training loss
            accuracy: Training accuracy
            corrections_count: Number of corrections in this retrain
            buffer_size: Buffer size used
        """
        # Calculate metrics
        evaluation = {
            'retrain_id': retrain_id,
            'timestamp': datetime.now().isoformat(),
            'loss': loss,
            'accuracy': accuracy,
            'corrections_count': corrections_count,
            'buffer_size': buffer_size,
        }
        
        # Compare with previous
        if self.evaluations:
            prev = self.evaluations[-1]
            evaluation['loss_change'] = loss - prev['loss']
            evaluation['accuracy_change'] = accuracy - prev['accuracy']
            evaluation['loss_change_percent'] = (loss - prev['loss']) / prev['loss'] * 100 if prev['loss'] > 0 else 0
            evaluation['accuracy_change_percent'] = (accuracy - prev['accuracy']) / prev['accuracy'] * 100 if prev['accuracy'] > 0 else 0
            
            # Status
            if evaluation['accuracy_change'] > 0.05:  # +5%
                evaluation['status'] = 'improving'
            elif evaluation['accuracy_change'] < -0.05:  # -5%
                evaluation['status'] = 'degrading'
            else:
                evaluation['status'] = 'stable'
        else:
            evaluation['loss_change'] = 0
            evaluation['accuracy_change'] = 0
            evaluation['loss_change_percent'] = 0
            evaluation['accuracy_change_percent'] = 0
            evaluation['status'] = 'baseline'
        
        # Add to history
        self.evaluations.append(evaluation)
        
        # Save
        self.save()
        
        # Print summary
        self._print_evaluation(evaluation)
        
        return evaluation
    
    def _print_evaluation(self, eval_data: Dict):
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print(f"[AutoEval] Retrain #{eval_data['retrain_id']} Evaluation")
        print(f"{'='*60}")
        print(f"Loss:     {eval_data['loss']:.4f} ({eval_data['loss_change']:+.4f}, {eval_data['loss_change_percent']:+.1f}%)")
        print(f"Accuracy: {eval_data['accuracy']:.2%} ({eval_data['accuracy_change']:+.2%}, {eval_data['accuracy_change_percent']:+.1f}%)")
        print(f"Status:   {eval_data['status'].upper()}")
        print(f"{'='*60}\n")
    
    def save(self):
        """Save evaluations to file."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluations, f, indent=2, ensure_ascii=False)
    
    def get_summary(self) -> Dict:
        """Get summary statistics.
        
        Returns:
            Dict with summary stats
        """
        if not self.evaluations:
            return {
                'total_retrains': 0,
                'best_accuracy': 0,
                'worst_accuracy': 0,
                'current_accuracy': 0,
                'trend': 'unknown'
            }
        
        accuracies = [e['accuracy'] for e in self.evaluations]
        losses = [e['loss'] for e in self.evaluations]
        
        # Calculate trend (last 5 retrains)
        recent = self.evaluations[-5:] if len(self.evaluations) >= 5 else self.evaluations
        if len(recent) >= 2:
            recent_acc = [e['accuracy'] for e in recent]
            trend_slope = (recent_acc[-1] - recent_acc[0]) / len(recent_acc)
            if trend_slope > 0.01:
                trend = 'improving'
            elif trend_slope < -0.01:
                trend = 'degrading'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        # Calculate variance
        if len(accuracies) >= 2:
            variance = np.std(accuracies) * 100  # as percentage
        else:
            variance = 0
        
        return {
            'total_retrains': len(self.evaluations),
            'best_accuracy': max(accuracies),
            'worst_accuracy': min(accuracies),
            'current_accuracy': accuracies[-1],
            'average_accuracy': np.mean(accuracies),
            'best_loss': min(losses),
            'worst_loss': max(losses),
            'current_loss': losses[-1],
            'average_loss': np.mean(losses),
            'variance': variance,
            'trend': trend,
            'improving_count': sum(1 for e in self.evaluations if e['status'] == 'improving'),
            'degrading_count': sum(1 for e in self.evaluations if e['status'] == 'degrading'),
            'stable_count': sum(1 for e in self.evaluations if e['status'] == 'stable'),
        }
    
    def get_alerts(self) -> List[Dict]:
        """Get alerts for issues.
        
        Returns:
            List of alert dicts
        """
        alerts = []
        
        if len(self.evaluations) < 2:
            return alerts
        
        latest = self.evaluations[-1]
        
        # Alert: Big accuracy drop
        if latest['accuracy_change'] < -0.10:  # -10%
            alerts.append({
                'type': 'warning',
                'message': f"Large accuracy drop: {latest['accuracy_change']:.1%}",
                'severity': 'high'
            })
        
        # Alert: Loss increasing
        if latest['loss_change'] > 0.5:
            alerts.append({
                'type': 'warning',
                'message': f"Loss increased significantly: +{latest['loss_change']:.2f}",
                'severity': 'medium'
            })
        
        # Alert: Degrading trend
        if len(self.evaluations) >= 3:
            recent_3 = self.evaluations[-3:]
            if all(e['status'] == 'degrading' for e in recent_3):
                alerts.append({
                    'type': 'error',
                    'message': "Model degrading for 3 consecutive retrains",
                    'severity': 'high'
                })
        
        # Alert: High variance
        summary = self.get_summary()
        if summary['variance'] > 15:  # 15% variance
            alerts.append({
                'type': 'info',
                'message': f"High variance detected: {summary['variance']:.1f}%",
                'severity': 'low'
            })
        
        # Good news: Improving trend
        if latest['status'] == 'improving':
            alerts.append({
                'type': 'success',
                'message': f"Model improving: +{latest['accuracy_change']:.1%}",
                'severity': 'info'
            })
        
        return alerts


# Singleton instance
_auto_evaluator = None


def get_auto_evaluator() -> AutoEvaluator:
    """Get singleton auto evaluator."""
    global _auto_evaluator
    if _auto_evaluator is None:
        _auto_evaluator = AutoEvaluator()
    return _auto_evaluator
