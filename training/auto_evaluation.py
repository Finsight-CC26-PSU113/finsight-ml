"""
OCR FinSight - Auto Evaluation
Track and evaluate model performance across retrain sessions.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from app.config import ROOT_DIR


class AutoEvaluator:

    def __init__(self, save_path: Path = None):
        if save_path is None:
            save_path = ROOT_DIR / "data" / "auto_evaluation.json"
        self.save_path = save_path
        self.evaluations = []
        if self.save_path.exists():
            with open(self.save_path, 'r', encoding='utf-8') as f:
                self.evaluations = json.load(f)

    def evaluate_retrain(self, retrain_id: int, loss: float, accuracy: float,
                         corrections_count: int, buffer_size: int) -> Dict:
        evaluation = {
            'retrain_id': retrain_id,
            'timestamp': datetime.now().isoformat(),
            'loss': loss, 'accuracy': accuracy,
            'corrections_count': corrections_count, 'buffer_size': buffer_size,
        }
        if self.evaluations:
            prev = self.evaluations[-1]
            evaluation['loss_change'] = loss - prev['loss']
            evaluation['accuracy_change'] = accuracy - prev['accuracy']
            if evaluation['accuracy_change'] > 0.05:
                evaluation['status'] = 'improving'
            elif evaluation['accuracy_change'] < -0.05:
                evaluation['status'] = 'degrading'
            else:
                evaluation['status'] = 'stable'
        else:
            evaluation.update({'loss_change': 0, 'accuracy_change': 0, 'status': 'baseline'})

        self.evaluations.append(evaluation)
        self.save()
        self._print_evaluation(evaluation)
        return evaluation

    def _print_evaluation(self, e: Dict):
        print(f"\n{'='*50}")
        print(f"[AutoEval] Retrain #{e['retrain_id']}: "
              f"loss={e['loss']:.4f} ({e['loss_change']:+.4f})  "
              f"acc={e['accuracy']:.2%} ({e['accuracy_change']:+.2%})  "
              f"status={e['status'].upper()}")
        print('='*50)

    def save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluations, f, indent=2, ensure_ascii=False)

    def get_alerts(self) -> List[Dict]:
        alerts = []
        if len(self.evaluations) < 2:
            return alerts
        latest = self.evaluations[-1]
        if latest['accuracy_change'] < -0.10:
            alerts.append({'type': 'warning', 'message': f"Large accuracy drop: {latest['accuracy_change']:.1%}", 'severity': 'high'})
        if latest['loss_change'] > 0.5:
            alerts.append({'type': 'warning', 'message': f"Loss increased: +{latest['loss_change']:.2f}", 'severity': 'medium'})
        if len(self.evaluations) >= 3 and all(e['status'] == 'degrading' for e in self.evaluations[-3:]):
            alerts.append({'type': 'error', 'message': "Model degrading for 3 consecutive retrains", 'severity': 'high'})
        if latest['status'] == 'improving':
            alerts.append({'type': 'success', 'message': f"Model improving: +{latest['accuracy_change']:.1%}", 'severity': 'info'})
        return alerts


_auto_evaluator = None


def get_auto_evaluator() -> AutoEvaluator:
    global _auto_evaluator
    if _auto_evaluator is None:
        _auto_evaluator = AutoEvaluator()
    return _auto_evaluator
