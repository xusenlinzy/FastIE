from typing import (
    List,
    Tuple,
    Dict,
    Literal,
    Set,
)

from .precision_recall_fscore import (
    _precision_recall_fscore,
    extract_tp_actual_correct,
    extract_tp_actual_correct_for_event
)
from ..base import Metric


class ExtractionScore(Metric):

    def __init__(self, average="micro"):
        self.average = average
        self.reset()

    def update(self, y_true: List[Set[Tuple[str, int, int, str]]], y_pred: List[Set[Tuple[str, int, int, str]]]):
        pred_sum, tp_sum, true_sum = extract_tp_actual_correct(y_true, y_pred)
        self.pred_sum += pred_sum
        self.tp_sum += tp_sum
        self.true_sum += true_sum

    def value(self) -> Dict[Literal["precision", "recall", "f1"], float]:
        precision, recall, f1 = _precision_recall_fscore(self.pred_sum, self.tp_sum, self.true_sum)
        return {"precision": precision, "recall": recall, "f1": f1}

    def name(self) -> str:
        return "extraction_score"

    def reset(self):
        self.pred_sum = 0
        self.tp_sum = 0
        self.true_sum = 0


class EventExtractionScore(Metric):

    def __init__(self):
        self.reset()

    def update(
        self,
        y_true: List[List[List[Tuple[str, str, str, int, int]]]],
        y_pred: List[List[List[Tuple[str, str, str, int, int]]]]
    ):
        y_true = [[[tuple(j[:3]) for j in i] for i in d] for d in y_true]
        y_pred = [[[tuple(j[:3]) for j in i] for i in d] for d in y_pred]
        ex, ey, ez, ax, ay, az = extract_tp_actual_correct_for_event(y_true, y_pred)
        self.ex += ex
        self.ey += ey
        self.ez += ez

        self.ax += ax
        self.ay += ay
        self.az += az

    def value(self) -> Dict[Literal["event_precision", "event_recall", "event_f1", "argu_precision", "argu_recall", "argu_f1"], float]:
        event_score = _precision_recall_fscore(self.ey, self.ex, self.ez)
        argu_score = _precision_recall_fscore(self.ay, self.ax, self.az)

        return {
            "event_precision": event_score[0],
            "event_recall": event_score[1],
            "event_f1": event_score[2],
            "argu_precision": argu_score[0],
            "argu_recall": argu_score[1],
            "argu_f1": argu_score[2],
        }

    def name(self) -> str:
        return "event_extraction_score"

    def reset(self):
        # 事件级别
        self.ex = 0
        self.ey = 0
        self.ez = 0

        # 论元级别
        self.ax = 0
        self.ay = 0
        self.az = 0
