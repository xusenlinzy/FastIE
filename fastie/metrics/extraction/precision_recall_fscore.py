from typing import List, Tuple, Set


class DedupList(list):
    """ 定义去重的 list """
    def append(self, x):
        if x not in self:
            super(DedupList, self).append(x)


def _precision_recall_fscore(pred_sum: int, tp_sum: int, true_sum: int) -> Tuple[float, float, float]:
    recall = tp_sum / true_sum if true_sum > 0 else 0.0
    precision = tp_sum / pred_sum if pred_sum > 0 else 0.0
    f_score = 0.0 if (recall + precision) == 0.0 else 2 * recall * precision / (recall + precision)
    return precision, recall, f_score


def extract_tp_actual_correct(
    y_true: List[Set[Tuple[str, int, int, str]]],
    y_pred: List[Set[Tuple[str, int, int, str]]]
) -> Tuple[int, int, int]:
    entities_true = set()
    entities_pred = set()
    for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        for d in y_t:
            entities_true.add((i, d))
        for d in y_p:
            entities_pred.add((i, d))

    tp_sum = len(entities_true & entities_pred)
    pred_sum = len(entities_pred)
    true_sum = len(entities_true)
    return pred_sum, tp_sum, true_sum


def extract_tp_actual_correct_for_event(
    y_true: List[List[List[Tuple[str, str, str, int, int]]]],
    y_pred: List[List[List[Tuple[str, str, str, int, int]]]],
) -> Tuple[int, int, int, int, int, int]:
    ex, ey, ez = 0, 0, 0  # 事件级别
    ax, ay, az = 0, 0, 0  # 论元级别

    for events, pred_events in zip(y_true, y_pred):
        R, T = DedupList(), DedupList()
        # 事件级别
        for event in pred_events:
            if any([argu[1] == "触发词" for argu in event]):
                R.append(list(sorted(event)))
        for event in events:
            T.append(list(sorted(event)))
        for event in R:
            if event in T:
                ex += 1

        ey += len(R)
        ez += len(T)

        # 论元级别
        R, T = DedupList(), DedupList()
        for event in pred_events:
            for argu in event:
                if argu[1] != "触发词":
                    R.append(argu)
        for event in events:
            for argu in event:
                if argu[1] != "触发词":
                    T.append(argu)
        for argu in R:
            if argu in T:
                ax += 1

        ay += len(R)
        az += len(T)

    return ex, ey, ez, ax, ay, az
