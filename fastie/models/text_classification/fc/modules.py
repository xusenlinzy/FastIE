import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSampleDropout(nn.Module):
    """ multisample dropout (wut): https://arxiv.org/abs/1905.09788 """
    def __init__(self, hidden_size, num_labels, K=5, p=0.5):
        super().__init__()
        self.K = K
        self.dropout = nn.Dropout(p)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, inputs):
        logits = torch.stack([self.classifier(self.dropout(inputs)) for _ in range(self.K)], dim=0)
        return torch.mean(logits, dim=0)


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        return F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last", "first_token_transform"], \
            f"unrecognized pooling type {self.pooler_type}"

    def forward(self, outputs, attention_mask):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type in ["cls_before_pooler", "cls", "first_token_transform"]:
            return last_hidden[:, 0]
        elif self.pooling == "pooler":
            return outputs.pooler_output
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class RDropLoss(nn.Module):
    """R-Drop的Loss实现，官方项目：https://github.com/dropreg/R-Drop"""
    def __init__(self, alpha=4, rank="adjacent"):
        super().__init__()
        self.alpha = alpha

        # 支持两种方式，一种是奇偶相邻排列，一种是上下排列
        assert rank in {"adjacent", "updown"}, "rank kwarg only support 'adjacent' and 'updown' "
        self.rank = rank

        self.loss_sup = nn.CrossEntropyLoss()
        self.loss_rdrop = nn.KLDivLoss(reduction="none")

    def forward(self, *args):
        """支持两种方式: 一种是y_pred, y_true, 另一种是y_pred1, y_pred2, y_true
        y_pred: torch.Tensor, 第一种方式的样本预测值, shape=[btz*2, num_labels]
        y_true: torch.Tensor, 样本真实值, 第一种方式shape=[btz*2,], 第二种方式shape=[btz,]
        y_pred1: torch.Tensor, 第二种方式的样本预测值, shape=[btz, num_labels]
        y_pred2: torch.Tensor, 第二种方式的样本预测值, shape=[btz, num_labels]
        """
        assert len(args) in {2, 3}, "RDropLoss only support 2 or 3 input args"
        # y_pred是1个Tensor
        if len(args) == 2:
            y_pred, y_true = args
            loss_sup = self.loss_sup(y_pred, y_true)  # 两个都算

            if self.rank == "adjacent":
                y_pred1 = y_pred[1::2]
                y_pred2 = y_pred[::2]
            else:
                half_btz = y_true.shape[0] // 2
                y_pred1 = y_pred[:half_btz]
                y_pred2 = y_pred[half_btz:]

        # y_pred是两个tensor
        else:
            y_pred1, y_pred2, y_true = args
            loss_sup = (self.loss_sup(y_pred1, y_true) + self.loss_sup(y_pred2, y_true)) / 2

        loss_rdrop1 = self.loss_rdrop(F.log_softmax(y_pred1, dim=-1), F.softmax(y_pred2, dim=-1))
        loss_rdrop2 = self.loss_rdrop(F.log_softmax(y_pred2, dim=-1), F.softmax(y_pred1, dim=-1))

        return loss_sup + torch.mean(loss_rdrop1 + loss_rdrop2) / 4 * self.alpha
