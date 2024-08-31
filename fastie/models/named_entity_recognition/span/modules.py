import torch.nn as nn


class SpanLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, preds, target, masks):
        # assert if inp and target has both start and end values
        assert len(preds) == 2, "start and end logits should be present for spn losses calc"
        assert len(target) == 2, "start and end logits should be present for spn losses calc"
        assert masks is not None, "masks should be provided."

        active_loss = masks.view(-1) == 1
        start_logits, end_logits = preds
        start_positions, end_positions = target

        start_logits = start_logits.view(-1, start_logits.size(-1))
        end_logits = end_logits.view(-1, start_logits.size(-1))

        active_start_logits = start_logits[active_loss]
        active_end_logits = end_logits[active_loss]
        active_start_labels = start_positions.view(-1)[active_loss]
        active_end_labels = end_positions.view(-1)[active_loss]

        start_loss = self.loss_fct(active_start_logits, active_start_labels)
        end_loss = self.loss_fct(active_end_logits, active_end_labels)

        return (start_loss + end_loss) / 2
