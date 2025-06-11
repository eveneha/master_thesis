import torch 
import torch.nn.functional as F

class NonUniformLabelSmoothingLoss(nn.Module):
    def __init__(self, class_counts, smoothing=0.03, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

        class_weights = 1.0 / (class_counts.float()** 1.5)
        class_weights = class_weights / class_weights.sum()
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits, target):
        with torch.no_grad():
            true_dist = self.class_weights.unsqueeze(0).repeat(target.size(0), 1) 
            true_dist = true_dist * self.smoothing  
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing) 

        log_probs = F.log_softmax(logits, dim=1)
        loss = -(true_dist * log_probs).sum(dim=1)


        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  