import torch
import torch.nn.functional as F


class LabelSmoothFocalLoss(torch.nn.Module):
    def __init__(self, num_classes, smooth_ratio=0.08, alpha=0.8, gamma=2):
        super(LabelSmoothFocalLoss, self).__init__()
        self.smooth_ratio = smooth_ratio
        self.v = self.smooth_ratio / num_classes
        self.alpha, self.gamma = alpha, gamma
    
    def forward(self, logits: torch.Tensor, targets):
        one_hot = torch.ones(logits.shape, device=logits.device) * self.v
        one_hot.scatter_(1, targets.view(-1, 1), 1 - self.smooth_ratio + self.v)
        
        prob = F.softmax(logits, dim=1)
        log_prob = prob.log()
        ce_loss = -(one_hot * log_prob)
        weight = (1 - prob) ** self.gamma
        
        focal_loss = (self.alpha * weight * ce_loss).sum(dim=1).mean()
        return focal_loss


class LabelSmoothFocalLossV2(LabelSmoothFocalLoss):
    def forward(self, logits, targets):
        one_hot = torch.ones(logits.shape, device=logits.device) * self.v
        one_hot.scatter_(1, targets.view(-1, 1), 1 - self.smooth_ratio + self.v)
        
        log_prob = F.log_softmax(logits, dim=1)
        ce_loss = -(log_prob * one_hot).sum(dim=1)
        weight = (1 - torch.exp(-ce_loss)) ** self.gamma
        
        focal_loss = (self.alpha * weight * ce_loss).mean()
        return focal_loss


class _LabelSmoothCELoss(torch.nn.Module):
    def __init__(self, smooth_ratio, num_classes):
        super(_LabelSmoothCELoss, self).__init__()
        self.smooth_ratio = smooth_ratio
        self.v = self.smooth_ratio / num_classes
        
        self.logsoft = torch.nn.LogSoftmax(dim=1)
    
    def forward(self, logits, targets):
        one_hot = torch.zeros_like(logits)
        one_hot.fill_(self.v)
        y = targets.to(torch.long).view(-1, 1)
        one_hot.scatter_(1, y, 1 - self.smooth_ratio + self.v)
        
        loss = - torch.sum(self.logsoft(logits) * (one_hot.detach())) / logits.size(0)
        return loss


def __focal_loss(logits, targets, alpha=0.8, gamma=2):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')  # important to add reduction='none' to keep per-batch-item loss
    focal_loss = (alpha * (1 - torch.exp(-ce_loss)) ** gamma * ce_loss).mean()
    return focal_loss


if __name__ == '__main__':
    logits = torch.rand(5, 10)
    targets = torch.randperm(10)[:5]
    logits[:, 3] += 10
    targets[...] = 3
    print(targets)
    
    print('FCE          =', __focal_loss(logits, targets))
    print('FCE + ls0    =', LabelSmoothFocalLoss(num_classes=10, smooth_ratio=0.)(logits, targets))
    print('FCE2 + ls0   =', LabelSmoothFocalLossV2(num_classes=10, smooth_ratio=0.)(logits, targets))
    print('CE           =', F.cross_entropy(logits, targets))
    print('CE  + ls0.1  =', _LabelSmoothCELoss(num_classes=10, smooth_ratio=0.1)(logits, targets))
    print('FCE + ls0.1  =', LabelSmoothFocalLoss(num_classes=10, smooth_ratio=0.1)(logits, targets))
    print('FCE2 + ls0.1 =', LabelSmoothFocalLossV2(num_classes=10, smooth_ratio=0.1)(logits, targets))
