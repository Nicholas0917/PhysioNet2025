import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        targets = targets.view(-1, 1).float()
        
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class SampleWeightedLoss(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        self.beta = beta
    
    def forward(self, logits, targets):
        if len(targets.shape) < len(logits.shape):
            targets = targets.view(-1, 1)
        
        pos_weight = (1-self.beta)/(self.beta) * (targets==0).sum()/(targets==1).sum()
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pos_weight
        )
