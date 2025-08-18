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
        # # Check for NaN/Inf in inputs
        # if torch.isnan(inputs).any() or torch.isinf(inputs).any():
        #     print(f"WARNING: FocalLoss inputs contains NaN/Inf values")
        #     print(f"Inputs stats - min: {inputs.min().item():.4f}, max: {inputs.max().item():.4f}, mean: {inputs.mean().item():.4f}")
        
        # # Check for NaN/Inf in targets
        # if torch.isnan(targets).any() or torch.isinf(targets).any():
        #     print(f"WARNING: FocalLoss targets contains NaN/Inf values")
        #     print(f"Targets stats - min: {targets.min().item():.4f}, max: {targets.max().item():.4f}, mean: {targets.mean().item():.4f}")

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
        # # Check for NaN/Inf in logits
        # if torch.isnan(logits).any() or torch.isinf(logits).any():
        #     print(f"WARNING: SampleWeightedLoss logits contains NaN/Inf values")
        #     print(f"Logits stats - min: {logits.min().item():.4f}, max: {logits.max().item():.4f}, mean: {logits.mean().item():.4f}")
        
        # # Check for NaN/Inf in targets
        # if torch.isnan(targets).any() or torch.isinf(targets).any():
        #     print(f"WARNING: SampleWeightedLoss targets contains NaN/Inf values")
        #     print(f"Targets stats - min: {targets.min().item():.4f}, max: {targets.max().item():.4f}, mean: {targets.mean().item():.4f}")

        if len(targets.shape) < len(logits.shape):
            targets = targets.view(-1, 1)
        
        # Check for division by zero in pos_weight
        num_pos = (targets==1).sum()
        if num_pos == 0:
            print("WARNING: No positive samples in batch for pos_weight calculation")
        pos_weight = (1-self.beta)/(self.beta) * (targets==0).sum()/max(num_pos, 1)
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pos_weight
        )

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Apply label smoothing: convert 1 -> 1 - eps, and 0 -> eps
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

class BinaryLDAMLoss(nn.Module):
    def __init__(self, cls_num_list, device, margin=0.1, s=30):
        super().__init__()
        # Ensure cls_num_list is in tensor format
        if not isinstance(cls_num_list, torch.Tensor):
            cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
        
        # Calculate margin list based on class counts
        # For binary classification, we only care about the margin for positive samples; the margin for negative samples is 0.
        # Here, m_list will have two elements: m_list[0] for negative samples, m_list[1] for positive samples.
        # We can calculate m_list according to the original LDAMLoss calculation method.
        m_list = 1.0 / torch.sqrt(torch.sqrt(cls_num_list.float()))
        m_list = m_list * (margin / torch.max(m_list)) # Use the passed margin as the maximum margin
        m_list = m_list.to(device)
        
        self.m_list = m_list
        self.s = s
        self.device = device

    def forward(self, x, target):
        # Check for NaN/Inf in inputs
        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     print(f"WARNING: BinaryLDAMLoss inputs contains NaN/Inf values")
        #     print(f"Inputs stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}")
        
        # # Check for NaN/Inf in targets
        # if torch.isnan(target).any() or torch.isinf(target).any():
        #     print(f"WARNING: BinaryLDAMLoss targets contains NaN/Inf values")
            # print(f"Targets stats - min: {target.min().item():.4f}, max: {target.max().item():.4f}, mean: {target.mean().item():.4f}")

        # The shape of target is usually (batch_size, 1) or (batch_size,)
        # We need to select the margin from m_list based on the target value (0 or 1).
        # If target is (batch_size, 1), use directly.
        # If target is (batch_size,), it needs to be viewed as (-1, 1).
        
        # Ensure target is long type for indexing
        target_long = target.long()
        
        # Select the corresponding margin based on the target value
        # m_list[0] corresponds to target=0 (negative samples)
        # m_list[1] corresponds to target=1 (positive samples)
        # Use gather or direct indexing
        batch_m = self.m_list[target_long]
        
        # Ensure batch_m's shape matches x for subtraction
        # If x is (batch_size, 1), batch_m should also be (batch_size, 1)
        if batch_m.dim() == 1:
            batch_m = batch_m.view(-1, 1)
        
        # Check margin value
        # if torch.isnan(batch_m).any() or torch.isinf(batch_m).any():
        #     print(f"WARNING: Invalid batch_m value: {batch_m}")
            
        # Apply margin
        x_m = x - batch_m
        
        # Check output values
        # if torch.isnan(x_m).any() or torch.isinf(x_m).any():
        #     print("WARNING: Invalid values in x_m (x - margin)")
        
        # Calculate loss using binary_cross_entropy_with_logits
        return F.binary_cross_entropy_with_logits(self.s * x_m, target.float())

class BinaryLMFLoss(nn.Module):
    def __init__(self, cls_num_list, device, alpha=0.5, beta=0.5, focal_gamma=2,
                 ldam_margin=0.1, ldam_s=30, label_smoothing=0.0):  # Add label_smoothing
        super().__init__()

        # Ensure cls_num_list is in tensor format
        if not isinstance(cls_num_list, torch.Tensor):
            cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
        
        # Calculate alpha weight for Focal Loss based on class counts
        # For binary classification, cls_num_list is typically [neg_samples, pos_samples]
        # In the Focal Loss paper, alpha is usually set to the proportion of positive samples, or a fixed value.
        # Here, we calculate alpha based on the sample ratio.
        neg_samples = cls_num_list[0].item()
        pos_samples = cls_num_list[1].item()
        total_samples = neg_samples + pos_samples
        focal_alpha = neg_samples / total_samples if total_samples > 0 else 0.5

        # Focal Loss component, using the calculated weight
        self.focal_loss = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma, label_smoothing=label_smoothing)
        self.ldam_loss = BinaryLDAMLoss(cls_num_list, device, margin=ldam_margin, s=ldam_s)

        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        # Check for NaN/Inf in output
        # if torch.isnan(output).any() or torch.isinf(output).any():
        #     print(f"WARNING: BinaryLMFLoss output contains NaN/Inf values")
        #     print(f"Output stats - min: {output.min().item():.4f}, max: {output.max().item():.4f}, mean: {output.mean().item():.4f}")
        
        # Check for NaN/Inf in target
        # if torch.isnan(target).any() or torch.isinf(target).any():
        #     print(f"WARNING: BinaryLMFLoss target contains NaN/Inf values")
        #     print(f"Target stats - min: {target.min().item():.4f}, max: {target.max().item():.4f}, mean: {target.mean().item():.4f}")

        focal_loss_output = self.focal_loss(output, target)
        ldam_loss_output = self.ldam_loss(output, target)
        
        total_loss = self.alpha * focal_loss_output + self.beta * ldam_loss_output
        return total_loss

# Helper function to create BinaryLMFLoss
def create_binary_lmf_loss(pos_samples, neg_samples, device, 
                           alpha=None, beta=None, focal_gamma=None, 
                           ldam_margin=None, ldam_s=None, label_smoothing=0.0):
    """
    Creates a binary LMF loss function.
    
    Args:
        pos_samples: Number of positive samples.
        neg_samples: Number of negative samples.
        device: Device to run on.
        alpha: Weight for the Focal Loss part of the total LMF loss.
        beta: Weight for the LDAM Loss part of the total LMF loss.
        focal_gamma: Gamma parameter for Focal Loss.
        ldam_margin: Margin for LDAM.
        ldam_s: Scale parameter for LDAM.
        label_smoothing: Label smoothing parameter for Focal Loss.
    """
    cls_num_list = [neg_samples, pos_samples]
    
    return BinaryLMFLoss(
        cls_num_list=cls_num_list, 
        device=device, 
        alpha=alpha, 
        beta=beta, 
        focal_gamma=focal_gamma,
        ldam_margin=ldam_margin, 
        ldam_s=ldam_s,
        label_smoothing=label_smoothing
    )

class ConfusionLoss(nn.Module):
    def __init__(self, task=0):
        super(ConfusionLoss, self).__init__()
        self.task = task

    def forward(self, x, target):
        # We only care about x
        # Apply sigmoid to x to ensure values are between 0 and 1 before taking log
        # This assumes x are logits.
        x = torch.sigmoid(x)

        # Add a small epsilon to x to prevent log(0) which results in NaN
        epsilon = 1e-12
        log = torch.log(x + epsilon)
        
        # Check for NaN/Inf in log
        if torch.isnan(log).any() or torch.isinf(log).any():
            print(f"WARNING: ConfusionLoss log contains NaN/Inf values")
            print(f"x stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}")
            print(f"log stats - min: {log.min().item():.4f}, max: {log.max().item():.4f}, mean: {log.mean().item():.4f}")

        log_sum = torch.sum(log, dim=1)
        
        # Check for NaN/Inf in log_sum
        if torch.isnan(log_sum).any() or torch.isinf(log_sum).any():
            print(f"WARNING: ConfusionLoss log_sum contains NaN/Inf values")
            print(f"log_sum stats - min: {log_sum.min().item():.4f}, max: {log_sum.max().item():.4f}, mean: {log_sum.mean().item():.4f}")

        normalised_log_sum = torch.div(log_sum,  x.size()[1])
        
        # Check for NaN/Inf in normalised_log_sum
        if torch.isnan(normalised_log_sum).any() or torch.isinf(normalised_log_sum).any():
            print(f"WARNING: ConfusionLoss normalised_log_sum contains NaN/Inf values")
            print(f"normalised_log_sum stats - min: {normalised_log_sum.min().item():.4f}, max: {normalised_log_sum.max().item():.4f}, mean: {normalised_log_sum.mean().item():.4f}")

        loss = torch.mul(torch.sum(normalised_log_sum, dim=0), -1)
        
        # Check for NaN/Inf in final loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"WARNING: ConfusionLoss final loss contains NaN/Inf values")
            print(f"Loss value: {loss.item():.4f}")

        return loss

class ELRLoss(nn.Module):
    """
    Early Learning Regularization (ELR) Loss.
    
    This loss function is designed to handle noisy labels by combining a standard 
    classification loss with a regularization term that encourages the model's
    predictions to conform to a moving average of its own past predictions.
    
    Args:
        num_examples (int): The total number of examples in the training dataset.
        num_classes (int): The number of classes. Default: 1 (for binary classification).
        elr_lambda (float): The weight of the ELR regularization term.
        elr_beta (float): The momentum parameter for updating the soft target labels.
    """
    def __init__(self, num_examples, num_classes=1, elr_lambda=3.0, elr_beta=0.7, device='cpu'):
        super(ELRLoss, self).__init__()
        self.num_classes = num_classes
        self.elr_lambda = elr_lambda
        self.elr_beta = elr_beta
        self.device = device

        self.register_buffer(
            'target_labels', 
            torch.zeros(num_examples, self.num_classes).to(self.device)
        )

    def forward(self, index, output, target):
        """
        Calculates the ELR loss.
        
        Args:
            index (torch.Tensor): A tensor of indices for the current batch of data.
                                  This is crucial for updating the correct soft labels.
            output (torch.Tensor): The raw logits from the model (batch_size, num_classes).
            target (torch.Tensor): The ground truth (potentially noisy) labels.
            
        Returns:
            torch.Tensor: The final computed ELR loss.
        """
        if target.dim() < output.dim():
            target = target.view(-1, 1).float()

        classification_loss = F.binary_cross_entropy_with_logits(output, target)
        
        current_soft_labels = self.target_labels[index]
        
        probs = torch.sigmoid(output) if self.num_classes == 1 else F.softmax(output, dim=1)
        
        reg_loss = self.elr_lambda * torch.mean((probs - current_soft_labels.detach())**2)

        self.target_labels[index] = self.elr_beta * self.target_labels[index] + (1. - self.elr_beta) * probs.detach()
        
        return classification_loss + reg_loss
