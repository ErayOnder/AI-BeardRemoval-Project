import torch
import torch.nn as nn

# Loss functions
bce = nn.BCELoss()        # GAN loss
l1 = nn.L1Loss()          # Reconstruction loss

def gan_loss(pred, real=True):
    """
    Calculate the GAN loss for the discriminator or generator.
    
    Args:
        pred (torch.Tensor): Discriminator predictions
        real (bool): Whether to use real (1) or fake (0) targets
        
    Returns:
        torch.Tensor: The calculated loss
    """
    targets = torch.ones_like(pred) if real else torch.zeros_like(pred)
    return bce(pred, targets) 