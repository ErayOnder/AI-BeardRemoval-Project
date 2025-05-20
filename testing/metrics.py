import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


def tensor_to_numpy(tensor):
    """
    Convert a PyTorch tensor to a numpy array and scale to 0-255 range.
    
    Args:
        tensor (torch.Tensor): Input tensor in range [-1, 1]
        
    Returns:
        np.ndarray: Numpy array in range [0, 255]
    """
    # Convert to numpy and move channels to last dimension
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    # Scale from [-1, 1] to [0, 255]
    img = ((img + 1) * 127.5).astype(np.uint8)
    return img


def compute_ssim(img1, img2):
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1 (torch.Tensor): First image tensor in range [-1, 1]
        img2 (torch.Tensor): Second image tensor in range [-1, 1]
        
    Returns:
        float: SSIM score
    """
    # Convert tensors to numpy arrays
    img1_np = tensor_to_numpy(img1)
    img2_np = tensor_to_numpy(img2)
    
    # Compute SSIM
    return ssim(img1_np, img2_np, channel_axis=2, data_range=255)


def compute_psnr(img1, img2):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1 (torch.Tensor): First image tensor in range [-1, 1]
        img2 (torch.Tensor): Second image tensor in range [-1, 1]
        
    Returns:
        float: PSNR score in dB
    """
    # Convert tensors to numpy arrays in range [0, 1]
    img1_np = (img1.cpu().numpy() + 1) / 2
    img2_np = (img2.cpu().numpy() + 1) / 2
    
    # Compute MSE
    mse = np.mean((img1_np - img2_np) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    
    # Compute PSNR
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr 