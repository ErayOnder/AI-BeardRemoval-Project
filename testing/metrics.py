import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms as T
from PIL import Image

def get_transform():
    """
    Get the standard transform pipeline for normalizing images.
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

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
        img1 (PIL.Image or torch.Tensor): First image
        img2 (PIL.Image or torch.Tensor): Second image
        
    Returns:
        float: SSIM score between 0 and 1
    """
    # Convert PIL images to tensors if needed
    transform = get_transform()
    if isinstance(img1, Image.Image):
        img1 = transform(img1)
    if isinstance(img2, Image.Image):
        img2 = transform(img2)
    
    # Convert tensors to numpy arrays
    img1_np = tensor_to_numpy(img1)
    img2_np = tensor_to_numpy(img2)
    
    # Compute SSIM
    return ssim(img1_np, img2_np, channel_axis=2, data_range=255)

def compute_psnr(img1, img2):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1 (PIL.Image or torch.Tensor): First image
        img2 (PIL.Image or torch.Tensor): Second image
        
    Returns:
        float: PSNR score in dB
    """
    # Convert PIL images to tensors if needed
    transform = get_transform()
    if isinstance(img1, Image.Image):
        img1 = transform(img1)
    if isinstance(img2, Image.Image):
        img2 = transform(img2)
    
    # Convert tensors to numpy arrays in range [0, 1]
    img1_np = (img1.cpu().numpy() + 1) / 2
    img2_np = (img2.cpu().numpy() + 1) / 2
    
    # Compute PSNR
    return psnr(img1_np, img2_np, data_range=1.0) 