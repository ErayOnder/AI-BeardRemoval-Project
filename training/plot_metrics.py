import matplotlib.pyplot as plt
import torch

def plot_training_metrics(metrics):
    """
    Plot training and validation metrics.
    
    Args:
        metrics (dict): Dictionary containing tracked metrics
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot generator and discriminator losses
    axs[0, 0].plot(metrics['epoch'], metrics['G_loss'], 'b-', label='Generator Loss')
    axs[0, 0].plot(metrics['epoch'], metrics['D_loss'], 'r-', label='Discriminator Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Generator and Discriminator Losses')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot generator component losses
    axs[0, 1].plot(metrics['epoch'], metrics['G_GAN'], 'g-', label='GAN Loss')
    axs[0, 1].plot(metrics['epoch'], metrics['G_L1'], 'm-', label='L1 Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_title('Generator Component Losses')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot validation SSIM
    axs[1, 0].plot(metrics['epoch'], metrics['val_ssim'], 'c-')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('SSIM')
    axs[1, 0].set_title('Validation SSIM')
    axs[1, 0].grid(True)
    
    # Plot validation PSNR
    axs[1, 1].plot(metrics['epoch'], metrics['val_psnr'], 'y-')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('PSNR (dB)')
    axs[1, 1].set_title('Validation PSNR')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def load_metrics_from_checkpoint(checkpoint_path, device='cpu'):
    """
    Load metrics from a saved checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        device (str): Device to load the checkpoint to
        
    Returns:
        dict: Dictionary containing metrics or None if no metrics found
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'metrics' in checkpoint:
        return checkpoint['metrics']
    else:
        print("No metrics found in checkpoint")
        return None 