import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from training.dataset import BeardDataset
from training.models import UNetGenerator, PatchDiscriminator
from training.losses import gan_loss, l1


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pix2Pix for beard removal")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lambda_l1", type=float, default=100.0, help="Weight for L1 loss")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default="models", help="Directory for model checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, cpu)")
    return parser.parse_args()


def validate(model, val_loader, device):
    """Run validation and compute metrics."""
    model.eval()
    total_ssim = 0
    total_psnr = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_img = batch["input"].to(device)
            target_img = batch["target"].to(device)
            
            # Generate fake image
            fake_img = model(input_img)
            
            # Convert to numpy for metric computation
            fake_np = fake_img.cpu().numpy()
            target_np = target_img.cpu().numpy()
            
            # Compute metrics for each image in batch
            for i in range(fake_np.shape[0]):
                # Convert from [-1,1] to [0,1] range
                fake = (fake_np[i].transpose(1,2,0) + 1) / 2
                target = (target_np[i].transpose(1,2,0) + 1) / 2
                
                # Compute metrics
                total_ssim += ssim(fake, target, channel_axis=2, data_range=1.0)
                total_psnr += psnr(target, fake, data_range=1.0)
                num_samples += 1
    
    # Return average metrics
    return total_ssim / num_samples, total_psnr / num_samples


def train(args):
    """Train the model"""
    # Set device - Use the provided device if available
    if hasattr(args, 'device') and args.device:
        # If it's already a torch.device object, use it directly
        if isinstance(args.device, torch.device):
            device = args.device
        else:
            # Otherwise convert string to torch.device
            device = torch.device(args.device)
    else:
        # Fallback to automatic detection
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create datasets and dataloaders
    train_dataset = BeardDataset(args.data_dir, split="train")
    val_dataset = BeardDataset(args.data_dir, split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize models
    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    
    # Initialize optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Initialize learning rate schedulers
    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='max', factor=0.5, patience=5, verbose=True)
    scheduler_D = ReduceLROnPlateau(optimizer_D, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Track best validation metrics
    best_ssim = 0.0
    start_epoch = 0
    
    # Check for previously saved models and load if available
    best_model_path = checkpoint_dir / "best_generator.pth"    
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        if 'best_ssim' in checkpoint:
            best_ssim = checkpoint['best_ssim']
            print(f"Previous best SSIM: {best_ssim:.4f}")
    
    # Initialize metrics tracking
    metrics = {
        'epoch': [],
        'D_loss': [],
        'G_loss': [],
        'G_GAN': [],
        'G_L1': [],
        'val_ssim': [],
        'val_psnr': []
    }
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        generator.train()
        discriminator.train()
        
        # Initialize progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        # Initialize running averages
        running_loss_D = 0
        running_loss_G = 0
        running_loss_G_GAN = 0
        running_loss_G_L1 = 0
        
        for batch in pbar:
            # Get images
            input_img = batch["input"].to(device)
            target_img = batch["target"].to(device)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Generate fake image
            fake_img = generator(input_img)
            
            # Real images
            pred_real = discriminator(input_img, target_img)
            loss_real = gan_loss(pred_real, True)
            
            # Fake images
            pred_fake = discriminator(input_img, fake_img.detach())
            loss_fake = gan_loss(pred_fake, False)
            
            # Total discriminator loss
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()
            
            # Generate fake image again
            fake_img = generator(input_img)
            
            # Adversarial loss
            pred_fake = discriminator(input_img, fake_img)
            loss_G_GAN = gan_loss(pred_fake, True)
            
            # L1 loss
            loss_G_L1 = l1(fake_img, target_img) * args.lambda_l1
            
            # Total generator loss
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()
            
            # Update running averages
            running_loss_D += loss_D.item()
            running_loss_G += loss_G.item()
            running_loss_G_GAN += loss_G_GAN.item()
            running_loss_G_L1 += loss_G_L1.item()
            
            # Update progress bar
            pbar.set_postfix({
                "D_loss": running_loss_D / (pbar.n + 1),
                "G_loss": running_loss_G / (pbar.n + 1),
                "G_GAN": running_loss_G_GAN / (pbar.n + 1),
                "G_L1": running_loss_G_L1 / (pbar.n + 1)
            })
        
        # Calculate average losses for the epoch
        avg_loss_D = running_loss_D / len(train_loader)
        avg_loss_G = running_loss_G / len(train_loader)
        avg_loss_G_GAN = running_loss_G_GAN / len(train_loader)
        avg_loss_G_L1 = running_loss_G_L1 / len(train_loader)
        
        # Validation
        val_ssim, val_psnr = validate(generator, val_loader, device)
        print(f"\nValidation - SSIM: {val_ssim:.4f}, PSNR: {val_psnr:.4f}")
        
        # Store metrics
        metrics['epoch'].append(epoch + 1)
        metrics['D_loss'].append(avg_loss_D)
        metrics['G_loss'].append(avg_loss_G)
        metrics['G_GAN'].append(avg_loss_G_GAN)
        metrics['G_L1'].append(avg_loss_G_L1)
        metrics['val_ssim'].append(val_ssim)
        metrics['val_psnr'].append(val_psnr)
        
        # Step the schedulers based on validation SSIM
        scheduler_G.step(val_ssim)
        scheduler_D.step(val_ssim)
        
        # Save best model if we have a new best SSIM
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer_G.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
                'val_ssim': val_ssim,
                'val_psnr': val_psnr,
                'best_ssim': best_ssim,
                'metrics': metrics,
            }, best_model_path)
            print(f"New best model saved with SSIM: {best_ssim:.4f}")
    
    return metrics, generator


if __name__ == "__main__":
    args = parse_args()
    train(args) 