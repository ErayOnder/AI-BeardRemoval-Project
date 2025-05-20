import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import argparse

from training.models import UNetGenerator


def load_model(model_path, device):
    """
    Load the trained generator model.
    
    Args:
        model_path (str): Path to the model checkpoint
        device (torch.device): Device to load the model on
        
    Returns:
        UNetGenerator: Loaded model
    """
    model = UNetGenerator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def get_transform(img_size=256):
    """
    Get the same transform pipeline used during training.
    
    Args:
        img_size (int): Size to resize images to
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # Range: [-1, 1]
    ])


def tensor_to_pil(tensor):
    """
    Convert a PyTorch tensor to a PIL Image.
    
    Args:
        tensor (torch.Tensor): Input tensor in range [-1, 1]
        
    Returns:
        PIL.Image: Output image
    """
    # Convert to numpy and move channels to last dimension
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    # Scale from [-1, 1] to [0, 1]
    img = (img + 1) / 2
    # Scale to 0-255 and convert to uint8
    img = (img * 255).astype('uint8')
    return Image.fromarray(img)


def predict(img_path, model_path, device=None):
    """
    Generate a beard-removed version of the input image.
    
    Args:
        img_path (str): Path to input image
        model_path (str): Path to model checkpoint
        device (torch.device, optional): Device to run inference on
        
    Returns:
        PIL.Image: Generated image
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(model_path, device)
    
    # Load and transform image
    transform = get_transform()
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Generate output
    with torch.no_grad():
        output_tensor = model(img_tensor)
    
    # Convert to PIL Image
    output_img = tensor_to_pil(output_tensor[0])
    
    return output_img


def main():
    parser = argparse.ArgumentParser(description="Generate beard-removed images")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="models/generator_epoch_200.pth", help="Path to model checkpoint")
    parser.add_argument("--output", type=str, help="Path to save output image (if not specified, will show image)")
    args = parser.parse_args()
    
    # Generate output
    output_img = predict(args.input, args.model)
    
    # Save or show
    if args.output:
        output_img.save(args.output)
        print(f"Saved output to {args.output}")
    else:
        output_img.show()


if __name__ == "__main__":
    main() 