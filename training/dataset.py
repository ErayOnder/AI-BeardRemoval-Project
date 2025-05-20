import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from typing import Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BeardDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, split: str = "train", img_size: int = 256):
        """
        Initialize the beard removal dataset.
        
        Args:
            root_dir (str): Root directory containing 'train/beard', 'train/no_beard', 'test/beard', 'test/no_beard' subfolders
            split (str): Dataset split ('train' or 'test')
            img_size (int): Size to resize images to (both width and height)
        """
        self.root_dir = Path(root_dir) / split
        self.img_size = img_size
        
        # Get paths to beard and no-beard images
        beard_dir = self.root_dir / "beard"
        no_beard_dir = self.root_dir / "no_beard"
        
        # Check if directories exist
        if not beard_dir.exists():
            logger.warning(f"Beard directory not found: {beard_dir}")
        if not no_beard_dir.exists():
            logger.warning(f"No-beard directory not found: {no_beard_dir}")
        
        # Collect matching pairs
        self.pairs = []
        for beard_path in beard_dir.glob("*_beard.png"):
            # Get corresponding no-beard image path
            no_beard_path = no_beard_dir / beard_path.name.replace("_beard.png", "_nobeard.png")
            if no_beard_path.exists():
                self.pairs.append((beard_path, no_beard_path))
        
        logger.info(f"Found {len(self.pairs)} image pairs in {split} set")
        
        # Define image transformations
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # Range: [-1, 1]
        ])
    
    def __len__(self) -> int:
        """Return the number of image pairs in the dataset."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a pair of images (beard and no-beard).
        
        Args:
            idx (int): Index of the image pair
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing input (beard) and target (no-beard) tensors
        """
        beard_path, no_beard_path = self.pairs[idx]
        
        # Load and transform images
        try:
            beard_img = Image.open(beard_path).convert("RGB")
            no_beard_img = Image.open(no_beard_path).convert("RGB")
            
            # Apply the same transform to both images to ensure they have the same size
            beard_tensor = self.transform(beard_img)
            no_beard_tensor = self.transform(no_beard_img)
            
            # Verify tensor shapes
            if beard_tensor.shape != no_beard_tensor.shape:
                logger.warning(f"Tensor shape mismatch: {beard_tensor.shape} vs {no_beard_tensor.shape}")
                # Force resize if needed
                if beard_tensor.shape != torch.Size([3, self.img_size, self.img_size]):
                    beard_tensor = T.functional.resize(beard_tensor, (self.img_size, self.img_size))
                if no_beard_tensor.shape != torch.Size([3, self.img_size, self.img_size]):
                    no_beard_tensor = T.functional.resize(no_beard_tensor, (self.img_size, self.img_size))
            
            return {
                "input": beard_tensor,
                "target": no_beard_tensor
            }
        except Exception as e:
            logger.error(f"Error loading images {beard_path} or {no_beard_path}: {e}")
            # Return a default tensor pair in case of error
            default_tensor = torch.zeros(3, self.img_size, self.img_size)
            return {"input": default_tensor, "target": default_tensor} 