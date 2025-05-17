import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from typing import Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StableDiffusionGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        Initialize the Stable Diffusion pipeline.
        
        Args:
            model_id (str): The model ID from Hugging Face Hub
        """
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.pipe = self.pipe.to(self.device)
            logger.info("Successfully loaded Stable Diffusion model")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _count_existing_pairs(self, output_dir: str, split: str) -> int:
        """
        Count the number of existing image pairs in the specified directory.
        
        Args:
            output_dir (str): Base directory for the dataset
            split (str): Dataset split ('train' or 'test')
            
        Returns:
            int: Number of complete pairs found
        """
        beard_dir = os.path.join(output_dir, split, "beard")
        no_beard_dir = os.path.join(output_dir, split, "no_beard")
        
        if not os.path.exists(beard_dir) or not os.path.exists(no_beard_dir):
            return 0
            
        beard_files = set(f for f in os.listdir(beard_dir) if f.endswith('_beard.png'))
        no_beard_files = set(f.replace('_beard.png', '_nobeard.png') for f in beard_files)
        
        # Count only complete pairs
        complete_pairs = sum(1 for f in beard_files if f.replace('_beard.png', '_nobeard.png') in no_beard_files)
        return complete_pairs

    def generate_image(self, prompt: str, seed: int, num_inference_steps: int = 50) -> Image.Image:
        """
        Generate a single image using Stable Diffusion.
        
        Args:
            prompt (str): Text prompt for image generation
            seed (int): Random seed for reproducibility
            num_inference_steps (int): Number of denoising steps
            
        Returns:
            PIL.Image: Generated image
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        try:
            image = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images[0]
            return image
        except Exception as e:
            logger.error(f"Failed to generate image: {str(e)}")
            raise

    def generate_paired_images(
        self,
        base_prompt: str,
        seed: int,
        beard_prompt: str = "with a thick brown beard",
        no_beard_prompt: str = "clean-shaven face",
        output_dir: str = "dataset",
        split: str = "train",
        pair_id: int = 0,
        save_images: bool = True
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Generate a pair of images (with and without beard) using the same seed.
        
        Args:
            base_prompt (str): Base prompt describing the person
            seed (int): Random seed for reproducibility
            beard_prompt (str): Additional prompt for bearded version
            no_beard_prompt (str): Additional prompt for clean-shaven version
            output_dir (str): Directory to save images
            split (str): Dataset split ('train' or 'test')
            pair_id (int): ID for the image pair
            save_images (bool): Whether to save images to disk
            
        Returns:
            Tuple[PIL.Image, PIL.Image]: Pair of generated images (bearded, clean-shaven)
        """
        # Check if the pair already exists
        beard_path = os.path.join(output_dir, split, "beard", f"person{pair_id:03d}_beard.png")
        no_beard_path = os.path.join(output_dir, split, "no_beard", f"person{pair_id:03d}_nobeard.png")
        
        if os.path.exists(beard_path) and os.path.exists(no_beard_path):
            logger.info(f"Pair {pair_id} already exists in {split} set, skipping generation")
            return Image.open(beard_path), Image.open(no_beard_path)
        
        # Construct full prompts
        bearded_prompt = f"{base_prompt}, {beard_prompt}"
        clean_prompt = f"{base_prompt}, {no_beard_prompt}"
        
        # Generate images
        bearded_image = self.generate_image(bearded_prompt, seed)
        clean_image = self.generate_image(clean_prompt, seed)
        
        if save_images:
            # Create output directories if they don't exist
            beard_dir = os.path.join(output_dir, split, "beard")
            no_beard_dir = os.path.join(output_dir, split, "no_beard")
            os.makedirs(beard_dir, exist_ok=True)
            os.makedirs(no_beard_dir, exist_ok=True)
            
            # Save images
            bearded_image.save(beard_path)
            clean_image.save(no_beard_path)
            
        return bearded_image, clean_image

    def generate_dataset(
        self,
        num_pairs: int = 100,
        test_split: float = 0.2,
        base_prompt: str = "A portrait photo of a young man, smiling, studio lighting",
        output_dir: str = "dataset",
        start_seed: int = 42
    ) -> None:
        """
        Generate a dataset of paired images with train/test splits.
        
        Args:
            num_pairs (int): Number of image pairs to generate
            test_split (float): Proportion of pairs to use for testing (default: 0.2)
            base_prompt (str): Base prompt for image generation
            output_dir (str): Directory to save the dataset
            start_seed (int): Starting seed for reproducibility
        """
        # Calculate number of pairs for each split
        num_test_pairs = int(num_pairs * test_split)
        num_train_pairs = num_pairs - num_test_pairs
        
        # Check existing pairs
        existing_train_pairs = self._count_existing_pairs(output_dir, "train")
        existing_test_pairs = self._count_existing_pairs(output_dir, "test")
        
        logger.info(f"Found {existing_train_pairs} existing training pairs and {existing_test_pairs} existing test pairs")
        
        # Create all necessary directories
        for split in ['train', 'test']:
            for subdir in ['beard', 'no_beard']:
                os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
        
        # Generate pairs
        total_generated = 0
        for i in range(num_pairs):
            seed = start_seed + i
            try:
                # Determine if this pair goes to train or test
                split = "test" if i < num_test_pairs else "train"
                pair_id = i - num_test_pairs if split == "train" else i
                
                # Skip if we already have enough pairs for this split
                if (split == "train" and pair_id < existing_train_pairs) or \
                   (split == "test" and pair_id < existing_test_pairs):
                    continue
                
                self.generate_paired_images(
                    base_prompt=base_prompt,
                    seed=seed,
                    output_dir=output_dir,
                    split=split,
                    pair_id=pair_id
                )
                total_generated += 1
                
                if (total_generated + existing_train_pairs + existing_test_pairs) % 10 == 0:
                    logger.info(f"Generated {total_generated} new pairs")
            except Exception as e:
                logger.error(f"Failed to generate pair {i}: {str(e)}")
                continue
        
        logger.info(f"Dataset generation complete! Total pairs: {num_pairs} "
                   f"(Train: {num_train_pairs}, Test: {num_test_pairs})")

# Example usage:
if __name__ == "__main__":
    # Initialize the generator
    generator = StableDiffusionGenerator()
    
    # Generate a single pair (for testing)
    bearded, clean = generator.generate_paired_images(
        base_prompt="A portrait photo of a young man, smiling, studio lighting",
        seed=42,
        split="train",
        pair_id=0
    )
    
    # Or generate a full dataset with train/test splits
    # generator.generate_dataset(num_pairs=100, test_split=0.2) 