import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
import numpy as np
from PIL import Image, ImageDraw
import os
from typing import Tuple, Optional
import logging
try:
    import face_recognition
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    logging.warning("face_recognition not installed. Dynamic beard masking will not be available.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StableDiffusionGenerator:
    def __init__(self, model_id: str = "sd-legacy/stable-diffusion-v1-5"):
        """
        Initialize the Stable Diffusion pipeline.
        
        Args:
            model_id (str): The model ID from Hugging Face Hub
        """
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            # Initialize pipelines
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.pipe = self.pipe.to(self.device)
            self.inpaint_pipe = self.inpaint_pipe.to(self.device)
            logger.info("Successfully loaded Stable Diffusion models")
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

    def _detect_face_landmarks(self, image: Image.Image) -> Optional[dict]:
        """
        Detect facial landmarks using face_recognition library.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            Optional[dict]: Dictionary with facial landmarks or None if no face detected
        """
        if not FACE_DETECTION_AVAILABLE:
            return None
            
        # Convert PIL image to numpy array for face_recognition
        image_np = np.array(image)
        
        # Find all faces in the image
        face_locations = face_recognition.face_locations(image_np)
        
        if not face_locations:
            logger.warning("No face detected in the image")
            return None
            
        # Get the first face (assuming one person in the image)
        face_location = face_locations[0]  # (top, right, bottom, left)
        
        # Get facial landmarks
        face_landmarks = face_recognition.face_landmarks(image_np, face_locations=[face_location])
        
        if not face_landmarks:
            logger.warning("Face detected but no landmarks found")
            return None
            
        landmarks = face_landmarks[0]
        
        # Add the face bounding box to the landmarks
        result = {
            "landmarks": landmarks,
            "face_box": face_location  # (top, right, bottom, left)
        }
        
        return result

    def _create_beard_mask(self, 
                           image: Image.Image, 
                           margin_sides_factor: float = 0.2, 
                           margin_top_factor: float = 0.3, 
                           margin_bottom_factor: float = 0.1, 
                           shape: str = "ellipse",
                           face_landmarks: Optional[dict] = None) -> Image.Image:
        """
        Create a mask for the beard area of a face, optionally using face detection.
        
        Args:
            image (PIL.Image): Input image
            margin_sides_factor (float): Margin factor for sides of the mask (0.25 means 25% margin)
            margin_top_factor (float): Margin factor for top of the mask (0.2 means 20% margin)
            margin_bottom_factor (float): Margin factor for bottom of the mask (0.3 means 30% margin)
            shape (str): Shape of the mask ("ellipse" or "rectangle")
            face_landmarks (Optional[dict]): Facial landmarks from _detect_face_landmarks
            
        Returns:
            PIL.Image: Binary mask image with white in the beard area
        """
        # Create a blank mask
        mask = Image.new("RGB", image.size, (0, 0, 0))
        draw = ImageDraw.Draw(mask)
        
        # Calculate dimensions
        width, height = image.size
        
        # Initialize mask coordinates to prevent UnboundLocalError in edge cases
        mask_left, mask_top, mask_right, mask_bottom = 0, 0, width, height

        used_precise_landmarks = False
        if face_landmarks and "landmarks" in face_landmarks:
            landmarks_data = face_landmarks["landmarks"]
            
            # Get chin points (these are the most important for beard area)
            chin_points = landmarks_data.get("chin", [])
            
            if chin_points:
                # Get the bottom lip points to determine the top boundary
                bottom_lip_points = landmarks_data.get("bottom_lip", [])
                
                # Calculate the region based on chin points
                x_coords = [p[0] for p in chin_points]
                y_coords = [p[1] for p in chin_points]
                
                if x_coords and y_coords:
                    min_x_region = min(x_coords)
                    max_x_region = max(x_coords)
                    min_y_region = min(y_coords)
                    max_y_region = max(y_coords)
                    
                    region_width = max_x_region - min_x_region
                    region_height = max_y_region - min_y_region
                    
                    # Calculate the top boundary using bottom lip points
                    if bottom_lip_points:
                        bottom_lip_y = min(p[1] for p in bottom_lip_points)
                        # Use the bottom lip as the top boundary, with a small margin
                        mask_top = max(0, bottom_lip_y - int(margin_top_factor * region_height))
                    else:
                        # Fallback if no bottom lip points
                        mask_top = max(0, min_y_region - int(margin_top_factor * region_height))
                    
                    # Calculate other boundaries with margins
                    mask_bottom = min(height, max_y_region + int(margin_bottom_factor * region_height))
                    mask_left = max(0, min_x_region - int(margin_sides_factor * region_width))
                    mask_right = min(width, max_x_region + int(margin_sides_factor * region_width))
                    
                    logger.info("Using precise chin landmarks for beard mask.")
                    used_precise_landmarks = True
            else:
                logger.info("Chin landmarks not found. Attempting fallback.")

        if not used_precise_landmarks:
            face_top_factor = 0.5
            face_bottom_factor = 0.25
            face_width_factor = 0.33
            if face_landmarks and "face_box" in face_landmarks:
                logger.info("Using face bounding box for beard mask (precise landmarks not available or failed).")
                top_fb, right_fb, bottom_fb, left_fb = face_landmarks["face_box"]
                face_width_val = right_fb - left_fb
                face_height_val = bottom_fb - top_fb
                face_center_x = (left_fb + right_fb) // 2
                
                mask_top = top_fb + int(face_height_val * face_top_factor)
                mask_bottom = bottom_fb + int(face_height_val * face_bottom_factor * 0.5)
                mask_left = face_center_x - int(face_width_val * face_width_factor)
                mask_right = face_center_x + int(face_width_val * face_width_factor)
            else:
                # Fallback to the original method if face detection fails entirely
                logger.info("Using fallback method for beard mask generation (no face detection or landmarks).")
                mask_left = int(width * face_width_factor)
                mask_right = int(width - (width * face_width_factor))
                mask_top = int(height * face_top_factor)
                mask_bottom = int(height - (height * face_bottom_factor))
        
        # Ensure mask coordinates are valid before drawing
        if mask_left >= mask_right or mask_top >= mask_bottom:
            logger.warning(
                f"Invalid mask dimensions: L={mask_left}, R={mask_right}, T={mask_top}, B={mask_bottom}. "
                f"This may result in a tiny or no mask. Consider adjusting parameters or image input."
            )
            # Ensure drawable, though Pillow might handle it by drawing nothing or a line
            mask_right = mask_left + 1 if mask_left >= mask_right else mask_right
            mask_bottom = mask_top + 1 if mask_top >= mask_bottom else mask_bottom

        # Draw the beard area in white based on the selected shape
        if shape.lower() == "rectangle":
            draw.rectangle([(mask_left, mask_top), (mask_right, mask_bottom)], fill=(255, 255, 255))
        else:  # Default to ellipse
            draw.ellipse([(mask_left, mask_top), (mask_right, mask_bottom)], fill=(255, 255, 255))
        
        # Convert to grayscale
        mask = mask.convert("L")
        
        return mask

    def generate_paired_images(
        self,
        base_prompt: str,
        seed: int,
        output_dir: str,
        beard_prompt: str = "with a thick brown beard",
        no_beard_prompt: str = "clean-shaven face",
        split: str = "train",
        pair_id: int = 0,
        save_images: bool = True,
        num_inference_steps: int = 75,
        mask_width_factor: float = 0.2,
        mask_height_top_factor: float = 0.3,
        mask_height_bottom_factor: float = 0.1,
        mask_shape: str = "ellipse",
        use_face_detection: bool = True
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Generate a pair of images (with and without beard) using inpainting for consistency.
        
        Args:
            base_prompt (str): Base prompt describing the person
            seed (int): Random seed for reproducibility
            beard_prompt (str): Additional prompt for bearded version
            no_beard_prompt (str): Additional prompt for clean-shaven version
            output_dir (str): Directory to save images
            split (str): Dataset split ('train' or 'test')
            pair_id (int): ID for the image pair
            save_images (bool): Whether to save images to disk
            num_inference_steps (int): Number of inference steps
            mask_width_factor (float): Controls width of beard mask
            mask_height_top_factor (float): Controls top position of beard mask
            mask_height_bottom_factor (float): Controls bottom position of beard mask
            mask_shape (str): Shape of the mask ("ellipse" or "rectangle")
            use_face_detection (bool): Whether to use face detection for dynamic masking
            
        Returns:
            Tuple[PIL.Image, PIL.Image]: Pair of generated images (bearded, clean-shaven)
        """
        # Check if the pair already exists
        beard_path = os.path.join(output_dir, split, "beard", f"person{pair_id:03d}_beard.png")
        no_beard_path = os.path.join(output_dir, split, "no_beard", f"person{pair_id:03d}_nobeard.png")
        
        if os.path.exists(beard_path) and os.path.exists(no_beard_path):
            logger.info(f"Pair {pair_id} already exists in {split} set, skipping generation")
            return Image.open(beard_path), Image.open(no_beard_path)
        
        # Set up the random generator
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        try:
            # Step 1: Generate a base face first, we'll use the clean-shaven version as our base
            clean_base_prompt = f"{base_prompt}, {no_beard_prompt}" if no_beard_prompt not in base_prompt else base_prompt
            clean_image = self.pipe(
                clean_base_prompt,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images[0]
            
            # Detect face if enabled
            face_landmarks = None
            if use_face_detection and FACE_DETECTION_AVAILABLE:
                face_landmarks = self._detect_face_landmarks(clean_image)
                if face_landmarks:
                    logger.info("Face detected successfully. Using dynamic beard masking.")
                else:
                    logger.warning("Face detection failed. Falling back to fixed mask.")
            
            # Create a mask for the beard area
            beard_mask = self._create_beard_mask(
                image=clean_image,
                margin_sides_factor=mask_width_factor,
                margin_top_factor=mask_height_top_factor,
                margin_bottom_factor=mask_height_bottom_factor,
                shape=mask_shape,
                face_landmarks=face_landmarks
            )
            
            # Step 2: Use inpainting to add a beard
            full_beard_prompt = f"{base_prompt}, {beard_prompt}" if beard_prompt not in base_prompt else base_prompt
            bearded_image = self.inpaint_pipe(
                prompt=full_beard_prompt,
                image=clean_image,
                mask_image=beard_mask,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images[0]
            
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
            
        except Exception as e:
            logger.error(f"Failed to generate image pair: {str(e)}")
            raise

    def generate_dataset(
        self,
        num_pairs: int = 100,
        test_split: float = 0.2,
        base_prompt: str = "High-quality 8K, modern head-and-shoulders mugshot photo of a man",
        beard_prompt: str = "with a thick, long, wavy beard",
        no_beard_prompt: str = "with a beardless, smooth clean-shaven face",
        output_dir: str = "dataset",
        start_seed: int = 0,
        mask_width_factor: float = 0.2,
        mask_height_top_factor: float = 0.3,
        mask_height_bottom_factor: float = 0.1,
        mask_shape: str = "ellipse",
        num_inference_steps: int = 50,
        use_face_detection: bool = True
    ) -> None:
        """
        Generate a dataset of paired images with train/test splits.
        
        Args:
            num_pairs (int): Number of image pairs to generate
            test_split (float): Proportion of pairs to use for testing (default: 0.2)
            base_prompt (str): Base prompt for face generation
            beard_prompt (str): Prompt for adding beard
            no_beard_prompt (str): Prompt for clean-shaven face
            output_dir (str): Directory to save the dataset
            start_seed (int): Starting seed for reproducibility
            mask_width_factor (float): Controls width of beard mask
            mask_height_top_factor (float): Controls top position of beard mask
            mask_height_bottom_factor (float): Controls bottom position of beard mask
            mask_shape (str): Shape of the mask ("ellipse" or "rectangle")
            num_inference_steps (int): Number of inference steps
            use_face_detection (bool): Whether to use face detection for dynamic masking
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
                pair_id = i if split == "test" else i - num_test_pairs
                
                # Skip if we already have enough pairs for this split
                if (split == "train" and pair_id < existing_train_pairs) or \
                   (split == "test" and pair_id < existing_test_pairs):
                    continue
                
                self.generate_paired_images(
                    base_prompt=base_prompt,
                    beard_prompt=beard_prompt,
                    no_beard_prompt=no_beard_prompt,
                    seed=seed,
                    output_dir=output_dir,
                    split=split,
                    pair_id=pair_id,
                    num_inference_steps=num_inference_steps,
                    mask_width_factor=mask_width_factor,
                    mask_height_top_factor=mask_height_top_factor,
                    mask_height_bottom_factor=mask_height_bottom_factor,
                    mask_shape=mask_shape,
                    use_face_detection=use_face_detection
                )
                total_generated += 1
                
                if (total_generated + existing_train_pairs + existing_test_pairs) % 10 == 0:
                    logger.info(f"Generated {total_generated} new pairs")
            except Exception as e:
                logger.error(f"Failed to generate pair {i}: {str(e)}")
                continue
        
        logger.info(f"Dataset generation complete! Total pairs: {num_pairs} "
                   f"(Train: {num_train_pairs}, Test: {num_test_pairs})")
