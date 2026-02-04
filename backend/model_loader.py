import os
import torch
import numpy as np
import nibabel as nib
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai import transforms
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class BrainTumorSegmentationModel:
    """
    Brain Tumor Segmentation Model using SwinUNETR
    Handles model loading, preprocessing, inference, and postprocessing
    """
    
    def __init__(self, model_path, device=None):
        """
        Initialize the model
        
        Args:
            model_path: Path to the trained model weights (.pth file)
            device: Device to run the model on (cuda/cpu)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = 128
        self.model = None
        self.model_path = model_path
        
        # Initialize preprocessing transforms
        self.preprocess_transform = transforms.Compose([
            transforms.LoadImage(image_only=True),
            transforms.EnsureChannelFirst(),
            transforms.NormalizeIntensity(nonzero=True, channel_wise=True),
        ])
        
        print(f"Using device: {self.device}")
        self.load_model()
    
    def load_model(self):
        """Load the SwinUNETR model with pre-trained weights"""
        try:
            print("Loading SwinUNETR model...")
            
            # Initialize model architecture
            # Note: img_size parameter removed in MONAI 1.5.2+
            # Model now accepts dynamic input sizes (must be divisible by 32)
            self.model = SwinUNETR(
                spatial_dims=3,     # 3D medical images
                in_channels=4,      # 4 MRI modalities: FLAIR, T1, T1ce, T2
                out_channels=3,     # 3 tumor classes: TC, WT, ET
                feature_size=48,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                use_checkpoint=True,
            ).to(self.device)
            
            # Load pre-trained weights
            if os.path.exists(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Model weights loaded from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            self.model.eval()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_nifti(self, file_paths):
        """
        Preprocess NIfTI files for model input
        
        Args:
            file_paths: List of paths to 4 NIfTI files [FLAIR, T1, T1ce, T2]
                       OR single path to a 4-channel NIfTI file
        
        Returns:
            Preprocessed tensor ready for model input
        """
        try:
            if isinstance(file_paths, str):
                # Single 4-channel file - load with nibabel directly
                print(f"Loading single 4-channel file: {file_paths}")
                nii = nib.load(file_paths)
                data = nii.get_fdata()
                print(f"Loaded shape: {data.shape}")
                
                # Convert to tensor
                image = torch.from_numpy(data).float()
                
                # Ensure shape is (4, H, W, D)
                if image.ndim == 4:
                    # Already 4D, check if channels are first
                    if image.shape[0] == 4:
                        print(f"Channels already first: {image.shape}")
                    elif image.shape[-1] == 4:
                        # Channels last, move to first
                        image = image.permute(3, 0, 1, 2)
                        print(f"Moved channels to first: {image.shape}")
                    else:
                        raise ValueError(f"Cannot determine channel dimension in shape {image.shape}")
                else:
                    raise ValueError(f"Expected 4D array, got {image.ndim}D with shape {image.shape}")
                
                # Resize to model input size to reduce memory usage
                # From (4, 240, 240, 155) to (4, 128, 128, 128)
                print(f"Resizing from {image.shape} to (4, {self.img_size}, {self.img_size}, {self.img_size})")
                
                # Use PyTorch's interpolate (expects input: (N, C, D, H, W))
                # Add batch dimension: (1, 4, 240, 240, 155)
                image = image.unsqueeze(0)
                
                # Resize using trilinear interpolation
                import torch.nn.functional as F
                image = F.interpolate(
                    image,
                    size=(self.img_size, self.img_size, self.img_size),
                    mode='trilinear',
                    align_corners=False
                )
                
                # Remove batch dimension: (4, 128, 128, 128)
                image = image.squeeze(0)
                print(f"Resized shape: {image.shape}")
                
                # Normalize intensity per channel
                for c in range(image.shape[0]):
                    channel_data = image[c]
                    # Normalize non-zero values
                    nonzero_mask = channel_data > 0
                    if nonzero_mask.any():
                        mean = channel_data[nonzero_mask].mean()
                        std = channel_data[nonzero_mask].std()
                        if std > 0:
                            image[c][nonzero_mask] = (channel_data[nonzero_mask] - mean) / std
                
            else:
                # Multiple single-channel files
                print(f"Loading {len(file_paths)} separate files")
                images = []
                for file_path in file_paths:
                    img = self.preprocess_transform(file_path)
                    images.append(img)
                image = torch.cat(images, dim=0)
            
            # Ensure correct shape: (4, H, W, D)
            print(f"Final preprocessed shape: {image.shape}")
            if image.shape[0] != 4:
                raise ValueError(f"Expected 4 channels, got {image.shape[0]}")
            
            # Add batch dimension: (1, 4, H, W, D)
            image = image.unsqueeze(0)
            
            return image.to(self.device)
            
        except Exception as e:
            print(f"Error preprocessing NIfTI file: {str(e)}")
            raise
    
    def predict_segmentation(self, image_tensor):
        """
        Run segmentation inference on preprocessed image
        
        Args:
            image_tensor: Preprocessed image tensor (1, 4, D, H, W)
        
        Returns:
            Prediction tensor with 3 channels (TC, WT, ET)
        """
        try:
            with torch.no_grad():
                # Use sliding window inference for better results
                # sw_batch_size=1 to reduce memory usage
                prediction = sliding_window_inference(
                    inputs=image_tensor,
                    roi_size=(self.img_size, self.img_size, self.img_size),
                    sw_batch_size=1,  # Reduced from 3 to save memory
                    predictor=self.model,
                    overlap=0.5
                )
                
                # Apply sigmoid and threshold
                prediction = torch.sigmoid(prediction)
                prediction = (prediction > 0.5).float()
            
            return prediction.cpu()
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
    
    def postprocess_prediction(self, prediction, original_image):
        """
        Post-process prediction for visualization
        
        Args:
            prediction: Model output tensor (1, 3, D, H, W)
            original_image: Original input tensor (1, 4, D, H, W)
        
        Returns:
            Dictionary with visualization images as base64 strings
        """
        try:
            # Remove batch dimension
            pred = prediction[0].numpy()  # (3, D, H, W)
            img = original_image[0].cpu().numpy()  # (4, D, H, W)
            
            # Find the slice with maximum tumor area
            slice_idx = self._find_best_slice(pred)
            
            # Create visualizations
            results = {
                'slice_index': int(slice_idx),
                'visualizations': []
            }
            
            # Channel names
            modality_names = ['FLAIR', 'T1', 'T1ce', 'T2']
            tumor_names = ['Tumor Core (TC)', 'Whole Tumor (WT)', 'Enhancing Tumor (ET)']
            
            # Visualize input modalities
            for i, name in enumerate(modality_names):
                img_b64 = self._create_visualization(
                    img[i, :, :, slice_idx],
                    title=f'Input: {name}',
                    cmap='gray'
                )
                results['visualizations'].append({
                    'type': 'input',
                    'name': name,
                    'image': img_b64
                })
            
            # Visualize predictions
            for i, name in enumerate(tumor_names):
                img_b64 = self._create_visualization(
                    pred[i, :, :, slice_idx],
                    title=f'Prediction: {name}',
                    cmap='hot'
                )
                results['visualizations'].append({
                    'type': 'prediction',
                    'name': name,
                    'image': img_b64
                })
            
            # Create overlay visualization
            overlay_b64 = self._create_overlay_visualization(
                img[2, :, :, slice_idx],  # Use T1ce as background
                pred[:, :, :, slice_idx]
            )
            results['visualizations'].append({
                'type': 'overlay',
                'name': 'Overlay',
                'image': overlay_b64
            })
            
            return results
            
        except Exception as e:
            print(f"Error during post-processing: {str(e)}")
            raise
    
    def _find_best_slice(self, prediction):
        """Find the slice with maximum tumor area"""
        # Sum across all tumor classes and spatial dimensions
        slice_sums = prediction.sum(axis=(0, 1, 2))
        
        if slice_sums.max() > 0:
            return int(np.argmax(slice_sums))
        else:
            # If no tumor detected, return middle slice
            return prediction.shape[-1] // 2
    
    def _create_visualization(self, image_slice, title, cmap='gray'):
        """Create a visualization and return as base64 string"""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_slice, cmap=cmap)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buffer.seek(0)
        
        img_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        return img_b64
    
    def _create_overlay_visualization(self, background, masks):
        """Create overlay visualization with all tumor classes"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Show background (T1ce)
        ax.imshow(background, cmap='gray', alpha=1.0)
        
        # Define colors for each tumor class
        colors = ['red', 'green', 'blue']
        labels = ['TC', 'WT', 'ET']
        
        # Overlay each tumor class with different color
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = masks[i]
            if mask.max() > 0:
                # Create colored overlay
                overlay = np.zeros((*mask.shape, 4))
                if color == 'red':
                    overlay[mask > 0] = [1, 0, 0, 0.5]
                elif color == 'green':
                    overlay[mask > 0] = [0, 1, 0, 0.5]
                elif color == 'blue':
                    overlay[mask > 0] = [0, 0, 1, 0.5]
                
                ax.imshow(overlay, alpha=0.5)
        
        ax.set_title('Segmentation Overlay', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, label='Tumor Core'),
            Patch(facecolor='green', alpha=0.5, label='Whole Tumor'),
            Patch(facecolor='blue', alpha=0.5, label='Enhancing Tumor')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buffer.seek(0)
        
        img_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        return img_b64
    
    def process_file(self, file_path):
        """
        Complete pipeline: preprocess -> predict -> postprocess
        
        Args:
            file_path: Path to NIfTI file(s)
        
        Returns:
            Dictionary with visualization results
        """
        # Preprocess
        image_tensor = self.preprocess_nifti(file_path)
        
        # Predict
        prediction = self.predict_segmentation(image_tensor)
        
        # Postprocess
        results = self.postprocess_prediction(prediction, image_tensor)
        
        return results


# Singleton instance
_model_instance = None

def get_model(model_path):
    """Get or create model instance (singleton pattern)"""
    global _model_instance
    if _model_instance is None:
        _model_instance = BrainTumorSegmentationModel(model_path)
    return _model_instance
