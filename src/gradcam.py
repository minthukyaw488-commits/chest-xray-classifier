"""
Grad-CAM implementation for ConvNeXt visualization.
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, class_idx=None):
        """Generate CAM heatmap for the given input."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
        
        # Compute weights (global average pool of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Upsample to input size
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, class_idx


def overlay_heatmap(pil_image, heatmap, alpha=0.4):
    """
    Overlay a heatmap on the original image.
    
    Args:
        pil_image: PIL Image (original X-ray)
        heatmap: 2D numpy array with values in [0, 1]
        alpha: Blend factor (0 = only image, 1 = only heatmap)
    
    Returns:
        PIL Image with heatmap overlay
    """
    # Resize original image to 224x224
    img = pil_image.convert("RGB").resize((224, 224))
    img_array = np.array(img).astype(float) / 255.0
    
    # Convert heatmap to RGB (jet colormap manually — no matplotlib needed)
    heatmap_rgb = jet_colormap(heatmap)
    
    # Blend
    blended = (1 - alpha) * img_array + alpha * heatmap_rgb
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)


def jet_colormap(values):
    """
    Convert 2D array [0, 1] to RGB using jet-like colormap.
    Manual implementation to avoid matplotlib dependency.
    """
    values = np.clip(values, 0, 1)
    
    # Jet colormap approximation
    r = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    
    rgb = np.stack([r, g, b], axis=-1)
    return rgb