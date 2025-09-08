import torch
import numpy as np
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
from scipy.ndimage import map_coordinates

def _linear_interpolate(value: float, points: list):
    lower_pt = int(np.floor(value))
    upper_pt = int(np.ceil(value))
    
    if lower_pt == upper_pt:
        return points[lower_pt]

    weight = value - lower_pt
    return (1 - weight) * points[lower_pt] + weight * points[upper_pt]

def gaussian_noise(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_levels = [0, 0.04, 0.06, 0.08, 0.09, 0.10]
    scale = _linear_interpolate(severity, c_levels)
    if scale == 0: return image_tensor
    noise = torch.randn_like(image_tensor) * scale
    return torch.clamp(image_tensor + noise, 0, 1)

def shot_noise(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    c_levels = [float('inf'), 500, 250, 100, 75, 50]
    scale = _linear_interpolate(severity, c_levels)
    if scale == float('inf'): 
        return image_tensor

    device = image_tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    image_unnormalized = image_tensor * std + mean
    image_unnormalized = torch.clamp(image_unnormalized, 0, 1)
    corrupted_unnormalized = torch.clamp(torch.poisson(image_unnormalized * scale) / scale, 0, 1)
    corrupted_normalized = (corrupted_unnormalized - mean) / std

    return corrupted_normalized

def contrast(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_levels = [1.0, 0.75, 0.5, 0.4, 0.3, 0.2]
    scale = _linear_interpolate(severity, c_levels)
    if scale == 1.0: return image_tensor
    mean = torch.mean(image_tensor, dim=[-2, -1], keepdim=True)
    return torch.clamp((image_tensor - mean) * scale + mean, 0, 1)

def brightness(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    scale = _linear_interpolate(severity, c_levels)
    if scale == 0: return image_tensor
    return torch.clamp(image_tensor + scale, 0, 1)

def impulse_noise(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_levels = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    amount = _linear_interpolate(severity, c_levels)
    if amount == 0: return image_tensor
    salt_mask = torch.rand_like(image_tensor) < (amount / 2.0)
    pepper_mask = torch.rand_like(image_tensor) < (amount / 2.0)
    out = image_tensor.clone()
    out[salt_mask] = 1.0
    out[pepper_mask] = 0.0
    return out

def elastic_transform(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_alpha = [0, 30, 45, 60, 75, 90]
    c_sigma = [0, 3.5, 4.0, 4.5, 5.0, 5.5]
    
    alpha = _linear_interpolate(severity, c_alpha)
    sigma = _linear_interpolate(severity, c_sigma)
    if alpha == 0: 
        return image_tensor

    image_np = image_tensor.permute(1, 2, 0).numpy()
    shape = image_np.shape

    random_state = np.random.RandomState(None)
    noise_x = random_state.rand(*shape) * 2 - 1
    noise_y = random_state.rand(*shape) * 2 - 1

    dx = scipy_gaussian_filter(noise_x, sigma, mode="reflect") * alpha
    dy = scipy_gaussian_filter(noise_y, sigma, mode="reflect") * alpha

    y, x, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = (np.reshape(y + dy, (-1, 1)), 
               np.reshape(x + dx, (-1, 1)), 
               np.reshape(z, (-1, 1)))

    distorted_np = map_coordinates(image_np, indices, order=1, mode='reflect').reshape(shape)
    
    return torch.from_numpy(distorted_np).permute(2, 0, 1)

CORRUPTION_FUNCS = {
    'gaussian_noise': gaussian_noise, 
    'shot_noise': shot_noise, 
    'impulse_noise': impulse_noise,
    'brightness': brightness,
    'contrast': contrast, 
    'elastic_transform': elastic_transform, 
}

BATCH_CORRUPTION_FUNCS = ['fog'] 
def apply_corruption(image_tensor: torch.Tensor, corruption_name: str, severity: float = 1) -> torch.Tensor:
    if severity == 0 or corruption_name.lower() == 'none':
        return image_tensor
    if not (0 < severity <= 5):
        raise ValueError(f"Severity must be between 0 (exclusive) and 5 (inclusive), but got {severity}")
    if corruption_name not in CORRUPTION_FUNCS:
        raise ValueError(f"Unknown corruption type: {corruption_name}")

    original_device = image_tensor.device
    corruption_func = CORRUPTION_FUNCS[corruption_name]

    if corruption_name in BATCH_CORRUPTION_FUNCS:
        if image_tensor.dim() == 4:
            return corruption_func(image_tensor, severity)
        elif image_tensor.dim() == 3:
            return corruption_func(image_tensor.unsqueeze(0), severity).squeeze(0)
    else:
        image_tensor_cpu = image_tensor.cpu()
        if image_tensor_cpu.dim() == 4: # Batch
            corrupted_images = [corruption_func(img, severity) for img in image_tensor_cpu]
            return torch.stack(corrupted_images).to(original_device)
        elif image_tensor_cpu.dim() == 3: # Single image
            return corruption_func(image_tensor_cpu, severity).to(original_device)
        else:
            raise ValueError("Input tensor must have 3 or 4 dimensions")
