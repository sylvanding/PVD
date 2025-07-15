import os
from typing import List, Optional, Tuple

import imageio
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter


def render_to_2d_image(
    points: np.ndarray,
    image_size: Tuple[int, int],
    psf_sigma_px: float,
    z_range: Optional[Tuple[float, float]] = None,
    intensity_scale: float = 1.0,
    add_background_noise: bool = False,
    background_noise_level: float = 0.05,
    output_dtype=np.uint16,
) -> np.ndarray:
    """
    Render 3D point cloud to 2D image by applying Gaussian blur to simulate microscope point spread function (PSF).
    All spatial coordinates for rendering (points x, y and PSF sigma) are expected in pixel units.

    Args:
        points (np.ndarray): Point cloud array of shape (N, 3). The first two columns (x, y) are in pixel
                             coordinates. The third column (z) is used for filtering and can be in any unit.
        image_size (Tuple[int, int]): Output image dimensions (width, height), in pixels.
        psf_sigma_px (float): Standard deviation of Gaussian PSF, in pixels.
        z_range (Optional[Tuple[float, float]]): A (min, max) tuple for filtering points by Z depth.
                                                 If None, includes all points.
        intensity_scale (float): Factor to scale point intensity before blurring.
        add_background_noise (bool): If True, add Gaussian noise to the final image. Defaults to False.
        background_noise_level (float): Standard deviation for Gaussian noise, relative to max intensity.
                                        Effective only when `add_background_noise` is True. Defaults to 0.05.
        output_dtype: Output image data type (e.g., np.uint8, np.uint16).

    Returns:
        np.ndarray: Rendered 2D image as NumPy array.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input `points` must be an array of shape (N, 3).")

    img_height, img_width = image_size[1], image_size[0]

    # Filter points by Z depth if Z range is specified
    if z_range:
        mask = (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
        points = points[mask]

    if points.shape[0] == 0:
        if not add_background_noise:
            return np.zeros((img_height, img_width), dtype=output_dtype)
        # If noise is requested, generate a noise-only image
        canvas = np.zeros((img_height, img_width), dtype=np.float32)

    else:
        # Point coordinates are already in pixels
        coords_px = points[:, :2]

        # Create a floating-point canvas for rendering
        canvas = np.zeros((img_height, img_width), dtype=np.float32)

        # Get integer pixel coordinates and check boundaries
        x_px = np.floor(coords_px[:, 0]).astype(int)
        y_px = np.floor(coords_px[:, 1]).astype(int)

        valid_indices = (x_px >= 0) & (x_px < img_width) & (y_px >= 0) & (y_px < img_height)
        x_px, y_px = x_px[valid_indices], y_px[valid_indices]

        # Add point intensities to canvas
        # Use np.add.at for efficient addition at specified indices
        np.add.at(canvas, (y_px, x_px), intensity_scale)

    # Apply Gaussian blur to simulate PSF
    blurred_image = gaussian_filter(canvas, sigma=psf_sigma_px)

    # Normalize to [0, 1]
    max_intensity = blurred_image.max()
    if max_intensity > 0:
        normalized_image = blurred_image / max_intensity
    else:
        normalized_image = blurred_image  # This is a zero image

    # Add background noise if enabled
    if add_background_noise:
        noise = np.random.normal(loc=0.0, scale=background_noise_level, size=normalized_image.shape).astype(np.float32)
        normalized_image += noise

    # Clip to [0, 1] range to handle noise that pushes values out of bounds
    np.clip(normalized_image, 0, 1, out=normalized_image)

    # # Scale to output data type
    # max_val = np.iinfo(output_dtype).max
    # scaled_image = normalized_image * max_val

    return normalized_image.astype(output_dtype)


def save_image(image: np.ndarray, output_path: str):
    # image is between 0 and 1
    image = (image * 255).astype(np.uint8)

    if image.ndim == 4:
        for i in range(image.shape[0]):
            imageio.imwrite(output_path.replace('.png', f'_{i}.png'), image[i].squeeze(0))
    else:
        imageio.imwrite(output_path, image.squeeze(0))

    print(f"Image saved to {output_path}")
