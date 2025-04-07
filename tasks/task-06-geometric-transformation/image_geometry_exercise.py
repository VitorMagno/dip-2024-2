# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    H, W = img.shape
    result = {}

    # 1. Translated image (shift right and down)
    shift_y, shift_x = 20, 30  # Example shift
    translated = np.zeros_like(img)
    translated[shift_y:, shift_x:] = img[:H - shift_y, :W - shift_x]
    result['translated'] = translated

    # 2. Rotated image (90 degrees clockwise)
    rotated = np.rot90(img, k=-1)
    result['rotated'] = rotated

    # 3. Horizontally stretched image (scale width by 1.5)
    new_W = int(W * 1.5)
    stretched = np.zeros((H, new_W), dtype=img.dtype)
    for y in range(H):
        for x in range(new_W):
            src_x = int(x / 1.5)
            if src_x < W:
                stretched[y, x] = img[y, src_x]
    result['stretched'] = stretched

    # 4. Horizontally mirrored image (flip along vertical axis)
    mirrored = img[:, ::-1]
    result['mirrored'] = mirrored

    # 5. Barrel distorted image (radial distortion)
    distorted = np.zeros_like(img)
    center_y, center_x = H / 2, W / 2
    k = -0.0005  # Barrel distortion coefficient

    for y in range(H):
        for x in range(W):
            # Normalize coordinates to center
            dx = x - center_x
            dy = y - center_y
            r = np.sqrt(dx**2 + dy**2)
            factor = 1 + k * r**2
            src_x = int(center_x + dx * factor)
            src_y = int(center_y + dy * factor)
            if 0 <= src_x < W and 0 <= src_y < H:
                distorted[y, x] = img[src_y, src_x]

    result['distorted'] = distorted

    return result