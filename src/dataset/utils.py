import cv2
import numpy as np

from skimage.transform import AffineTransform, warp


def crop_image(image: np.ndarray, threshold=5. / 255.) -> np.ndarray:
    assert image.ndim == 2
    is_black = image > threshold
    is_black_vertical = np.sum(is_black, axis=0) > 0
    is_black_horizontal = np.sum(is_black, axis=1) > 0

    left = np.argmax(is_black_horizontal)
    right = np.argmax(is_black_horizontal[::-1])
    top = np.argmax(is_black_vertical)
    bottom = np.argmax(is_black_vertical[::-1])
    height, width = image.shape
    cropped_image = image[left:height - right, top:width - bottom]
    return cropped_image


def resize(image, size=(128, 128)) -> np.ndarray:
    return cv2.resize(image, size)


def crop_and_embed(image: np.ndarray, size=(128, 128), threshold=20. / 255.):
    cropped = crop_image(image, threshold)
    height, width = cropped.shape
    aspect_ratio = height / width
    embedded = np.zeros(size)
    if aspect_ratio > 1.0:
        if height > size[0]:
            new_height = size[0]
            new_width = int(size[0] * 1 / aspect_ratio)
            image = resize(cropped, size=(new_width, new_height))

            margin = size[1] - new_width
            head = margin // 2
            embedded[:, head:head + new_width] = image
        else:
            margin = size[0] - height

            new_height = height + np.random.randint(0, margin)
            new_width = int(new_height * 1 / aspect_ratio)
            image = resize(cropped, size=(new_width, new_height))

            margin_height = size[0] - new_height
            margin_width = size[1] - new_width

            head_height = margin_height // 2
            head_width = margin_width // 2
            embedded[head_height:head_height +
                     new_height, head_width:head_width + new_width] = image
    else:
        if width > size[1]:
            new_width = size[1]
            new_height = int(size[1] * aspect_ratio)
            image = resize(cropped, size=(new_width, new_height))

            margin = size[0] - new_height
            head = margin // 2
            embedded[head:head + new_height, :] = image
        else:
            margin = size[1] - width

            new_width = width + np.random.randint(0, margin)
            new_height = int(new_width * aspect_ratio)
            image = resize(cropped, size=(new_width, new_height))

            margin_height = size[0] - new_height
            margin_width = size[1] - new_width

            head_height = margin_height // 2
            head_width = margin_width // 2
            embedded[head_height:head_height +
                     new_height, head_width:head_width + new_width] = image

    return embedded


def normalize(image: np.ndarray):
    if image.ndim == 3:
        image = image[:, :, 0]
    image = (255 - image).astype(np.float32) / 255.0
    return image


def affine_image(image: np.ndarray):
    assert image.ndim == 2
    min_scale = 0.8
    max_scale = 1.2
    sx = np.random.uniform(min_scale, max_scale)
    sy = np.random.uniform(min_scale, max_scale)

    max_rot_angle = 10
    rot_angle = np.random.uniform(-max_rot_angle, max_rot_angle) * np.pi / 180.

    max_shear_angle = 10
    shear_angle = np.random.uniform(-max_shear_angle,
                                    max_shear_angle) * np.pi / 180.

    max_translation = 4
    tx = np.random.randint(-max_translation, max_translation)
    ty = np.random.randint(-max_translation, max_translation)

    tform = AffineTransform(
        scale=(sx, sy),
        rotation=rot_angle,
        shear=shear_angle,
        translation=(tx, ty))
    transformed_image = warp(image, tform)
    return transformed_image


def random_erosion_or_dilation(image: np.ndarray):
    dice = np.random.randint(0, 3)
    if dice == 0:
        return image
    elif dice == 1:
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.erode(image, kernel, iterations=1)
    else:
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.dilate(image, kernel, iterations=1)
