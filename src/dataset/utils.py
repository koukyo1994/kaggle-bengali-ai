import cv2
import numpy as np


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


def crop_and_embed(image: np.ndarray, size=(128, 128)):
    cropped = crop_image(image)
    height, width = cropped.shape
    aspect_ratio = height / width
    embedded = np.zeros(size)
    if aspect_ratio > 1.0:
        if height > size[0]:
            new_height = size[0]
            new_width = int(size[0] * 1 / aspect_ratio)
            image = resize(cropped, size=(new_height, new_width))

            margin = size[1] - new_width
            head = margin // 2
            embedded[:, head:head + new_width] = image
        else:
            image = cropped
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
            image = resize(cropped, size=(new_height, new_width))

            margin = size[0] - new_height
            head = margin // 2
            embedded[head:head + new_height, :] = image
        else:
            image = cropped
            margin_height = size[0] - new_height
            margin_width = size[1] - new_width

            head_height = margin_height // 2
            head_width = margin_width // 2
            embedded[head_height:head_height +
                     new_height, head_width:head_width + new_width] = image

    return embedded
