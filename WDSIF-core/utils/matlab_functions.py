import numpy as np


def bgr2ycbcr(img, y_only=False):

    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img,
            [
                [24.966, 112.0, -18.214],
                [128.553, -74.203, -93.786],
                [65.481, -37.797, 112.0],
            ],
        ) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2bgr(img):

    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = np.matmul(
        img,
        [
            [0.00456621, 0.00456621, 0.00456621],
            [0, -0.00153632, 0.00791071],
            [0.00625893, -0.00318811, 0],
        ],
    ) * 255.0 + [-222.921, 135.576, -276.836]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def _convert_input_type_range(img):

    img_type = img.dtype
    img_range = img.max() - img.min()
    if img_type == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img_type == np.float32:
        img = img.astype(np.float32)
    else:
        raise TypeError(
            f"The img type should be np.float32 or np.uint8, but got {img_type}"
        )
    if img_range > 1.0:
        img = img / 255.0
    return img


def _convert_output_type_range(img, dst_type):

    if dst_type not in (np.uint8, np.float32):
        raise TypeError(
            f"The dst_type should be np.float32 or np.uint8, but got {dst_type}"
        )
    if dst_type == np.uint8:
        img = img.round()
    else:
        img = img / 255.0
    return img.astype(dst_type)


def rgb2ycbcr(img, y_only=False):

    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        out_img = np.matmul(
            img,
            [
                [65.481, -37.797, 112.0],
                [128.553, -74.203, -93.786],
                [24.966, 112.0, -18.214],
            ],
        ) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2rgb(img):

    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = np.matmul(
        img,
        [
            [0.00456621, 0.00456621, 0.00456621],
            [0, -0.00153632, 0.00791071],
            [0.00625893, -0.00318811, 0],
        ],
    ) * 255.0 + [-222.921, 135.576, -276.836]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img
