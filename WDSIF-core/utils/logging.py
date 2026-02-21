import torch
import shutil
import os
import torchvision.utils as tvu


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)


def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + ".pth.tar")


def load_checkpoint(path, device=None):
    map_loc = device if device is not None else torch.device("cpu")
    try:
        return torch.load(path, map_location=map_loc, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_loc)
