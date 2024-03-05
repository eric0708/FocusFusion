import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class PokemonDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = [os.path.join(root, file) for file in os.listdir(root) if file.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGBA')
        image = np.array(image)[:, :, :3]
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, 0