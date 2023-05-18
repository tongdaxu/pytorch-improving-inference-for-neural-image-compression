import os
import torch
import torchvision 

class KodakDataset(torch.utils.data.Dataset):
    def __init__(self, kodak_root):
        self.img_dir = kodak_root
        self.img_fname = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_fname)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_fname[idx])
        image = torchvision.io.read_image(img_path)
        image = image.to(dtype=torch.float32) / 255.0
        return image
