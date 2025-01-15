import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw
from torchvision import datasets, transforms
from torchvision.transforms import functional as F

'''
def add_backdoor(tensor_img, box_color=(255, 0, 0), box_size=5):
    img = F.to_pil_image(tensor_img)

    # Add backdoor using ImageDraw
    draw = ImageDraw.Draw(img)
    width, height = img.size
    x_start = width - box_size - 1
    y_start = height - box_size - 1
    draw.rectangle([x_start, y_start, x_start + box_size, y_start + box_size], fill=box_color)

    # Convert back to tensor
    tensor_img = F.to_tensor(img)
    return tensor_img

'''

def add_backdoor(img_path, output_path):
    """
    Adds a backdoor to a PPM image and saves it back.
    Args:
        img_path (str): Path to the input PPM image.
        output_path (str): Path to save the modified PPM image.
    """
    # Open the PPM image
    img = Image.open(img_path)
    
    # Add a backdoor (e.g., a white square at a specific location)
    draw = ImageDraw.Draw(img)
    draw.rectangle([5, 5, 15, 15], fill="white")  # Add a small white square

    # Save the modified image back as PPM
    img.save(output_path, format='PPM')

class BelgiumTS(Dataset):
    base_folder = 'BelgiumTS'

    def __init__(self, root_dir, train=False, transform=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        self.sub_directory = 'Training' if train else 'Testing'
        self.csv_file_name = 'train_data.csv' if train else 'test_data.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[idx, 0])
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId

class BackdoorTestDataset(Dataset):
    def __init__(self, csv_data, root_dir, transform=None):
        """
        Args:
            csv_data (DataFrame): DataFrame containing image paths and labels.
            root_dir (string): Directory containing the images.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.csv_data = csv_data
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214),
                                (0.2724, 0.2608, 0.2669))
        ])

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        
        img_name = os.path.join(self.root_dir, self.csv_data.iloc[idx, 0])
        
        image = Image.open(img_name)

        label = self.csv_data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label        