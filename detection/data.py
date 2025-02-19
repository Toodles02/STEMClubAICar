import json
import os
import torch
import torchvision.transforms as transforms 
import random 
from torch.utils.data import Dataset, Subset, DataLoader
from PIL import Image


PATH = "D:/Rohan Stuff/Datasets/Traffic_Dataset/annotations/"


def remove_image(image_folder, image_key): 
    image_path = os.path.join(image_folder, f"{image_key}.jpg")
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"DELETED -------> {image_key}")

# 347 Labels, 42k images 

class TrafficSignDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, transform=None):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.image_keys = [f.split('.')[0] for f in os.listdir(self.image_folder)] 
        self.transform = transform
    def load_annotation(self, image_key):
        annotation_path = os.path.join(self.annotation_folder, f"{image_key}.json")
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                return json.load(f)
        raise FileNotFoundError(f'Could not find annotation for image key {image_key}')
    def __len__(self):
        return len(self.image_keys)
    def __getitem__(self, idx):
        image_key = self.image_keys[idx]
        image_path = os.path.join(self.image_folder, f"{image_key}.jpg")

        image = Image.open(image_path).convert('RGB')
    
        anno = self.load_annotation(image_key)
        label = anno['objects'][0]['label'] 

        if self.transform: 
            image = self.transform(image)

        return image, label
    

image_folder = 'images'
annotations_folder = 'mtsd_v2_fully_annotated/annotations' 


if __name__ == '__main__':

    mean = [0.4270, 0.4716, 0.4838]
    std = [0.2461, 0.2567, 0.2902] 
    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(), 
    transforms.Normalize(mean, std)
    ])

    dataset = TrafficSignDataset(PATH + image_folder, PATH + annotations_folder, transform=transform)
    print(len(dataset))

    sample_size = 5000 

    indices = random.sample(range(len(dataset)), sample_size)

    sample = Subset(dataset, indices)



