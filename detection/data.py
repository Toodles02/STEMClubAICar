import os
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO


PATH = "C:/Users/rohan/OneDrive/Desktop/Datasets/Traffic/"

# 10k images 

class TrafficSignDataset(Dataset):
    def __init__(self, image_folder, annotations_path, transform=None):
        self.image_folder = image_folder
        self.annotations_path = annotations_path 
        self.transform = transform
        self.images = [] 

        coco = COCO(os.path.join(image_folder, annotations_path))
        for img in coco.imgs.values(): 
            img_id = img['id']
            ann_ids = coco.getAnnIds(imgIds=img_id)
            annotations = coco.loadAnns(ann_ids)

            if annotations:
                label = annotations[0]['category_id']
            else:
                continue 

            img_path = os.path.join(image_folder, img['file_name'])
            

            self.images.append((img_path, label))

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path, label = self.images[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
    



def load_data():
    print("Loading data...")

    mean = [0.3774, 0.3560, 0.3700]
    std = [0.1897, 0.1858, 0.1921]
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    augmentation = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    annotations_file = "_annotations.coco.json"

    train_path = os.path.join(PATH, 'train')
    train_set = TrafficSignDataset(train_path, os.path.join(train_path, annotations_file), augmentation)

    valid_path = os.path.join(PATH, 'valid')
    valid_set = TrafficSignDataset(valid_path, os.path.join(valid_path, annotations_file), transform)

    test_path = os.path.join(PATH, 'test')
    test_set = TrafficSignDataset(test_path, os.path.join(test_path, annotations_file), transform)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4) 
    
    print("Loaded data!")
    
    return train_loader, valid_loader, test_loader
