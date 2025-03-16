import os
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import ConcatDataset, Subset
from collections import Counter 
from sklearn.model_selection import train_test_split

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

class SplitTrafficDataset():
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
train_path = os.path.join(PATH, 'train')
valid_path = os.path.join(PATH, 'valid')
test_path = os.path.join(PATH, 'test')

annotations_file = "_annotations.coco.json"

mean = [0.3774, 0.3560, 0.3700]
std = [0.1897, 0.1858, 0.1921]
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

augmentation = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(25),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

def load_data():
    print("Loading data...")

    train_set = TrafficSignDataset(train_path, os.path.join(train_path, annotations_file))

    valid_set = TrafficSignDataset(valid_path, os.path.join(valid_path, annotations_file))

    test_set = TrafficSignDataset(test_path, os.path.join(test_path, annotations_file))

    # I have to combine and resplit in order to even them out as orignal dataset splits suck
    combined_set = ConcatDataset([train_set, valid_set, test_set])

    indices = list(range(len(combined_set)))
    labels = [label for _, label in combined_set]

    train_idx, temp_idx, _, temp_labels = train_test_split(indices, labels, test_size=0.3, stratify=labels, random_state=42)

    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=42)

    train_subset = SplitTrafficDataset(Subset(combined_set, train_idx), augmentation)
    valid_subset = SplitTrafficDataset(Subset(combined_set, val_idx), transform) 
    test_subset = SplitTrafficDataset(Subset(combined_set, test_idx), transform) 

    train_labels = [combined_set[i][1] for i in train_idx]
    val_labels = [combined_set[i][1] for i in val_idx]
    test_labels = [combined_set[i][1] for i in test_idx]

    print("Training Distribution:", Counter(train_labels))
    print("Validation Distribution:", Counter(val_labels))
    print("Test Distribution:", Counter(test_labels))

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_subset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4) 
    
    print("Loaded data!")
    
    return train_loader, valid_loader, test_loader

def process(image):
    return transform(image)

def human_label(label): 
    anno_f = os.path.join(train_path, annotations_file)
    coco = COCO(anno_f)
    category = coco.loadCats(label)[0]
    return category['name']