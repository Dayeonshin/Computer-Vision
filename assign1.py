import json
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.models import alexnet
import torchvision.transforms as transforms

class ImageNet(data.Dataset):
    def __init__(self, path):
        self.path = path
        self.folder_paths = glob("{}/*/".format(self.path))
        self.json_path = "{}/imagenet_class_index.json".format(self.path)

        with open("{}/imagenet_class_index.json".format(self.path), "r") as f:
            self.lbl_dic = json.load(f)
        self.lbl_dic = {v[0]: int(k) for k, v in self.lbl_dic.items()}

        self.img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.imgs = []
        self.lbls = []
        for folder_path in self.folder_paths:
            image_paths = glob("{}/*".format(folder_path))
            self.imgs += image_paths
            self.lbls += [self.lbl_dic[folder_path.split("/")[-2]]] * len(image_paths)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        img = self.img_transforms(img)
        lbl = self.lbls[index]
        return img, lbl

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    model = alexnet(pretrained=True)
    model.eval()
    data_loader = torch.utils.data.DataLoader (ImageNet("imagenet_12"), batch_size=2, shuffle=False)
    total = 0
    correct = 0
    for images, labels in data_loader:
        predicted = model(images)
        predicted_label = torch.argmax(predicted, dim=1)
    
        correct += int(torch.sum(predicted_label == labels))
        total += len(images)
        
    acc = correct/total
    print(acc)
