import json
import os
import time
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.models import resnet18
import torchvision.transforms as transforms

class ImageNet(data.Dataset):
    def __init__(self, path, is_validation_dataset = False):
#        self.path = path
        self.folder_path = path
        self.is_validation_dataset = is_validation_dataset
        self.path = ""
        
        if self.is_validation_dataset == True:
            # look in the validation folder
            self.path = "{}/imagenet_12_val/*/".format(self.folder_path)
        else:
            # look in the training folder
            self.path = "{}/imagenet_12_train/*/".format(self.folder_path)

        self.img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224, 224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.imgs = []
        self.lbls = []

        for folder_path_g in glob(self.path):
            image_paths = glob("{}/*".format(folder_path_g))
            self.imgs += image_paths
            if folder_path_g == "n02109961":
                self.lbls+= [1] *len(image_paths)
            else:
                self.lbls+= [0] *len(image_paths)
    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        print(img)
        img = self.img_transforms(img)
        lbl = self.lbls[index]
        return img, lbl
        
    def __len__(self):
        return len(self.imgs)
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        pretrained_resnet = resnet18()
        self.resnet = nn.Sequential(*list(pretrained_resnet.children())[:-1])
        self.linear = nn.Linear(512, 2)
        
    def forward(self, x):
        x = self.resnet(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, 512)
        x = self.linear(x)
        return x

training_dataset = ImageNet("./imagenet_12", is_validation_dataset=False)
validation_dataset = ImageNet("./imagenet_12", is_validation_dataset= True)

train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=16)
val_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=16)

#dataset = ImageNet("./")
#train_num =int(len(dataset)*0.8)
#val_num=len(dataset)-train_num
#train_dataset, val_dataset = torch.utils.data.random_split(dataset, (train_num, val_num)) # don't need to do this
#test_dataset = ImageNet("./")

model = Net()
loss = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.003)

#test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle = False, batch_size=16)

for epoch in range(20):
    model.train()
    for batch_idx, (imgs, lbls) in enumerate(train_dataloader, 0):
#        imgs = imgs
#        lbls = lbls
    
        print(batch_idx)
        optimizer.zero_grad()
        output = model(imgs)
        calculated_loss = loss(output, lbls)
        calculated_loss.backward()
        optimizer.step()
        
        print("Epoch {} train: {}/{} loss: {:.5f} ({:.3f}s)".format (
        epoch+1,
        batch_idx+1,
        len(train_dataloader),
        calculated_loss.item(),
        time.time() ,end ="\r"))

    print("Epoch {} total loss: {:.5f}".format(
        epoch+1,
        total_loss / len(train_dataset)))

    for batch_idx, (imgs, lbls) in enumerate(test_dataloader):
        output = model(imgs)
        loss = loss_fn(output, lbls)

        print("Epoch {} val: {}/{} loss: {:.5f} ({:.3f}s)".format (
        epoch+1,
        batch_idx+1,
        len(test_dataloader),
        loss.item(),
        time.time() ,end ="\r"))

    print("Epoch {} total loss: {:.5f}".format(
        epoch+1,
        total_val_loss / len(test_dataset),
        correct / len(test_dataset)))
