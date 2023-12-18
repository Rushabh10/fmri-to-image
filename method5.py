import pickle
import os
import cv2
import scipy.io
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

data = scipy.io.loadmat('./CSI1_ROIs_TR1.mat')
data2 = scipy.io.loadmat('./CSI1_ROIs_TR2.mat')
data3 = scipy.io.loadmat('./CSI1_ROIs_TR3.mat')
data4 = scipy.io.loadmat('./CSI1_ROIs_TR4.mat')

bands = ['LHPPA', 'RHLOC', 'LHLOC', 'RHEarlyVis', 'RHRSC', 'RHOPA',
         'RHPPA', 'LHEarlyVis', 'LHRSC', 'LHOPA']

total_data = []
for i in range(5254):
  curr_image_data = []
  for band in bands:
    curr_image_data.extend(data[band][i])
  total_data.append(curr_image_data)
  
for i in range(5254):
  curr_image_data = []
  for band in bands:
    curr_image_data.extend(data2[band][i])
  total_data.append(curr_image_data)
  
for i in range(5254):
  curr_image_data = []
  for band in bands:
    curr_image_data.extend(data3[band][i])
  total_data.append(curr_image_data)
  
for i in range(5254):
  curr_image_data = []
  for band in bands:
    curr_image_data.extend(data4[band][i])
  total_data.append(curr_image_data)

file_path = "./CSI01_stim_lists.txt"  # Replace with the path to your specific text file
file_strings_temp = []

with open(file_path, "r") as file:
    lines = file.readlines()
    cleaned_lines = [line.strip() for line in lines]  # Remove leading/trailing whitespace
    file_strings_temp.extend(cleaned_lines)
    
file_strings = []    
file_strings.extend(file_strings_temp)
file_strings.extend(file_strings_temp)
file_strings.extend(file_strings_temp)
file_strings.extend(file_strings_temp)

def normalize(data):
    # Convert the list of lists to a NumPy array for easy manipulation
    array_data = np.array(data, dtype=float)

    # Calculate the minimum and maximum values for each column
    min_values = np.min(array_data, axis=0)
    max_values = np.max(array_data, axis=0)

    # Normalize each column to the range [-1, 1]
    normalized_data = -1 + 2 * (array_data - min_values) / (max_values - min_values)

    # Convert the NumPy array back to a list of lists
    normalized_list_of_lists = normalized_data.tolist()

    return normalized_list_of_lists

coco_location = "/coco/train2014/"
imagenet_location = "/imagenet/train/"

valid_indices = []
for i in range(len(file_strings)):
  string = file_strings[i]
  if string.startswith("rep_n"):
      string = string.replace("rep_", "")
      parts = string.split("_")
      file_strings[i] = imagenet_location + parts[0] + '/' + string
      valid_indices.append(i)
  elif string.startswith("n0"):
      parts = string.split("_")
      file_strings[i] = imagenet_location + parts[0] + '/' + string
      valid_indices.append(i)
  elif string.startswith("COCO"):
      file_strings[i] = coco_location + string
      valid_indices.append(i)

total_data = normalize(total_data)
# Extract the subset of rows
fmri_data = [total_data[i] for i in valid_indices]
image_data = [file_strings[i] for i in valid_indices]

print("Number of images = ", len(image_data))
print("Number of fmri data = ", len(fmri_data))

class CustomDataset(Dataset):
    def __init__(self, data, image_paths, transform=None):
        self.data = data
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the image from the file path
        image = Image.open(self.image_paths[idx])
        
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)

        # Extract the corresponding data and label
        data_point = torch.tensor(self.data[idx], dtype=torch.float32)

        return data_point, image

# Define the standard transforms for the images
image_transforms = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize the image to match the input size of the generator
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# Create an instance of the CustomDataset and DataLoader
# Might need to convert fmri data and image data to numpy arrays
custom_dataset = CustomDataset(data=fmri_data, image_paths=image_data, transform=image_transforms)
dataloader = DataLoader(custom_dataset, batch_size=128, shuffle=True, num_workers = 2, pin_memory = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class CFG:
    debug = False
    batch_size = 32
    num_workers = 2
    head_lr = 5e-4
    image_encoder_lr = 5e-5
    text_encoder_lr = 5e-6
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 150
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet18'
    image_embedding = 512
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 64

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 512
    dropout = 0.1
    
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
        
class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()

        # Fully connected layers with batch normalization
        self.fc1 = nn.Linear(1685, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(1024, 768)
        self.bn3 = nn.BatchNorm1d(768)

    def forward(self, x):
        # Input shape: (batch_size, 1685)

        # Fully connected layers with batch normalization and dropout
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))

        return x
        
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
        
class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, imgs, fmris):
        # Getting Image and Text Features
        image_features = self.image_encoder(imgs)
        text_features = self.text_encoder(fmris)
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
        
def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        imgs = batch[1]
        fmris = batch[0]
        imgs = imgs.to(device)
        fmris = fmris.to(device)
        loss = model(imgs, fmris)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = imgs.size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter

losses = []
def main():
    train_loader = dataloader
    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        lr_scheduler.step(train_loss.avg)
        print(train_loss.avg)
        losses.append(train_loss.avg)
        if epoch%20 == 0:
            torch.save(model.state_dict(), "./method5_" + str(epoch) + ".pth")
            with open('method5_losses_' + str(epoch) + '.pkl', 'wb') as f:
                pickle.dump(losses, f)
    torch.save(model.state_dict(), "./method5.pth")
    with open('method5_losses.pkl', 'wb') as f:
        pickle.dump(losses, f)
main()
