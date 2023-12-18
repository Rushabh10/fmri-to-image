
import numpy as np
import pandas as pd
import scipy.io
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle

data = scipy.io.loadmat('./CSI1_ROIs_TR1.mat')
data2 = scipy.io.loadmat('./CSI1_ROIs_TR2.mat')

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

file_path = "./CSI01_stim_lists.txt"  # Replace with the path to your specific text file
file_strings = []

with open(file_path, "r") as file:
    lines = file.readlines()
    cleaned_lines = [line.strip() for line in lines]  # Remove leading/trailing whitespace
    file_strings.extend(cleaned_lines)
    
file_strings.extend(file_strings)

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
dataloader = DataLoader(custom_dataset, batch_size=256, shuffle=True)

# Define a generator model with convolutional layers
        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.instance_norm2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.instance_norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.instance_norm2(out)

        out += residual
        out = self.relu(out)

        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc = nn.Linear(1685, 4 * 4 * 256)
        self.instance_norm = nn.InstanceNorm2d(256)

        self.residual_blocks = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(128)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.instance_norm2 = nn.InstanceNorm2d(64)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.instance_norm3 = nn.InstanceNorm2d(3)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.instance_norm(x)

        x = self.residual_blocks(x)

        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.instance_norm1(x)
        x = F.relu(x)

        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.instance_norm2(x)
        x = F.relu(x)

        x = self.upsample3(x)
        x = self.conv3(x)
        x = self.upsample3(x)
        x = self.instance_norm3(x)
        x = torch.tanh(x)

        return x
        

def ssim_mse_loss(output, target, alpha=0.25):
    """
    Combination of SSIM and MSE loss.
    
    Args:
    - output (torch.Tensor): The generated image.
    - target (torch.Tensor): The ground truth image.
    - alpha (float): Weight factor for the SSIM loss. (Default: 0.8)
    
    Returns:
    - loss (torch.Tensor): Combined loss value.
    """
    mse_loss = F.mse_loss(output, target)
    ssim = StructuralSimilarityIndexMeasure(data_range=target.max() - target.min()).to(device)
    ssim_loss = 1 - ssim(output, target)
    
    # Weighted sum of SSIM and MSE
    loss = alpha * ssim_loss + (1 - alpha) * mse_loss
    
    return loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Initialize the generator
generator = Generator()
generator = generator.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
losses = []

# Training loop
for epoch in range(200):
    for batch in dataloader:
        fmri, target_image = batch

        # Move data to GPU
        fmri, target_image = fmri.to(device), target_image.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: compute predicted image by passing fmri through the generator
        generated_image = generator(fmri)
        
        '''print("////////////")
        print(generated_image.shape)
        print(target_image.shape)'''

        # Compute the loss
        loss = ssim_mse_loss(generated_image, target_image, alpha=0)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Update the weights
        optimizer.step()
    
    scheduler.step(loss)
    # Print the loss after each epoch
    print(f'Epoch [{epoch+1}/{200}], Loss: {loss.item():.4f}')
    losses.append(loss.item())

torch.save(generator.state_dict(),  "./method1_model.pth")
with open('method1_losses.pkl', 'wb') as f:
    pickle.dump(losses, f)
print("Code execution complete!")
