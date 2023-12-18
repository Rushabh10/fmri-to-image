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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class VGGAutoEncoder(nn.Module):

    def __init__(self):

        super(VGGAutoEncoder, self).__init__()

        # VGG without Bn as AutoEncoder is hard to train
        configs = [2, 2, 3, 3, 3]
        self.encoder = VGGEncoder(configs=configs,       enable_bn=True)
        self.decoder = VGGDecoder(configs=configs[::-1], enable_bn=True)
        self.flatten = nn.Flatten()


    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x
        
class VGG(nn.Module):

    def __init__(self, configs, num_classes=1000, img_size=224, enable_bn=False):
        super(VGG, self).__init__()

        self.encoder = VGGEncoder(configs=configs, enable_bn=enable_bn)

        self.img_size = img_size / 32

        self.fc = nn.Sequential(
            nn.Linear(in_features=int(self.img_size*self.img_size*512), out_features=4096),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.encoder(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x
        
class VGGEncoder(nn.Module):

    def __init__(self, configs, enable_bn=False):

        super(VGGEncoder, self).__init__()

        if len(configs) != 5:

            raise ValueError("There should be 5 stage in VGG")

        self.conv1 = EncoderBlock(input_dim=3,   output_dim=64,  hidden_dim=64,  layers=configs[0], enable_bn=enable_bn)
        self.conv2 = EncoderBlock(input_dim=64,  output_dim=128, hidden_dim=128, layers=configs[1], enable_bn=enable_bn)
        self.conv3 = EncoderBlock(input_dim=128, output_dim=256, hidden_dim=256, layers=configs[2], enable_bn=enable_bn)
        self.conv4 = EncoderBlock(input_dim=256, output_dim=512, hidden_dim=512, layers=configs[3], enable_bn=enable_bn)
        self.conv5 = EncoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=configs[4], enable_bn=enable_bn)
        self.flatten = nn.Flatten()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)

        return x
        
class VGGDecoder(nn.Module):

    def __init__(self, configs, enable_bn=False):

        super(VGGDecoder, self).__init__()

        if len(configs) != 5:

            raise ValueError("There should be 5 stage in VGG")

        self.conv1 = DecoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=configs[0], enable_bn=enable_bn)
        self.conv2 = DecoderBlock(input_dim=512, output_dim=256, hidden_dim=512, layers=configs[1], enable_bn=enable_bn)
        self.conv3 = DecoderBlock(input_dim=256, output_dim=128, hidden_dim=256, layers=configs[2], enable_bn=enable_bn)
        self.conv4 = DecoderBlock(input_dim=128, output_dim=64,  hidden_dim=128, layers=configs[3], enable_bn=enable_bn)
        self.conv5 = DecoderBlock(input_dim=64,  output_dim=3,   hidden_dim=64,  layers=configs[4], enable_bn=enable_bn)
        self.gate = nn.Sigmoid()

    def forward(self, x):

        x = x.view(-1, 512, 4, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x
        
class EncoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(EncoderBlock, self).__init__()

        if layers == 1:

            layer = EncoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('0 EncoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = EncoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)

                self.add_module('%d EncoderLayer' % i, layer)

        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.add_module('%d MaxPooling' % layers, maxpool)

    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class DecoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(DecoderBlock, self).__init__()

        upsample = nn.ConvTranspose2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=2, stride=2)

        self.add_module('0 UpSampling', upsample)

        if layers == 1:

            layer = DecoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('1 DecoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = DecoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)

                self.add_module('%d DecoderLayer' % (i+1), layer)

    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class EncoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super(EncoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):

        return self.layer(x)

class DecoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super(DecoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(input_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.layer = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, x):

        return self.layer(x)

checkpoint = torch.load("imagenet-vgg16.pth")
new_dict = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
autoencoder = VGGAutoEncoder()
model_dict = autoencoder.state_dict()
model_dict.update(new_dict)
autoencoder.load_state_dict(model_dict)
autoencoder = autoencoder.to(device)

class ComplexNetwork(nn.Module):
    def __init__(self):
        super(ComplexNetwork, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(1685, 1024)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 2048)
        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(2048, 2048)
        self.batch_norm3 = nn.BatchNorm1d(2048)

    def forward(self, x):
        # Input shape: (batch_size, 1685)

        # Fully connected layers with dropout
        x = F.leaky_relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.leaky_relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.batch_norm3(self.fc3(x)))

        return x
        
def custom_loss(t1, t2):
    # Calculate mean and standard deviation of your data
    tensor1 = t1
    tensor2 = t2
    mean = tensor1.mean()
    std = tensor1.std()
    
    # Normalize your data
    tensor1_normalized = (tensor1 - mean) / std
    tensor2_normalized = (tensor2 - mean) / std
    
    # Calculate MSE loss on normalized data
    criterion = nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)
    mse_loss = criterion(tensor1_normalized, tensor2_normalized)
    return mse_loss
        
mriEncoder = ComplexNetwork()
mriEncoder = mriEncoder.to(device)

#criterion = nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)
optimizer = optim.Adam(mriEncoder.parameters(), lr=0.00001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
losses = []
        
# Training loop
def cosine_similarity_loss(x, y):
    # Normalize the input vectors
    x_normalized = F.normalize(x, p=2, dim=-1)
    y_normalized = F.normalize(y, p=2, dim=-1)
    
    # Compute cosine similarity
    similarity = torch.sum(x_normalized * y_normalized, dim=-1)
    
    # Invert the similarity to turn it into a similarity loss (maximize similarity)
    similarity_loss = 1 - similarity
    
    return similarity_loss.mean()

# Example usage in your training loop
for epoch in range(200):
    for batch in dataloader:
        fmri, target_image = batch

        # Move data to GPU
        fmri, target_image = fmri.to(device), target_image.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: compute predicted image by passing fmri through the generator
        img_encoding = autoencoder.encoder(target_image)
        
        fmri_encoding = mriEncoder(fmri)

        # Compute the cosine similarity loss
        loss = custom_loss(img_encoding, fmri_encoding)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Update the weights
        optimizer.step()
    
    scheduler.step(loss)
    # Print the loss after each epoch
    print(f'Epoch [{epoch+1}/{200}], Cosine Similarity Loss: {loss.item():.4f}')
    losses.append(loss.item())

torch.save(mriEncoder.state_dict(),  "./method3_model.pth")
with open('method3_losses.pkl', 'wb') as f:
    pickle.dump(losses, f)
print("Code execution complete!")
