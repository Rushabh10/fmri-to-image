import random
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
from torch.autograd.variable import Variable

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
dataloader = DataLoader(custom_dataset, batch_size=64, shuffle=True, num_workers = 2, pin_memory = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 1685

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
        
fMRI_size = 1685  # Change this based on the actual size of your fMRI signal
dummy_fMRI_signal = torch.randn(1, fMRI_size)

# Initialize your generator
image_channels = 3  # Assuming RGB images

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
        
discriminator = Discriminator()
generator = Generator()
discriminator = discriminator.to(device)
generator = generator.to(device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
generator.apply(weights_init)
discriminator.apply(weights_init)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.00005, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.000075, betas=(0.5, 0.999))
scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5, verbose=True)
scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5, verbose=True)

real_label = 1.
fake_label = 0.

# Training loop
num_epochs = 200
num_discriminator_iterations = 8  # Number of times to train the discriminator before updating the generator
batch_size = 64
losses = {}
losses["d_real"] = []
losses["d_fake"] = []
losses["gen"] = []

for epoch in range(num_epochs):
    label_smoothing_factor = random.uniform(0, 0.1)
    for batch in dataloader:  # You need to implement your own dataloader
        fMRI_signals, real_images = batch
        # move to device
        fMRI_signals = fMRI_signals.to(device)
        real_images = real_images.to(device)
        batch_size = fMRI_signals.shape[0]
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        for _ in range(num_discriminator_iterations):
            # Train discriminator on real images
            optimizer_D.zero_grad()
            label = torch.full((batch_size,), real_label - label_smoothing_factor, dtype=torch.float)
            label = label.to(device)
            output = discriminator(real_images).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            # Train discriminator on generated images
            fake = generator(fMRI_signals.reshape(-1, 1685, 1, 1))
            label.fill_(fake_label)
            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()

            optimizer_D.step()
        #scheduler_D.step(errD_real + errD_fake)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # Train generator
        optimizer_G.zero_grad()
        label.fill_(real_label - label_smoothing_factor)
        fake = generator(fMRI_signals.reshape(-1, 1685, 1, 1))
        output = discriminator(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizer_G.step()
    scheduler_G.step(errG)

    # Print loss after each epoch
    print(f"Epoch {epoch}, Loss D Real: {errD_real.item()}, Loss D Fake: {errD_fake.item()}, Loss G: {errG.item()}")
    losses["d_real"].append(errD_real.item())
    losses["d_fake"].append(errD_fake.item())
    losses["gen"].append(errG.item())
    if epoch%20 == 0:
        torch.save(generator.state_dict(), "./new_method4_gen_" + str(epoch) + ".pth")
        torch.save(discriminator.state_dict(),  "./new_method4_dis_" + str(epoch) + ".pth")
        with open('new_method4_losses_' + str(epoch) + '.pkl', 'wb') as f:
            pickle.dump(losses, f)
    
torch.save(generator.state_dict(), "./new_method4_gen.pth")
torch.save(discriminator.state_dict(),  "./new_method4_dis.pth")
with open('new_method4_losses.pkl', 'wb') as f:
    pickle.dump(losses, f)
print("Code execution complete!")
