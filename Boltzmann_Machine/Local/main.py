# Importing neccassary torch libraries
import torch
import torch.nn.functional as F
import torch.nn as NN
import torch.utils.data
import torch.optim as optim

# numpy for numerical operations and matplotlib for plotting the image
import matplotlib.pyplot as plt
import numpy as np

# libraries for handling datasets, image transformations, and visualizing data.
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid

from Local.bm import BM # importing by RMB class from rbn.py
from Local.train import train, calculate_mse # importing training from train.py

# HYPER-PARAMETERS (experimented with various hyperparameters below)
batch_size = 32
num_epochs = 2
learning_rate = 0.001
num_hidden = 500
num_visible = 784

# Addinitional function to save and show image while executing
def show_and_save(img, file_name):
    pic = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    plt.imshow(pic, cmap='gray')
    plt.imsave(f, pic)

# This code initializes a DataLoader for the MNIST dataset, downloading it if necessary,
# onverting images to tensors, and organizing them into batches of a specified size for training. Saving results in ./ouput file
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./output', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor() ])),
    batch_size=batch_size
)

model = BM(num_visible, num_hidden)
optimizer = torch.optim.adam(model.parameters(), lr=0.001)  # Adap optimizer

# Train the Boltzmann Machine (using CUDA-GPU in google colab if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train(model, train_loader, optimizer, num_epochs, device=device)

images = next(iter(train_loader))[0]
images = images.float()
images = images.view(-1, 784)  # Reshape the input images to (batch_size, 784)
v, v_gibbs = model.forward(images)

# Saving and showing original image
show_and_save(make_grid(v.view(batch_size, 1, 28, 28).data), 'output/original_images')

# Saving and showing generated image by RBM
show_and_save(make_grid(v_gibbs.view(batch_size, 1, 28, 28).data), 'output/generated_image')