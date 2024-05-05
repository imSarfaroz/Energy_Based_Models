import torch
from torchvision import datasets 
from torchvision import transforms
from torchvision.utils import make_grid # for creating  grid of images from a list of images
import torch.nn.functional as F

import matplotlib.pyplot as plt

from rbm import RBM # importing by RMB class from rbn.py
from train import train # importing training from train.py

# HYPER-PARAMETERS (experimented with various hyperparameters below)
batch_size = 32 
num_epochs = 2
lr = 0.01 
n_hid = 200 
n_vis = 784

# Addinitional function to save and show image once executed
def show_and_save(img, file_name):

    pic = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.jpg" % file_name

    plt.imshow(pic, cmap='gray') 
    plt.imsave(f, pic)


model = RBM(num_visible=n_vis, num_hidden=n_hid, k=1)

# This code initializes a DataLoader for the MNIST dataset, downloading it if necessary,
# onverting images to tensors, and organizing them into batches of a specified size for training. Saving results in ./ouput file
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./output', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor() ])),
    batch_size=batch_size
)

model = train(model, train_loader, n_epochs=num_epochs, learning_rate=lr)

images = next(iter(train_loader))[0]
v, v_gibbs = model(images.view(-1, 784))

# Saving and showing original image
show_and_save(make_grid(v.view(batch_size, 1, 28, 28).data), 'output/original_images')

# Saving and showing generated image by RBM
show_and_save(make_grid(v_gibbs.view(batch_size, 1, 28, 28).data), 'output/generated_image')