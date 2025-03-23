
import pickle
import random

import torch
import numpy as np
import torch.nn as nn
import torchvision
from sklearn.metrics import root_mean_squared_error
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.io import ImageReadMode
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, random_split
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("########### using device : ",device)
img_dir='../Datas/imgResized/'
train_loader = None
test_loader = None
batch_size=125

os.makedirs('models', exist_ok=True)


transform = transforms.Compose([ #transforms.ToPILImage(),
                                 transforms.Grayscale(),
                                 transforms.Resize((64,64)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.0], std=[1.0])
                               ])# TODO: compose transforms here

# Define transform

dataset = datasets.ImageFolder(img_dir, transform=transform) # TODO: create the ImageFolder
print("Classes : ",dataset.classes)
print("Classes : ",dataset.class_to_idx)



train_dataset, test_dataset = random_split(dataset, [0.8,0.2])
train_loader = DataLoader(train_dataset, batch_size=batch_size,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,drop_last=True)

images, labels = next(iter(test_loader))
print("Train Images : ",len(test_loader))
print('Images shape: ',images[0].shape)
latent_dim=256
class VAE(nn.Module):

    def __init__(self, input_dim=4096, hidden_dim=1024, latent_dim=256, device=device):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat, mean, log_var

model = VAE().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')

    ## To see ####""
    #print("Input x range: ", x.min().item(), x.max().item())
    #print("Output x_hat range: ", x_hat.min().item(), x_hat.max().item())

    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def train(model, optimizer, epochs, device,batch_size, x_dim=4096):
    model.train()
    l_overall_loss=[]
    for epoch in range(epochs):
        overall_loss = 0
        batches_nbr=len(train_loader)
        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx == batches_nbr-2:
                break

            x = x.view(batch_size, x_dim).to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        l_overall_loss.append(overall_loss/(batch_idx*batch_size))
        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
        torch.save(model.state_dict(), 'models/vae.pth')

        file = open('models/l_overall_loss.dmp', 'wb')
        pickle.dump(l_overall_loss, file)
        file.close()

    return overall_loss
doTrain=1
if doTrain:
    train(model, optimizer, epochs=5000, batch_size=batch_size, device=device)
else:
    model.load_state_dict(torch.load('models/vae-256-5000.pth', weights_only=True))

def generate_digit(mean, var):
    z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)
    x_decoded = model.decode(z_sample)
    digit = x_decoded.detach().cpu().reshape(64, 64) # reshape vector to 2d array
    plt.title(f'[{mean},{var}]')
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
    plt.show()

#img1: mean0, var1 / img2: mean1, var0
(
generate_digit(0.0, 1.0),
generate_digit(.5, 0.10),
generate_digit(.25, 0.5)
 )

def plot_latent_space(model, scale=5.0, n=4, digit_size=64, figsize=15):
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            x_decoded = model.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Visualization with {}'.format(scale))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

plot_latent_space(model, scale=1.0)
#plot_latent_space(model, scale=.75)
#plot_latent_space(model, scale=.65)
#plot_latent_space(model, scale=.5)
#plot_latent_space(model, scale=.25)


images, labels = next(iter(test_loader))
print("Train Images : ",len(test_loader))
print('Images shape: ',images[0].shape)
plt.imshow(make_grid(images[0:9], nrow=3).permute(1, 2, 0))
plt.show()

########################################################################
def plot_two_figs(img1,img2):
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 2, 1)

    # showing image
    plt.imshow(img1, cmap="Greys_r")
    plt.axis('off')
    plt.title("True")

    # Adds a subplot at the 2nd position
    fig.add_subplot(1, 2, 2)

    # showing image
    plt.imshow(img2, cmap="Greys_r")
    plt.axis('off')
    plt.title("Reconstructed")
    plt.show()


########################################################################
images, labels = next(iter(test_loader))
img = images[0].to(device)
print(img.shape,type(img),img.dtype)

img = torchvision.io.read_image('../Datas/noTum_observation/7.png').to(device)
transform =transforms.Compose([
    transforms.ToPILImage(),transforms.Grayscale(),
    transforms.ToTensor()
])
img= transform(img).to(device)
rec_images,_,_= model(img.view(1, 64*64))
plot_two_figs(img.detach().cpu().reshape(64, 64),rec_images.detach().cpu().reshape(64, 64))
plt.show()

#####################################################

images, labels = next(iter(test_loader))
img = images[0].to(device)
encoded_images= model.encoder(img.view(1, 64*64))
print(encoded_images.shape)

labels=[]
encoded_imgs=[]
for batch_idx, (img, label) in enumerate(test_loader):
    label = label.detach().cpu().numpy()
    labels.append(label)
    img = img.to(device)
    #print("img.shape : ",img.shape)
    encoded_images= model.encoder(img.view(batch_size, 64*64))
    encoded_imgs.append(encoded_images.detach().cpu().numpy())

encoded_imgs = np.array(encoded_imgs).reshape(-1, latent_dim)
print("encoded_imgs.shape : ",encoded_imgs.shape)
labels = np.array(labels).flatten()
print("labels.shape : ",labels.shape)

file = open('dataset/test_encoded_dataset.dmp','wb')
# dump information to that file
pickle.dump([encoded_imgs,labels], file)
file.close()
print("Dump Done.")
