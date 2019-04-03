'''Define all the model architectures.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    '''Define network architecture for mnist.'''
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Linear(nn.Module):
    '''Define linear model for mnist.'''
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(28*28, num_classes)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc(x)
        return x


class Cifar10CNN(nn.Module):
    '''Define network architecture for cifar10.
    The model is taken from the Keras example: cifar10_cnn.py
    Link: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
    '''
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # output: 32 * 32 * 32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3) # output: 32 * 30 * 30
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        # Flatten the tensor
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




# Architectures for Autoencoders
def reparameterize(mu, logvar):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Args
        mu: mean
        logvar: log of variance
    # Returns
        z: sampled latent vector
    """
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

class VAE_MLP(nn.Module):
    def __init__(self, latent_dim=10, dense_size=400, img_dim=784):
        super().__init__()
        self.img_dim=img_dim
        self.fc1 = nn.Linear(img_dim, dense_size)
        self.fc21 = nn.Linear(dense_size, latent_dim)
        self.fc22 = nn.Linear(dense_size, latent_dim)
        self.fc3 = nn.Linear(latent_dim, dense_size)
        self.fc4 = nn.Linear(dense_size, img_dim)

    def encode(self, x):
        x = x.view(-1, self.img_dim)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAE_CONV(nn.Module):
    def __init__(self, latent_dim=2, num_channels=32):
        super().__init__()
        self.num_channels = num_channels
        # for encoder
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(num_channels * 16 * 16, 128)
        self.mu = nn.Linear(128, latent_dim) 
        self.logvar = nn.Linear(128, latent_dim) 
        # for decoder
        self.fc2 = nn.Linear(latent_dim, 128)
        self.fc3 = nn.Linear(128, num_channels * 16 * 16)
        self.deconv1 = nn.ConvTranspose2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1) 
        self.deconv2 = nn.ConvTranspose2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(num_channels, 3, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        x = F.relu(self.conv1(x)) # 32 * 32 * 32
        x = F.relu(self.conv2(x)) # 32 * 16 * 16
        x = F.relu(self.conv3(x)) # 32 * 16 * 16
        x = F.relu(self.conv4(x)) # 32 * 16 * 16
        x = x.view(-1, self.num_channels * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.mu(x), self.logvar(x)

    def decode(self, x):
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, self.num_channels, 16, 16) # 32, 16, 16
        x = F.relu(self.deconv1(x)) # 32, 16, 16
        x = F.relu(self.deconv2(x)) # 32, 16, 16
        x = F.relu(self.deconv3(x)) # 32, 32, 32
        x = torch.sigmoid(self.deconv4(x)) # 3, 32, 32
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        latent = reparameterize(mu, logvar)
        return self.decode(latent), mu, logvar