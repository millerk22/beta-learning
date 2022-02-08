#Code for training CNN VAE and saving model 

import numpy as np
import os
import torch
from torchvision.utils import save_image
from torchinfo import summary
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.optim.lr_scheduler import StepLR
import graphlearning as gl

#Our package imports
import utils
import models

# Training function (1 epoch)
def train(model, data, batch_size, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for idx in range(0,len(data),batch_size):
        data_batch = data[idx:idx+batch_size]
        data_batch = data_batch.to(device)

        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data_batch)
        bce_loss = criterion(reconstruction, data_batch)
        loss = bce_loss - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss.backward()
        running_loss += loss.item()
        optimizer.step()

    train_loss = running_loss / len(data)
    return train_loss, reconstruction


def train_vae(data):
    data = torch.from_numpy(data).float()

    #Randomly shuffle data
    P = torch.randperm(len(data))
    data = data[P,:,:,:]

    # set the learning parameters
    lr = 0.001
    epochs = 500
    batch_size = 64
    cuda = True
    gamma = 0.99

    use_cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = models.CVAE().to(device)

    summary(model, input_size=(25, 1, 28, 28))
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction='sum')
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    #Create directory for output figures, if doesn't already exist
    #if not os.path.exists('../figures/CNNVAE/'):
    #    os.makedirs('../figures/CNNVAE/')

    for epoch in range(epochs):

        train_epoch_loss, recon_images = train(model, data, batch_size, device, optimizer, criterion)
        scheduler.step()
        #save_image(recon_images.cpu(), '../figures/CNNVAE/recon_epoch%d.jpg'%epoch)

        print(f"Epoch {epoch+1} of {epochs}: ",end='')
        print(f"Train Loss: {train_epoch_loss:.4f}")


    #Save model
    torch.save(model, './models/MNIST49_CNNVAE.pt')
    return