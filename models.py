import torch.nn as nn
import torch.nn.functional as F
import torch



#CNN Variational Autoencoder
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
 
        kernel_size = 3 # (3, 3) kernel
        num_classes = 2
        init_channels = 8 # initial number of filters
        image_channels = 1
        latent_dim = 32 # latent dimension for sampling

        # encoder
        self.enc1 = nn.Conv2d(image_channels, init_channels, kernel_size, padding=1)
        self.enc2 = nn.Conv2d(init_channels, 2*init_channels, kernel_size+2, padding=2)
        self.enc3 = nn.Conv2d(2*init_channels, 4*init_channels, kernel_size+2, padding=2)

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)

        # decoder 
        self.dec1 = nn.ConvTranspose2d(64, 2*init_channels, kernel_size, output_padding=1, dilation=2)
        self.dec2 = nn.ConvTranspose2d(2*init_channels, init_channels, kernel_size, stride=2)
        self.dec3 = nn.ConvTranspose2d(init_channels, image_channels, kernel_size, stride=2, output_padding=1)
        #Dropout
        self.dropout1 = nn.Dropout(0.5)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample

    def encode(self, x):
        x = F.relu(self.enc1(x)) #28
        x = F.max_pool2d(x, 2) #14
        x = F.relu(self.enc2(x)) #14
        x = F.max_pool2d(x, 2) #7
        x = F.relu(self.enc3(x)) #7
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        return x

    def decode(self, x):
        x = x.view(-1, 64, 1, 1)
        x = F.relu(self.dec1(x)) #6
        x = F.relu(self.dec2(x)) #13
        x = self.dropout1(x)
        x = torch.sigmoid(self.dec3(x)) #28

        return x

    def forward(self, x): #88

        #encode
        x = self.encode(x)
        hidden = self.fc1(x) #(32*7*7, 128)

        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var) #(32, )
        z = self.fc2(z) #(64, 64)

        #decode
        reconstruction = self.decode(z)
        
        return reconstruction, mu, log_var


