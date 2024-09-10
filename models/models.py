import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from einops import einsum
from sys import getsizeof

kernel_size = 4 # (4, 4) kernel
init_channels = 8 # initial number of filters
image_channels = 1 # MNIST images are grayscale
latent_dim = 16 # latent dimension for sampling
device = torch.device("cpu")

class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        # encoders 
        self.vector_quantization = VectorQuantizer(400,16,0.25)
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(image_channels, init_channels, kernel_size, stride=1, padding=1),
			nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, stride=2, padding=1),
			nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, stride=2, padding=1),
			nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size, stride=2, padding=0)
        )
        self.beta = 0.25
        self.correlation_encoder = torch.nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1),
			nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, stride=2, padding=1),
			nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, stride=2, padding=1),
			nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size, stride=2, padding=0),
			nn.BatchNorm2d(64),
            nn.Tanh()
        )
        

        # fully connected layers for learning representations
        self.latent_fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        
		
        
        self.cor_interpretation = torch.nn.Sequential(
			nn.Linear(64,128),
			nn.LeakyReLU(),
            nn.Linear(128,latent_dim**2),
			nn.LeakyReLU(),
			nn.Linear(latent_dim**2,latent_dim**2)
        )
        
        # decoder 
        self.decoder = torch.nn.Sequential(
			nn.ConvTranspose2d(in_channels=16, out_channels=init_channels*8, kernel_size=kernel_size, stride=1, padding=0),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			nn.ConvTranspose2d(in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, stride=2, padding=1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),
			nn.ConvTranspose2d(in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, stride=2, padding=1),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(),
			nn.ConvTranspose2d(in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, stride=2, padding=1),
			nn.Sigmoid()
        )
		
        self.embedding = nn.Embedding(num_embeddings=400,embedding_dim=16)
    
    def kld(self, mu, cov_matrix):
        # the scale tril has to be given to the distribution as opposed to the covariance matrix because the condition that the cov matrix must be psd is 
        # awful and finds that psd matrices arent psd even when they absolutely are
        standard = torch.distributions.MultivariateNormal(loc=torch.zeros(16),covariance_matrix=torch.diag_embed(torch.ones(16)))
        multi = torch.distributions.MultivariateNormal(loc=mu[-1:],scale_tril=torch.tril(cov_matrix[-1::]))
        
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # modified kld for the multivariate distribution so that the kld and the entropy rate are calculated at the same time and the rate distortion can be properly learned
        KLD = torch.distributions.kl_divergence(standard,multi)
        KLD = torch.mean(KLD)
        
        # i am aware that this is absolutely not the correct way to get the actual bits per pixel but im hoping it may give a ballpark estimate for bugfixing
        

        # the log2 of e converts the nats that pytorch calculates into bits
        
        # KLD = 0.5 * (1+torch.log2(torch.det(cov_matrix[-1::]))-latent_dim+(torch.sum(logvar.exp())/torch.det(cov_matrix[-1::]))+mu[-1:]*torch.inverse(cov_matrix[-1::])*torch.transpose(mu[-1:],dim0=0,dim1=1))
        return KLD
    
    def forward(self, x):
        
        # encodes the features that will be used to get the latent distributions
        lat=self.encoder(x)
        batch, _, _, _ = lat.shape
        lat = F.adaptive_avg_pool2d(lat, 1).reshape(batch, -1)
        hidden_latent = self.latent_fc1(lat)

        # the cor matrix holds the value of the correlation coefficients for the lower triangle of the correlation matrix
        # it is also clamped to avoid some numerical instability issues that it was having when the correlation values got very close to 0
        cor = self.correlation_encoder(x)
        cor = F.adaptive_avg_pool2d(cor, 1).reshape(batch, -1)
        cor = self.cor_interpretation(cor) 
        
        
        
        #gets a list of all the outputs of the convolutional layers for the correlation encoder and reshapes them into a square matrix 
        #and gets the lower triangle
        
        cor = cor.reshape(batch,latent_dim,latent_dim)
        cor_tril = torch.tril(cor,diagonal=-1)
        
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden_latent)
        log_var = self.fc_log_var(hidden_latent)

        std = log_var.exp().sqrt()

        # multilies each row and column by the corresponding standard deviation to get the lower traingle of the estimated covariance matrix
        cov_tril = cor_tril[:None]*std[-1:]
        cov_tril = cov_tril[::]*std.unsqueeze(1)

        # adds the lower triangle covariance with its transposed self and with a diagonal filled with the variances from the latent distributions
        cov_matrix = cov_tril+torch.transpose(cov_tril,1,2)+torch.diag_embed(std)
        cov_tril = cov_tril+torch.diag_embed(torch.square(std))

        multi = torch.distributions.MultivariateNormal(mu,scale_tril=cov_tril)
        # this is the reparameterization for this model
        quant_input = multi.rsample()

        kullback = self.kld(quant_input,cov_matrix)
        quant_input = quant_input.view(-1,16,1,1)
        
        embedding_loss, quant_output , perplexity, _, _, bits_loss = self.vector_quantization(quant_input,kullback)

		## Decoder part
        decoder_input = quant_output
        reconstruction = self.decoder(decoder_input)
        bits = torch.log(perplexity)
        
        return reconstruction, mu, log_var, cov_matrix, embedding_loss, bits, perplexity, bits_loss
    
class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        
        

    def forward(self, z, kld):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.sum((z_q.detach()-z)**2) + self.beta * \
            torch.sum((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        entropy = torch.log(perplexity)
        bits_loss = torch.nn.functional.relu(entropy-kld)+kld
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices,bits_loss
    
    


