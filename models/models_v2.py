import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import config



with open("attempt.txt",'r') as infile:
    attempt = int(infile.read())
    config.attempt = attempt
kernel_size = config.kernel_size # (4, 4) kernel
init_channels = config.init_channels # initial number of filters
image_channels = config.image_channels # MNIST images are grayscale
latent_dim = config.latent_dim # latent dimension for sampling
device = config.hardware
name = f'attempt_{config.attempt}'


class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        # since the vector quantizer is essentially its own networ it has to be initialized in the conv vae initialization
         
        self.vector_quantization = VectorQuantizer(config.num_embeddings,config.embedding_dimensions,config.beta)
        
        

        self.encoder = torch.nn.Sequential(
            nn.Conv2d(image_channels,8,kernel_size,1,config.zero_padding),
            nn.LeakyReLU(),
            nn.Conv2d(8,8,3,2,1),
            nn.LeakyReLU(),

            

            nn.Conv2d(8,16,kernel_size,1,config.zero_padding),
            nn.LeakyReLU(),
            nn.Conv2d(16,16,3,2,1),
            nn.LeakyReLU(),

            

            nn.Conv2d(16,32,kernel_size,1,config.zero_padding),
            nn.LeakyReLU(),
            nn.Conv2d(32,32,3,2,1),
            nn.LeakyReLU(),

        
 
            nn.Conv2d(32,64,kernel_size,1,config.zero_padding),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.LeakyReLU(),

            
            
        )
        
        self.latent_interpretation = torch.nn.Sequential(
            nn.Linear(2304,1152),
            nn.LeakyReLU(),

            nn.Linear(1152,576),
            nn.LeakyReLU(),

            nn.Linear(576,288),
            nn.LeakyReLU(),

            nn.Linear(288,latent_dim),
            nn.LeakyReLU()
        )
        
        

        self.correlation_encoder  = torch.nn.Sequential(
            nn.Conv2d(image_channels,8,config.kernel_size,2,1),
            nn.LeakyReLU(),
            nn.Conv2d(8,8,3,1,1),
            nn.LeakyReLU(),

            nn.Conv2d(8,16,config.kernel_size,2,1),
            nn.LeakyReLU(),
            nn.Conv2d(16,16,3,1,1),
            nn.LeakyReLU(),

            nn.Conv2d(16,32,config.kernel_size,2,1),
            nn.LeakyReLU(),
            nn.Conv2d(32,32,3,1,1),
            nn.LeakyReLU(),

            nn.Conv2d(32,64,config.kernel_size,1,1),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.LeakyReLU()
        )

        self.cor_interpretation = torch.nn.Sequential(
            nn.Linear(5184,5184),
            nn.LeakyReLU(),

            nn.Linear(5184,latent_dim**2),
            nn.Tanh()
        )



        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim, latent_dim)

        

        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(64,32,5,2,2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32,32,5,1,2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32,16,5,2,2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16,16,5,1,2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(16,8,2,2,2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8,8,5,1,2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(8,image_channels,2,2,2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(image_channels,image_channels,5,1,2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(image_channels,image_channels,4,1,3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(image_channels,image_channels,3,1,1),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(image_channels,image_channels,3,1,0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(image_channels,image_channels,2,1,0),
            nn.LeakyReLU()

        )

        self.post_quantization_linears = torch.nn.Sequential(
            nn.Linear(latent_dim,288),
            nn.LeakyReLU(),

            nn.Linear(288,576),
            nn.LeakyReLU(),

            nn.Linear(576,1152),
            nn.LeakyReLU(),

            nn.Linear(1152,2304),
            nn.LeakyReLU(),

        )
    
    def kld(self, mu, cov_matrix):
        # finds the kld between a standard multivariate distribution and the multivariate distribution predicted by the encoder 
        # which will be factored into the loss of the encoder as the rate 
        standard = torch.distributions.MultivariateNormal(loc=torch.zeros(config.latent_dim),covariance_matrix=torch.diag_embed(torch.ones(config.latent_dim)))
        multi = torch.distributions.MultivariateNormal(loc=mu[-1:],scale_tril=torch.tril(cov_matrix[-1::]))
        
        
        KLD = torch.distributions.kl_divergence(standard,multi)
        KLD = torch.mean(KLD)
    
        return config.tradeoff*KLD
    
    def forward(self, x):
        
        # encodes the features that will be used to get the latent distributions
        lat=self.encoder(x)
        batch, _, _, _ = lat.shape
        
        lat = torch.flatten(lat,1,3)
        hidden_latent = self.latent_interpretation(lat)

        # the cor matrix holds the value of the correlation coefficients for the lower triangle of the correlation matrix
        # it is also clamped to avoid some numerical instability issues that it was having when the correlation values got very close to 0
        cor = self.correlation_encoder(x)
        cor = torch.flatten(cor,1,3)
    
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

        # finds the average kld for the threshold in the vectorquantizer because the calculated codebook entropy is for the
        # whole batch rather than for an individual forward pass and therefore does not scale to the value of the other components 
        # of the final loss

        kullback = self.kld(quant_input,cov_matrix)
        
        quant_input = quant_input.view(-1,latent_dim,1,1)
        
        embedding_loss, quant_output , perplexity, _, _, bits_loss = self.vector_quantization(quant_input,kullback)
        
        quant_output = quant_output.reshape(batch, -1)
        
		## Decoder part
        quant_output = self.post_quantization_linears(quant_output)

        decoder_input = torch.reshape(quant_output,(batch,64,6,6))

        
        #print(quant_output.shape)

        

        

        reconstruction = self.decoder(decoder_input)
        bits = torch.log(perplexity)
        
        return reconstruction, x, mu, cov_matrix, embedding_loss, bits, perplexity, bits_loss
    

    
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

        
        bits_loss = torch.nn.functional.relu(entropy-kld)
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices,bits_loss

# prevents the saturation of the tanh at the end of the correlation encoder as shown in yann lecunns paper "Efficient Back-Prop"
class new_tanh(nn.Module):
    def __init__(self):    
        super().__init__()
    def forward(self,x):
        return 1.7159 * F.tanh( 2/3 * x) 
        