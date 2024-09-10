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
        
        self.layers = []

        # this creates the first block of convolutional layers which are outised the for loop because the image channels is 
        # a constant that cannot be integrated into the for loop

        self.layers.append(nn.Conv2d(in_channels=image_channels,
                                    out_channels=init_channels,
                                    kernel_size=kernel_size,
                                    stride=config.stride,
                                    padding=config.zero_padding))
        self.layers.append(nn.BatchNorm2d(init_channels))
        self.layers.append(nn.LeakyReLU())
        
        # this for loop creates a list of layers depending on what is set in the config file that can then be passed to torch.sequential
        # for a dynamic architecture controlled by the numbers set in config

        for i in range(config.encoder_convolutions-1) :
            self.layers.append(nn.Conv2d(in_channels=init_channels*(2**i),
                                        out_channels=init_channels*(2**(i+1)), 
                                        kernel_size=kernel_size, 
                                        stride=config.stride, 
                                        padding=config.zero_padding))
            self.layers.append(nn.BatchNorm2d(init_channels*(2**(i+1))))
            self.layers.append(nn.LeakyReLU())
        
        # this layer is the last convolutional layer that happens in the encoder it is outisde the for loop because
        # it cant have an activation or batch normalization associated with it
        
        self.layers.append(nn.Conv2d(in_channels=init_channels*(2**(config.encoder_convolutions-1)),
                                    out_channels=init_channels*(2**config.encoder_convolutions),
                                    kernel_size=kernel_size, 
                                    stride=2, 
                                    padding=0))
        
        # the asterisk forces the arguments out into a set of individual arguments instead of passing them as a list object
        # this sets up the basic encoder
        
        self.encoder = torch.nn.Sequential(*self.layers)
        
        # this does the same thing as the regular encoder but ends in a tanh
        # because the values will be treated as correlation coefficients

        self.layers.clear()

        self.layers.append(nn.Conv2d(in_channels=image_channels, 
                                     out_channels=init_channels, 
                                     kernel_size=kernel_size, 
                                     stride=config.stride, 
                                     padding=config.zero_padding))
        self.layers.append(nn.BatchNorm2d(init_channels))
        self.layers.append(nn.LeakyReLU())

        for i in range(config.correlation_convolutions-1):
            self.layers.append(nn.Conv2d(in_channels=init_channels*(2**i), 
                                        out_channels=init_channels*(2**(i+1)), 
                                        kernel_size=kernel_size, 
                                        stride=config.stride, 
                                        padding=config.zero_padding))
            self.layers.append(nn.BatchNorm2d(init_channels*(2**(i+1))))
            self.layers.append(nn.LeakyReLU())

        self.layers.append(nn.Conv2d(in_channels=init_channels*(2**(config.correlation_convolutions-1)), 
                                    out_channels=init_channels*(2**config.correlation_convolutions), 
                                    kernel_size=kernel_size, 
                                    stride=2, 
                                    padding=0))
        self.layers.append(nn.BatchNorm2d(init_channels*(2**config.correlation_convolutions)))
        self.layers.append(nn.Tanh())

        self.correlation_encoder = torch.nn.Sequential(*self.layers)
        
        self.layers.clear()
        
        # these fully connected layers come after the convolutions to allow for better interpretation of the values of the correlation coefficients
        
        

        for i in range(config.correlation_linears):    
            self.layers.append(nn.Linear(init_channels*(2**config.correlation_convolutions),
                                         init_channels*(2**config.correlation_convolutions)))
            self.layers.append(nn.LeakyReLU())

        # makes sure that the output of the correlation linear layers is equal to the latent dimensions squared so that
        # there is enough values to make a correlation matrix

        self.layers.append(nn.Linear(init_channels*(2**config.correlation_convolutions),latent_dim**2))

        self.cor_interpretation = torch.nn.Sequential(*self.layers)

        self.layers.clear()

        for i in range(config.encoder_linears):
                self.layers.append(nn.Linear(init_channels*(2**config.encoder_convolutions),
                                            init_channels*(2**config.encoder_convolutions)))
                self.layers.append(nn.LeakyReLU())
        
        self.latent_interpretation =torch.nn.Sequential(*self.layers)

        self.fc_mu = nn.Linear(init_channels*(2**config.encoder_convolutions), latent_dim)
        self.fc_log_var = nn.Linear(init_channels*(2**config.encoder_convolutions), latent_dim)

        self.layers.clear()

       

        self.layers.append(nn.ConvTranspose2d(in_channels=init_channels*(2**config.encoder_convolutions),
                                              out_channels=init_channels*(2**(config.encoder_convolutions-1)),
                                              kernel_size = kernel_size,
                                              stride=config.stride,
                                              padding=0))
        self.layers.append(nn.BatchNorm2d(init_channels*(2**(config.encoder_convolutions-1))))
        self.layers.append(nn.LeakyReLU())

        for i in range(config.encoder_convolutions-1):
            self.layers.append(nn.ConvTranspose2d(in_channels=init_channels*(2**(config.encoder_convolutions-(i+1))),
                                                out_channels=init_channels*(2**(config.encoder_convolutions-(i+2))),
                                                kernel_size=kernel_size,
                                                stride=config.stride,
                                                padding=config.zero_padding))
            self.layers.append(nn.BatchNorm2d(init_channels*(2**(config.encoder_convolutions-(i+2)))))
            self.layers.append(nn.LeakyReLU())

        # ends with a sigmoid because all image pixel values from test and training dataset are between 0 and 1 rather than for 
        # some theoretical reason

        self.layers.append(nn.ConvTranspose2d(in_channels = init_channels, 
                                            out_channels = image_channels, 
                                            kernel_size = 4 , 
                                            stride = 2, 
                                            padding = 0))
        self.layers.append(nn.Sigmoid())

        self.decoder = torch.nn.Sequential(*self.layers)

        self.layers.clear()

        self.layers.append(nn.Linear(latent_dim, init_channels*(2**config.encoder_convolutions)))

        self.post_quantization_linears = torch.nn.Sequential(*self.layers)
    
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
        lat = F.adaptive_avg_pool2d(lat, 1).reshape(batch, -1)
        hidden_latent = self.latent_interpretation(lat)

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

        # finds the average kld for the threshold in the vectorquantizer because the calculated codebook entropy is for the
        # whole batch rather than for an individual forward pass and therefore does not scale to the value of the other components 
        # of the final loss

        kullback = self.kld(quant_input,cov_matrix)
        
        quant_input = quant_input.view(-1,latent_dim,1,1)
        
        embedding_loss, quant_output , perplexity, _, _, bits_loss = self.vector_quantization(quant_input,kullback)
        
        
        quant_output = F.adaptive_avg_pool2d(quant_output, 1).reshape(batch, -1)
        
		## Decoder part
        quant_output = self.post_quantization_linears(quant_output)
        #print(quant_output.shape)

        decoder_input = quant_output.view(-1,init_channels*(2**(config.encoder_convolutions)),1,1)

        

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


