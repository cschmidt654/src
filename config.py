from torch import device
# definattempt = 2erent numbers associated with the convolutional layer

latent_dim = 32
kernel_size = 3
init_channels = 8
image_channels = 1
zero_padding = 0
stride = 2

# defines the numbers associated with the architecture of the variational autoencoder part of the neural network

encoder_convolutions = 3
correlation_convolutions = 3
encoder_linears = 4
correlation_linears = 4

# defines the numbers associated with the vector quantization network

num_embeddings = 6000
embedding_dimensions = 16
beta = 0.25

# defines the number assocciated with training and evaluation

# this is the lambda for rate distortion tradeoff but lambda is a special word in python
tradeoff = 2.0
reduction  = "sum"
lr = 1e-3
epochs = 200
batch_size = 64
hardware = device("cpu")
attempt = 0