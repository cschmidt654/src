import torch
import torch.optim as optim
import torch.nn as nn
from models.models_v2 import ConvVAE
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import engine
import utils
import config
from datasets.cld_high_dataset import CLD_high_dataset
import os
import torchvision


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initialize the model
model = ConvVAE()
model.to(device)
# set the learning parameters
# also setting the learning rate too high has a tendency to send gigantic negative gradients through the network and cause its averages to be nan 
lr = config.lr
epochs = config.epochs
batch_size = config.batch_size
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss(reduction=config.reduction)
# a list to save all the reconstructed images in PyTorch grid format
grid_images = []
ground_truth = []

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
# training set and train data loader
#data = CLD_high_dataset()
trainset = CLD_high_dataset()

# trainset = torchvision.datasets.MNIST(
#     root='../input', train=True, download=True, transform=transform
# )

trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)

# validation set and validation data loader
testset = CLD_high_dataset()

# testset = torchvision.datasets.MNIST(
#     root='../input', train=False, download=True, transform=transform
# )

testloader = DataLoader(
    testset, batch_size=batch_size, shuffle=True
)

train_loss = []
valid_loss = []
# this part of the code is for printing out the information for this training attempt
with open("attempt.txt",'r') as infile:
    attempt = int(infile.read())
    config.attempt = attempt
infile.close()

with open("attempt.txt","r+") as attempts:
    contents = attempts.read()
    attempts.seek(0)
    attempts.truncate()
    attempts.write(f'{config.attempt+1}')    
attempts.close()

name = f'attempt_{config.attempt}'
directory = name
parent_directory = "outputs\\runs"
path = os.path.join(parent_directory, directory)
os.makedirs(path)

with open("outputs\\runs\\"+name+"\\description.txt",'w') as outfile:
    
    outfile.write(f'''latent_dim:{config.latent_dim}
kernel_size:{config.kernel_size}
init_channels:{config.init_channels}
image_channels:{config.image_channels}
zero_padding:{config.zero_padding}
stride:{config.stride}

encoder_convolutions:{config.encoder_convolutions}
correlation_convolutions:{config.correlation_convolutions}
encoder_linears:{config.encoder_linears}
correlation_linears:{config.encoder_linears}

num_embeddings:{config.num_embeddings}
embedding_dimensions:{config.embedding_dimensions}
beta:{config.beta}

tradeoff:{config.tradeoff}
reduction:{config.reduction}
learning_rate:{config.lr}
epochs:{config.epochs}
batch_size:{config.batch_size}
device:{config.hardware}''')
    
outfile.close()

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_BPP = engine.train(
        model, trainloader, trainset, device, optimizer, criterion
    )
    valid_epoch_loss, recon_images, comp_images, validate_BPP = engine.validate(
        model, testloader, testset, device, criterion
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    # save the reconstructed images from the validation loop
    utils.save_reconstructed_images(recon_images, epoch+1)
    utils.save_ground_truth(comp_images, epoch+1)
    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())
    ground_truth_grid = make_grid(comp_images.detach().cpu())
    grid_images.append(image_grid)
    ground_truth.append(ground_truth_grid)
    average_train_BPP = sum(train_BPP)/len(train_BPP)
    average_validate_BPP = sum(validate_BPP)/len(validate_BPP)
    
    with open("outputs\\runs\\"+name+"\\description.txt",'a') as outfile:
        outfile.write(f'''
epoch:{epoch+1}
codebook_entropy:{average_train_BPP*784}
''')
    outfile.close()
    
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")
    print(average_train_BPP)
    print(average_validate_BPP)
    
# save the reconstructions as a .gif file
utils.image_to_vid(grid_images)
# save the loss plots to disk
utils.save_loss_plot(train_loss, valid_loss)

print('TRAINING COMPLETE')