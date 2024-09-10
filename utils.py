import imageio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import config

to_pil_image = transforms.ToPILImage()

with open("attempt.txt",'r') as infile:
    attempt = int(infile.read())
    config.attempt = attempt
infile.close

name = f'attempt_{config.attempt}'

def image_to_vid(images):
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave(f'outputs/runs/{name}/generated_images.gif', imgs)

def save_reconstructed_images(recon_images, epoch):
    save_image(recon_images.cpu(), f"outputs\\runs\\{name}\\output{epoch}.png")

def save_ground_truth(comp_images, epoch):
    save_image(comp_images.cpu(), f"outputs\\runs\\{name}\\ground_truth{epoch}.png")

def save_loss_plot(train_loss, valid_loss):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'outputs/runs/{name}/loss.png')
    plt.show()


