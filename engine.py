from tqdm import tqdm
import torch 
import math
import config
with open("attempt.txt",'r') as infile:
    attempt = int(infile.read())
    config.attempt = attempt
infile.close
name = f'attempt_{config.attempt}'

def final_loss(mse_loss, mu, cov_matrix, bits):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence and the entropy of the joint distribution.
    :param bce_loss: recontruction loss
    :param logvar: log variance from the latent vector
    :param cov_matrix: the predicted covariance matrix
    """
    MSE = mse_loss 
    
    # the scale tril has to be given to the distribution as opposed to the covariance matrix because the condition that the cov matrix must be psd is 
    # awful and finds that psd matrices arent psd even when they absolutely are
    standard = torch.distributions.MultivariateNormal(loc=torch.zeros(config.latent_dim),covariance_matrix=torch.diag_embed(torch.ones(config.latent_dim)))
    multi = torch.distributions.MultivariateNormal(loc=mu[-1:],scale_tril=torch.tril(cov_matrix[-1::]))
    
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # modified kld for the multivariate distribution so that the kld and the entropy rate are calculated at the same time and the rate distortion can be properly learned
    KLD = torch.distributions.kl_divergence(standard,multi)
    KLD = torch.sum(KLD)
    
    
    # i am aware that this is absolutely not the correct way to get the actual bits per pixel but im hoping it may give a ballpark estimate for bugfixing
    BPP = bits/784.0
    
    # the log2 of e converts the nats that pytorch calculates into bits
    entropy = torch.mean(bits)*math.log2(math.e)
    # KLD = 0.5 * (1+torch.log2(torch.det(cov_matrix[-1::]))-latent_dim+(torch.sum(logvar.exp())/torch.det(cov_matrix[-1::]))+mu[-1:]*torch.inverse(cov_matrix[-1::])*torch.transpose(mu[-1:],dim0=0,dim1=1))
    
    return MSE + config.tradeoff*KLD, BPP

def train(model, dataloader, dataset, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    counter = 0
    bits_per_pixel = []
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data[0]
        
        
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, inputs, mu, cov_matrix, quant_loss, bits, perplexity, bits_loss = model(data)
        mse_loss = criterion(reconstruction, data)
        loss, BPP = final_loss(mse_loss, mu, cov_matrix, bits)
        loss = loss+quant_loss+config.batch_size*bits_loss
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        bits_per_pixel.append(BPP)
    train_loss = running_loss / counter
    with open("outputs\\runs\\"+name+"\\description.txt",'a') as outfile:
        outfile.write(f'''
                      
mse_loss:{mse_loss}

quantization_loss:{quant_loss}

batch_bits_loss:{config.batch_size*bits_loss}

bits_loss:{bits_loss}

''')
    outfile.close()
    return train_loss, bits_per_pixel

def validate(model, dataloader, dataset, device, criterion):
    model.eval()
    running_loss = 0.0
    counter = 0
    bits_per_pixel = []
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data= data[0]
            
            data = data.to(device)
            reconstruction, inputs, mu, cov_matrix, quant_loss, bits, perplexity, bits_loss = model(data)
            mse_loss = criterion(reconstruction, data)
            loss, BPP = final_loss(mse_loss, mu, cov_matrix, bits)
            
            loss = loss+quant_loss+config.batch_size*bits_loss
            running_loss += loss.item()
            bits_per_pixel.append(BPP)
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction
                comp_images = inputs
                
    val_loss = running_loss / counter
    return val_loss, recon_images, comp_images, bits_per_pixel