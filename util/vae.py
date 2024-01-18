import torch
import torch.nn as nn
import torch.nn.functional as F
import os,sys,gc
import numpy as np
import pickle
from tqdm.auto import tqdm
import random

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  
#setup_seed(42)

class BranchEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(BranchEncoder, self).__init__()

        self.fc_mean = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        mu = self.fc_mean(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim1, latent_dim2, output_dim):
        super(Decoder, self).__init__()

        # Linear layers for both latent spaces to directly output to the regression output
        self.fc1 = nn.Linear(latent_dim1, output_dim, bias=True)
        self.fc2 = nn.Linear(latent_dim2, output_dim, bias=True)

    def forward(self, z1, z2):
        # Compute linear regression outputs from both latent spaces
        out1 = self.fc1(z1)
        out2 = self.fc2(z2)

        # Combine the outputs (sum them up)
        return out1 + out2


class VAE(nn.Module):
    def __init__(self, input_dim1, input_dim2, latent_dim1, latent_dim2, output_dim, brchindices):
        super(VAE, self).__init__()

        self.encoder1 = BranchEncoder(input_dim1, latent_dim1)
        self.encoder2 = BranchEncoder(input_dim2, latent_dim2)
        self.decoder = Decoder(latent_dim1, latent_dim2, output_dim)
        self.brchindices = brchindices

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, X):
        brchindex = list(np.asarray(self.brchindices).cumsum())#[0,50,38,50,8,50,20,20]).cumsum())
        X_u, X_v, X_w, X_th = X[:,brchindex[0]:brchindex[1]],X[:,brchindex[1]:brchindex[2]],X[:,brchindex[2]:brchindex[3]],X[:,brchindex[3]:brchindex[4]]
        X_hdia, X_lw, X_sw = X[:,brchindex[4]:brchindex[5]],X[:,brchindex[5]:brchindex[6]],X[:,brchindex[6]:brchindex[7]]
        mu1, log_var1 = self.encoder1(X_lw)
        mu2, log_var2 = self.encoder2(X_sw)

        z1 = self.reparameterize(mu1, log_var1)
        z2 = self.reparameterize(mu2, log_var2)

        return self.decoder(z1, z2), mu1, log_var1, mu2, log_var2

def vae_loss(reconstructed_x, x, mu1, log_var1, mu2, log_var2, coeff):
    recon_loss = F.l1_loss(reconstructed_x, x, reduction='sum')
    kl_loss1 = -0.5 * torch.sum(1 + log_var1 - mu1.pow(2) - log_var1.exp())
    kl_loss2 = -0.5 * torch.sum(1 + log_var2 - mu2.pow(2) - log_var2.exp())
    return coeff*recon_loss + (1-coeff)*(kl_loss1 + kl_loss2), coeff*recon_loss, (1-coeff)*(kl_loss1 + kl_loss2)


def train_model(model=None,optimizer=None,scheduler=None,numepochs=None,early_stopper=None,variance_store=None,lossfunc=None,regularization='None',l1_lambda=0.01,l2_lambda=0.1,train_loader=None,val_loader=None,test_loader=None,count=None,
               vaeloss_coeff=1):
    # Custom loss: MSE_physicalLoss(eigenvectors,wcomps,variance_store)
    #liveloss = PlotLosses()
    schedulerCY,schedulerLS = scheduler[1],scheduler[0]
    train_losses,trainrecon_losses,trainkl_losses = [],[],[]
    val_losses,valrecon_losses,valkl_losses = [],[],[]
    val_NSEs = []
    statedicts = []
    for epoch in (range(int(numepochs))):
        """
        Initialize loss
        """
        train_loss = 0
        trainrecon_loss = 0
        trainkl_loss = 0
        """
        Operate per batch
        """
        for features, labels in train_loader:
            optimizer.zero_grad()
            
            reconX,mu1,logvar1,mu2,logvar2 = model(features)
            batch_loss,recon_loss,kl_loss = vae_loss(reconX, labels.unsqueeze(1),mu1,logvar1,mu2,logvar2,vaeloss_coeff)
            
            batch_loss.backward()                
            
            optimizer.step()
            schedulerCY.step()
            
            train_loss += batch_loss.item()
            trainrecon_loss += recon_loss.item()
            trainkl_loss += kl_loss.item()
            
            
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        trainrecon_loss = trainrecon_loss / len(train_loader)
        trainrecon_losses.append(trainrecon_loss)
        trainkl_loss = trainkl_loss / len(train_loader)
        trainkl_losses.append(trainkl_loss)
        
        model.train()
        criterion = vae_loss
        val_loss,valrecon_loss,valkl_loss = eval_model(model,
                                                       val_loader,
                                                       criterion,
                                                       l2_lambda,
                                                       vaeloss_coeff)
        schedulerLS.step(val_loss)
        statedicts.append(model.state_dict())
        
        ##################################################################
        # Early Stopping (valid / train)
        ##################################################################
        counter = 0
        if len(val_losses)>=1:
            best_score = val_losses[-1]
            if val_loss > best_score:
                counter += 1
                #val_NSEs.append(val_NSE)
                val_losses.append(val_loss)
                valrecon_losses.append(valrecon_loss)
                valkl_losses.append(valkl_loss)
                if counter >= count:
                    break
            else:
                #val_NSEs.append(val_NSE)
                val_losses.append(val_loss)
                valrecon_losses.append(valrecon_loss)
                valkl_losses.append(valkl_loss)
        else:
            #val_NSEs.append(val_NSE)
            val_losses.append(val_loss)
            valrecon_losses.append(valrecon_loss)
            valkl_losses.append(valkl_loss)
            
        if early_stopper:
            if early_stopper.__call__(val_loss, model):
                break
        
        if epoch % 300 == 0:
            print(((train_loss),(val_loss)))
            
    #return model, {'train':train_losses,'utrain':trainu_losses,'vtrain':trainv_losses,'wtrain':trainw_losses,'thtrain':trainth_losses,'val':val_losses} 
    return model, {'trainALL':train_losses,'valALL':val_losses,'trainRECON':trainrecon_losses,'valRECON':valrecon_losses,'trainKL':trainkl_losses,'valKL':valkl_losses}, statedicts

def eval_model(model, dataloader, loss_func, l2_lambda, vaeloss_coeff):
    with torch.no_grad():
        loss,loss2,loss3 = 0,0,0
        metric = 0
        
        global_sum = 0
        label_size = 0
        for feature, labels in dataloader:
            global_sum += labels.sum()
            label_size += len(labels)
            
        global_mean = global_sum / label_size
        model.train()
        for features, labels in dataloader:
            reconX,mu1,logvar1,mu2,logvar2 = model(features)
            batch_loss,recon_loss,kl_loss = vae_loss(reconX, labels.unsqueeze(1),mu1,logvar1,mu2,logvar2,vaeloss_coeff)
            
            #l2_parameters = []
            #for parameter in model.parameters():
            #    l2_parameters.append(parameter.view(-1))
            #    l2 = l2_lambda * model.compute_l2_loss(torch.cat(l2_parameters))
            #batch_loss += l2
            loss+=batch_loss.item()
            loss2+=recon_loss.item()
            loss3+=kl_loss.item()
            
        num_batches = len(dataloader)
        
        loss = loss/num_batches
        loss2 = loss2/num_batches
        loss3 = loss3/num_batches
        return loss,loss2,loss3