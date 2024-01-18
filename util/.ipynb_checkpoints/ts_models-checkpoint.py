import os,sys,gc
import numpy as np
import pickle
import torch
#from livelossplot import PlotLosses
from tqdm.auto import tqdm


class OptimMLR_lwswhdia_3D_ts(torch.nn.Module):
    def __init__(self):
        #super(OptimMLR_all_2D, self).__init__()
        super(OptimMLR_lwswhdia_3D_ts, self).__init__()
        ############################################################
        # Input channels
        ############################################################
        brchsize = [82,20,20]#[50,38,91,8,82,20,20]
        self.dense1 = torch.nn.Linear(brchsize[0], 1)
        self.dense2 = torch.nn.Linear(brchsize[1], 1)
        self.dense3 = torch.nn.Linear(brchsize[2], 1)
        #self.dense4 = torch.nn.Linear(brchsize[3], 1)
        #self.dense5 = torch.nn.Linear(brchsize[4], 1)
        #self.dense6 = torch.nn.Linear(brchsize[5], 1)
        #self.dense7 = torch.nn.Linear(brchsize[6], 1)
        ############################################################
        # Final Dense Layer
        ############################################################
        self.denseout = torch.nn.Linear(3,1)#106)
        
    def forward(self,X):
        brchindex = list(np.asarray([0,50,38,50,8,50,20,20]).cumsum())
        X_u, X_v, X_w, X_th = X[:,brchindex[0]:brchindex[1]],X[:,brchindex[1]:brchindex[2]],X[:,brchindex[2]:brchindex[3]],X[:,brchindex[3]:brchindex[4]]
        X_hdia, X_lw, X_sw = X[:,brchindex[4]:brchindex[5]],X[:,brchindex[5]:brchindex[6]],X[:,brchindex[6]:brchindex[7]]
        ############################################################
        # Optimal PC layer
        ############################################################
        besthdia = self.dense1(X_hdia)
        bestlw = self.dense2(X_lw)
        bestsw = self.dense3(X_sw)
        ############################################################
        # Concat
        ############################################################
        bestPC = torch.cat((besthdia,bestlw,bestsw),1)
        ############################################################
        # Prediction layer
        ############################################################
        outpred = self.denseout(bestPC)
        return outpred

    def compute_l2_loss(self, w):
        return torch.square(w).sum()


class OptimMLR_lwsw_3D_ts_dropout2(torch.nn.Module):
    def __init__(self,droprate,brchindices):
        #super(OptimMLR_all_2D, self).__init__()
        super(OptimMLR_lwsw_3D_ts_dropout2, self).__init__()
        self.brchindices = brchindices
        ############################################################
        # Input channels
        ############################################################
        brchsize = self.brchindices[-2:]#[20,20]#[50,38,91,8,82,20,20]
        self.dense1 = torch.nn.Linear(brchsize[0], 1)
        self.dense2 = torch.nn.Linear(brchsize[1], 1)
        #self.dense3 = torch.nn.Linear(brchsize[2], 1)
        #self.dense4 = torch.nn.Linear(brchsize[3], 1)
        #self.dense5 = torch.nn.Linear(brchsize[4], 1)
        #self.dense6 = torch.nn.Linear(brchsize[5], 1)
        #self.dense7 = torch.nn.Linear(brchsize[6], 1)
        self.dropout1 = torch.nn.Dropout(droprate)
        self.dropout2 = torch.nn.Dropout(droprate)
        self.dropout3 = torch.nn.Dropout(droprate)
        ############################################################
        # Final Dense Layer
        ############################################################
        self.denseout = torch.nn.Linear(2,1)#106)
        
    def forward(self,X):
        brchindex = list(np.asarray(self.brchindices).cumsum())#[0,50,38,50,8,50,20,20]).cumsum())
        X_u, X_v, X_w, X_th = X[:,brchindex[0]:brchindex[1]],X[:,brchindex[1]:brchindex[2]],X[:,brchindex[2]:brchindex[3]],X[:,brchindex[3]:brchindex[4]]
        X_hdia, X_lw, X_sw = X[:,brchindex[4]:brchindex[5]],X[:,brchindex[5]:brchindex[6]],X[:,brchindex[6]:brchindex[7]]
        ############################################################
        # Optimal PC layer
        ############################################################
        X_lwc = self.dropout1(X_lw)
        bestlw = self.dense1(X_lwc)
        X_swc = self.dropout2(X_sw)
        bestsw = self.dense2(X_swc)
        ############################################################
        # Concat
        ############################################################
        bestPC = torch.cat((bestlw,bestsw),1)
        ############################################################
        # Prediction layer
        ############################################################
        bestPC = self.dropout3(bestPC)
        outpred = self.denseout(bestPC)
        return outpred
    
    def compute_l2_loss(self, w):
        return torch.square(w).sum()
    
    def compute_l2_loss(self, w):
        return torch.square(w).sum()
    
class OptimMLR_lwsw_3D_ts_dropout2_nonln(torch.nn.Module):
    def __init__(self,droprate,brchindices,nonlinear_num):
        #super(OptimMLR_all_2D, self).__init__()
        super(OptimMLR_lwsw_3D_ts_dropout2_nonln, self).__init__()
        self.brchindices = brchindices
        self.nonlinear_num = nonlinear_num
        ############################################################
        # Input channels
        ############################################################
        brchsize = self.brchindices[-2:]#[20,20]#[50,38,91,8,82,20,20]
        self.dense1 = torch.nn.Linear(brchsize[0], 1)
        self.dense2 = torch.nn.Linear(brchsize[1], 1)
        self.dropout1 = torch.nn.Dropout(droprate)
        self.dropout2 = torch.nn.Dropout(droprate)
        self.dropout3 = torch.nn.Dropout(droprate)
        ############################################################
        # Nonlinear channels
        ############################################################        
        self.nonln = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(len(brchsize),len(brchsize)),
                                torch.nn.LeakyReLU()) for i in range(self.nonlinear_num)])
        ############################################################
        # Final Dense Layer
        ############################################################
        self.denseout = torch.nn.Linear(len(brchsize),1)#106)
        
    def forward(self,X):
        brchindex = list(np.asarray(self.brchindices).cumsum())#[0,50,38,50,8,50,20,20]).cumsum())
        X_u, X_v, X_w, X_th = X[:,brchindex[0]:brchindex[1]],X[:,brchindex[1]:brchindex[2]],X[:,brchindex[2]:brchindex[3]],X[:,brchindex[3]:brchindex[4]]
        X_hdia, X_lw, X_sw = X[:,brchindex[4]:brchindex[5]],X[:,brchindex[5]:brchindex[6]],X[:,brchindex[6]:brchindex[7]]
        ############################################################
        # Optimal PC layer
        ############################################################
        X_lwc = self.dropout1(X_lw)
        bestlw = self.dense1(X_lwc)
        X_swc = self.dropout2(X_sw)
        bestsw = self.dense2(X_swc)
        ############################################################
        # Concat
        ############################################################
        bestPC = torch.cat((bestlw,bestsw),1)
        bestPC = self.dropout3(bestPC)
        bestPC_proc = bestPC
        ############################################################
        # Nonlinear layer
        ############################################################
        for nonln in self.nonln:
            bestPC_proc = nonln(bestPC_proc)
        ############################################################
        # Prediction layer
        ############################################################
        bestPC_proc = self.dropout3(bestPC_proc)
        outpred = self.denseout(bestPC_proc)
        return outpred
    
    def compute_l2_loss(self, w):
        return torch.square(w).sum()
    
    def compute_l2_loss(self, w):
        return torch.square(w).sum()
    
    
class OptimMLR_lwswinten_3D_ts_dropout(torch.nn.Module):
    def __init__(self,droprate,brchindices):
        #super(OptimMLR_all_2D, self).__init__()
        super(OptimMLR_lwswinten_3D_ts_dropout, self).__init__()
        self.brchindices = brchindices
        ############################################################
        # Input channels
        ############################################################
        brchsize = self.brchindices[-3:-1]#[20,20]#[50,38,91,8,82,20,20]
        self.dense1 = torch.nn.Linear(brchsize[0], 1)
        self.dense2 = torch.nn.Linear(brchsize[1], 1)
        self.intensity = torch.nn.Linear(1,1)
        self.dropout1 = torch.nn.Dropout(droprate)
        self.dropout2 = torch.nn.Dropout(droprate)
        self.dropout3 = torch.nn.Dropout(droprate)
        ############################################################
        # Final Dense Layer
        ############################################################
        self.denseout = torch.nn.Linear(3,1)#106)
        
    def forward(self,X):
        brchindex = list(np.asarray(self.brchindices).cumsum())#[0,50,38,50,8,50,20,20]).cumsum())
        X_u, X_v, X_w, X_th = X[:,brchindex[0]:brchindex[1]],X[:,brchindex[1]:brchindex[2]],X[:,brchindex[2]:brchindex[3]],X[:,brchindex[3]:brchindex[4]]
        X_hdia, X_lw, X_sw = X[:,brchindex[4]:brchindex[5]],X[:,brchindex[5]:brchindex[6]],X[:,brchindex[6]:brchindex[7]]
        X_inten = X[:,brchindex[7]:]
        ############################################################
        # Optimal PC layer
        ############################################################
        X_lwc = self.dropout1(X_lw)
        bestlw = self.dense1(X_lwc)
        X_swc = self.dropout2(X_sw)
        bestsw = self.dense2(X_swc)
        ############################################################
        # Intensity layer
        ############################################################
        inten = self.intensity(X_inten)
        ############################################################
        # Concat
        ############################################################
        bestPC = torch.cat((bestlw,bestsw,inten),1)
        ############################################################
        # Prediction layer
        ############################################################
        bestPC = self.dropout3(bestPC)
        outpred = self.denseout(bestPC)
        return outpred
    
    def compute_l2_loss(self, w):
        return torch.square(w).sum()
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# Customzied evaluation metric NSE for validation set and test set # 
def calc_nse(sim: torch.FloatTensor, obs: torch.FloatTensor, global_obs_mean: torch.FloatTensor) -> float:
    """Calculate the Nash-Sutcliff-Efficiency coefficient.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :param global_obs_mean: mean of the whole observation series
    :return: NSE value.
    """
    numerator = torch.square(sim - obs).sum()
    #denominator = torch.square(obs - global_obs_mean).sum()
    #nse_val = 1 - numerator / denominator

    return numerator

def eval_model(model, dataloader, loss_func, l2_lambda):
    with torch.no_grad():
        loss = 0
        metric = 0
        
        global_sum = 0
        label_size = 0
        for feature, labels in dataloader:
            global_sum += labels.sum()
            label_size += len(labels)
            
        global_mean = global_sum / label_size
        model.train()
        for features, labels in dataloader:
            pred = model(features)
            batch_loss = loss_func(pred, labels.unsqueeze(1))
            
            l2_parameters = []
            for parameter in model.parameters():
                l2_parameters.append(parameter.view(-1))
                l2 = l2_lambda * model.compute_l2_loss(torch.cat(l2_parameters))
            batch_loss += l2
            loss+=batch_loss.item()
            
        num_batches = len(dataloader)
        
        loss = loss/num_batches
        return loss
    
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_model(model=None,optimizer=None,scheduler=None,numepochs=None,early_stopper=None,variance_store=None,lossfunc=None,regularization='None',l1_lambda=0.01,l2_lambda=0.1,train_loader=None,val_loader=None,test_loader=None):
    # Custom loss: MSE_physicalLoss(eigenvectors,wcomps,variance_store)
    #liveloss = PlotLosses()
    schedulerCY,schedulerLS = scheduler[1],scheduler[0]
    train_losses,trainu_losses,trainv_losses,trainw_losses,trainth_losses = [],[],[],[],[]
    val_losses = []
    val_NSEs = []
    statedicts = []
    for epoch in (range(int(numepochs))):
        """
        Initialize loss
        """
        train_loss = 0
        """
        Operate per batch
        """
        for features, labels in train_loader:
            optimizer.zero_grad()
            
            prediction = model(features)
            batch_loss = lossfunc(prediction, labels.unsqueeze(1))#loss_func(prediction, labels)
            
            if regularization=='L1':
                #L1 regularization-------------------------------
                l1_norm = sum(abs(p).sum() for p in model.parameters())
                #-------------------------------------------------
                batch_loss = batch_loss + l1_lambda * l1_norm
                optimizer.zero_grad()
                batch_loss.backward()
            elif regularization=='L2':
                # Compute l2 loss component
                l2_parameters = []
                for parameter in model.parameters():
                    l2_parameters.append(parameter.view(-1))
                    l2 = l2_lambda * model.compute_l2_loss(torch.cat(l2_parameters))
                batch_loss += l2
                batch_loss.backward() 
            elif regularization=='None':
                batch_loss.backward()                
            
            optimizer.step()
            schedulerCY.step()
            
            train_loss += batch_loss.item()
            
            
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        model.train()
        val_loss = eval_model(model,
                val_loader,
                lossfunc,
                l2_lambda)
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
                if counter >= 10:
                    break
            else:
                #val_NSEs.append(val_NSE)
                val_losses.append(val_loss)
        else:
            #val_NSEs.append(val_NSE)
            val_losses.append(val_loss)
            
        if early_stopper:
            if early_stopper.__call__(val_loss, model):
                break
        
        if epoch % 300 == 0:
            print(((train_loss),(val_loss)))
            
    #return model, {'train':train_losses,'utrain':trainu_losses,'vtrain':trainv_losses,'wtrain':trainw_losses,'thtrain':trainth_losses,'val':val_losses} 
    return model, {'train':train_losses,'val':val_losses}, statedicts
