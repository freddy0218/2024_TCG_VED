import os,sys,gc
import numpy as np
import pickle
import torch
from tqdm.auto import tqdm
sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED/util/')
import derive_var,read_and_proc
import glob

import random
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  
setup_seed(42)

class spread_error_diagram:
    def __init__(self,criteria,typee,model,droprate):
        self.criteria=criteria
        self.typee=typee
        self.model=model
        self.droprate=droprate
    
    def _get_spreadcriteria(self,inmodel=None):
        return [np.percentile(inmodel[self.typee],obj) for obj in self.criteria]

    def get_spreadcriteria(self,inmodel=None):
        return [self._get_spreadcriteria(obj) for obj in inmodel]
    
    def get_allcriteria(self):
        store = []
        for ind,numdrop in enumerate(self.droprate):
            temp = self.model[numdrop]
            temptemp = [self.get_spreadcriteria(temp[i]['spreads']) for i in range(len(temp))]
            store.append(temptemp)
        return store
    
    def get_criteria_flatten(self):
        setup_seed(42)
        storetestcrit = []
        for ind, numdrop in enumerate(self.droprate):
            tempp = []
            hihihi = ([obj['spreads'] for obj in self.model[numdrop]])
            for i in range(len(self.model[numdrop])):
                temp = hihihi[i]
                tempp.append(read_and_proc.flatten([obj['valid'] for obj in temp]))
            testst = [[np.percentile(np.asarray((tempp[k])),j) for j in self.criteria] for k in range(len(self.model[numdrop]))]
            storetestcrit.append(testst)
        return storetestcrit
    
    def _get_rmse_curve(self,criterias=None,modelset=[0,0,0]):
        setup_seed(42)
        storermse = []
        modelsetindex = np.abs(np.asarray(self.droprate)-modelset[0]).argmin()
        criterianum = criterias[modelsetindex][modelset[1]]

        ssRel = 0
        nPts = np.asarray(self.model[modelset[0]][modelset[1]]['truth'][modelset[2]][self.typee]).size
        for i in (range(len(criterianum)-1)):
            ytruth = np.asarray(self.model[modelset[0]][modelset[1]]['truth'][modelset[2]][self.typee])[np.where(np.logical_and(self.model[modelset[0]][modelset[1]]['spreads'][modelset[2]][self.typee]>criterianum[i],
                                                                                                                             self.model[modelset[0]][modelset[1]]['spreads'][modelset[2]][self.typee]<criterianum[i+1]))]
            if self.typee=='valid':
                valstorename = 'meanvals'
            elif self.typee=='test':
                valstorename = 'meantests'
            elif self.typee=='train':
                valstorename = 'meantrains'
                
            nPtsbin = ytruth.size
            
            ypred = self.model[modelset[0]][modelset[1]][valstorename][modelset[2]][np.where(np.logical_and(self.model[modelset[0]][modelset[1]]['spreads'][modelset[2]][self.typee]>criterianum[i],
                                                                                                          self.model[modelset[0]][modelset[1]]['spreads'][modelset[2]][self.typee]<criterianum[i+1]))]
            spreadss = self.model[modelset[0]][modelset[1]]['spreads'][modelset[2]][self.typee][np.where(np.logical_and(self.model[modelset[0]][modelset[1]]['spreads'][modelset[2]][self.typee]>criterianum[i],
                                                                                                                             self.model[modelset[0]][modelset[1]]['spreads'][modelset[2]][self.typee]<criterianum[i+1]))]
            try:
                storermse.append(np.sqrt(mean_squared_error(ytruth,ypred)))
                ssRel += (nPtsbin/nPts) * np.abs(np.sqrt(mean_squared_error(ytruth,ypred)) - np.mean(spreadss))
            except:
                storermse.append(np.nan)
                ssRel += 0
        return storermse, ssRel
    
    def get_rmse_curve(self,criterias=None):
        setup_seed(42)
        out2,out2ssrel = {},{}
        for inddrop,numdrop in (enumerate(self.droprate)):
            out1,out1ssrel = [],[]
            for indsplit in (range(len(self.model[numdrop]))):
                out1.append([self._get_rmse_curve(criterias,[numdrop,indsplit,indmodel])[0] for indmodel in range(len(self.model[numdrop][indsplit]['models']))])
                out1ssrel.append([self._get_rmse_curve(criterias,[numdrop,indsplit,indmodel])[1] for indmodel in range(len(self.model[numdrop][indsplit]['models']))])
            out2[inddrop] = out1
            out2ssrel[inddrop] = out1ssrel
        return out2,out2ssrel
    
    def _get_ssrat(self,modelset=[0,0,0]):
        if self.typee=='valid':
            valstorename = 'meanvals'
        elif self.typee=='test':
            valstorename = 'meantests'
        elif self.typee=='train':
            valstorename = 'meantrains'
        spreads = self.model[modelset[0]][modelset[1]]['spreads'][modelset[2]][self.typee]
        ytruth,ypred = np.asarray(self.model[modelset[0]][modelset[1]]['truth'][modelset[2]][self.typee]),self.model[modelset[0]][modelset[1]][valstorename][modelset[2]]
        
        ssrel = np.mean(spreads) / np.sqrt(mean_squared_error(ytruth, ypred))
        return ssrel
    
    def get_ssrat(self):
        out2 = {}
        for inddrop,numdrop in enumerate(self.droprate):
            out1 = []
            for indsplit in range(len(self.model[numdrop])):
                out1.append([self._get_ssrat([numdrop,indsplit,indmodel]) for indmodel in range(len(self.model[numdrop][indsplit]['models']))])
            out2[inddrop] = out1
        return out2