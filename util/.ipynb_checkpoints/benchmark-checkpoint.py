import os,sys,gc
sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED/util/')
import derive_var,read_and_proc
import numpy as np
import pickle
import torch
from tqdm.auto import tqdm
import pandas as pd
import seaborn as sns
import derive_var,read_and_proc
sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/parallel/')
import properscoring as ps
import glob
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import random

def prepare_tensors(filepath='./haiyan_store',splitnum=None,explag=23,notensor='No'):
    if 'haiyan_store' in filepath:
        Xtrain = read_and_proc.depickle(glob.glob(filepath+'/X/'+str(splitnum)+'/*intensity*')[0])['uvwthhdialwsw']['train']
        Xvalid = read_and_proc.depickle(glob.glob(filepath+'/X/'+str(splitnum)+'/*intensity*')[0])['uvwthhdialwsw']['valid']
        Xtest = read_and_proc.depickle(glob.glob(filepath+'/X/'+str(splitnum)+'/*intensity*')[0])['uvwthhdialwsw']['test']
        ytrain = read_and_proc.depickle(glob.glob(filepath+'/y/'+str(splitnum)+'/*intensity*')[0])['train'][explag]
        yvalid = read_and_proc.depickle(glob.glob(filepath+'/y/'+str(splitnum)+'/*intensity*')[0])['valid'][explag]
        ytest = read_and_proc.depickle(glob.glob(filepath+'/y/'+str(splitnum)+'/*intensity*')[0])['test'][explag]
    elif 'maria_store' in filepath:
        Xtrain = read_and_proc.depickle(glob.glob(filepath+'/X/'+str(splitnum)+'/*corr*')[0])['uvwthhdialwsw']['train']
        Xvalid = read_and_proc.depickle(glob.glob(filepath+'/X/'+str(splitnum)+'/*corr*')[0])['uvwthhdialwsw']['valid']
        Xtest = read_and_proc.depickle(glob.glob(filepath+'/X/'+str(splitnum)+'/*corr*')[0])['uvwthhdialwsw']['test']
        ytrain = read_and_proc.depickle(glob.glob(filepath+'/y/'+str(splitnum)+'/*corr*')[0])['train'][explag]
        yvalid = read_and_proc.depickle(glob.glob(filepath+'/y/'+str(splitnum)+'/*corr*')[0])['valid'][explag]
        ytest = read_and_proc.depickle(glob.glob(filepath+'/y/'+str(splitnum)+'/*corr*')[0])['test'][explag]
    
    X_totrain, y_totrain = read_and_proc.delete_padding(Xtrain,ytrain)
    X_tovalid, y_tovalid = read_and_proc.delete_padding(Xvalid,yvalid)
    X_totest, y_totest = read_and_proc.delete_padding(Xtest,ytest)
    
    if notensor=='No':
        calc_device = 'cpu'
        ###################################################################################
        # Convert numpy arrays into tensors
        ###################################################################################
        train_Xtensor = torch.FloatTensor(X_totrain).to(calc_device)
        train_ytensor = torch.FloatTensor(y_totrain).to(calc_device)
        val_Xtensor = torch.FloatTensor(X_tovalid).to(calc_device)
        val_ytensor = torch.FloatTensor(y_tovalid).to(calc_device)
        test_Xtensor = torch.FloatTensor(X_totest).to(calc_device)
        test_ytensor = torch.FloatTensor(y_totest).to(calc_device)
        #train_data = torch.utils.data.TensorDataset(train_Xtensor, train_ytensor)
        #val_data = torch.utils.data.TensorDataset(val_Xtensor, val_ytensor)
        #test_data = torch.utils.data.TensorDataset(test_Xtensor, test_ytensor)
        
        return {'train':[train_Xtensor,train_ytensor],'valid':[val_Xtensor,val_ytensor],'test':[test_Xtensor,test_ytensor]}
    elif notensor=='Yes':
        return {'train':[X_totrain, y_totrain],'valid':[X_tovalid, y_tovalid],'test':[X_totest, y_totest]}

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  

def model_outweights_all(model=None):
    params,names = [],[]
    for name, param in model.named_parameters():
        params.append(param)
        names.append(name)
    return params, names

def model_outweights(model=None):
    params,names = [],[]
    for name, param in model.named_parameters():
        if ".weight" not in name:
            continue            
        else:
            params.append(param)
            names.append(name)
    return params, names

def grab_predictions(model=None,Xtensors=None,trial=20,TYPE='all'):
    if TYPE=='all':
        return [np.squeeze(model.train()(Xtensors).detach().numpy().transpose()) for i in range(trial)],np.nanmean(np.asarray([np.squeeze(model.train()(Xtensors).detach().numpy().transpose()) for i in range(trial)]),axis=0)
    elif TYPE=='Drop1_2':
        model.dropout1.train()
        model.dropout2.train()
        storeweights = [model_outweights(model) for i in range(trial)]
        return [np.squeeze(model(Xtensors).detach().numpy().transpose()) for i in range(trial)],np.nanmean(np.asarray([np.squeeze(model(Xtensors).detach().numpy().transpose()) for i in range(trial)]),axis=0),storeweights
    elif TYPE=='ensemble':
        return [np.squeeze(model(Xtensors).detach().numpy().transpose()) for i in range(trial)],np.nanmean(np.asarray([np.squeeze(model(Xtensors).detach().numpy().transpose()) for i in range(trial)]),axis=0),model_outweights(model)
    elif TYPE=='vae':
        return [np.squeeze(model(Xtensors)[0].detach().numpy().transpose()) for i in range(trial)],np.nanmean(np.asarray([np.squeeze(model(Xtensors)[0].detach().numpy().transpose()) for i in range(trial)]),axis=0),model_outweights(model)
    
def get_meanr2(X=None,y=None):
    return r2_score(y,X)
def get_meanrmse(X=None,y=None):
    return np.sqrt(mean_squared_error(y,X))
def get_meanmae(X=None,y=None):
    return mean_absolute_error(y,X)

def get_performances_vae(datasets=None,datasets_notensor=None,modelpath='./store/dropout/',suffix='_*',numsplits=33,
                     droprate=None,metric='r2',trailnums=50,withspread=False,output_type='Drop1_2',seed=42):
    setup_seed(seed)
    trailnums = trailnums
    allstoredicts_drop1_2 = []
    for splitnum in (range(numsplits)):
        y_totrain = datasets_notensor[splitnum]['train'][1]
        y_tovalid = datasets_notensor[splitnum]['valid'][1]
        y_totest = datasets_notensor[splitnum]['test'][1]
        ###################################################################################
        # Convert numpy arrays into tensors
        ###################################################################################
        train_Xtensor = datasets[splitnum]['train'][0]
        train_ytensor = datasets[splitnum]['train'][1]
        val_Xtensor = datasets[splitnum]['valid'][0]
        val_ytensor = datasets[splitnum]['valid'][1]
        test_Xtensor = datasets[splitnum]['test'][0]
        test_ytensor = datasets[splitnum]['test'][1]
        ###################################################################################
        # Models
        ###################################################################################
        models,models_orig = [],[]
        for exp in ['exp1a','exp1b','exp1c','exp1d','exp1e','exp1f','exp1g','exp1h','exp1i']:
            #print(glob.glob(modelpath+str(splitnum)+'/modelstest'+str(splitnum)+'_*_'+str(exp)+'_best.pk')[0])
            if suffix=='_*best*':
                #print(glob.glob(modelpath+str(splitnum)+'/modelstest'+str(splitnum)+'_*_'+str(exp)+'_best.pk')[0])
                models.append(torch.load(glob.glob(modelpath+str(splitnum)+'/modelstest'+str(splitnum)+'_*_'+str(exp)+'_best.pk')[0]))
                models_orig.append(torch.load(glob.glob(modelpath+str(splitnum)+'/modelstest'+str(splitnum)+'_*_'+str(exp)+'.pk')[0]))
            else:
                models.append(torch.load(glob.glob(modelpath+str(splitnum)+'/modelstest'+str(splitnum)+'_*_'+str(exp)+'.pk')[0]))
                models_orig.append(torch.load(glob.glob(modelpath+str(splitnum)+'/modelstest'+str(splitnum)+'_*_'+str(exp)+'.pk')[0]))                
        ###################################################################################
        # Make predictions
        ###################################################################################
        alltrains,meantrains,allvals,meanvals,alltests,meantests,r2trains,r2vals,r2tests,weights = [],[],[],[],[],[],[],[],[],[]
        spreaddicts,truth,modelout = [],[],[]
        for ind,model_dict in enumerate(models):
            try:
                alltrain,meantrain,weight_train = grab_predictions(model_dict[0],train_Xtensor,trailnums,output_type)
                allval,meanval,weight_val = grab_predictions(model_dict[0],val_Xtensor,trailnums,output_type)
                alltest,meantest,weight_test = grab_predictions(model_dict[0],test_Xtensor,trailnums,output_type)
                modelout.append(model_dict[0])
            except:
                orig_model = models_orig[ind][0]
                orig_dict = orig_model.state_dict()
                new_dict = {k: v for k, v in model_dict.items() if k in orig_dict}
                orig_dict.update(new_dict)
                orig_model.load_state_dict(orig_dict)
                modelout.append(orig_model)
                
                alltrain,meantrain,weight_train = grab_predictions(orig_model,train_Xtensor,trailnums,output_type)
                allval,meanval,weight_val = grab_predictions(orig_model,val_Xtensor,trailnums,output_type)
                alltest,meantest,weight_test = grab_predictions(orig_model,test_Xtensor,trailnums,output_type)                
            if metric=='r2':
                meanr2_train = get_meanr2(meantrain,y_totrain)
                meanr2_val = get_meanr2(meanval,y_tovalid)
                meanr2_test = get_meanr2(meantest,y_totest)
            elif metric=='rmse':
                meanr2_train = get_meanrmse(meantrain,y_totrain)
                meanr2_val = get_meanrmse(meanval,y_tovalid)
                meanr2_test = get_meanrmse(meantest,y_totest)
            elif metric=='mae':
                meanr2_train = get_meanmae(meantrain,y_totrain)
                meanr2_val = get_meanmae(meanval,y_tovalid)
                meanr2_test = get_meanmae(meantest,y_totest)
            elif metric=='crps':
                meanr2_train = ps.crps_ensemble(np.asarray(y_totrain).transpose(),np.asarray(alltrain).transpose()).mean()
                meanr2_val = ps.crps_ensemble(np.asarray(y_tovalid).transpose(),np.asarray(allval).transpose()).mean()
                meanr2_test = ps.crps_ensemble(np.asarray(y_totest).transpose(),np.asarray(alltest).transpose()).mean()
            if withspread:
                meanstd_train = np.std(np.asarray(alltrain),axis=0)
                meanstd_val = np.std(np.asarray(allval),axis=0)
                meanstd_test = np.std(np.asarray(alltest),axis=0)
                meanstd_dict = {'train':meanstd_train,'valid':meanstd_val,'test':meanstd_test}
            else:
                meanstd_dict = {'train':None,'valid':None,'test':None}
            alltrains.append(alltrain)
            meantrains.append(meantrain)
            allvals.append(allval)
            meanvals.append(meanval)
            alltests.append(alltest)
            meantests.append(meantest)
            r2trains.append(meanr2_train)
            r2vals.append(meanr2_val)
            r2tests.append(meanr2_test)
            weights.append({'train':weight_train,'val':weight_val,'test':weight_test})
            spreaddicts.append(meanstd_dict)
            truth.append({'train':y_totrain,'valid':y_tovalid,'test':y_totest})
        
        allstoredicts_drop1_2.append({'models':modelout,'alltrains':alltrains,'meantrains':meantrains,'allvals':allvals,'meanvals':meanvals,'alltests':alltests,'meantests':meantests,\
                                      'r2trains':r2trains,'r2vals':r2vals,'r2tests':r2tests,'weights':weights,'spreads':spreaddicts,'truth':truth})
    return allstoredicts_drop1_2


def get_performances_retrain(datasets=None,datasets_notensor=None,modelpath='./store/dropout/',suffix='_*',numsplits=33,
                     droprate=None,metric='r2',trailnums=50,withspread=False,output_type='Drop1_2'):
    setup_seed(42)
    trailnums = trailnums
    allstoredicts_drop1_2 = []
    for splitnum in (range(numsplits)):
        y_totrain = datasets_notensor[splitnum]['train'][1]
        y_tovalid = datasets_notensor[splitnum]['valid'][1]
        y_totest = datasets_notensor[splitnum]['test'][1]
        ###################################################################################
        # Convert numpy arrays into tensors
        ###################################################################################
        train_Xtensor = datasets[splitnum]['train'][0]
        train_ytensor = datasets[splitnum]['train'][1]
        val_Xtensor = datasets[splitnum]['valid'][0]
        val_ytensor = datasets[splitnum]['valid'][1]
        test_Xtensor = datasets[splitnum]['test'][0]
        test_ytensor = datasets[splitnum]['test'][1]
        ###################################################################################
        # Models
        ###################################################################################
        models,models_orig = [],[]
        for exp in ['exp1a','exp1b','exp1c','exp1d','exp1e','exp1f','exp1g','exp1h','exp1i']:
            models.append(torch.load(glob.glob(modelpath+str(splitnum)+'/'+str(droprate)+'/modelstest'+str(splitnum)+'_*_'+str(exp)+'_best.pt')[0]))
            models_orig.append(torch.load(glob.glob(modelpath+str(splitnum)+'/'+str(droprate)+'/modelstest'+str(splitnum)+'_*_'+str(exp)+'.pt')[0]))
        ###################################################################################
        # Make predictions
        ###################################################################################
        alltrains,meantrains,allvals,meanvals,alltests,meantests,r2trains,r2vals,r2tests,weights = [],[],[],[],[],[],[],[],[],[]
        spreaddicts,truth,modelout = [],[],[]
        for ind,model_dict in enumerate(models):
            try:
                alltrain,meantrain,weight_train = grab_predictions(model[0],train_Xtensor,trailnums,output_type)
                allval,meanval,weight_val = grab_predictions(model[0],val_Xtensor,trailnums,output_type)
                alltest,meantest,weight_test = grab_predictions(model[0],test_Xtensor,trailnums,output_type)
                modelout.append(model[0])
            except:
                orig_model = models_orig[ind][0]
                orig_dict = orig_model.state_dict()
                new_dict = {k: v for k, v in model_dict.items() if k in orig_dict}
                orig_dict.update(new_dict)
                orig_model.load_state_dict(orig_dict)
                modelout.append(orig_model)
                
                alltrain,meantrain,weight_train = grab_predictions(orig_model,train_Xtensor,trailnums,output_type)
                allval,meanval,weight_val = grab_predictions(orig_model,val_Xtensor,trailnums,output_type)
                alltest,meantest,weight_test = grab_predictions(orig_model,test_Xtensor,trailnums,output_type)                
            if metric=='r2':
                meanr2_train = get_meanr2(meantrain,y_totrain)
                meanr2_val = get_meanr2(meanval,y_tovalid)
                meanr2_test = get_meanr2(meantest,y_totest)
            elif metric=='rmse':
                meanr2_train = get_meanrmse(meantrain,y_totrain)
                meanr2_val = get_meanrmse(meanval,y_tovalid)
                meanr2_test = get_meanrmse(meantest,y_totest)
            elif metric=='mae':
                meanr2_train = get_meanmae(meantrain,y_totrain)
                meanr2_val = get_meanmae(meanval,y_tovalid)
                meanr2_test = get_meanmae(meantest,y_totest)
            elif metric=='crps':
                meanr2_train = ps.crps_ensemble(np.asarray(y_totrain).transpose(),np.asarray(alltrain).transpose()).mean()
                meanr2_val = ps.crps_ensemble(np.asarray(y_tovalid).transpose(),np.asarray(allval).transpose()).mean()
                meanr2_test = ps.crps_ensemble(np.asarray(y_totest).transpose(),np.asarray(alltest).transpose()).mean()
            if withspread:
                meanstd_train = np.std(np.asarray(alltrain),axis=0)
                meanstd_val = np.std(np.asarray(allval),axis=0)
                meanstd_test = np.std(np.asarray(alltest),axis=0)
                meanstd_dict = {'train':meanstd_train,'valid':meanstd_val,'test':meanstd_test}
            else:
                meanstd_dict = {'train':None,'valid':None,'test':None}
            alltrains.append(alltrain)
            meantrains.append(meantrain)
            allvals.append(allval)
            meanvals.append(meanval)
            alltests.append(alltest)
            meantests.append(meantest)
            r2trains.append(meanr2_train)
            r2vals.append(meanr2_val)
            r2tests.append(meanr2_test)
            weights.append({'train':weight_train,'val':weight_val,'test':weight_test})
            spreaddicts.append(meanstd_dict)
            truth.append({'train':y_totrain,'valid':y_tovalid,'test':y_totest})
        
        allstoredicts_drop1_2.append({'models':modelout,'alltrains':alltrains,'meantrains':meantrains,'allvals':allvals,'meanvals':meanvals,'alltests':alltests,'meantests':meantests,\
                                      'r2trains':r2trains,'r2vals':r2vals,'r2tests':r2tests,'weights':weights,'spreads':spreaddicts,'truth':truth})
    return allstoredicts_drop1_2
        

def get_performances_ensemble(datasets=None,datasets_notensor=None,modelpath='./store/dropout/',numsplits=33,
                     droprate=None,metric='r2',trailnums=50,withspread=False,output_type='Drop1_2'):
    setup_seed(42)
    trailnums = trailnums
    allstoredicts_drop1_2 = []
    for splitnum in tqdm(range(numsplits)):
        y_totrain = datasets_notensor[splitnum]['train'][1]
        y_tovalid = datasets_notensor[splitnum]['valid'][1]
        y_totest = datasets_notensor[splitnum]['test'][1]
        ###################################################################################
        # Convert numpy arrays into tensors
        ###################################################################################
        train_Xtensor = datasets[splitnum]['train'][0]
        train_ytensor = datasets[splitnum]['train'][1]
        val_Xtensor = datasets[splitnum]['valid'][0]
        val_ytensor = datasets[splitnum]['valid'][1]
        test_Xtensor = datasets[splitnum]['test'][0]
        test_ytensor = datasets[splitnum]['test'][1]
        ###################################################################################
        # Models
        ###################################################################################
        models = [torch.load(obj) for obj in glob.glob(modelpath+str(splitnum)+'/models'+str(splitnum)+'_*')]
        #[torch.load(obj) for obj in glob.glob('../tmp/deterministic/lwsw2/'+str(splitnum)+'/models'+str(splitnum)+'_*')]
        #models = glob.glob('../tmp/torch_try/ts/lwsw_drop/0/'+str(droprate)+'/modelstest'+str(splitnum)+'_*')]
        ###################################################################################
        # Make predictions
        ###################################################################################
        alltrains,meantrains,allvals,meanvals,alltests,meantests,r2trains,r2vals,r2tests,weights = [],[],[],[],[],[],[],[],[],[]
        spreaddicts,truth = [],[]
        for model in models:
            alltrain,meantrain,weight_train = grab_predictions(model[0],train_Xtensor,trailnums,output_type)
            allval,meanval,weight_val = grab_predictions(model[0],val_Xtensor,trailnums,output_type)
            alltest,meantest,weight_test = grab_predictions(model[0],test_Xtensor,trailnums,output_type)
            if metric=='r2':
                meanr2_train = get_meanr2(meantrain,y_totrain)
                meanr2_val = get_meanr2(meanval,y_tovalid)
                meanr2_test = get_meanr2(meantest,y_totest)
            elif metric=='rmse':
                meanr2_train = get_meanrmse(meantrain,y_totrain)
                meanr2_val = get_meanrmse(meanval,y_tovalid)
                meanr2_test = get_meanrmse(meantest,y_totest)
            elif metric=='mae':
                meanr2_train = get_meanmae(meantrain,y_totrain)
                meanr2_val = get_meanmae(meanval,y_tovalid)
                meanr2_test = get_meanmae(meantest,y_totest)
            elif metric=='crps':
                meanr2_train = ps.crps_ensemble(np.asarray(y_totrain).transpose(),np.asarray(alltrain).transpose()).mean()
                meanr2_val = ps.crps_ensemble(np.asarray(y_tovalid).transpose(),np.asarray(allval).transpose()).mean()
                meanr2_test = ps.crps_ensemble(np.asarray(y_totest).transpose(),np.asarray(alltest).transpose()).mean()
            if withspread:
                meanstd_train = np.std(np.asarray(alltrain),axis=0)
                meanstd_val = np.std(np.asarray(allval),axis=0)
                meanstd_test = np.std(np.asarray(alltest),axis=0)
                meanstd_dict = {'train':meanstd_train,'valid':meanstd_val,'test':meanstd_test}
            else:
                meanstd_dict = {'train':None,'valid':None,'test':None}
            alltrains.append(alltrain)
            meantrains.append(meantrain)
            allvals.append(allval)
            meanvals.append(meanval)
            alltests.append(alltest)
            meantests.append(meantest)
            r2trains.append(meanr2_train)
            r2vals.append(meanr2_val)
            r2tests.append(meanr2_test)
            weights.append({'train':weight_train,'val':weight_val,'test':weight_test})
            spreaddicts.append(meanstd_dict)
            truth.append({'train':y_totrain,'valid':y_tovalid,'test':y_totest})
        
        allstoredicts_drop1_2.append({'models':models,'alltrains':alltrains,'meantrains':meantrains,'allvals':allvals,'meanvals':meanvals,'alltests':alltests,'meantests':meantests,'r2trains':r2trains,'r2vals':r2vals,'r2tests':r2tests,'weights':weights,'spreads':spreaddicts,'truth':truth})
    return allstoredicts_drop1_2

class analysis_patterns:
    def __init__(self,bestdropout_index=7,dropout_rates=None):
        self.bestdropout_index=bestdropout_index
        self.dropout_rates=dropout_rates
        
    def _get_best_model(self,crps_performance=None,numsplit=40):
        performances = [(([obj[i]['r2vals'] for i in range(numsplit)])) for obj in list(crps_performance.values())][self.bestdropout_index]
        storems = {}
        for i in range(len(performances[0])):
            storems[i] = ([performances[j][i] for j in range(len(performances))])
            
        best_dropout = self.dropout_rates[self.bestdropout_index]
        modelnum = np.asarray([np.mean(np.asarray(storems[key])) for key in storems.keys()]).argmin()
        valsplit = np.asarray(storems[modelnum]).argmin()
        try:
            bestmodel = crps_performance[best_dropout][int(valsplit)]['models'][int(modelnum)][0]
        except:
            bestmodel = crps_performance[best_dropout][int(valsplit)]['models'][int(modelnum)]
        return bestmodel,best_dropout,modelnum,valsplit,storems
    
    def new_structure(self,X_totrain=None,bestmodel=None,TYPE='LW',varINDX=[-40,-20]):
        LW,SW = (np.asarray(X_totrain)[:,varINDX[0]:varINDX[1]]),(np.asarray(X_totrain)[:,varINDX[1]:])
        store = []
        for i in range(np.abs(varINDX[1])):
            if TYPE=='LW':
                term1 = model_outweights_all(bestmodel)[0][0][0].detach().numpy()[i]#benchmark.model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[0]*benchmark.model_outweights_all(bestmodel)[0][0][0].detach().numpy()[i]
                sumss = np.sqrt(np.sum((model_outweights_all(bestmodel)[0][0][0].detach().numpy()/np.std(LW,axis=0))**2))
                term2 = (np.std(LW[:,i])*sumss)
                store.append(np.sign(model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[0])*term1/term2)
            elif TYPE=='SW':
                term1 = model_outweights_all(bestmodel)[0][2][0].detach().numpy()[i]#benchmark.model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[1]*benchmark.model_outweights_all(bestmodel)[0][2][0].detach().numpy()[i]
                sumss = np.sqrt(np.sum((model_outweights_all(bestmodel)[0][2][0].detach().numpy()/np.std(SW,axis=0))**2))
                term2 = (np.std(SW[:,i])*sumss)
                store.append(np.sign(model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[1])*term1/term2)
        return store
    
    def new_structure_vae(self,X_totrain=None,bestmodel=None,TYPE='LW',varINDX=[-40,-20]):
        LW,SW = (np.asarray(X_totrain)[:,varINDX[0]:varINDX[1]]),(np.asarray(X_totrain)[:,varINDX[1]:])
        store = []
        for i in range(np.abs(varINDX[1])):
            if TYPE=='LW':
                term1 = model_outweights_all(bestmodel)[0][0][0].detach().numpy()[i]#benchmark.model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[0]*benchmark.model_outweights_all(bestmodel)[0][0][0].detach().numpy()[i]
                sumss = np.sqrt(np.sum((model_outweights_all(bestmodel)[0][0][0].detach().numpy()/np.std(LW,axis=0))**2))
                term2 = (np.std(LW[:,i])*sumss)
                store.append(np.sign(model_outweights_all(bestmodel)[0][-4][0].detach().numpy()[0])*term1/term2)
            elif TYPE=='SW':
                term1 = model_outweights_all(bestmodel)[0][4][0].detach().numpy()[i]#benchmark.model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[1]*benchmark.model_outweights_all(bestmodel)[0][2][0].detach().numpy()[i]
                sumss = np.sqrt(np.sum((model_outweights_all(bestmodel)[0][4][0].detach().numpy()/np.std(SW,axis=0))**2))
                term2 = (np.std(SW[:,i])*sumss)
                store.append(np.sign(model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[0])*term1/term2)
            elif TYPE=='LW_logvar':
                term1 = model_outweights_all(bestmodel)[0][2][0].detach().numpy()[i]#benchmark.model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[0]*benchmark.model_outweights_all(bestmodel)[0][0][0].detach().numpy()[i]
                sumss = np.sqrt(np.sum((model_outweights_all(bestmodel)[0][2][0].detach().numpy()/np.std(LW,axis=0))**2))
                term2 = (np.std(LW[:,i])*sumss)
                store.append(term1/term2)
                #store.append(np.sign(model_outweights_all(bestmodel)[0][-4][0].detach().numpy()[0])*term1/term2)   
            elif TYPE=='SW_logvar':
                term1 = model_outweights_all(bestmodel)[0][6][0].detach().numpy()[i]#benchmark.model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[0]*benchmark.model_outweights_all(bestmodel)[0][0][0].detach().numpy()[i]
                sumss = np.sqrt(np.sum((model_outweights_all(bestmodel)[0][6][0].detach().numpy()/np.std(SW,axis=0))**2))
                term2 = (np.std(SW[:,i])*sumss)
                store.append(term1/term2)
                #store.append(np.sign(model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[0])*term1/term2)    
        return store
    
    def new_coeff(self,X_totrain=None,bestmodel=None,TYPE='LW',varINDX=[-40,-20]):
        LW,SW = (np.asarray(X_totrain)[:,varINDX[0]:varINDX[1]]),(np.asarray(X_totrain)[:,varINDX[1]:])
        if TYPE=='LW':
            sumss = np.sqrt(np.sum((model_outweights_all(bestmodel)[0][0][0].detach().numpy()/np.std(LW,axis=0))**2))
            store = (np.abs(model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[0])*sumss)
        elif TYPE=='SW':
            sumss = np.sqrt(np.sum((model_outweights_all(bestmodel)[0][2][0].detach().numpy()/np.std(SW,axis=0))**2))
            store = (np.abs(model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[1])*sumss)
        return store
    
    def new_coeff_vae(self,X_totrain=None,bestmodel=None,TYPE='LW',varINDX=[-40,-20]):
        LW,SW = (np.asarray(X_totrain)[:,varINDX[0]:varINDX[1]]),(np.asarray(X_totrain)[:,varINDX[1]:])
        if TYPE=='LW':
            sumss = np.sqrt(np.sum((model_outweights_all(bestmodel)[0][0][0].detach().numpy()/np.std(LW,axis=0))**2))
            store = (np.abs(model_outweights_all(bestmodel)[0][-4][0].detach().numpy()[0])*sumss)
        elif TYPE=='SW':
            sumss = np.sqrt(np.sum((model_outweights_all(bestmodel)[0][4][0].detach().numpy()/np.std(SW,axis=0))**2))
            store = (np.abs(model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[0])*sumss)
        elif TYPE=='LW_logvar':
            sumss = np.sqrt(np.sum((model_outweights_all(bestmodel)[0][2][0].detach().numpy()/np.std(LW,axis=0))**2))
            store = (np.abs(model_outweights_all(bestmodel)[0][-4][0].detach().numpy()[0])*sumss)
        elif TYPE=='SW_logvar':
            sumss = np.sqrt(np.sum((model_outweights_all(bestmodel)[0][6][0].detach().numpy()/np.std(SW,axis=0))**2))
            store = (np.abs(model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[0])*sumss)
        return store
    
    def new_b(self,X_totrain=None,bestmodel=None,varINDX=[-40,-20]):
        LW,SW = (np.asarray(X_totrain)[:,varINDX[0]:varINDX[1]]),(np.asarray(X_totrain)[:,varINDX[1]:])
        store = []
        b2 = float(model_outweights_all(bestmodel)[0][-1][0].detach().numpy())
        b1lw = float(model_outweights_all(bestmodel)[0][1][0].detach().numpy())
        a2lw = model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[0]
        b1sw = float(model_outweights_all(bestmodel)[0][3][0].detach().numpy())
        a2sw = model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[1]
                
        sumss_lw = np.sum((model_outweights_all(bestmodel)[0][0][0].detach().numpy()/np.std(LW,axis=0)*np.mean(LW,axis=0)))
        sumss_sw = np.sum((model_outweights_all(bestmodel)[0][2][0].detach().numpy()/np.std(SW,axis=0)*np.mean(SW,axis=0)))
            
        b = b2 + a2lw * (b1lw - sumss_lw) + a2sw * (b1sw - sumss_sw)
        return b
    
    def new_b_vae(self,X_totrain=None,bestmodel=None,varINDX=[-40,-20]):
        LW,SW = (np.asarray(X_totrain)[:,varINDX[0]:varINDX[1]]),(np.asarray(X_totrain)[:,varINDX[1]:])
        store = []
        b2 = float(model_outweights_all(bestmodel)[0][-1][0].detach().numpy())+float(model_outweights_all(bestmodel)[0][-3][0].detach().numpy())
        b1lw = float(model_outweights_all(bestmodel)[0][1][0].detach().numpy())
        a2lw = model_outweights_all(bestmodel)[0][-4][0].detach().numpy()[0]
        b1sw = float(model_outweights_all(bestmodel)[0][5][0].detach().numpy())
        a2sw = model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[0]
                
        sumss_lw = np.sum((model_outweights_all(bestmodel)[0][0][0].detach().numpy()/np.std(LW,axis=0)*np.mean(LW,axis=0)))
        sumss_sw = np.sum((model_outweights_all(bestmodel)[0][2][0].detach().numpy()/np.std(SW,axis=0)*np.mean(SW,axis=0)))
            
        b = b2 + a2lw * (b1lw - sumss_lw) + a2sw * (b1sw - sumss_sw)
        return b
    
def get_spread_skill_score_model(seeds_store=None,seeds=None,snum=None,coeff=None,splitno=None,modelno=None,category='valid',meancat='meanvals'):
    aaaaaa = [np.percentile(seeds_store[seeds[snum]][coeff][splitno]['spreads'][modelno][category],i) for i in np.arange(0,95,5)]
    
    hihihi = []
    hihihib = []

    ssRel = 0
    nPts = np.asarray(seeds_store[seeds[snum]][coeff][splitno]['truth'][modelno][category]).size
    for i in range(len(aaaaaa)-1):
        ytruth = np.asarray(seeds_store[seeds[snum]][coeff][splitno]['truth'][modelno][category])[np.where(np.logical_and(seeds_store[seeds[snum]][coeff][splitno]['spreads'][modelno][category]>aaaaaa[i],
                                                                                                                         seeds_store[seeds[snum]][coeff][splitno]['spreads'][modelno][category]<aaaaaa[i+1]))]
        nPtsbin = ytruth.size
        ypred = np.asarray(seeds_store[seeds[snum]][coeff][splitno][meancat][modelno])[np.where(np.logical_and(seeds_store[seeds[snum]][coeff][splitno]['spreads'][modelno][category]>aaaaaa[i],
                                                                                                                  seeds_store[seeds[snum]][coeff][splitno]['spreads'][modelno][category]<aaaaaa[i+1]))]
        spreadss = np.asarray(seeds_store[seeds[snum]][coeff][splitno]['spreads'][modelno][category])[np.where(np.logical_and(seeds_store[seeds[snum]][coeff][splitno]['spreads'][modelno][category]>aaaaaa[i],
                                                                                                                             seeds_store[seeds[snum]][coeff][splitno]['spreads'][modelno][category]<aaaaaa[i+1]))] 
        try:
            hihihi.append(np.sqrt(mean_squared_error(ytruth,ypred)))
            ssRel += (nPtsbin/nPts) * np.abs(np.sqrt(mean_squared_error(ytruth,ypred)) - np.mean(spreadss))
        except:
            hihihi.append(np.nan)
            ssRel += 0
    return {'spreadlim':aaaaaa,'error':hihihi,'ssRel':ssRel}