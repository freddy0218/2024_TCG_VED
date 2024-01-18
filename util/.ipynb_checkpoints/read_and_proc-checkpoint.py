import xarray as xr
import numpy as np
import concurrent.futures
import json,pickle,marshal
import glob

def flatten(l):
    return [item for sublist in l for item in sublist]

def read_some_azimuth_fields(fileloc=None,fieldname=None):
    dict_name = {}
    for inx,obj in enumerate(fileloc):
        field_read = xr.open_dataset(obj)
        dict_name[fieldname[inx]] = field_read
    return dict_name

def add_ctrl_before_senstart(CTRLvar=None,SENvar=None,exp='ncrf_36h',firstdo='Yes'):
    if firstdo=='Yes':
        if (exp=='ncrf_36h') or (exp=='lwcrf'):
            return np.concatenate((CTRLvar[0:36],SENvar))
        elif exp=='ncrf_60h':
            return np.concatenate((CTRLvar[0:60],SENvar))
        elif exp=='ncrf_96h':
            return np.concatenate((CTRLvar[0:96],SENvar))
        else:
            return SENvar
            
def add_ctrl_before_senstart_ctrlbase(CTRLvar=None,SENvar=None,exp='ncrf_36h',firstdo='Yes'):
    if firstdo=='Yes':
        if (exp=='ncrf_36h') or (exp=='lwcrf'):
            return np.concatenate((CTRLvar[0:37],SENvar[1:]))
        elif exp=='ncrf_60h':
            return np.concatenate((CTRLvar[0:61],SENvar[1:]))
        elif exp=='ncrf_96h':
            return np.concatenate((CTRLvar[0:97],SENvar[1:]))
    else:
        return SENvar
            
def flatten(t):
    #https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
    return [item for sublist in t for item in sublist]

#######################################################################################
# Save files
#######################################################################################
def save_to_pickle(loc=None,var=None,TYPE='PICKLE'):
    if TYPE=='PICKLE':
        with open(loc,"wb") as f:
            pickle.dump(var,f)
        return None
    elif TYPE=='JSON':
        #dumpedvar = json.dumps(var, cls=NumpyEncoder)
        with open(loc,"wb") as f:
            json.dump(var.tolist(),f)
        return None
    elif TYPE=='MARSHAL':
        with open(loc,"wb") as f:
            marshal.dump(var.tolist(),f)
        return None 

def depickle(fileloc=None):
    output = []
    with open(fileloc,'rb') as f:
        output.append(pickle.load(f))    
    return output[0]

#######################################################################################
# Polar to cartesian
#######################################################################################

import scipy
def azimuth2angle(azimuth=None):
    """
    https://math.stackexchange.com/questions/926226/conversion-from-azimuth-to-counterclockwise-angle
    """
    angletest = 450-azimuth
    for index,item in enumerate(angletest):
        if item>360:
            angletest[index] = item-360
        else:
            continue
    return angletest

def closest_index(array=None,target=None):
    return np.abs(array-target).argmin()

def polar2cartesian(outcoords, inputshape, origin):
    """Coordinate transform for converting a polar array to Cartesian coordinates. 
    inputshape is a tuple containing the shape of the polar array. origin is a
    tuple containing the x and y indices of where the origin should be in the
    output array."""
    
    xindex, yindex = outcoords
    x0, y0 = origin
    x = xindex - x0
    y = yindex - y0
    
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta_index = np.round((theta + np.pi) * inputshape[1] / (2 * np.pi))
    return (r,theta_index)

def proc_tocart(polarfield=None,angle=None,twoD=True,standard=False):
    if twoD==True:
        PWnew = [np.asarray(polarfield)[int(np.abs(angle-360).argmin()),:]]
        for i in np.linspace(0,358,359):
            PWnew.append(np.asarray(polarfield)[int(np.abs(angle-i).argmin()),:])
        PWnew = np.swapaxes(np.asarray(PWnew),0,1)
        del i
        
        if standard==True:
            PWnew = (PWnew-np.nanmean(PWnew))/np.nanstd(PWnew)
        else:
            PWnew=PWnew
        test_2cartesian = scipy.ndimage.geometric_transform(PWnew,polar2cartesian,order=0,mode='constant',output_shape =(PWnew.shape[0]*2,PWnew.shape[0]*2),\
                                                            extra_keywords = {'inputshape':PWnew.shape,'origin':(PWnew.shape[0],PWnew.shape[0])})
        return ((test_2cartesian))

def proc_radial(polarfield=None,angle=None,twoD=True,standard=False):
    if twoD==True:
        PWnew = [np.asarray(polarfield)[int(np.abs(angle-360).argmin()),:]]
        for i in np.linspace(0,358,359):
            PWnew.append(np.asarray(polarfield)[int(np.abs(angle-i).argmin()),:])
        PWnew = np.swapaxes(np.asarray(PWnew),0,1)
        del i
        return PWnew

def proc_tocartUV(polarfield=None,angle=None,twoD=True,standard=False):
    if twoD==True:
        PWnew=polarfield.copy()
        test_2cartesian = scipy.ndimage.geometric_transform(PWnew,polar2cartesian,order=0,mode='constant',output_shape =(PWnew.shape[0]*2,PWnew.shape[0]*2),\
                                                            extra_keywords = {'inputshape':PWnew.shape,'origin':(PWnew.shape[0],PWnew.shape[0])})
        #print('Finish processing')
        return ((test_2cartesian))

#######################################################################################
# Processed time series
#######################################################################################

def _get_exp_name(folderpath=None,splitnum=None,folder=2,TYPE='varimax'):
    if TYPE=='varimax':
        return sorted(glob.glob(folderpath+'varimaxpca/X/random/Xtrain*'))[splitnum][:-7].split('/')[-1][6:],sorted(glob.glob(folderpath+'varimaxpca/X/random/Xtrain*'))[splitnum][:-7].split('/')[-1][6:].split('_')
    elif TYPE=='orig':
        return sorted(glob.glob(folderpath+'pca/X/random/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:],sorted(glob.glob(folderpath+'pca/X/random/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:].split('_')
    elif TYPE=='keras':
        return sorted(glob.glob(folderpath+'keras/X/random/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:],sorted(glob.glob(folderpath+'keras/X/random/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:].split('_')
    elif TYPE=='fixTEST':
        return sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:],sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:].split('_')
        

def real_random(folderpath=None,index=None,folder=2,TYPE=None,yfolder=None):
    toextract = _get_exp_name(folderpath,index,folder,TYPE)[0]
    # X
    if TYPE=='varimax':
        Xtestpath,Xtrainpath,Xvalidpath = sorted(glob.glob(folderpath+'varimaxpca/X/random/*'+str(toextract)+'*'))
        yallpath = sorted(glob.glob(folderpath+'varimaxpca/y/random/*'+str(toextract)+'*'))
    elif TYPE=='orig':
        Xtestpath,Xtrainpath,Xvalidpath = sorted(glob.glob(folderpath+'pca/X/random/'+str(folder)+'/*'+str(toextract)+'*'))
        yallpath = sorted(glob.glob(folderpath+'pca/y/random/'+str(folder)+'/*'+str(toextract)+'*'))
    elif TYPE=='keras':
        Xtestpath,Xtrainpath,Xvalidpath = sorted(glob.glob(folderpath+'keras/X/random/'+str(folder)+'/*'+str(toextract)+'*'))
        yallpath = sorted(glob.glob(folderpath+'keras/y/random/'+str(yfolder)+'/*'+str(toextract)+'*'))
    elif TYPE=='fixTEST':
        Xtrainpath = sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/Xtrain'+str(toextract)+'*'))
        Xvalidpath = sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/Xvalid'+str(toextract)+'*'))
        Xtestpath = sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/Xtest'+str(toextract)+'*'))
        #Xtestpath,Xtrainpath,Xvalidpath = sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/*'+str(toextract)+'*'))
        yallpath = sorted(glob.glob(folderpath+'keras/ynew/'+str(yfolder)+'/allY'+str(toextract)+'*'))
    
    Xtest,Xtrain,Xvalid = [depickle(obj) for obj in [Xtestpath[0],Xtrainpath[0],Xvalidpath[0]]]
    yall = depickle(yallpath[0])
    return Xtest,Xtrain,Xvalid,yall

def real_random_y(folderpath=None,index=None,folder=2,TYPE=None,yfolder=None):
    toextract = _get_exp_name(folderpath,index,folder,TYPE)[0]
    # X
    if TYPE=='orig':
        yallpath = sorted(glob.glob(folderpath+'pca/y/random/'+str(folder)+'/*'+str(toextract)+'*'))
    elif TYPE=='keras':
        yallpath = sorted(glob.glob(folderpath+'keras/y/random/'+str(yfolder)+'/*'+str(toextract)+'*'))
    elif TYPE=='fixTEST':
        yallpath = sorted(glob.glob(folderpath+'keras/ynew/'+str(yfolder)+'/tsY'+str(toextract)+'*'))
    yall = depickle(yallpath[0])
    return yall

def delete_padding(inTS=None,outTS=None):
    output_nozero,input_nozero = [],[]
    if len(outTS.shape)>1:
        for i in range(len(outTS[:,0])):
            temp = outTS[i,:]
            tempin = inTS[i,:]
            if temp.all()==0:
                continue
            else:
                output_nozero.append(temp)
                input_nozero.append(tempin)
        return input_nozero,output_nozero
    else:
        for i in range(len(outTS[:])):
            temp = outTS[i]
            tempin = inTS[i,:]
            if temp.all()==0:
                continue
            else:
                output_nozero.append(temp)
                input_nozero.append(tempin)
        return input_nozero,output_nozero 
    
def train_valid_test(expvarlist=None,validindex=None,testindex=None,concat='Yes'):
    X_valid, X_test = [expvarlist[i] for i in validindex], [expvarlist[i] for i in testindex]
    X_traint = expvarlist.copy()
    popindex = validindex+testindex
    X_train = [X_traint[i] for i in range(len(X_traint)) if i not in popindex]
    #assert len(X_train)==16, 'wrong train-valid-test separation!'
    if concat=='Yes':
        return np.concatenate([X_train[i] for i in range(len(X_train))],axis=0), np.concatenate([X_valid[i] for i in range(len(X_valid))],axis=0), np.concatenate([X_test[i] for i in range(len(X_test))],axis=0)
    else:
        return X_train, X_valid, X_test