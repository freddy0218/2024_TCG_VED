from metpy.calc import pressure_to_height_std
from metpy.units import units
from metpy.calc import potential_temperature,relative_humidity_from_mixing_ratio
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED/util/')
import read_and_proc

##############################################################################################################
# Numerical
##############################################################################################################
def backward_secondorder(arrayin=None,delta=None,axis=None):
    result = []
    if axis==0:
        result.append((arrayin[1]-arrayin[0])/3600)
        for i in range(2,arrayin.shape[axis]):
            temp = (3*arrayin[i,:]-4*arrayin[i-1,:]+arrayin[i-2,:])/(2*delta)
            result.append(temp)
        return np.asarray(result)
    elif axis==1:
        result.append((arrayin[:,1,:]-arrayin[:,0,:])/3600)
        for i in range(2,arrayin.shape[axis]):
            temp = (3*arrayin[:,i,:]-4*arrayin[:,i-1,:]+arrayin[:,i-2,:])/(2*delta)
            result.append(temp)
        return np.asarray(result)   
    elif axis==2:
        result.append((arrayin[:,:,1,:]-arrayin[:,:,0,:])/3600)
        for i in range(2,arrayin.shape[axis]):
            temp = (3*arrayin[:,:,i,:]-4*arrayin[:,:,i-1,:]+arrayin[:,:,i-2,:])/(2*delta)
            result.append(temp)
        return np.asarray(result)
    elif axis==3:
        result.append((arrayin[:,:,:,1]-arrayin[:,:,:,0])/3600)
        for i in range(2,arrayin.shape[axis]):
            temp = (3*arrayin[:,:,:,i]-4*arrayin[:,:,:,i-1]+arrayin[:,:,:,i-2])/(2*delta)
            result.append(temp)
        return np.asarray(result)

def forward_diff(arrayin=None,delta=None,axis=None,LT=1):
    result = []
    if axis==0:
        for i in range(0,arrayin.shape[axis]-LT):
            temp = (arrayin[i+LT,:]-arrayin[i,:])/(LT*delta)
            result.append(temp)
            return np.asarray(result)
    
def nearest_index(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx.values
    
##############################################################################################################
# Theta
##############################################################################################################
def do_theta(var=None,presaxis=None): #ctrlvar_dict['T']['T']
    """
    var: Temperature in Kelvin
    presaxis: 1D pressure axis in hPa
    """
    theta_out = []
    try:
        pres3d = np.tile(presaxis.data[:, np.newaxis, np.newaxis], (1,var[0,0].shape[0],var[0,0].shape[1]))
    except:
        pres3d = np.tile(np.asarray(presaxis)[:, np.newaxis, np.newaxis], (1,var[0,0].shape[0],var[0,0].shape[1]))            
    for i in (range(len(presaxis))):
        temppres = pres3d[i,:,:]
        temp_T = var[:,i,:,:].data
        theta = np.asarray([potential_temperature(units.Quantity(temppres,'hPa'),units.Quantity(temp_T[indexin,:,:],'K')) for indexin in range(temp_T.shape[0])])
        theta_out.append(theta)
    return theta_out
    
def output_dtheta(uvtpath=None,originpath=None,expname=None,sigma=[3,0,0,0],addsenmethod='new'):
    theta = gaussian_filter(np.swapaxes(np.asarray(read_and_proc.depickle(uvtpath+str(expname)+'/theta')),0,1),sigma=sigma)
    ###############################################################################################
    # dtheta
    ###############################################################################################
    if expname=='ctl':
        dtheta = forward_diff(gaussian_filter(theta,sigma=sigma),60*60,0)
    elif (expname=='ncrf_36h') or (expname=='lwcrf'):
        ctrl_thetaA = gaussian_filter(np.swapaxes(np.asarray(read_and_proc.depickle(uvtpath+'/ctl/theta')),0,1),sigma=sigma)
        if addsenmethod=='orig':
            thetaL = read_and_proc.add_ctrl_before_senstart(ctrl_thetaA,theta,'NCRF36','Yes')
        elif addsenmethod=='new':
            thetaL = read_and_proc.add_ctrl_before_senstart_ctrlbase(ctrl_thetaA,theta,'NCRF36','Yes')
        dtheta = forward_diff(thetaL,60*60,0)
        del ctrl_thetaA,thetaL
        gc.collect()
    elif (expname=='ncrf_60h'):
        ctrl_thetaA = gaussian_filter(np.swapaxes(np.asarray(read_and_proc.depickle(uvtpath+'/ctl/theta')),0,1),sigma=sigma)
        if addsenmethod=='orig':
            thetaL = read_and_proc.add_ctrl_before_senstart(ctrl_thetaA,theta,'NCRF60','Yes')
        elif addsenmethod=='new':
            thetaL = read_and_proc.add_ctrl_before_senstart_ctrlbase(ctrl_thetaA,theta,'NCRF60','Yes')
        dtheta = forward_diff(thetaL,60*60,0)
        del ctrl_thetaA,thetaL
        gc.collect()
    del theta
    gc.collect()
    return dtheta

##############################################################################################################
# urad/vtan
##############################################################################################################
def ruppert_vtmax_calc(dataU=None,dataV=None,outer_limit=None,azimuth_in=None,simulation=None,r500=None):
    d2r = np.pi/180
    Uradout,Vtanout,Vtanmaxout = [],[],[]
    if len(dataU.shape)==4:
        try:  az = dataU[:,0,:,:].azmiuth.values
        except: az=azimuth_in
        
        azt = np.moveaxis(np.tile(az,(dataU[:,0,:,:].shape[0],1,outer_limit,1)),-1,-2)[:,0,:,:]
        for heightindices in (range(39)):
            try:
                wdir = (np.arctan(dataV[:,heightindices,:,:outer_limit]/dataU[:,heightindices,:,:outer_limit])/d2r)
            except:
                print('Cannot find wind field arrays!')
            loc_neg = (dataU[:,heightindices,:,:outer_limit]<0)
            wdir[loc_neg] += 180
            wspd = np.sqrt(dataU[:,heightindices,:,:outer_limit]**2+dataV[:,heightindices,:,:outer_limit]**2)
            Urad = wspd*np.cos((wdir-azt)*d2r)
            Vtan = wspd*np.sin((wdir-azt)*d2r)
            Vtan_max = np.max(np.mean(Vtan[:,:,:outer_limit],axis=1),axis=1)
            del loc_neg,wspd,wdir
            ##################################################################
            # Populate Empty Lists
            Uradout.append(Urad)
            Vtanout.append(Vtan)
            Vtanmaxout.append(Vtan_max)
        del Urad,Vtan
        return (Uradout),(Vtanout),Vtanmaxout    
    elif len(dataU.shape)==3:
        try:
            wdir = (np.arctan(dataV[:,:,:outer_limit]/dataU[:,:,:outer_limit])/d2r).values	
        except:
            print('Cannot find wind field arrays!')
        loc_neg = (dataU<0)
        wdir[loc_neg] += 180
        wspd = np.sqrt(dataU[:,heightindices,:,:outer_limit]**2+dataV[:,heightindices,:,:outer_limit]**2).values
        az = dataU.azmiuth.values
        azt = np.moveaxis(np.tile(az,(dataU.shape[0],1,167,1)),-1,-2)[:,0,:,:]
        Urad = wspd*np.cos((wdir-azt)*d2r)
        Vtan = wspd*np.sin((wdir-azt)*d2r)
        Vtan_max = np.max(np.mean(Vtan[:,:,:outer_limit],axis=1),axis=1)		
        return Urad,Vtan,Vtan_max

##############################################################################################################
# Relative Humidity
##############################################################################################################
def do_rh(qv=None,temperature=None,presaxis=None): #ctrlvar_dict['T']['T']
    """
    var: Temperature in Kelvin
    presaxis: 1D pressure axis in hPa
    """
    rh_out = []
    try:
        pres3d = np.tile(presaxis.data[:, np.newaxis, np.newaxis], (1,temperature[0,0].shape[0],temperature[0,0].shape[1]))
    except:
        pres3d = np.tile(np.asarray(presaxis)[:, np.newaxis, np.newaxis], (1,temperature[0,0].shape[0],temperature[0,0].shape[1]))            
    for i in (range(len(presaxis))):
        temppres = pres3d[i,:,:]
        temp_T = temperature[:,i,:,:].data
        temp_qv = qv[:,i,:,:].data
        rh = np.asarray([relative_humidity_from_mixing_ratio(units.Quantity(temppres,'hPa'),units.Quantity(temp_T[indexin,:,:],'K'),\
                                                               units.Quantity(temp_qv[indexin,:,:],'kg/kg')) for indexin in range(temp_T.shape[0])])
        rh_out.append(rh)
    return rh_out
    
from scipy.ndimage import gaussian_filter
import gc
def do_gauss_smooth(var=None,gaussian=[3,0,0]):
    normal_var = []
    for presindex in range(len(var[0,:,0,0])):
        normal_var.append(gaussian_filter((var[:,presindex,:,:]),sigma=gaussian))
    normal_var = np.swapaxes(np.asarray(normal_var),0,1)
    print("Finished!")
    return normal_var

def windrates_real(u=None,v=None,w=None,LT=None):
    # dudt
    dudtT = [forward_diff(uobj,60*60,0,LT) for uobj in u]
    #print([obj.shape[0] for obj in dudtT])
    dudt = np.concatenate((dudtT[0],dudtT[1][(36-23):],dudtT[2][(60-23):],dudtT[3][(36-23):]),axis=0)
    del dudtT
    gc.collect()
    # dvdt
    dvdtT = [forward_diff(vobj,60*60,0,LT) for vobj in v]
    dvdt = np.concatenate((dvdtT[0],dvdtT[1][(36-23):],dvdtT[2][(60-23):],dvdtT[3][(36-23):]),axis=0)
    del dvdtT 
    gc.collect()
    # dwdt
    dwdtT = [forward_diff(wobj,60*60,0,LT) for wobj in w]
    dwdt = np.concatenate((dwdtT[0],dwdtT[1][(36-23):],dwdtT[2][(60-23):],dwdtT[3][(36-23):]),axis=0)
    del dwdtT 
    return dudt,dvdt,dwdt

