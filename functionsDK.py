import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
#matplotlib.use('Agg')
plt.rc("font",**{"size":18})
import collections
import copy
import os.path
import pickle
from collections import defaultdict


####################################################################################################
# Global parameters
####################################################################################################
M_e = gv.gvar('0.5109989461(31)*1e-3')#GeV
M_mu = gv.gvar('105.6583745(24)*1e-3')#GeV
GF = gv.gvar('1.1663787(6)*1e-5') #Gev-2
Metasphys = gv.gvar('0.6885(22)')   # 1303.1670
Metacphys = gv.gvar('2.9766(12)')# gv.gvar('2.98390(50)')  # From Christine not PDG
Metas_C = gv.gvar('0.432855(40)')#fitted from Judd's data 
Metas_VCp = gv.gvar('0.52680(8)')
Metas_VC = gv.gvar('0.54024(15)') # 1408.4169
Metas_Cp = gv.gvar('0.42310(3)')#from 1408.4169 
Metas_Fp = gv.gvar('0.30480(4)')#from 1408.4169  
Metas_F = gv.gvar('0.314015(89)') #from BsEtas fits
Metas_SF = gv.gvar('0.207021(65)')#from new BsEtas fit
Metas_UF = gv.gvar('0.154107(88)') #from new BsEast fit
Metas_Fs = Metas_F#gv.gvar('0.314015(89)') #from BsEtas fits
Metas_SFs = Metas_SF#gv.gvar('0.207021(65)')#from new BsEtas fit
Metas_UFs = Metas_UF#gv.gvar('0.154107(88)') #from new BsEast fit
#MKphys = gv.gvar('0.497611(13)') #PD K^0
MKphys0 = gv.gvar('0.497611(13)') #PD K^0
MKphys = gv.gvar('0.493677(13)')#PD K^pm  we are doing D_0 to K^pm 
MBsphys = gv.gvar('5.36688(14)') # PDG 
MDsphys = gv.gvar('1.968340(70)')  #PDG
MDs0phys = gv.gvar('2.3178(5)')
MDsstarphys = gv.gvar('2.1122(4)')  #PDG 
MBphys = gv.gvar('5.27965(12)') # PDG B0  
#MDphys = gv.gvar('1.86965(5)')  #PDG Dpm
MDphyspm = gv.gvar('1.86965(5)')  #PDG Dpm
MDphys = gv.gvar('1.86483(5)')  #PDG D0 
Mpiphys = gv.gvar('0.1349770(5)')  #PDG
MBsstarphys = gv.gvar('5.4158(15)') #PDG
#tensornorm = gv.gvar('1.09024(56)') # from Dan
w0 = gv.gvar('0.1715(9)')  #fm
hbar = gv.gvar('6.58211928(15)') # x 10^-25 GeV s
clight = 2.99792458 #*10^23 fm/s
slratio = gv.gvar('27.18(10)')
MetacC = gv.gvar('1.876536(48)') #2005.01845
MetacVC = gv.gvar('2.331899(72)') # correct mass 2005.01845
MetacVCp = gv.gvar('2.283452(45)')# from Bp's data 
MetacCp = gv.gvar('1.833947(14)')# from fitting Judd's data '1.833950(18)' 2005.01845   
MetacFp = gv.gvar('1.32929(3)')# adjusted from '1.32929(3)' 1408.4169 for am_h =0.433 can adjust to 1.327173(30) for 0.432
MetacF = gv.gvar('1.364919(40)')  #adjusted from '1.367014(40)'for 0.45 not 0.449 not correct
MetacSF = gv.gvar('0.896675(24)')       #2005.01845 
MetacUF = gv.gvar('0.666818(39)')       #2005.01845
MetacFs = MetacF#gv.gvar('1.367014(40)')        #lattice units
MetacSFs = MetacSF#gv.gvar('0.896806(48)')       #where are these from? 
MetacUFs = MetacUF#gv.gvar('0.666754(39)')       #All from Mclean 1906.00701
deltaFVC = gv.gvar('0.018106911(16)')
deltaFVVC = gv.gvar('0.05841163(17)')
deltaFVVCp = gv.gvar('0.12907825(82)')
deltaFVCp = gv.gvar('0.04894993(12)')
deltaFVFp = gv.gvar('0.06985291(24)')
deltaFVF = gv.gvar('0.020801419(21)')#80812089(1)')
deltaFVSF = gv.gvar('0.020801419(21)')#80812089(1)')
deltaFVUF = gv.gvar('0.027538708(37)')#753275(1)') #from code Chris sent
deltaFVFs = deltaFVF#gv.gvar('0.020801419(21)')#80812089(1)')
deltaFVSFs = deltaFVSF#gv.gvar('0.020801419(21)')#80812089(1)')
deltaFVUFs = deltaFVUF#gv.gvar('0.027538708(37)')#753275(1)') #from code Chris sent
etaEW2 = gv.gvar('1.009(2)')**2
deltaEM2 = gv.gvar('1.00(1)')
cor2 = etaEW2*deltaEM2
#deltaEMDpm2 = gv.gvar('1.000(5)')
tauD0 = gv.gvar('410.1(1.5)e-6')#ns # PDG
tauDpm = gv.gvar('1040(7)e-6')#ns # PDG 1040(7)*1e-15s
B = gv.gvar('0.03999(44)') #HFLAV pg 340
Banorm = B/tauD0
#x =  MBsphys*(MBsstarphys-MBsphys)  #GeV^2 
LQCD = 0.5
mbphys = gv.gvar('4.18(04)') # b mass GeV
qsqmaxphys = (MBsphys-Metasphys)**2
qsqmaxphysBK = (MBphys-MKphys)**2
qsqmaxphysDK = (MDphys-MKphys)**2
#Dels = (MDs0phys-MDphys).mean # 0.4 +0.1 in control too
mlmsfac = 5.63  #gives ml/(mlmsfac*ms) 10 originally, now 5.63
#####################################################################################################
############################### Other data #########################################################
dataf0maxBK = None  #only works for BsEtas for now
datafpmaxBK = None
datafTmaxBK = None
dataf00BK = None
dataf0max1BsEtas = gv.gvar('0.811(17)') # only BsEtas works for now 1510.07446 
dataf0max2BsEtas =  gv.gvar('0.816(35)') #chris 1406.2279
datafpmaxBsEtas =  gv.gvar('2.293(91)') #chris 1406.2279
dataf00BsEtas =  gv.gvar('0.297(47)') #chris 1406.2279
#####################################################################################################

def unmake_gvar_vec(vec):
    #A function which extracts the mean and standard deviation of a list of gvars
    mean = []
    sdev = []
    for element in vec:
        mean.append(element.mean)
        sdev.append(element.sdev)
    return(mean,sdev)

####################################################################################################

def make_upp_low(vec):
    #A function which extracts the upper and lower error bars of a list of gvars
    upp = []
    low = []
    for element in vec:
        upp.append(element.mean + element.sdev)
        low.append(element.mean - element.sdev)
    return(upp,low)

####################################################################################################

def convert_Gev(a):
    #converts lattice spacings from fm to GeV-1
    aGev = gv.gvar(a)/(gv.gvar(hbar)*clight*1e-2) #bar in x10^-25 GeV seconds so hence 1e-2 
    return(aGev)

####################################################################################################
def make_Z_T():
    mean = gv.gvar(['0.9592(42)','0.9840(42)','1.0132(43)','1.0439(43)','1.0585(41)'])
    corr = [[1.0,0.99737,0.99847,0.99431,0.92883],[0.99737,1.0,0.99765,0.99388,0.92941],[0.99847,0.99765,1.0,0.99576,0.93216],[0.99430,0.99388,0.99576,1.0,0.92851],[0.92883,0.92940,0.93216,0.92851,1.0]]
    x = gv.correlate(mean,corr)
    Z_T = gv.BufferDict()
    Z_T['VC'] = x[0]
    Z_T['C'] = x[1]
    Z_T['F'] = x[2] # all from Dan
    Z_T['Fp'] = x[2] # these are sea mass independent
    Z_T['SF'] = x[3]
    Z_T['UF'] = x[4]
    return(Z_T)
    
####################################################################################################

def make_params_BK(Fits,Masses,Twists):
    for Fit in Fits:
        Fit['momenta'] = []
        daughters = []
        Fit['a'] = w0/(hbar*clight*0.01*Fit['w0/a'])
        j = 0
        for i in range(len(Fit['masses'])):
            if i not in Masses[Fit['conf']]:
                del Fit['masses'][i-j]
                del Fit['Zdisc'][i-j]
                j += 1
        j = 0
        for i in range(len(Fit['twists'])):
            if i not in Twists[Fit['conf']]:
                del Fit['daughter-Tag'][i-j]
                del Fit['twists'][i-j]
                j += 1
        for i,twist in enumerate(Fit['twists']):
            Fit['momenta'].append(np.sqrt(3)*np.pi*float(twist)/Fit['L'])
            daughters.append(Fit['daughter-Tag'][i].format(twist))
        Fit['daughter-Tag'] = daughters
    return()    

####################################################################################################

def get_results(Fit,thpts):
    p = gv.load(Fit['filename'],method='pickle')
    if Fit['conf'] in ['Fs','SFs','UFs']:
        pl = gv.load(Fit['Hlfilename'],method='pickle')
    # We should only need goldstone masses and energies here
    Fit['M_parent_m{0}'.format(Fit['m_c'])] = p['dE:{0}'.format(Fit['parent-Tag'].format(Fit['m_s'],Fit['m_c']))][0]
    for mass in Fit['masses']:
        Fit['M_parent_m{0}'.format(mass)] = p['dE:{0}'.format(Fit['parent-Tag'].format(Fit['m_s'],mass))][0]
        NGtag = '{0}G5T-G5T{1}'.format((Fit['parent-Tag'].format(Fit['m_s'],mass)).split('G5-G5')[0],(Fit['parent-Tag'].format(Fit['m_s'],mass)).split('G5-G5')[1])
        Fit['GNGsplit_m{0}'.format(mass)] = p['dE:{0}'.format(NGtag)][0] - p['dE:{0}'.format(Fit['parent-Tag'].format(Fit['m_s'],mass))][0]
        if Fit['conf'] in ['Fs','SFs','UFs']:
            mass2 = '{0}'.format(float(mass))
            Fit['Ml_m{0}'.format(mass)] = pl['dE:{0}'.format(Fit['Hltag'].format(Fit['m_s'],mass2))][0]
    if Fit['conf'] in ['Fs','SFs','UFs']:
        Fit['M_Kaon'] = pl['dE:{0}'.format(Fit['ldaughtertag'][0].format('0'))][0]
    Fit['M_daughter'] = p['dE:{0}'.format(Fit['daughter-Tag'][0])][0]
    for t,twist in enumerate(Fit['twists']):
        #Fit is the actual measured value, theory is obtained from the momentum
        Fit['E_daughter_tw{0}_fit'.format(twist)] = p['dE:{0}'.format(Fit['daughter-Tag'][t])][0]
        Fit['E_daughter_tw{0}_theory'.format(twist)] = gv.sqrt(Fit['M_daughter']**2+Fit['momenta'][t]**2)
        for m, mass in enumerate(Fit['masses']):
            for thpt in thpts[Fit['conf']]:
                if twist != '0' or thpt != 'T':
                    if Fit['conf'] in  ['Fs','SFs','UFs']: # were not extracted with 2 in them
                        Fit['{0}_m{1}_tw{2}'.format(thpt,mass,twist)] = 2 * 2 * Fit['Zdisc'][m] * gv.sqrt(Fit['M_parent_m{0}'.format(mass)]*Fit['E_daughter_tw{0}_theory'.format(twist)]) * p['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)][0][0]
                        Fit['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)] = p['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)][0][0]
                    else:
                        Fit['{0}_m{1}_tw{2}'.format(thpt,mass,twist)] =  2 * Fit['Zdisc'][m] * gv.sqrt(Fit['M_parent_m{0}'.format(mass)]*Fit['E_daughter_tw{0}_theory'.format(twist)]) * p['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)][0][0]
                        Fit['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)] =  p['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)][0][0]
                    #check zdisc is correctly implemented here
    return()

###################################################################################################

#def make_MHsstar(MH_s,a=1.0): # need one of these for lattice units and GeV set a=1.0 to give GeV
    # Take MH_s and MH so we can use either. May be better to use MH throughout but then have BsEtas problem until we have UF data. However, this will solve problems with finephysical etc. 
#    DeltaDs = MDsstarphys - MDsphys
#    DeltaBs = MBsstarphys - MBsphys
#    MHsstar = MH_s + a**2*MDsphys*DeltaDs/MH_s + a*MBsphys/MH_s * ( (MH_s-a*MDsphys)/(MBsphys-MDsphys) * (DeltaBs - MDsphys/MBsphys * DeltaDs) )
#    return(MHsstar)

####### MH version
def make_MHsstar(MH,p,a=1.0): # need one of these for lattice units and GeV set a=1.0 to give GeV
    if a == 0:
        a = 1.0
    #DeltaD = p['MDsstarphys'] - p['MDphys']
    #DeltaB = p['MBsstarphys'] - p['MBphys']
    #old_MHsstar = MH + a**2*p['MDphys']*DeltaD/MH + a*p['MBphys']/MH * ( (MH-a*p['MDphys'])/(p['MBphys']-p['MDphys']) * (DeltaB - p['MDphys']/p['MBphys'] * DeltaD) )
    #MHsstar = p['MDsstarphys'] * a
    #print('Dsstar',old_MHsstar,MHsstar)
    return(MHsstar)
    
####################################################################################################

def make_fs(Fit,fs,thpts,Z_T):
    for m,mass in enumerate(Fit['masses']):
        Z_v = (float(mass) - float(Fit['m_s']))*Fit['S_m{0}_tw0'.format(mass)]/((Fit['M_parent_m{0}'.format(mass)] - Fit['M_daughter']) * Fit ['V_m{0}_tw0'.format(mass)])
        fs['Z_v_m{0}'.format(mass)] = Z_v
        print(Fit['conf'],mass,Z_v)
        for t,twist in enumerate(Fit['twists']):
            delta = (float(mass) - float(Fit['m_s']))*(Fit['M_parent_m{0}'.format(mass)]-Fit['E_daughter_tw{0}_theory'.format(twist)])
            qsq = (Fit['M_parent_m{0}'.format(mass)]-Fit['E_daughter_tw{0}_theory'.format(twist)])**2 - Fit['momenta'][t]**2
            f0 = ((float(mass) - float(Fit['m_s']))*(1/(Fit['M_parent_m{0}'.format(mass)]**2 - Fit['M_daughter']**2))*Fit['S_m{0}_tw{1}'.format(mass,twist)])
            
            A = Fit['M_parent_m{0}'.format(mass)] + Fit['E_daughter_tw{0}_theory'.format(twist)]
            B = (Fit['M_parent_m{0}'.format(mass)]**2 - Fit['M_daughter']**2)*(Fit['M_parent_m{0}'.format(mass)] - Fit['E_daughter_tw{0}_theory'.format(twist)])/qsq
            fp = None
            fT = None
            if twist != '0':
                fp = (1/(A-B))*(Z_v*Fit['V_m{0}_tw{1}'.format(mass,twist)] - B*f0)
                if 'T' in thpts[Fit['conf']]:
                    fT = np.sqrt(3)*Z_T[Fit['conf']]*Fit['T_m{0}_tw{1}'.format(mass,twist)]*(Fit['M_parent_m{0}'.format(mass)]+Fit['M_daughter'])/(2*Fit['M_parent_m{0}'.format(mass)]*Fit['momenta'][t])
            fs['qsq_m{0}_tw{1}'.format(mass,twist)] = qsq
            fs['f0_m{0}_tw{1}'.format(mass,twist)] = f0
            fs['fp_m{0}_tw{1}'.format(mass,twist)] = fp
            fs['fT_m{0}_tw{1}'.format(mass,twist)] = fT
    return()
#######################################################################################################
def make_t_plus(M_H,M_K): # This should ALWAYS be M_H,M_K because it is sea mass based
    t_plus = (M_H + M_K)**2
    return(t_plus)
#####################################################################################################
def make_t_0(t0,M_H,M_K,M_parent,M_daughter):
    #note that t_- is qsqmax
    t_plus = (M_H+M_K)**2
    t_minus = (M_parent-M_daughter)**2 #i.e. qsqmax
    if t0 == '0':
        t_0 = 0
    elif t0 == 'rev':
        t_0 = t_minus
    elif t0 == 'min': #not sure what this should be
        t_0 = t_plus * (1- (1 - (t_minus/t_plus))**(0.5))
    elif type(t0) == gv._gvarcore.GVar:
        t_0 = t0
    elif type(t0) == np.float64:
        t_0 = t0
    else:
        print('t_0 needs to be 0, rev or min or a gvar t_0 = {0} type: {1}'.format(t0,type(t0)))
    return(t_0)
######################################################################################################

def make_z(qsq,t0,M_H,M_K,M_parent=None,M_daughter=None): # this is give M_H and M_K in each case, then M_parent and M_daughter are given if different
    if M_parent == None:
        M_parent = M_H
    if M_daughter == None:
        M_daughter = M_K
    t_plus = make_t_plus(M_H,M_K)
    t_0 = make_t_0(t0,M_H,M_K,M_parent,M_daughter)
    z = (gv.sqrt(t_plus - qsq) - gv.sqrt(t_plus - t_0)) / (gv.sqrt(t_plus - qsq) + gv.sqrt(t_plus - t_0))
    if z.mean == 0 and z.sdev == 0:
        z = 0 #gv.gvar(0,1e-8) # ensures not 0(0)
    #print(qsq,z,M_H,M_K,M_parent,M_daughter,t0,t_0,t_plus)
    return(z)
###################################################################################################

def make_phi_fp(qsq,t0,M_H,M_K,M_parent=None,M_daughter=None,alat=1.0):
    #we require all 3 values of t_0 here
    if alat == 0:
        alat = 1.0    
    if t0 != 'min':
        print('Warning: Tring to use phi without t_0 = min')
    m_c = alat*1.25
    t_plus = (M_H+M_K)**2
    if M_parent == None:
        M_parent = M_H
    if M_daughter == None:
        M_daughter = M_K
    t_minus = (M_parent-M_daughter)**2
    t_0 = make_t_0(t0,M_H,M_K,M_parent,M_daughter)
    z_0 = make_z(qsq,'0',M_H,M_K,M_parent=M_parent,M_daughter=M_daughter)
    z_t_0 = make_z(qsq,t0,M_H,M_K,M_parent=M_parent,M_daughter=M_daughter)
    z_t_minus = make_z(qsq,'rev',M_H,M_K,M_parent=M_parent,M_daughter=M_daughter)
    if qsq == 0:
        phi = np.sqrt(np.pi/3)* m_c * (1/(4*t_plus))**(5/2) * (z_t_0/(t_0-qsq))**(-1/2) * (z_t_minus/(t_minus-qsq))**(-3/4) * (t_plus-qsq)/((t_plus-t_0)**(1/4))

    elif qsq == t_minus:
        phi = np.sqrt(np.pi/3)* m_c * (z_0/(-qsq))**(5/2) * (z_t_0/(t_0-qsq))**(-1/2) * (1/(4*(t_plus-qsq)))**(-3/4)* (t_plus-qsq)/((t_plus-t_0)**(1/4))
    elif qsq == t_0:
        phi = np.sqrt(np.pi/3)* m_c * (z_0/(-qsq))**(5/2) * (1/(4*(t_plus-qsq)))**(-1/2) * (z_t_minus/(t_minus-qsq))**(-3/4) * (t_plus-qsq)/((t_plus-t_0)**(1/4))
    else:
        phi = np.sqrt(np.pi/3)* m_c * (z_0/(-qsq))**(5/2) * (z_t_0/(t_0-qsq))**(-1/2) * (z_t_minus/(t_minus-qsq))**(-3/4) * (t_plus-qsq)/((t_plus-t_0)**(1/4))
    phi = phi # no 30 here

    #phitest = np.sqrt(np.pi/3)* m_c *(gv.sqrt(t_plus-qsq)+gv.sqrt(t_plus-t_0)) * (t_plus-qsq)/((t_plus-t_0)**(1/4)) * (gv.sqrt(t_plus-qsq)+gv.sqrt(t_plus-t_minus))**(3/2)/(gv.sqrt(t_plus-qsq)+gv.sqrt(t_plus))**5
    #print(phi,phitest)
    return(phi)

######################################################################################################

def check_poles(Fits):
    #plt.figure()
    Del = MDs0phys - MDphys
    for Fit in Fits:
        for mass in Fit['masses']:
            qsqmax = ((Fit['M_parent_m{0}'.format(mass)]-Fit['M_daughter'])**2).mean
            if Fit['conf'] in ['Fs','SFs','UFs']:
                t_plus = (make_t_plus(Fit['Ml_m{0}'.format(mass)],Fit['M_Kaon'])).mean
                f0pole2 = ((Fit['Ml_m{0}'.format(mass)] + Fit['a']*Del)**2).mean
                fppole2 = ((make_MHsstar(Fit['Ml_m{0}'.format(mass)],Fit['a']))**2).mean
            else:
                t_plus = (make_t_plus(Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter'])).mean
                f0pole2 = ((Fit['M_parent_m{0}'.format(mass)] + Fit['a']*Del)**2).mean
                fppole2 = ((make_MHsstar(Fit['M_parent_m{0}'.format(mass)],Fit['a']))**2).mean
            if f0pole2 < qsqmax:
                print('f0 pole below qsqmax Fit {0} mass {1}'.format(Fit['conf'],mass))
            if fppole2 < qsqmax:
                print('fp pole below qsqmax Fit {0} mass {1}'.format(Fit['conf'],mass))
            if f0pole2 > t_plus:
                print('f0 pole above t_plus Fit {0} mass {1}'.format(Fit['conf'],mass))
            if fppole2 > t_plus:
                print('fp pole above t_plus Fit {0} mass {1}'.format(Fit['conf'],mass))
            
    return()                                                                                         
              

########################################################################################################

def make_prior_BK(fs_data,Fits,addrho,t_0,Npow,Nijk,Nm,rhopri,dpri,cpri,cvalpri,d000npri,di000pri,di10npri,adddata,constraint,w=1):
    prior = gv.BufferDict()
    f = gv.BufferDict()
    prior['g'] = gv.gvar('0.570(6)')#1304.5009 pg 19 
    prior['Metacphys'] = Metacphys
    prior['MDphys'] = MDphys
    prior['MKphys'] = MKphys
    prior['MDsstarphys'] =  MDsstarphys
    prior['MDs0phys'] = MDs0phys
    #prior['MBphys'] = MBphys
    #prior['MBsstarphys'] =  MBsstarphys
    prior['slratio'] = slratio
    for Fit in Fits:
        fit = Fit['conf']
        prior['a_{0}'.format(fit)] = Fit['a'] #used in 1406.2279
        #prior['LQCD_{0}'.format(fit)] = LQCD*Fit['a']#have to convert this now so can evaluate in GeV later
        ms0 = Fit['m_ssea']
        ml0 = Fit['m_lsea'] #becuase valuence and sea same, only use one
        ms0val = float(Fit['m_s']) # valence untuned s mass
        ml0val = float(Fit['m_l']) # valence untuned s mass
        Metas = globals()['Metas_{0}'.format(fit)]/prior['a_{0}'.format(fit)] # in GeV
        prior['Metac_{0}'.format(fit)] = globals()['Metac{0}'.format(fit)]/Fit['a'] #in GeV
        prior['deltaFV_{0}'.format(fit)] = globals()['deltaFV{0}'.format(fit)]
        prior['mstuned_{0}'.format(fit)] = ms0val*(Metasphys/Metas)**2
        prior['ml10ms_{0}'.format(fit)] = ml0val/(mlmsfac*prior['mstuned_{0}'.format(fit)])
        mltuned = prior['mstuned_{0}'.format(fit)]/prior['slratio'] 
        #prior['MD_{0}'.format(fit)] = Fit['M_parent_m{0}'.format(Fit['m_c'])] #lat units
        prior['deltas_{0}'.format(fit)] = ms0-prior['mstuned_{0}'.format(fit)]     
        prior['deltasval_{0}'.format(fit)] = ms0val-prior['mstuned_{0}'.format(fit)]
        prior['deltal_{0}'.format(fit)] = ml0-mltuned
        prior['deltalval_{0}'.format(fit)] = ml0val-mltuned
        prior['MK_{0}'.format(fit)] = Fit['M_daughter']
        for mass in Fit['masses']:
            prior['MH_{0}_m{1}'.format(fit,mass)] = Fit['M_parent_m{0}'.format(mass)]
            ######
            #Del =  prior['MDs0phys'] - prior['MDphys']
            #MHs01 = prior['MH_{0}_m{1}'.format(fit,mass)] + prior['a_{0}'.format(fit)]*Del
            #MHs02 = prior['a_{0}'.format(fit)] * prior['MDs0phys']
            #print(Fit['conf'],MHs01,MHs02,MHs02-MHs01)
            ##########################
            if Fit['conf'] in ['Fs','SFs','UFs']:
                print('ERROR: Trying to use BsEtas data, not enabled here')
            for twist in Fit['twists']:
                tag = '{0}_m{1}_tw{2}'.format(fit,mass,twist)
                qsq = fs_data[fit]['qsq_m{0}_tw{1}'.format(mass,twist)]                    
                prior['qsq_{0}'.format(tag)] = qsq
                f['f0_{0}'.format(tag)] = fs_data[fit]['f0_m{0}_tw{1}'.format(mass,twist)]   # y values go in f   
                f['fp_{0}'.format(tag)] = fs_data[fit]['fp_m{0}_tw{1}'.format(mass,twist)]
    if constraint:
        f['constraint'] = gv.gvar(0,1e-8) #1e-8 works here
    keys = []
    for key in f:
        if f[key] == None:
            keys.append(key)   #removes vector tw 0 etc
    for key in keys:
        del f[key]
    if addrho:
        prior['0rho'] =gv.gvar(Npow*[rhopri])*w
        prior['prho'] =gv.gvar(Npow*[rhopri])*w
    prior['0d'] = gv.gvar(Nijk[0]*[Nijk[1]*[Nijk[2]*[Npow*[dpri]]]])*w
    prior['0cs'] = gv.gvar(Npow*[cpri])*w
    prior['0cl'] = gv.gvar(Npow*[cpri])*w
    prior['0cc'] = gv.gvar(Npow*[cpri])*w
    prior['0csval'] = gv.gvar(Npow*[cvalpri])*w
    prior['0clval'] = gv.gvar(Npow*[cvalpri])*w
    prior['pd'] = gv.gvar(Nijk[0]*[Nijk[1]*[Nijk[2]*[Npow*[dpri]]]])*w
    for i in range(Nijk[0]):
        if i != 0:
            prior['0d'][i][0][0][0] = gv.gvar(di000pri)*w
            prior['pd'][i][0][0][0] = gv.gvar(di000pri)*w
        for n in range(Npow):
            prior['0d'][i][1][0][n] = gv.gvar(di10npri)*w
            prior['pd'][i][1][0][n] = gv.gvar(di10npri)*w
            prior['0d'][0][0][0][n] = gv.gvar(d000npri)*w 
            prior['pd'][0][0][0][n] = gv.gvar(d000npri)*w
    prior['pcs'] = gv.gvar(Npow*[cpri])*w
    prior['pcl'] = gv.gvar(Npow*[cpri])*w
    prior['pcc'] = gv.gvar(Npow*[cpri])*w
    prior['pcsval'] = gv.gvar(Npow*[cvalpri])*w
    prior['pclval'] = gv.gvar(Npow*[cvalpri])*w
    #print(prior)
    print(w)
    return(prior,f)
                
########################################################################################################

def make_an_BK(n,Nijk,Nm,addrho,p,tag,Fit,mass,amh,fpf0same,newdata=False,const=False,const2=False): # tag is 0,p,T in this way, we can set fp(0)=f0(0) by just putting 0 for n=0 alat is lattice spacing (mean) so we can use this to evaluate at different lattice spacings p is dict containing all values (prior or posterior or anything) # need to edit l valence mistuning
    fit = Fit['conf']
    an = 0    
    for i in range(Nijk[0]):
        for j in range(Nijk[1]):
            for k in range(Nijk[2]):
                tagsamed = tag
                tagsamerho = tag
                if tag == '0' or tag == 'T' or fpf0same == False:
                    pass
                elif n == 0:
                    tagsamerho = '0'
                    if j == 0 and k == 0 :
                        tagsamed = '0'
                if addrho:
                    if const:
                        an += (1 + p['{0}rho'.format(tagsamerho)][n]*gv.log(p['MDphys']/p['MDphys'])) *  p['{0}d'.format(tagsamed)][i][j][k][n] * (0)**int(i) * (0)**int(2*j) * (0)**int(2*k) 
                    elif const2:
                        an += (1 + p['{0}rho'.format(tagsamerho)][n]*gv.log(p['MH_{0}_m{1}'.format(fit,mass)]/p['MD_{0}'.format(fit)])) *  p['{0}d'.format(tagsamed)][i][j][k][n] * (0)**int(i) * (0)**int(2*j) * (0)**int(2*k) 
                    elif newdata:
                        print('Added external data in a{0}'.format(n), 'need to edit this only does BsEtas')
                        an += (1 + p['{0}rho'.format(tagsamerho)][n]*gv.log(p['MBsphys']/p['MDsphys'])) *  p['{0}d'.format(tagsamed)][i][j][k][n] * (0)**int(i) * (0)**int(2*j) * (0)**int(2*k) * (1+mlpowsBsEtas)
                    elif newdata == False and const == False and const2 == False:
                        an += (1 + p['{0}rho'.format(tagsamerho)][n]*gv.log(p['MH_{0}_m{1}'.format(fit,mass)]/p['MD_{0}'.format(fit)])) * (1 + (p['{0}csval'.format(tag)][n]*p['deltasval_{0}'.format(fit)] + p['{0}cs'.format(tag)][n]*p['deltas_{0}'.format(fit)] + 2*p['{0}cl'.format(tag)][n]*p['deltal_{0}'.format(fit)]+p['{0}clval'.format(tag)][n]*p['deltalval_{0}'.format(fit)])/(10*p['mstuned_{0}'.format(fit)])   + p['{0}cc'.format(tag)][n]*((p['Metac_{0}'.format(fit)] - p['Metacphys'])/p['Metacphys'])) * p['{0}d'.format(tagsamed)][i][j][k][n] * (0)**int(i) * (amh/np.pi)**int(2*j) * (LQCD*p['a_{0}'.format(fit)]/np.pi)**int(2*k)
                    else:
                        print('Error in make_an_BsEtas(): newdata = {0}, const = {1}, const2 = {2}'.format(newdata,const,const2))
                        
                else:
                    if const:
                        an += p['{0}d'.format(tagsamed)][i][j][k][n] * (0)**int(i) * (0)**int(2*j) * (0)**int(2*k)
                    elif const2:
                        an += p['{0}d'.format(tagsamed)][i][j][k][n] * (0)**int(i) * (0)**int(2*j) * (0)**int(2*k) 
                    elif newdata:
                        print('Added external data in a{0}'.format(n),'need to edit this to work properly')
                        an += p['{0}d'.format(tagsamed)][i][j][k][n] * (0)**int(i) * (0)**int(2*j) * (0)**int(2*k)
                    elif newdata == False and const == False and const2 == False:
                        an += (1 + (p['{0}csval'.format(tag)][n]*p['deltasval_{0}'.format(fit)] + p['{0}cs'.format(tag)][n]*p['deltas_{0}'.format(fit)]+2*p['{0}cl'.format(tag)][n]*p['deltal_{0}'.format(fit)]+p['{0}clval'.format(tag)][n]*p['deltalval_{0}'.format(fit)])/(10*p['mstuned_{0}'.format(fit)])  + p['{0}cc'.format(tag)][n]*((p['Metac_{0}'.format(fit)] - p['Metacphys'])/p['Metacphys'])) * p['{0}d'.format(tagsamed)][i][j][k][n] * (0)**int(i) * (amh/np.pi)**int(2*j) * (0*LQCD*p['a_{0}'.format(fit)]/np.pi)**int(2*k)
                    else:
                        print('Error in make_an_BK(): newdata = {0}, const = {1}, const2 = {2}'.format(newdata,const,const2))
                    
    return(an)

########################################################################################################

def make_logs(p,Fit):
    logs = 1 - ( (9/8) * p['g']**2 * p['ml10ms_{0}'.format(Fit['conf'])] * ( gv.log(p['ml10ms_{0}'.format(Fit['conf'])]) + p['deltaFV_{0}'.format(Fit['conf'])]))
    logs = 1
    return(logs)

########################################################################################################
def make_f0_BK(Nijk,Npow,Nm,addrho,p,Fit,qsq,t_0,mass,fpf0same,amh,newdata=False,const=False):
    fit = Fit['conf']
    f0 = 0
    logs = make_logs(p,Fit)
    Del =  p['MDs0phys']- p['MDphys']
    if p['a_{0}'.format(fit)] != 0:
        MHs0 = p['MH_{0}_m{1}'.format(fit,mass)] + p['a_{0}'.format(fit)]*Del
    else:
        MHs0 = p['MH_{0}_m{1}'.format(fit,mass)] + Del
    pole = 1-(qsq/MHs0**2)
    f0pole0 = 1.0 
    z  = make_z(qsq,t_0,p['MH_{0}_m{1}'.format(fit,mass)],p['MK_{0}'.format(fit)])# need to change this for B to K
    z0 = make_z(0,t_0,p['MH_{0}_m{1}'.format(fit,mass)],p['MK_{0}'.format(fit)])
    if newdata:
        logsBsEtas = 1 - ( (9/8) * p['g']**2 * 1/10 * ( gv.log(1/10) ) )
    if const:
        logsBK = 1 - ( (9/8) * p['g']**2 * 1/(10*p['slratio']) * ( gv.log(1/(10*p['slratio'])) ) )
        z0const = make_z(0,t_0,p['MDphys'],p['MKphys'])
    for n in range(Npow):
        if const:
            an = make_an_BK(n,Nijk,Nm,addrho,p,'0',Fit,mass,amh,fpf0same,const=const)
            f0 += (logsBK/f0pole0) * an * z0const**n
        elif newdata:
            an = make_an_BK(n,Nijk,Nm,addrho,p,'0',Fit,mass,amh,fpf0same,newdata=newdata)
            f0 += logsBsEtas/(1-qsq/(p['MBs0phys']**2)) * an * z**n
        elif newdata == False and const == False:
            an = make_an_BK(n,Nijk,Nm,addrho,p,'0',Fit,mass,amh,fpf0same)
            f0 += (logs/pole) * an * z**n
        else:
            print('Error in make_f0_BK(): newdata = {0}'.format(newdata))
    return(f0)

########################################################################################################

def make_fp_BK(Nijk,Npow,Nm,addrho,p,Fit,qsq,t_0,mass,fpf0same,amh,newdata=False,const=False,const2=False):
    fit = Fit['conf']
    fp = 0
    fp0 = 0
    f00 = 0
    logs = make_logs(p,Fit)
    #MHsstar = make_MHsstar(p['MH_{0}_m{1}'.format(fit,mass)],p,p['a_{0}'.format(fit)])
    Del = p['MDsstarphys'] - p['MDphys']
    if p['a_{0}'.format(fit)] != 0:
        MHsstar = p['MH_{0}_m{1}'.format(fit,mass)] + p['a_{0}'.format(fit)]*Del
    else:
        MHsstar = p['MH_{0}_m{1}'.format(fit,mass)] + Del
    z0 = make_z(0,t_0,p['MH_{0}_m{1}'.format(fit,mass)],p['MK_{0}'.format(fit)])
    z = make_z(qsq,t_0,p['MH_{0}_m{1}'.format(fit,mass)],p['MK_{0}'.format(fit)])
    pole = 1-(qsq/MHsstar**2)
    fppole0 = 1.0
    f0pole0 = 1.0
    if newdata:
        logsBsEtas = 1 - ( (9/8) * p['g']**2 * 1/10 * ( gv.log(1/10) ) )
    if const or const2:
        logsBK = 1 - ( (9/8) * p['g']**2 * 1/(10*p['slratio']) * ( gv.log(1/(10*p['slratio'])) ) )
    if const:
        z0const = make_z(0,t_0,p['MDphys'],p['MKphys'])
    for n in range(Npow):
        if const==True and const2 ==False and newdata ==False:
            an = make_an_BK(n,Nijk,Nm,addrho,p,'p',Fit,mass,amh,fpf0same,const=const)
            fp += (logsBK/fppole0) * an  * (z0const**n - (n/Npow) * (-1)**(n-Npow) *  z0const**Npow)
        elif newdata == True and const ==False and const2 ==False:
            an = make_an_BK(n,Nijk,Nm,addrho,p,'p',Fit,mass,amh,fpf0same,newdata=newdata)
            fp += logsBsEtas/(1-qsq/(p['MBsstarphys']**2)) * an  * (z**n - (n/Npow) * (-1)**(n-Npow) *  z**Npow)
        elif const2==True and const ==False and newdata==False:
            an = make_an_BK(n,Nijk,Nm,addrho,p,'p',Fit,mass,amh,fpf0same)
            fp += (logs/pole) * an  * (z**n - (n/Npow) * (-1)**(n-Npow) *  z**Npow)
            anp0 = make_an_BK(n,Nijk,Nm,addrho,p,'p',Fit,mass,amh,fpf0same,const2=const2)
            fp0 += (logsBK/fppole0) * anp0  * (z0**n - (n/Npow) * (-1)**(n-Npow) *  z0**Npow)
            an00 = make_an_BK(n,Nijk,Nm,addrho,p,'0',Fit,mass,amh,fpf0same,const2=const2)
            f00 += (logsBK/f0pole0) * an00 * z0**n
        elif newdata == False and const2 ==False and const ==False:
            an = make_an_BK(n,Nijk,Nm,addrho,p,'p',Fit,mass,amh,fpf0same)
            fp += (logs/pole) * an  * (z**n - (n/Npow) * (-1)**(n-Npow) *  z**Npow)
        else:
            print('Error in make_fp_BK(): newdata = {0} const = {1} const2 = {2}'.format(newdata,const,const2))
    fp = fp + f00 - fp0
    #print(fp0/f00)
    return(fp)

#########################################################################################################

def make_fT_BK(Nijk,Npow,Nm,addrho,p,Fit,alat,qsq,z,mass,fpf0same,amh,newdata=False):
    print('FT NOT MODIFIED ERROR')
    fT = 0
    logs = make_logs(p,Fit)
    for n in range(Npow):
        if newdata:
            print('Error,trying to input new data fT')
            an = make_an_BK(n,Nijk,Nm,addrho,p,'T',Fit,alat,mass,amh,fpf0same,newdata=newdata)
            fT += logs/(1-qsq/(p['MBsstarphys']**2)) * an  * (z**n - (n/Npow) * (-1)**(n-Npow) *  z**Npow)
        elif newdata == False:
            an = make_an_BK(n,Nijk,Nm,addrho,p,'T',Fit,alat,mass,amh,fpf0same)
            fT += logs/(1-qsq/(p['MHsstar_{0}_m{1}'.format(Fit['conf'],mass)]**2)) * an  * (z**n - (n/Npow) * (-1)**(n-Npow) *  z**Npow)
        else:
            print('Error in make_fT_BK(): newdata = {0}'.format(newdata))
    return(fT)

#######################################################################################################

def do_fit_BK(fs_data,adddata,Fits,f,Nijk,Npow,Nm,t_0,addrho,noise,prior,fpf0same,rhopri,dpri,cpri,cvalpri,d000npri,di000pri,di10npri,constraint,const2):
    def fcn(p):
        models = gv.BufferDict()
        if 'constraint' in f:
            models['constraint'] = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,t_0,Fits[0]['masses'][0],fpf0same,0,const=True)- make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,t_0,Fits[0]['masses'][0],fpf0same,0,const=True)  
        for Fit in Fits:
            for mass in Fit['masses']:
                for twist in Fit['twists']:
                    tag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)
                    if 'f0_{0}'.format(tag) in f:
                        models['f0_{0}'.format(tag)] = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fit,p['qsq_{0}'.format(tag)],t_0,mass,fpf0same,float(mass)) #second mass is amh
                    if 'fp_{0}'.format(tag) in f:
                        models['fp_{0}'.format(tag)] = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fit,p['qsq_{0}'.format(tag)],t_0,mass,fpf0same,float(mass),const2=const2) #second mass is amh
                    
                        
        return(models)
    #################################################
    def fitargs(w):
        prior,f = make_prior_BK(fs_data,Fits,addrho,t_0,Npow,Nijk,Nm,rhopri,dpri,cpri,cvalpri,d000npri,di000pri,di10npri,adddata,constraint,w=w)
        return dict(data=f, fcn=fcn, prior=prior)
    ##################################################
    #p0 = None
    #if os.path.isfile('Fits/pmeanBK{0}{1}{2}{3}.pickle'.format(addrho,Npow,Nijk,Nm)):
    #    p0 = gv.load('Fits/pmeanBK{0}{1}{2}{3}.pickle'.format(addrho,Npow,Nijk,Nm))
    p0 = None    
    fit = lsqfit.nonlinear_fit(data=f, prior=prior, p0=p0, fcn=fcn, svdcut=1e-12 ,noise=noise, maxit=500, tol=(1e-6,0.0,0.0),debug=True,fitter='gsl_multifit', alg='subspace2D', solver='cholesky' )
    gv.dump(fit.pmean,'Fits/pmeanBK{0}{1}{2}{3}.pickle'.format(addrho,Npow,Nijk,Nm))
    print(fit.format(maxline=True))
    print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.2f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
    #fit2, w = lsqfit.empbayes_fit(1.1, fitargs)
    #print(fit2.format(True))
    #print(w)
    return(fit.p)

######################################################################################################

def make_p_physical_point_DK(pfit,Fits,Dplus=False):
    #only need to evaluate at one Fit one mass but change all anyway
    # everything should now be in GeV
    p = gv.BufferDict()
    for Fit in Fits:
        fit = Fit['conf']
        p['a_{0}'.format(fit)] = 0
        if Dplus:
            p['MK_{0}'.format(fit)] = MKphys0
            p['MKphys'] = MKphys0
        else:
            p['MK_{0}'.format(fit)] = pfit['MKphys']
        #p['LQCD_{0}'.format(fit)] = LQCD
        p['Metac_{0}'.format(fit)] = pfit['Metacphys']
        p['deltas_{0}'.format(fit)] = 0
        p['deltal_{0}'.format(fit)] = 0
        p['deltasval_{0}'.format(fit)] = 0
        p['deltalval_{0}'.format(fit)] = 0
        p['ml10ms_{0}'.format(fit)] = 1/(mlmsfac*pfit['slratio'])
        p['deltaFV_{0}'.format(fit)] = 0
        for mass in Fit['masses']:
            if Dplus:
                p['MH_{0}_m{1}'.format(fit,mass)] = MDphyspm
                p['MDphys'] = MDphyspm
            else:
                p['MH_{0}_m{1}'.format(fit,mass)] = pfit['MDphys']
            #p['MD_{0}'.format(fit)] = pfit['MDphys']
    for key in pfit:
        if key not in p:
            p[key] = pfit[key]
    return(p)

######################################################################################################

def make_p_Mh_BK(pfit,Fits,MH):
    print('WARNING, ARE POLE POSITIONS CORRECT?')
    #only need to evaluate at one Fit one mass but change all anyway
    # everything should now be in GeV
    #Need to think about pole positions here should be at MHsstar ?  ? 
    p = gv.BufferDict()
    for Fit in Fits:
        fit = Fit['conf']
        p['a_{0}'.format(fit)] = 0
        p['MK_{0}'.format(fit)] = pfit['MKphys']
        #p['LQCD_{0}'.format(fit)] = LQCD
        p['Metac_{0}'.format(fit)] = pfit['Metacphys']
        p['deltas_{0}'.format(fit)] = 0
        p['deltal_{0}'.format(fit)] = 0
        p['deltalval_{0}'.format(fit)] = 0
        p['deltasval_{0}'.format(fit)] = 0
        p['ml10ms_{0}'.format(fit)] = 1/(mlmsfac*pfit['slratio'])
        p['deltaFV_{0}'.format(fit)] = 0
        for mass in Fit['masses']:
            p['MH_{0}_m{1}'.format(fit,mass)] = MH
            #p['MD_{0}'.format(fit)] = pfit['MDphys']
    for key in pfit:
        if key not in p:
            p[key] = pfit[key]
    return(p)

######################################################################################################


def fs_at_lims_DK(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    p = make_p_physical_point_DK(pfit,Fits)
    qsq0 = 0
    f00 = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq0,t_0,Fits[0]['masses'][0],fpf0same,0)
    fp0 = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq0,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
    qsq = qsqmaxphysDK.mean
    f0max = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0)
    fpmax = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
    print('f_+(0)/f_0(0) = {0}'.format(fp0/f00))
    print('f_0(0) = {0}  error: {1:.2%}'.format(f00,f00.sdev/f00.mean))
    print('f_+(0) = {0}  error: {1:.2%}'.format(fp0,fp0.sdev/fp0.mean))
    print('f_0(max) = {0}  error: {1:.2%}'.format(f0max,f0max.sdev/f0max.mean))
    print('f_+(max) = {0}  error: {1:.2%}'.format(fpmax,fpmax.sdev/fpmax.mean))
    HFLAV= gv.gvar('0.7180(33)')# from #1909.12524 p318
    BESmu = gv.gvar(0.7133,np.sqrt(0.0038**2+0.0030**2))#1810.03127
    correction = gv.sqrt(cor2)
    Vcsq20 = HFLAV/(fp0*correction)
    terr = Vcsq20.partialsdev(fp0)
    eerr = Vcsq20.partialsdev(HFLAV)
    cerr = Vcsq20.partialsdev(correction)
    print('A) HFLAV av. Vcs e at q^2=0 = {0}'.format(Vcsq20))
    print('Theory error = {0:.4f} Exp error = {1:.4f} Correction error = {2:.4f} Total = {3:.4f}'.format(terr,eerr,cerr,np.sqrt(eerr**2+terr**2+cerr**2)))
    Vcsq20 = BESmu/(fp0*correction)
    terr = Vcsq20.partialsdev(fp0)
    eerr = Vcsq20.partialsdev(BESmu)
    cerr = Vcsq20.partialsdev(correction)
    print('Vcs mu at q^2=0 = {0}'.format(Vcsq20))
    print('Theory error = {0:.4f} Exp error = {1:.4f} Correction error = {2:.4f} Total = {3:.4f}'.format(terr,eerr,cerr,np.sqrt(eerr**2+terr**2+cerr**2)))
    return()

######################################################################################################

def integrate_fp(qsq_min,qsq_max,pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2,m_lep,iters=100,Dplus=False):
    if qsq_min < (m_lep**2).mean:
        qsq_min = (m_lep**2).mean
    p = make_p_physical_point_DK(pfit,Fits,Dplus=Dplus)
    A = 3/8 * p['MDphys']**2 * (1 - (p['MKphys']**2/p['MDphys']**2) )**2 
    def integrand(qsq):
        eps = m_lep**2/qsq
        p2 = (qsq-p['MKphys']**2-p['MDphys']**2)**2/(4*p['MDphys']**2)-p['MKphys']**2
        if p2.mean <0:
            p2 = 0
        p3 = (p2)**(3/2)
        p1 = p2**(1/2)
        fp = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
        f0 = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0)
        integrand = (1-eps)**2 * (p3 * fp**2 * (1 + eps/2) + p1 * f0**2 * eps * A)
        return(integrand)
    del_qsq =  (qsq_max-qsq_min) /iters
    funcs = integrand(qsq_min) + integrand(qsq_max)
    for i in range(1,iters):
        funcs += 2*integrand(qsq_min+del_qsq*i)
    result = del_qsq*funcs/2
    return(result)

################################################################################################


def integrate_R_mu_e(qsq_min,qsq_max,pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2,iters=500):
    p = make_p_physical_point_DK(pfit,Fits)
    A = 3/8 * p['MDphys']**2 * (1 - (p['MKphys']**2/p['MDphys']**2) )**2
    #qsq_max = ((p['MDphys']-p['MKphys'])**2).mean
    def integrand(qsq):
        epse = M_e**2/qsq
        epsmu = M_mu**2/qsq
        p2 = (qsq-p['MKphys']**2-p['MDphys']**2)**2/(4*p['MDphys']**2)-p['MKphys']**2
        if p2.mean < 0:
            p2 = 0
        p3 = (p2)**(3/2)
        p1 = p2**(1/2)
        fp = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
        f0 = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0)
        inte = (1-epse)**2 * (p3 * fp**2 * (1 + epse/2) + p1 * f0**2 * epse * A)
        intmu = (1-epsmu)**2 * (p3 * fp**2 * (1 + epsmu/2) + p1 * f0**2 * epsmu * A)
        integrand = intmu/inte
        if qsq == qsqmaxphysDK.mean:
            integrand = epsmu*(1-epsmu)**2/(epse*(1-epse)**2)
        #print(qsq,p2,integrand)
        return(integrand)
    del_qsq =  (qsq_max-qsq_min) /iters
    funcs = integrand(qsq_min) + integrand(qsq_max)
    for i in range(1,iters):
        funcs += 2*integrand(qsq_min+del_qsq*i)
    result = del_qsq*funcs/2
    return(result)

#################################################################################################

def R_mu_e_integrals(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2):
    qsq_min = (M_mu**2).mean
    qsq_max = qsqmaxphysDK.mean
    mu = integrate_fp(qsq_min,qsq_max,pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2,M_mu,iters=500)
    qsq_min = (M_e**2).mean
    e = integrate_fp(qsq_min,qsq_max,pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2,M_e,iters=500)
    rat = mu/e
    print('R_mu/e = ', rat)
    return()

#################################################################################################

def comp_cleo(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2):
    #bins 0,0.2,0.4...1.6,inf
    p3integrals0 = []
    p3integralsp = []
    bins = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,qsqmaxphysDK.mean]
    partials = [17.82,15.83,13.91,11.69,9.36,7.08,5.34,3.09,1.28,17.79,15.62,14.02,12.28,8.92,8.17,4.96,2.67,1.19] ## D^0 to K^- followed by D^+ to K^0
    for i in range(len(bins)-1):
        p3integrals0.append(integrate_fp(bins[i],bins[i+1],pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2,M_e))
        p3integralsp.append(integrate_fp(bins[i],bins[i+1],pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2,M_e,Dplus=True))
    cov_mat = gv.load('covarience_matricies/CLEO.pickle')
    partials  = gv.gvar(partials,cov_mat)
    Vcss21 = []
    terr1 = []
    eerr1 = []
    Vcss22 = []
    terr2 = []
    eerr2 = []
    ints1 = []
    ints2 = []
    for i in range(int(len(partials)/2)):
        V2 =  24 * np.pi**3 * partials[i] * 6.582119569*1e-16 /(GF**2 * p3integrals0[i])
        Vcss21.append(V2)
        terr1.append(V2.partialsdev(p3integrals0[i]))
        eerr1.append(V2.partialsdev(partials[i]))
        ints1.append(p3integrals0[i])
    for i in range(int(len(partials)/2),len(partials)):
        V2 =  24 * np.pi**3 * partials[i] * 6.582119569*1e-16 /(GF**2 * p3integralsp[i-int(len(partials)/2)])
        Vcss22.append(V2)
        terr2.append(V2.partialsdev(p3integralsp[i-int(len(partials)/2)]))
        eerr2.append(V2.partialsdev( partials[i]))
        ints2.append(p3integralsp[i-int(len(partials)/2)])
        
    return(Vcss21,terr1,eerr1,Vcss22,terr2,eerr2,bins,ints1,ints2,partials)

############################################################################################

def comp_BES(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2):
    #bins 0,0.2,0.4...1.6,inf
    p3integrals = []
    bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,qsqmaxphysDK.mean]
    for i in range(len(bins)-1):
        p3integrals.append(integrate_fp(bins[i],bins[i+1],pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2,M_e))
    partials = [8.812,8.743,8.295,7.567,7.486,6.446,6.200,5.519,5.028,4.525,3.972,3.326,2.828,2.288,1.737,1.314,0.858,0.379] ## D^0 to K^-
    
    cov_mat = gv.load('covarience_matricies/BES.pickle')
    partials  = gv.gvar(partials,cov_mat)
    Vcss2 = []
    terr = []
    eerr = []
    ints = []
    for i in range(len(partials)):
        V2 =  24 * np.pi**3 * partials[i] * 6.582119569*1e-16 /(GF**2 * p3integrals[i])
        Vcss2.append(V2)
        terr.append(V2.partialsdev(p3integrals[i]))
        eerr.append(V2.partialsdev(partials[i]))
        ints.append(p3integrals[i])
    return(Vcss2,terr,eerr,bins,ints,partials)

##################################################################################################

def comp_BESDp(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2):
    #bins 0,0.2,0.4...1.6,inf
    p3integrals = []
    bins = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,qsqmaxphysDK.mean]
    for i in range(len(bins)-1):
        p3integrals.append(integrate_fp(bins[i],bins[i+1],pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2,M_e,Dplus=True))
    partials = [16.97, 15.29, 13.57, 11.65,  9.33,  7.06,  4.96,  2.97,  1.01]    
    cov_mat = gv.load('covarience_matricies/BESDp.pickle')
    partials  = gv.gvar(partials,cov_mat)
    Vcss2 = []
    terr = []
    eerr = []
    ints = []
    for i in range(len(partials)):
        V2 =  24 * np.pi**3 * partials[i] * 6.582119569*1e-16 /(GF**2 * p3integrals[i])
        Vcss2.append(V2)
        terr.append(V2.partialsdev(p3integrals[i]))
        eerr.append(V2.partialsdev(partials[i]))
        ints.append(p3integrals[i])
    return(Vcss2,terr,eerr,bins,ints,partials)

#######################BaBar #####################################################

def comp_BaBar(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2):
    data = gv.load('covarience_matricies/normless_Rless_BaBar.pickle') # contains [R,bf1,bf2...]
    R = data[0]
    bf = data[1:]
    #bf[-1] = 1 - sum(bf[:-1]) # to modify 
    partials = []
    for i in range(len(bf)):
        partials.append(bf[i]*R)
    p3integrals = []
    bins = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,qsqmaxphysDK.mean]
    for i in range(len(bins)-1):
        p3integrals.append(integrate_fp(bins[i],bins[i+1],pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2,M_e))
    #gv.dump(p3integrals,'../Misc/data/fps.pickle')
    #partials = [18.15, 16.63, 14.75, 12.67, 10.14, 7.90, 5.44, 3.32, 1.320, 0.0633] ## D^0 to K^-
    #partials = [0.1861, 0.1705, 0.1513, 0.1300, 0.1040, 0.0810, 0.0558, 0.0340, 0.01353, 0.000649]
    #cov_mat = gv.load('covarience_matricies/normless_BaBar.pickle')
    #print(cov_mat)
    #partials  = gv.gvar(partials,cov_mat)
    #print(partials)
    Vcss2 = []
    terr = []
    eerr = []
    ints = []
    for i in range(len(bf)):
        V2 = 24 * np.pi**3 * partials[i] * 6.582119569*1e-16 /(GF**2 * p3integrals[i])
        Vcss2.append(V2)
        terr.append((V2*Banorm).partialsdev(p3integrals[i]))
        eerr.append((V2*Banorm).partialsdev(partials[i]))
        ints.append(p3integrals[i])
    return(Vcss2,terr,eerr,bins,ints,partials)

##################################################################################################

def total(Cleo1V2,Cleo2V2,BESV2,BaBarV2,C1ints,C2ints,BESints,BaBarints,Cpars,BESpars,BaBarpars):
    total = []
    total.extend(np.array(Cleo1V2))
    total.extend(np.array(Cleo2V2))
    total.extend(np.array(BESV2))
    total.extend(np.array(BaBarV2))
    #av = gv.sqrt(lsqfit.wavg(total)/cor2)
    av = gv.sqrt(lsqfit.wavg([lsqfit.wavg(Cleo1V2),lsqfit.wavg(Cleo2V2),lsqfit.wavg(BESV2),lsqfit.wavg(BaBarV2)*Banorm])/cor2)
    print('B) Average V_cs from q^2 bins = ', av)
    ints = []
    pars = []
    ints.extend(C1ints)
    ints.extend(C2ints)
    ints.extend(BESints)
    ints.extend(BaBarints)
    pars.extend(Cpars)
    pars.extend(BESpars)
    pars.extend(BaBarpars)
    pars.append(Banorm)
    terr = av.partialsdev(tuple(ints))
    eerr = av.partialsdev(tuple(pars))
    cerr = av.partialsdev(cor2)
    print('Theory error = {0:.4f} Exp error = {1:.4f} Correction error = {2:.4f} Total = {3:.4f}'.format(terr,eerr,cerr,np.sqrt(eerr**2+terr**2+cerr**2)))
    return(av)

##################################################################################################
def comp(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2):
    Cleo1V2,Cleo1terr,Cleo1eerr,Cleo2V2,Cleo2terr,Cleo2eerr,Cleobins,C1ints,C2ints,Cpars = comp_cleo(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2)
    BESV2,BESterr,BESeerr,BESbins,BESints,BESpars = comp_BES(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2)
    BESDpV2,BESDpterr,BESDpeerr,BESDpbins,BESDpints,BESDppars = comp_BESDp(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2)
    BaBarV2,BaBarterr,BaBareerr,BaBarbins,BaBarints,BaBarpars = comp_BaBar(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2)
    Cleo1av = lsqfit.wavg(Cleo1V2)
    Cleo2av = lsqfit.wavg(Cleo2V2)
    BESav = lsqfit.wavg(BESV2)
    BESDpav = lsqfit.wavg(BESDpV2)
    BaBarav = lsqfit.wavg(BaBarV2,svdcut=0.01)*Banorm
    #avtest0 = lsqfit.wavg(BaBarV2,svdcut=0.01)
    #avtest999 = lsqfit.wavg(BaBarV2,svdcut=0.1)
    #avtest1 = lsqfit.wavg(BaBarV2)*Banorm
    #avtest2 = lsqfit.wavg(BaBarV2,svdcut=0.01)*Banorm
    #avtest3 = lsqfit.wavg(BaBarV2,svdcut=0.1)*Banorm
    #avtest4 = lsqfit.wavg(BaBarV2,svdcut=0.5)*Banorm
    #avtest5 = lsqfit.wavg(BaBarV2,svdcut=0.999)*Banorm
    #avtest0.fit.qqplot_residuals().savefig('DKPlots/QQplot_svd01.pdf')
    #plt.clf()
    #avtest999.fit.qqplot_residuals().savefig('DKPlots/QQplot_svd1.pdf')
    #print(avtest1,avtest2,avtest3,avtest4,avtest5)
    print('Cleo |V_cs| D^0 to K^- sqrt(weighted average) = ',gv.sqrt(Cleo1av/cor2))
    print('Cleo |V_cs| D^+ to K^0 sqrt(weighted average) = ',gv.sqrt(Cleo2av/cor2))
    print('BES |V_cs| D^0 to K^- sqrt(weighted average) = ',gv.sqrt(BESav/cor2))
    print('BES |V_cs| D^+ to K^-0 sqrt(weighted average) = ',gv.sqrt(BESDpav/cor2))
    print('BaBar |V_cs| D^0 to K^- sqrt(weighted average) = ',gv.sqrt(BaBarav/cor2))
    av = total(Cleo1V2,Cleo2V2,BESV2,BaBarV2,C1ints,C2ints,BESints,BaBarints,Cpars,BESpars,BaBarpars)
    d = collections.OrderedDict()
    d['Cleo1V2'] = Cleo1V2
    d['Cleo2V2'] = Cleo2V2
    d['BaBarV2'] = BaBarV2
    d['BESV2'] = BESV2
    d['BESDpV2'] = BESDpV2
    d['Cleo1av'] = Cleo1av
    d['Cleo2av'] = Cleo2av
    d['BESav'] = BESav
    d['BESDpav'] = BESDpav
    d['BaBarav'] = BaBarav
    d['Cleobins'] = Cleobins
    d['BESbins'] = BESbins
    d['BESDpbins'] = BESDpbins
    d['BaBarbins'] = BaBarbins
    d['average'] = av
    d['Cleo1terr'] = Cleo1terr
    d['Cleo2terr'] = Cleo2terr
    d['Cleo1eerr'] = Cleo1eerr
    d['Cleo2eerr'] = Cleo2eerr
    d['BESterr'] = BESterr
    d['BESeerr'] = BESeerr
    d['BESDpterr'] = BESDpterr
    d['BESDpeerr'] = BESDpeerr
    d['BaBarterr'] = BaBarterr
    d['BaBareerr'] = BaBareerr    
    return(d)

##################################################################################################

def make_refit_fp(qsq,p,Npow):
    z = make_z(qsq,'min',p['MDphys'],p['MKphys'])
    pole = make_z(qsq,p['MDsstarphys']**2,p['MDphys'],p['MKphys']) * make_phi_fp(qsq,'min',p['MDphys'],p['MKphys'])
    fp = 0
    for n in range(Npow):
        fp += (1/pole) * p['a'][n] * z**n #
    return(fp)

##################################################################################################

def re_fit_fp(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,noise,const2):
    nopts = 20
    prior = gv.BufferDict()
    f = gv.BufferDict()
    ###### get the original fit in terms of q^2 ###############################
    p = make_p_physical_point_DK(pfit,Fits)
    for qsq in np.linspace(0,qsqmaxphysDK.mean,nopts):
        fp = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
        f['fp_qsq{0}'.format(qsq)] = fp
        ##################### make prior ############################################
    prior['MKphys'] = copy.deepcopy(p['MKphys'])
    prior['MDphys'] = copy.deepcopy(p['MDphys'])
    prior['MDsstarphys'] = copy.deepcopy(p['MDsstarphys'])
    prior['a'] = gv.gvar(Npow*['0.0(0.5)'])
    def fcn(p):
        models = gv.BufferDict()
        for qsq in np.linspace(0,qsqmaxphysDK.mean,nopts):
            models['fp_qsq{0}'.format(qsq)] = make_refit_fp(qsq,p,Npow)
        return(models)
    p0 = None
    fit = lsqfit.nonlinear_fit(data=f, prior=prior, p0=p0, fcn=fcn, svdcut=1e-3 ,noise=noise, maxit=500, tol=(1e-6,0.0,0.0),fitter='gsl_multifit', alg='subspace2D', solver='cholesky',debug=True )
    print(fit.format(maxline=True))
    a0 = fit.p['a'][0]
    a1 = fit.p['a'][1]
    a2 = fit.p['a'][2]
    print('exp a0 = ',a0,'a1 = ',a1, 'a2 = ',a2,'a1/a0 = ',a1/a0,'a2/a0 = ',a2/a0)
    print('Cov:')
    print(gv.evalcov([a1/a0,a2/a0]))
    return(p,fit.p)

##################################################################################################


def output_error_DK(pfit,prior,Fits,Nijk,Npow,Nm,f,qsqs,t_0,addrho,fpf0same,const2):
    p = make_p_physical_point_DK(pfit,Fits)
    f0dict = collections.OrderedDict()
    fpdict = collections.OrderedDict()
    for i in range(1,5):
        f0dict[i] = []
        fpdict[i] = []
    disclist = []
    qmislist = []
    #heavylist = []
    dat = []
    extinputs = [prior['g'],prior['Metacphys'],prior['MDphys'],prior['MKphys'],prior['MDsstarphys'],prior['MDs0phys'],prior['slratio']]
    for Fit in Fits:
        extinputs.append(prior['Metac_{0}'.format(Fit['conf'])])
    for n in range(Npow):
        #if addrho:
        #    heavylist.append(prior['0rho'][n])
        qmislist.append(prior['0cl'][n])
        qmislist.append(prior['0cs'][n])
        qmislist.append(prior['0cc'][n])
        qmislist.append(prior['0csval'][n])
        qmislist.append(prior['0clval'][n])
        #if addrho:
        #    heavylist.append(prior['prho'][n])
        qmislist.append(prior['pcl'][n])
        qmislist.append(prior['pcs'][n])
        qmislist.append(prior['pcc'][n])
        qmislist.append(prior['pcsval'][n])
        qmislist.append(prior['pclval'][n])
        
        for i in range(Nijk[0]):
            for j in range(Nijk[1]):
                for k in range(Nijk[2]):
                    if j != 0 or k != 0:
                        disclist.append(prior['0d'][i][j][k][n])
                        disclist.append(prior['pd'][i][j][k][n])
                    #else:
                    #    heavylist.append(prior['0d'][i][j][k][n])
                    #    heavylist.append(prior['pd'][i][j][k][n])
    for key in prior:
        if not isinstance(prior[key],(list,tuple,np.ndarray)):
            if prior[key] not in disclist + qmislist  + extinputs:  # + heavylist:
                dat.append(prior[key])
    for key in f:
        dat.append(f[key])
    Fit = Fits[0]
    mass = Fit['masses'][0]
    fit = Fit['conf']
    for qsq in qsqs:
        f0 = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fit,qsq,t_0,mass,fpf0same,0)
        fp = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fit,qsq,t_0,mass,fpf0same,0,const2=const2)
        var1 = (100*(f0.partialsdev(tuple(extinputs)))/f0.mean)**2
        var2 = (100*(f0.partialsdev(tuple(qmislist)))/f0.mean)**2
        var3 = (100*(f0.partialsdev(tuple(dat)))/f0.mean)**2
        #var4 = (100*(f0.partialsdev(tuple(heavylist)))/f0.mean)**2
        var4 = (100*(f0.partialsdev(tuple(disclist)))/f0.mean)**2
        f0dict[1].append(var1)
        f0dict[2].append(var1+var2)
        f0dict[3].append(var1+var2+var3)
        f0dict[4].append(var1+var2+var3+var4)
        #f0dict[5].append(var1+var2+var3+var4+var5)
        var1 = (100*(fp.partialsdev(tuple(extinputs)))/fp.mean)**2
        var2 = (100*(fp.partialsdev(tuple(qmislist)))/fp.mean)**2
        var3 = (100*(fp.partialsdev(tuple(dat)))/fp.mean)**2
        #var4 = (100*(fp.partialsdev(tuple(heavylist)))/fp.mean)**2
        var4 = (100*(fp.partialsdev(tuple(disclist)))/fp.mean)**2
        fpdict[1].append(var1)
        fpdict[2].append(var1+var2)
        fpdict[3].append(var1+var2+var3)
        fpdict[4].append(var1+var2+var3+var4)
        #fpdict[5].append(var1+var2+var3+var4+var5)
    return(f0dict,fpdict)

#######################################################################################################

























































###########################Do stuff below here check stuff is for BK and t_0*a etc etc#######################################################
######################################################################################################

def ratio_fp_B_D_BsEtas(pfit,Fits,Nijk,Npow,addrho,fpf0same,t_0):
    p = make_p_Mh_BsEtas(pfit,Fits,MBsphys)
    z = make_z(0,t_0,MBsphys,Metasphys)
    fpBs = make_fp_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,0,z.mean,Fits[0]['masses'][0],fpf0same,0)
    p = make_p_Mh_BsEtas(pfit,Fits,MDsphys)
    z = make_z(0,t_0,MDsphys,Metasphys)
    fpDs = make_fp_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,0,z.mean,Fits[0]['masses'][0],fpf0same,0)
    thmean = (MDsphys/MBsphys)**(3/2)
    therror = 3*thmean * LQCD**2 * (1/MDsphys**2 - 1/MBsphys**2)
    theory = gv.gvar('{0}({1})'.format(thmean.mean,therror.mean))
    print('f_+^(Bs)(0)/f_+^(Ds)(0) = {0} (MDs/MBs)^3/2 = {1} ratio = {2}'.format(fpBs/fpDs,theory,fpBs/(fpDs*theory)))
    return()

######################################################################################################

def make_beta_delta_BsEtas(Fits,t_0,Nijk,Npow,addrho,p,fpf0same,MH_s):
    #an = make_an_BsEtas(n,Nijk,addrho,p,tag,Fit,alat,mass,amh,f0fpsame)
    z0 = make_z(0,t_0,MH_s,Metasphys).mean
    zHsstar = make_z(((MH_s+x/MH_s)**2).mean,t_0,MH_s,Metasphys).mean
    Fit = Fits[0]
    mass = Fit['masses'][0]
    fit = Fit['conf']
    fp0 = make_fp_BsEtas(Nijk,Npow,addrho,p,Fit,0,0,z0,mass,fpf0same,0)
    fpHsstar = make_an_BsEtas(0,Nijk,addrho,p,'p',Fit,0,mass,0,fpf0same) 
    f00 = make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,0,0,z0,mass,fpf0same,0)
    t_plus = (MH_s + Metasphys)**2
    t_nought = make_t_0(t_0,MH_s,MKphys)
    zprime = (-1) / (2* (t_plus + gv.sqrt( t_plus * (t_plus - t_nought) ) ) )
    f0prime = (f00/p['MHs0_{0}_m{1}'.format(fit,mass)]**2)
    fpprime = (fp0/p['MHsstar_{0}_m{1}'.format(fit,mass)]**2)
    for n in range(1,Npow):
        f0prime += zprime * n * make_an_BsEtas(n,Nijk,addrho,p,'0',Fit,0,mass,0,fpf0same) * z0**(n-1)
        fpprime += zprime * n * make_an_BsEtas(n,Nijk,addrho,p,'p',Fit,0,mass,0,fpf0same) * ( z0**(n-1) - (-1)**(n-Npow) * z0**(Npow-1))
        fpHsstar += make_an_BsEtas(n,Nijk,addrho,p,'p',Fit,0,mass,0,fpf0same) * ( zHsstar**(n) - (n/Npow) * (-1)**(n-Npow) * zHsstar**(Npow))
    delta = 1 - ((MH_s**2-Metasphys**2)/fp0) * (fpprime-f0prime)
    invbeta = ((MH_s**2-Metasphys**2)/fp0) * f0prime
    alpha = 1 - (fp0/fpHsstar)
    if MH_s == MBsphys.mean:
        print('delta at MBs = ',delta)
        print('alpha at MBs = ',alpha)
    if MH_s == MDsphys.mean:
        print('delta at MDs = ',delta)
        print('alpha at MDs = ',alpha)
    return(alpha,delta,invbeta)


#####################################################################################################

def eval_at_different_spacings_BsEtas(asfm,pfit,Fits,fpf0same,Npow,Nijk,addrho):
    #asfm is a list of lattice spacings in fm
    Fit = Fits[0]
    mass = Fit['masses'][0]
    fit = Fit['conf']
    p = make_p_physical_point_BsEtas(pfit,Fits)
    forchris = collections.OrderedDict()
    forchris['M_B_s^*'] = p['MHsstar_{0}_m{1}'.format(fit,mass)]
    forchris['M_B_s^0'] = p['MHs0_{0}_m{1}'.format(fit,mass)]
    forchris['M_B_s'] = MBsphys
    forchris['M_eta_s'] = Metasphys
    for afm in asfm:
        for n in range(Npow):
            forchris['a_plusa{0}n{1}'.format(afm,n)] = make_an_BsEtas(n,Nijk,addrho,p,'p',Fit,convert_Gev(afm).mean,mass,convert_Gev(afm).mean*mbphys,fpf0same)
            forchris['a_0a{0}n{1}'.format(afm,n)] = make_an_BsEtas(n,Nijk,addrho,p,'0',Fit,convert_Gev(afm).mean,mass,convert_Gev(afm).mean*mbphys,fpf0same)
    gv.dump(forchris,'Tables/forchris.pickle')
    return()


