import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

plt.rc("font",**{"size":18})
import collections
import copy
import os.path
import pickle
from collections import defaultdict
import math

####################################################################################################
# Global parameters
####################################################################################################
GF = gv.gvar('1.1663787(6)*1e-5') #Gev-2 #PDG
Metasphys = gv.gvar('0.6885(22)')   # 1303.1670
Metacphys = gv.gvar('2.9766(12)')# gv.gvar('2.98390(50)')  # From Christine not PDG
Metas_C = gv.gvar('0.432855(40)')#fitted from Judd's data0.432853(42)
Metas_VC = gv.gvar('0.54024(15)') # 1408.4169
Metas_VCp = gv.gvar('0.52680(8)')
Metas_Cp = gv.gvar('0.42310(3)')#from 1408.4169 
Metas_Fp = gv.gvar('0.30480(4)')#from 1408.4169  
Metas_F = gv.gvar('0.314015(89)') #from BsEtas fits
Metas_SF = gv.gvar('0.207021(65)')#from new BsEtas fit
Metas_UF = gv.gvar('0.154107(88)') #from new BsEast fit
Metas_Fs = Metas_F#gv.gvar('0.314015(89)') #from BsEtas fits
Metas_SFs = Metas_SF#gv.gvar('0.207021(65)')#from new BsEtas fit
Metas_UFs = Metas_UF#gv.gvar('0.154107(88)') #from new BsEast fit
MKphys0 = gv.gvar('0.497611(13)') #PDG K^0 doing B0 to K0 
MKphysp = gv.gvar('0.493677(13)') #PDG K+-
MBsphys = gv.gvar('5.36688(14)') # PDG
MDsphys = gv.gvar('1.968340(70)')  #PDG
MDsstarphys = gv.gvar('2.1122(4)')  #PDG 
MBphys0 = gv.gvar('5.27965(12)') # PDG this is B0
MBphysp = gv.gvar('5.27934(12)') #PDG B+-
MDphysp = gv.gvar('1.86965(5)')  #PDG D+-
MDphys0 = gv.gvar('1.86483(5)')  #PDG D0
Mpiphys = gv.gvar('0.1349770(5)')  #PDG
MBsstarphys = gv.gvar('5.4158(15)') #PDG
#tensornorm = gv.gvar('1.09024(56)') # from Dan
w0 = gv.gvar('0.1715(9)')  #fm
hbar = gv.gvar('6.58211928(15)') # x 10^-25 GeV s
clight = 2.99792458 #*10^23 fm/s
slratio = gv.gvar('27.18(10)') #* 3/2
MetacVCp = gv.gvar('2.283452(45)')# from Bp 
MetacCp = gv.gvar('1.833947(14)')# '1.833950(18)'2005.01845  1.833947(14) from Judd's data 
MetacFp = gv.gvar('1.32929(3)')# 1.32929(3) for 0.433 from 1408.4169 can adjust to 1.327173(30) for 0.432  
MetacC = gv.gvar('1.876536(48)') #2005.01845
MetacVC = gv.gvar('2.331899(72)') #2005.01845  
MetacF = gv.gvar('1.364919(40)') # adjusted from 1.367014(40)for 0.45 not 0.449 from Mclean
MetacSF = gv.gvar('0.896675(24)')  # 2005.01845
MetacUF = gv.gvar('0.666818(39)')  #2005.01845     
deltaFVC = gv.gvar('0.018106911(16)')
deltaFVVC = gv.gvar('0.05841163(17)')
deltaFVVCp = gv.gvar('0.12907825(82)')
deltaFVCp = gv.gvar('0.04894993(12)')
deltaFVFp = gv.gvar('0.06985291(24)')
deltaFVF = gv.gvar('0.020801419(21)')#80812089(1)')
deltaFVSF = gv.gvar('0.020801419(21)')#80812089(1)')
deltaFVUF = gv.gvar('0.027538708(37)')#753275(1)') #from code Chris sent
MetacFs = MetacF#gv.gvar('1.367014(40)')        #lattice units
MetacSFs = MetacSF#gv.gvar('0.896806(48)')       #where are these from? 
MetacUFs = MetacUF#gv.gvar('0.666754(39)')       #All from Mclean 1906.00701
deltaFVFs = deltaFVF#gv.gvar('0.020801419(21)')#80812089(1)')
deltaFVSFs = deltaFVSF#gv.gvar('0.020801419(21)')#80812089(1)')
deltaFVUFs = deltaFVUF#gv.gvar('0.027538708(37)')#753275(1)') #from code Chris sent
ginf = gv.gvar('0.48(11)') #0310050
gB = (gv.gvar('0.56(8)') + gv.gvar('0.449(51)') + gv.gvar('0.492(29)'))/3# 0.56(8) 1506.06413, 0.449(51) 1109.2480  0.492(29) 1404.6951
gD = gv.gvar('0.570(6)') # 1304.5009
#x =  MBsphys*(MBsstarphys-MBsphys)  #GeV^2
#MKphys = (MKphys0 + MKphysp)/2  #decide which to choose
#MBphys = (MBphys0 + MBphysp)/2 #decide which to choose
#MDphys = (MDphys0 + MDphysp)/2
tauB0GeV = gv.gvar('1.519(4)')/(6.582119569*1e-13)#1909.12524
tauBpmGeV = gv.gvar('1.638(4)')/(6.582119569*1e-13)#1909.12524
m_e = gv.gvar('0.5109989461(31)*1e-3').mean#GeV
m_mu = gv.gvar('105.6583745(24)*1e-3').mean#GeV
m_tau = gv.gvar('1.77686(12)')
m_tau = gv.gvar('1.77686(12)').mean
LQCD = 0.5
mbphys = gv.gvar('4.18(04)') # b mass GeV
qsqmaxphys = (MBsphys-Metasphys)**2
qsqmaxphysBK0 = (MBphys0-MKphys0)**2 # This is the smaller of the two
qsqmaxphysBKp = (MBphysp-MKphysp)**2
qsqmaxphysBK = (MBphys0+MBphysp-MKphys0-MKphysp)**2/4
print('q^2 B^0,B^+:',qsqmaxphysBK0,qsqmaxphysBKp)
qsqmaxphysDK = (MDphys0+MDphysp-MKphysp-MKphys0)**2/4 # This is smaller 
Del = 0.45 # 0.4 +0.5
mlmsfac = 5.63  #gives ml/(mlmsfac*ms) 10 originally, now 5.63
Z_T_running = gv.gvar('1.0773(17)')#calcuated to run from m_b to 2GeV using evolvetest.py
#####################################################################################################
############################### Other data #########################################################
dataf0maxBK = None  #only works for BsEtas for now
datafpmaxBK = None
datafTmaxBK = None
dataf00BK = None
dataf0max2BsEtas = None # gv.gvar('0.816(35)') #chris 1406.2279
# from Bs etas paper. Correlate these
dataf0maxBsEtas = gv.gvar('0.808(15)')  
datafpmaxBsEtas =  gv.gvar('2.58(28)') 
dataf00BsEtas =  gv.gvar('0.296(25)') 
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
    mean = gv.gvar(['0.9493(42)','0.9740(43)','1.0029(43)','1.0342(43)','1.0476(42)'])
    corr = [[1.0,0.99750,0.99854,0.99475,0.93231],[0.99750,1.0,0.99777,0.99430,0.93294],[0.99854,0.99777,1.0,0.99605,0.93562],[0.99475,0.99430,0.99605,1.0,0.93197],[0.93213,0.93294,0.93562,0.93197,1.0]]
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
            #print(Fit['conf'],(np.sqrt(3)*np.pi*float(twist)/Fit['L'])**2)
            daughters.append(Fit['daughter-Tag'][i].format(twist))
        Fit['daughter-Tag'] = daughters
    return()    

####################################################################################################

def get_results(Fit_in,thpts,Fit_s=None):
    if Fit_in['conf'] in ['Fs','SFs']:
        return()
    elif Fit_s != None:
        fits = [Fit_in,Fit_s]
    else:
        fits = [Fit_in]
    p = gv.load(Fit_in['filename'],method='pickle')
    for Fit in fits:
        # We should only need goldstone masses and energies here
        Fit['M_parent_m{0}'.format(Fit['m_c'])] = p['dE:{0}'.format(Fit['parent-Tag'].format(Fit['m_s'],Fit['m_c']))][0]
        for mass in Fit['masses']:
            Fit['M_parent_m{0}'.format(mass)] = p['dE:{0}'.format(Fit['parent-Tag'].format(Fit['m_s'],mass))][0]
            NGtag = '{0}G5T-G5T{1}'.format((Fit['parent-Tag'].format(Fit['m_s'],mass)).split('G5-G5')[0],(Fit['parent-Tag'].format(Fit['m_s'],mass)).split('G5-G5')[1])
            Fit['GNGsplit_m{0}'.format(mass)] = p['dE:{0}'.format(NGtag)][0] - Fit['M_parent_m{0}'.format(mass)] 
        Fit['M_daughter'] = p['dE:{0}'.format(Fit['daughter-Tag'][0])][0]
        if 'log(dE:K_G5-G5X_tw0)' in p and Fit['conf'] not in ['Fs','SFs']:
            #print(Fit['conf'],'Kaon G5X - G5 =',p['dE:K_G5-G5X_tw0'][0]-Fit['M_daughter'])
            Fit['Ksplit'] = p['dE:K_G5-G5X_tw0'][0]-Fit['M_daughter']
        for t,twist in enumerate(Fit['twists']):
            #Fit is the actual measured value, theory is obtained from the momentum
            Fit['E_daughter_tw{0}_fit'.format(twist)] = p['dE:{0}'.format(Fit['daughter-Tag'][t])][0]
            Fit['E_daughter_tw{0}_theory'.format(twist)] = gv.sqrt(Fit['M_daughter']**2+Fit['momenta'][t]**2)
            for m, mass in enumerate(Fit['masses']):
                for thpt in thpts[Fit['conf']]:
                    if twist != '0' or thpt != 'T':
                        if Fit['conf'] in  ['UFs'] and '{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist) in p: # were not extracted with 2 in them
                            Fit['{0}_m{1}_tw{2}'.format(thpt,mass,twist)] = 2 * 2 * Fit['Zdisc'][m] * gv.sqrt(Fit['M_parent_m{0}'.format(mass)]*Fit['E_daughter_tw{0}_theory'.format(twist)]) * p['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)][0][0]
                            Fit['unnorm_{0}_m{1}_tw{2}'.format(thpt,mass,twist)] = 2 * 2 * Fit['Zdisc'][m] *  p['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)][0][0]
                            #print(Fit['conf'],thpt,mass,twist,gv.evalcorr([p['{0}Vnn_m{1}_tw{2}'.format(thpt,Fit['masses'][0],Fit['twists'][1])][0][0],p['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)][0][0]])[0][1])
                        elif Fit['conf'] in ['Fs','SFs'] and '{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist) in p:
                            Fit['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)] = p['{0}sVnn_m{1}_tw{2}'.format(thpt,mass,twist)][0][0]
                            Fit['{0}_m{1}_tw{2}'.format(thpt,mass,twist)] =  2 * Fit['Zdisc'][m] * gv.sqrt(Fit['M_parent_m{0}'.format(mass)]*Fit['E_daughter_tw{0}_theory'.format(twist)]) * Fit['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)]
                            Fit['unnorm_{0}_m{1}_tw{2}'.format(thpt,mass,twist)] =  2 * Fit['Zdisc'][m] * Fit['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)]
                        elif '{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist) in p:
                            Fit['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)] = p['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)][0][0]
                            Fit['{0}_m{1}_tw{2}'.format(thpt,mass,twist)] =  2 * Fit['Zdisc'][m] * gv.sqrt(Fit['M_parent_m{0}'.format(mass)]*Fit['E_daughter_tw{0}_theory'.format(twist)]) * Fit['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)]
                            Fit['unnorm_{0}_m{1}_tw{2}'.format(thpt,mass,twist)] =  2 * Fit['Zdisc'][m] *  Fit['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)]
                            #print(Fit['conf'],thpt,mass,twist,gv.evalcorr([p['{0}Vnn_m{1}_tw{2}'.format(thpt,Fit['masses'][0],Fit['twists'][1])][0][0],p['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)][0][0]])[0][1])
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
    MDphysav = (p['MDphys0'] + p['MDphysp'])/2
    MBphysav = (p['MBphys0'] + p['MBphysp'])/2
    if a == 0:
        a = 1.0
    DeltaD = p['MDsstarphys'] - MDphysav
    DeltaB = p['MBsstarphys'] - MBphysav
    MHsstar = MH + a**2 * MDphysav * DeltaD/MH + a*MBphysav/MH * ( (MH-a*MDphysav)/(MBphysav-MDphysav) * (DeltaB - MDphysav/MBphysav * DeltaD) )
    return(MHsstar)
    
####################################################################################################

def make_fs(Fit,fs,thpts,Z_T):
    for m,mass in enumerate(Fit['masses']):
        Z_v = (float(mass) - float(Fit['m_s']))*Fit['S_m{0}_tw0'.format(mass)]/((Fit['M_parent_m{0}'.format(mass)] - Fit['M_daughter']) * Fit ['V_m{0}_tw0'.format(mass)])
        fs['Z_v_m{0}'.format(mass)] = Z_v
        print(Fit['conf'],mass,'Z_V = ',Z_v)
        for t,twist in enumerate(Fit['twists']):
            momi = Fit['momenta'][t]/np.sqrt(3)
            delta = (float(mass) - float(Fit['m_s']))*(Fit['M_parent_m{0}'.format(mass)]-Fit['E_daughter_tw{0}_theory'.format(twist)])
            qsq = (Fit['M_parent_m{0}'.format(mass)]-Fit['E_daughter_tw{0}_theory'.format(twist)])**2 - Fit['momenta'][t]**2
            f0 = ((float(mass) - float(Fit['m_s']))*(1/(Fit['M_parent_m{0}'.format(mass)]**2 - Fit['M_daughter']**2))*Fit['S_m{0}_tw{1}'.format(mass,twist)])
            
            A = Fit['M_parent_m{0}'.format(mass)] + Fit['E_daughter_tw{0}_theory'.format(twist)]
            A2 =   - momi
            B = (Fit['M_parent_m{0}'.format(mass)]**2 - Fit['M_daughter']**2)*(Fit['M_parent_m{0}'.format(mass)] - Fit['E_daughter_tw{0}_theory'.format(twist)])/qsq
            B2 =  (Fit['M_parent_m{0}'.format(mass)]**2 - Fit['M_daughter']**2)*( momi )/qsq
            fp = None
            fp2 = None
            fT = None
            #print(mass,twist,'| A-B',A-B,'A2-B2',A2-B2)
            if twist != '0':
                fp = (1/(A-B))*(Z_v*Fit['V_m{0}_tw{1}'.format(mass,twist)] - B*f0)
                if 'T' in thpts[Fit['conf']]:
                    fT = Z_T[Fit['conf']]*Fit['T_m{0}_tw{1}'.format(mass,twist)]*(Fit['M_parent_m{0}'.format(mass)]+Fit['M_daughter'])/(2*Fit['M_parent_m{0}'.format(mass)]*momi)
                    #q0 = Fit['M_parent_m{0}'.format(mass)]-Fit['E_daughter_tw{0}_theory'.format(twist)]
                    #ZVV0 = Z_v * Fit ['V_m{0}_tw{1}'.format(mass,twist)]
                    #pxZV = momi * Z_v * 3 # we have x y and z 
                    #RHS = (float(mass) - float(Fit['m_s'])) * Fit['S_m{0}_tw{1}'.format(mass,twist)]
                    #print(q0,ZVV0,pxZV,RHS,q0*ZVV0,)
                    #V_xfromV_0 = -(q0 * ZVV0 - RHS)/pxZV
                    #print('V_x from V_0',V_xfromV_0) # This is from the Z_V expression. 
                    
            if 'X_m{0}_tw{1}'.format(mass,twist) in Fit:
                fp2 = (1/(A2-B2))*(-1*Z_v*Fit['X_m{0}_tw{1}'.format(mass,twist)] - B2*f0)
                print('fp0: ',fp,'fp1: ',fp2,'fp1/fp0: ',fp2/fp)
                XV0 = (A2-B2)/(A-B) * (Z_v*Fit['V_m{0}_tw{1}'.format(mass,twist)] - B*f0) + f0*B2
                print('X = ', Fit['X_m{0}_tw{1}'.format(mass,twist)],'X from V^0 = ',-XV0)
            fs['qsq_m{0}_tw{1}'.format(mass,twist)] = qsq
            fs['f0_m{0}_tw{1}'.format(mass,twist)] = f0
            fs['fp_m{0}_tw{1}'.format(mass,twist)] = fp
            fs['fp2_m{0}_tw{1}'.format(mass,twist)] = fp2
            fs['fT_m{0}_tw{1}'.format(mass,twist)] = fT
            fs['S_m{0}_tw{1}'.format(mass,twist)] = Fit['unnorm_S_m{0}_tw{1}'.format(mass,twist)]
            if twist !='0':
                fs['ZVV_m{0}_tw{1}'.format(mass,twist)] = Fit['unnorm_V_m{0}_tw{1}'.format(mass,twist)] * Fit['unnorm_S_m{0}_tw0'.format(mass)]/Fit['unnorm_V_m{0}_tw0'.format(mass)]
                if 'T' in thpts[Fit['conf']]:
                    fs['T_m{0}_tw{1}'.format(mass,twist)] = Z_T[Fit['conf']] * Fit['unnorm_T_m{0}_tw{1}'.format(mass,twist)]
                else:
                    fs['T_m{0}_tw{1}'.format(mass,twist)] = None
            else:
                fs['ZVV_m{0}_tw{1}'.format(mass,twist)] = None
                fs['T_m{0}_tw{1}'.format(mass,twist)] = None
            if 'unnorm_X_m{0}_tw{1}'.format(mass,twist) in Fit:
                fs['ZVX_m{0}_tw{1}'.format(mass,twist)] = Fit['unnorm_X_m{0}_tw{1}'.format(mass,twist)] * Fit['unnorm_S_m{0}_tw0'.format(mass)]/Fit['unnorm_V_m{0}_tw0'.format(mass)]
            else: 
                fs['ZVX_m{0}_tw{1}'.format(mass,twist)] =  None
    #for mass in Fit['masses']:
    #    m1 = Fit['masses'][0]
    #    m2 = mass
    #    twist = Fit['twists'][1]
    #    print(Fit['conf'],m1,m2)
    #    print('---------')
    #    print('f0',gv.evalcorr([fs['f0_m{0}_tw{1}'.format(m1,twist)],fs['f0_m{0}_tw{1}'.format(m2,twist)]])[0][1])
    #    print('fp',gv.evalcorr([fs['fp_m{0}_tw{1}'.format(m1,twist)],fs['fp_m{0}_tw{1}'.format(m2,twist)]])[0][1])
    #    print('---------')
    return()

######################################################################################################

def make_function(p,Fit,Nijk,Npow,Nm,addrho,t_0,mass,twist,fpf0same,const2,element=None):
    fit = Fit['conf']
    t = Fit['twists'].index(twist)
    mh = float(mass)
    ms = float(Fit['m_s'])
    mom = Fit['momenta'][t]
    momi  = mom/np.sqrt(3)
    MH = p['MH_{0}_m{1}'.format(fit,mass)]
    MK = p['MK_{0}'.format(fit)]
    EK = gv.sqrt(MK**2+mom**2)
    qsq = (MH-EK)**2 - mom**2
    f0z = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fit,qsq,t_0,mass,fpf0same,mh) #second mass is amh
    fpz = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fit,qsq,t_0,mass,fpf0same,mh,const2=const2)
    fTz = make_fT_BK(Nijk,Npow,Nm,addrho,p,Fit,qsq,t_0,mass,fpf0same,mh)
    A = MH + EK
    A2 = -momi # why is this -ve? Because should be -ve in B2, so probably an overall - from the imaginary part
    B = ((MH**2-MK**2)/qsq) * (MH - EK)
    B2  = ((MH**2-MK**2)/qsq) * (momi)
    if element == 'S':
        S = 1/gv.sqrt(MH*EK) * f0z * (MH**2 - MK**2)/(mh-ms)  # first term accounts for normalisation
        #if fit == 'UF':
        #    S = 0.995*S
        return(S)
    elif element == 'V':
        V  = 1/gv.sqrt(MH*EK) * (MH - MK)/(mh-ms) * ( fpz * (A - B) + f0z * B )   # second term is from remainder of ZV
        return(V)
    elif element == 'X':
        X  = -1/gv.sqrt(MH*EK) * (MH - MK)/((mh-ms) * (1 + p['{0}_A_V1_m{1}'.format(fit,mass)]*(mom)**2)) * ( fpz * (A2 - B2) + f0z * B2 )# include Z_V disc term on top of Z_V
        #X  = -1/gv.sqrt(MH*EK) * (MH - MK)/(mh-ms) * ( fpz * (A2 - B2) + f0z * B2 )# don't include Z_V disc term on top of Z_V
        return(X)
    if element == 'T':
        T = 1/gv.sqrt(MH*EK) * fTz * (2 * MH * momi)/(MH+MK)  # first term accounts for normalisation
        return(T)
#######################################################################################################
def make_t_plus(M_H,M_K): # This should ALWAYS be M_H,M_K because it is sea mass based
    t_plus = (M_H + M_K)**2
    return(t_plus)
#####################################################################################################
def make_t_0(t0,M_H,M_K,M_parent,M_daughter): # remove opt_a
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
    if qsq == t_0:
        z = 0
    elif z.mean == 0 and z.sdev == 0:
        z = 0 #gv.gvar(0,1e-10) # ensures not 0(0)
    return(z)

######################################################################################################

def check_poles(Fits):
    #Not working currently
    #plt.figure()
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

def make_prior_BK(fs_data,Fits,addrho,t_0,Npow,Nijk,Nm,rhopri,dpri,cpri,cvalpri,d0000npri,di0000pri,di100npri,d000lnpri,adddata,constraint,w=1.0):
    prior = gv.BufferDict()
    f = gv.BufferDict()
    prior['ginf'] = ginf#gv.gvar('0.496(64)') #we construct g using g = c0 + c1 Lamda/MH + c2 Lambda/MH
    prior['c1'] = gv.gvar('0.5(1.0)')#gv.gvar('0.47(64)')
    prior['c2'] = gv.gvar('0.0(3.0)')#gv.gvar('-0.7(2.1)')
    prior['Metacphys'] = Metacphys
    prior['MDphys0'] = MDphys0
    prior['MKphys0'] = MKphys0
    prior['MBphys0'] = MBphys0
    prior['MDphysp'] = MDphysp
    prior['MKphysp'] = MKphysp
    prior['MBphysp'] = MBphysp
    prior['MDsstarphys'] =  MDsstarphys
    #prior['MDs0phys'] = MDs0phys
    prior['Metasphys'] = Metasphys
    prior['MBsphys'] = MBsphys
    prior['MDsphys'] = MDsphys
    prior['MBsstarphys'] =  MBsstarphys
    prior['slratio'] = slratio
    for Fit in Fits:
        fit = Fit['conf']
        prior['a_{0}'.format(fit)] = Fit['a']
        prior['LQCD_{0}'.format(fit)] = LQCD*Fit['a']#have to convert this now so can evaluate in GeV later
        ms0 = Fit['m_ssea']
        ml0 = Fit['m_lsea'] #becuase valuence and sea same, only use one
        ms0val = float(Fit['m_s']) # valence untuned s mass
        ml0val = float(Fit['m_l']) # valence untuned s mass
        Metas = globals()['Metas_{0}'.format(fit)]/Fit['a'] # in GeV
        prior['Metac_{0}'.format(fit)] = globals()['Metac{0}'.format(fit)]/Fit['a'] #in GeV
        prior['Metas_{0}'.format(fit)] = globals()['Metas_{0}'.format(fit)] #in lat
        prior['deltaFV_{0}'.format(fit)] = globals()['deltaFV{0}'.format(fit)]
        prior['mstuned_{0}'.format(fit)] = ms0val*(prior['Metasphys']/Metas)**2
        prior['ml10ms_{0}'.format(fit)] = ml0val/(mlmsfac*prior['mstuned_{0}'.format(fit)])
        #print(prior['ml10ms_{0}'.format(fit)]-1/(mlmsfac*prior['slratio']))
        prior['mltuned_{0}'.format(fit)] = prior['mstuned_{0}'.format(fit)]/prior['slratio'] 
        prior['MD_{0}'.format(fit)] = Fit['M_parent_m{0}'.format(Fit['m_c'])] #lat units think about this for s
        prior['deltas_{0}'.format(fit)] = ms0-prior['mstuned_{0}'.format(fit)]     
        prior['deltasval_{0}'.format(fit)] = ms0val-prior['mstuned_{0}'.format(fit)]
        #prior['deltalval_{0}'.format(fit)] = ml0val-prior['mltuned_{0}'.format(fit)]
        prior['deltal_{0}'.format(fit)] = ml0-prior['mltuned_{0}'.format(fit)]
        prior['MK_{0}'.format(fit)] = Fit['M_daughter']
        #if fit in ['Fs','SFs','UFs']:
        #    prior['MKaon_{0}'.format(fit)] = prior['MK_{0}'.format(fit.split('s')[0])]
        for mass in Fit['masses']:
            prior['MH_{0}_m{1}'.format(fit,mass)] = Fit['M_parent_m{0}'.format(mass)]
            #if fit in ['Fs','SFs','UFs']:
             #   mass2 = mass
             #   if fit == 'SFs' and mass == '0.450':
             #       mass2 = '0.45'
             #   prior['Ml_{0}_m{1}'.format(fit,mass)] = prior['MH_{0}_m{1}'.format(fit.split('s')[0],mass2)] 
            for twist in Fit['twists']:
                tag = '{0}_m{1}_tw{2}'.format(fit,mass,twist)
                f['S_{0}'.format(tag)] = fs_data[fit]['S_m{0}_tw{1}'.format(mass,twist)]
                f['V_{0}'.format(tag)] = fs_data[fit]['ZVV_m{0}_tw{1}'.format(mass,twist)]
                f['X_{0}'.format(tag)] = fs_data[fit]['ZVX_m{0}_tw{1}'.format(mass,twist)]
                f['T_{0}'.format(tag)] = fs_data[fit]['T_m{0}_tw{1}'.format(mass,twist)]
                #qsq = fs_data[fit]['qsq_m{0}_tw{1}'.format(mass,twist)]
                #prior['qsq_{0}'.format(tag)] = qsq
                #f['f0_{0}'.format(tag)] = fs_data[fit]['f0_m{0}_tw{1}'.format(mass,twist)]   # y values go in f   
                #f['fp_{0}'.format(tag)] = fs_data[fit]['fp_m{0}_tw{1}'.format(mass,twist)]
                #f['fT_{0}'.format(tag)] = fs_data[fit]['fT_m{0}_tw{1}'.format(mass,twist)]
        f['gB'] = gB
        f['gD'] = gD
    if constraint:
        f['constraint'] = gv.gvar(0,1e-4) #1e-5
        #f['constraint'] = make_f0_BK(Nijk,Npow,Nm,addrho,prior,Fits[0],0,t_0,Fits[0]['masses'][0],False,0,const=True)
    if adddata: #not fot fT at the moment Have a big think about if this works
        f['f0_qsq{0}'.format(qsqmaxphys)] = dataf0maxBsEtas # onlyone BsEtas works
        #f['f0_qsq{0}2'.format(qsqmaxphys)] = dataf0max2BsEtas # onlyone BsEtas works
        f['fp_qsq{0}'.format(qsqmaxphys)] = datafpmaxBsEtas
        f['f0_qsq{0}'.format(0)] = dataf00BsEtas
        prior['qsq_qsq{0}'.format(qsqmaxphys)] = qsqmaxphys
        #prior['z_qsq{0}'.format(qsqmaxphys)] = make_z(qsqmaxphys,t_0,MBphys,MKphys,MBsphys,Metasphys)
        #prior['z_qsq{0}'.format(0)] = make_z(0,t_0,MBphys,MKphys,MBsphys,Metasphys)
        #prior['MBsphys'] = MBsphys        
        #prior['MBs0phys'] = MBsphys + Del
        #prior['MBsstarphys'] = MBsstarphys
        #prior['MDsphys'] = MDsphys 
    # remove unwanted keys from f
    keys = []
    for key in f:
        if f[key] == None:
            keys.append(key)   #removes vector tw 0 etc
    for key in keys:
        del f[key]
    fpdisc = 1.0 # for increasing disc effects in f_+
    #prior['0mom'] =gv.gvar(Npow*['0(1)'])*w
    #prior['pmom'] =gv.gvar(Npow*['0(1)'])*w
    #prior['Tmom'] =gv.gvar(Npow*['0(1)'])*w
    wdisc = w
    wndisc = w# chnage this to decide where w used in empirical Bayes
    if addrho:
        prior['0rho'] =gv.gvar(Npow*[rhopri])*wndisc
        prior['prho'] =gv.gvar(Npow*[rhopri])*wndisc
        prior['Trho'] =gv.gvar(Npow*[rhopri])*wndisc
    prior['0d'] = gv.gvar(Nijk[0]*[Nijk[1]*[Nijk[2]*[Nijk[3]*[Npow*[dpri]]]]])*wndisc
    prior['0cs'] = gv.gvar(Npow*[cpri])
    prior['0cl'] = gv.gvar(Npow*[cpri])
    prior['0cc'] = gv.gvar(Npow*[cpri])
    prior['0csval'] = gv.gvar(Npow*[cvalpri])
    #prior['0clval'] = gv.gvar(Nm*[Npow*[cvalpri]])*w
    prior['Td'] = gv.gvar(Nijk[0]*[Nijk[1]*[Nijk[2]*[Nijk[3]*[Npow*[dpri]]]]])*wndisc
    prior['Tcs'] = gv.gvar(Npow*[cpri])
    prior['Tcl'] = gv.gvar(Npow*[cpri])
    prior['Tcc'] = gv.gvar(Npow*[cpri])
    prior['Tcsval'] = gv.gvar(Npow*[cvalpri])
    #prior['Tclval'] = gv.gvar(Nm*[Npow*[cvalpri]])*w
    prior['pd'] = gv.gvar(Nijk[0]*[Nijk[1]*[Nijk[2]*[Nijk[3]*[Npow*['0.0({0})'.format(gv.gvar(dpri).sdev*fpdisc)]]]]])*wndisc
    prior['SF_A_V1_m0.6'] = gv.gvar('0.0(1)')# coefficent of ap^2 in Z_V for V^1 case
    prior['SF_A_V1_m0.8'] = gv.gvar('0.0(1)')
    prior['UF_A_V1_m0.45'] = gv.gvar('0.0(1)')
    prior['UF_A_V1_m0.6'] = gv.gvar('0.0(1)')
    prior['UF_A_V1_m0.8'] = gv.gvar('0.0(1)')
    for i in range(Nijk[0]):
        if i != 0:
            prior['0d'][i][0][0][0][0] = gv.gvar(di0000pri)*wndisc
            prior['pd'][i][0][0][0][0] = gv.gvar(di0000pri)*wndisc
            prior['Td'][i][0][0][0][0] = gv.gvar(di0000pri)*wndisc
            for n in range(Npow):
                #prior['0d'][i][1][0][0][n] = gv.gvar(di100npri)*wdisc
                #prior['pd'][i][1][0][0][n] = gv.gvar(di100npri)*wdisc
                #prior['Td'][i][1][0][0][n] = gv.gvar(di100npri)*wdisc
                prior['0d'][0][0][0][0][n] = gv.gvar(d0000npri)*wndisc 
                prior['pd'][0][0][0][0][n] = gv.gvar(d0000npri)*wndisc
                prior['Td'][0][0][0][0][n] = gv.gvar(d0000npri)*wndisc
    for i in range(Nijk[0]):
        for l in range(Nijk[3]):
            for n in range(Npow):
                prior['0d'][i][1][0][l][n] = gv.gvar(di100npri)*wdisc 
                prior['pd'][i][1][0][l][n] = gv.gvar(di100npri)*wdisc
                prior['Td'][i][1][0][l][n] = gv.gvar(di100npri)*wdisc
                prior['0d'][i][0][1][l][n] = gv.gvar(di100npri)*wdisc 
                prior['pd'][i][0][1][l][n] = gv.gvar(di100npri)*wdisc
                prior['Td'][i][0][1][l][n] = gv.gvar(di100npri)*wdisc
                
    prior['pcs'] = gv.gvar(Npow*[cpri])
    prior['pcl'] = gv.gvar(Npow*[cpri])
    prior['pcc'] = gv.gvar(Npow*[cpri])
    prior['pcsval'] = gv.gvar(Npow*[cvalpri])
    ### special case #####
    #prior['0d'][0][0][0][1][0] = gv.gvar('0(1)')*w
    #prior['0d'][0][0][0][1][1] = gv.gvar('0(1)')*w
    #prior['0d'][0][0][0][2][0] = gv.gvar('0(1)')*w
    ######################
    #prior['pclval'] = gv.gvar(Nm*[Npow*[cvalpri]])*w
    #if Kwikfit == True:
    #    kwikp = gv.load('Fits/kwikfitp.pickle')
    #    prior['0d'] = kwikp['0d']
    #    prior['pd'] = kwikp['pd']
    #    prior['Td'] = kwikp['Td']
    print('wdisc = ',wdisc,'wndisc = ',wndisc)
    return(prior,f)
                
########################################################################################################

def make_an_BK(n,Nijk,Nm,addrho,p,tag,Fit,mass,amh,fpf0same,newdata=False,const=False,const2=False): # tag is 0,p,T in this way, we can set fp(0)=f0(0) by just putting 0 for n=0 alat is lattice spacing (mean) so we can use this to evaluate at different lattice spacings p is dict containing all values (prior or posterior or anything) # need to edit l valence mistuning
    fit = Fit['conf']
    an = 0
    #lvalbit = p['deltalval_{0}'.format(fit)]/(10*p['mstuned_{0}'.format(fit)])
    lvalerr = 0#p['{0}clval'.format(tag)][0][n] * lvalbit
    xpi = p['ml10ms_{0}'.format(fit)]
    xpicont = 1/(p['slratio']*mlmsfac)#continuum value
    xpithing = xpi - xpicont
    if xpithing.mean == 0 and xpithing.sdev == 0:
        xpithing = 0
    #for nm in range(1,Nm):
    #    lvalerr +=  p['{0}clval'.format(tag)][nm][n] * lvalbit**(nm+1)
    MDphysav = (p['MDphys0'] + p['MDphysp'])/2
    MBphysav = (p['MBphys0'] + p['MBphysp'])/2
    for i in range(Nijk[0]):
        for j in range(Nijk[1]):
            for k in range(Nijk[2]):
                for l in range(Nijk[3]):
                    tagsamed = tag
                    tagsamerho = tag
                    if tag == '0' or tag == 'T' or fpf0same == False:
                        pass
                    elif n == 0:
                        tagsamerho = '0'
                        if j == 0 and k == 0 and l == 0:
                            tagsamed = '0'
                    if addrho:
                        if const:
                            an += (1 + p['{0}rho'.format(tagsamerho)][n]*gv.log(MBphysav/MDphysav)) *  p['{0}d'.format(tagsamed)][i][j][k][l][n] * (LQCD/MBphysav)**int(i) * (0)**int(2*j) * (0)**int(2*k) * (0)**int(l) 
                        elif const2:
                            print('ERROR, WRONG MASS FOR s')
                            an += (1 + p['{0}rho'.format(tagsamerho)][n]*gv.log(p['MH_{0}_m{1}'.format(fit,mass)]/p['MD_{0}'.format(fit)])) *  p['{0}d'.format(tagsamed)][i][j][k][l][n] * (p['LQCD_{0}'.format(fit)]/p['MH_{0}_m{1}'.format(fit,mass)])**int(i) * (0)**int(2*j) * (0)**int(2*k) 
                        elif newdata:
                            print('Added external data in a{0}'.format(n), 'need to edit this to include quadratic quark mistunings - BROKEN')
                            an += (1 + p['{0}rho'.format(tagsamerho)][n]*gv.log(p['MBsphys']/p['MDsphys'])) *  p['{0}d'.format(tagsamed)][i][j][k][l][n] * (LQCD/p['MBsphys'])**int(i) * (0)**int(2*j) * (0)**int(2*k) * p['{0}clval'.format(tag)][0][n] * (1/10 - 1/(10*p['slratio']))
                        elif newdata == False and const == False and const2 == False :
                            an += (1 + p['{0}rho'.format(tagsamerho)][n]*gv.log(p['MH_{0}_m{1}'.format(fit,mass)]/p['MD_{0}'.format(fit)])) * (1 + (p['{0}csval'.format(tag)][n]*p['deltasval_{0}'.format(fit)] + p['{0}cs'.format(tag)][n]*p['deltas_{0}'.format(fit)] + 2*p['{0}cl'.format(tag)][n]*p['deltal_{0}'.format(fit)])/(10*p['mstuned_{0}'.format(fit)]) + lvalerr  + p['{0}cc'.format(tag)][n]*((p['Metac_{0}'.format(fit)] - p['Metacphys'])/p['Metacphys'])) * p['{0}d'.format(tagsamed)][i][j][k][l][n] * (p['LQCD_{0}'.format(fit)]/p['MH_{0}_m{1}'.format(fit,mass)])**int(i) * (amh/np.pi)**int(2*j) * (LQCD*p['a_{0}'.format(fit)]/np.pi)**int(2*k) * (xpithing)**int(l)
                           # an += (1 + p['{0}rho'.format(tagsamerho)][n]*gv.log(p['LQCD_{0}'.format(fit)]/p['MH_{0}_m{1}'.format(fit,mass)])) * (1 + (p['{0}csval'.format(tag)][n]*p['deltasval_{0}'.format(fit)] + p['{0}cs'.format(tag)][n]*p['deltas_{0}'.format(fit)] + 2*p['{0}cl'.format(tag)][n]*p['deltal_{0}'.format(fit)])/(10*p['mstuned_{0}'.format(fit)]) + lvalerr  + p['{0}cc'.format(tag)][n]*((p['Metac_{0}'.format(fit)] - p['Metacphys'])/p['Metacphys'])) * p['{0}d'.format(tagsamed)][i][j][k][l][n] * (p['LQCD_{0}'.format(fit)]/p['MH_{0}_m{1}'.format(fit,mass)])**int(i) * (amh/np.pi)**int(2*j) * (LQCD*p['a_{0}'.format(fit)]/np.pi)**int(2*k) * (xpithing)**int(l)
                            
                            
                        else:
                            print('Error in make_an_BK(): newdata = {0}, const = {1}, const2 = {2}'.format(newdata,const,const2))
                        
                    else:
                        if const:
                            an += p['{0}d'.format(tagsamed)][i][j][k][l][n] * (LQCD/MBphysav)**int(i) * (0)**int(2*j) * (0)**int(2*k) * (0)**int(l)
                        elif const2:
                            print('ERROR, WRONG MASS FOR s')
                            an += p['{0}d'.format(tagsamed)][i][j][k][l][n] * (p['LQCD_{0}'.format(fit)]/p['MH_{0}_m{1}'.format(fit,mass)])**int(i) * (0)**int(2*j) * (0)**int(2*k)
                        elif newdata:
                            print('Added external data in a{0}'.format(n),'need to edit this to work properly')
                            an += p['{0}d'.format(tagsamed)][i][j][k][l][n] * (LQCD/p['MBsphys'])**int(i) * (0)**int(2*j) * (0)**int(2*k) * p['{0}clval'.format(tag)][0][n] * (1/10 - 1/(10*p['slratio']))
                        elif newdata == False and const == False and const2 == False:
                            an += (1 + (p['{0}csval'.format(tag)][n]*p['deltasval_{0}'.format(fit)] + p['{0}cs'.format(tag)][n]*p['deltas_{0}'.format(fit)]+2*p['{0}cl'.format(tag)][n]*p['deltal_{0}'.format(fit)])/(10*p['mstuned_{0}'.format(fit)]) + lvalerr + p['{0}cc'.format(tag)][n]*((p['Metac_{0}'.format(fit)] - p['Metacphys'])/p['Metacphys'])) * p['{0}d'.format(tagsamed)][i][j][k][l][n] * (p['LQCD_{0}'.format(fit)]/p['MH_{0}_m{1}'.format(fit,mass)])**int(i) * (amh/np.pi)**int(2*j) * (LQCD*p['a_{0}'.format(fit)]/np.pi)**int(2*k) * (xpithing)**int(l)
                            
                        else:
                            print('Error in make_an_BsEtas(): newdata = {0}, const = {1}, const2 = {2}'.format(newdata,const,const2))

    return(an)

########################################################################################################
def make_g(p,mass,Fit):
    #require M_H not M_H_s
    fit = Fit['conf']
    if fit in ['Fs','SFs','UFs']:
        MH = p['MH_{0}_m{1}'.format(fit.split('s')[0],mass)]#p['Ml_{0}_m{1}'.format(fit,mass)]
    else:
        MH = p['MH_{0}_m{1}'.format(fit,mass)]  
        
    g = p['ginf'] + p['c1'] * (p['LQCD_{0}'.format(fit)]/MH) + p['c2'] * (p['LQCD_{0}'.format(fit)]/MH)**2
    return(g)

########################################################################################################

def make_logs(p,mass,Fit):
    g = make_g(p,mass,Fit)
    fit = Fit['conf']
    xpi = p['ml10ms_{0}'.format(fit)]
    Mpi2 = mlmsfac * p['ml10ms_{0}'.format(fit)] * p['Metas_{0}'.format(fit)]**2
    xK = xpi * p['MK_{0}'.format(fit)]**2/Mpi2
    Meta2 = (Mpi2 + 2*p['Metas_{0}'.format(fit)]**2)/3
    xeta = xpi * Meta2/Mpi2
    logs = 1 - ( (9/8) * g**2 * xpi * ( gv.log(xpi) + p['deltaFV_{0}'.format(Fit['conf'])])) - (1/2 + g**2 *3/4) * xK * gv.log(xK) - (1/6 + g**2/8) * xeta * gv.log(xeta)
    #print(Fit['conf'],'x_pi = ', xpi, 'x_K = ',xK,'x_eta = ',xeta,'logs = ',logs)
    #print(Fit['conf'],'x_pi*log(x_pi) = ', xpi*gv.log(xpi), 'x_K*log(x_K) = ', xK*gv.log(xK),'x_eta*log(x_eta) = ', xeta*gv.log(xeta))
    #logs = 1 
    return(logs)

########################################################################################################

def make_f0_BK(Nijk,Npow,Nm,addrho,p,Fit,qsq,t_0,mass,fpf0same,amh,newdata=False,const=False):
    MKphysav = (p['MKphys0'] + p['MKphysp'])/2
    MBphysav = (p['MBphys0'] + p['MBphysp'])/2
    fit = Fit['conf']
    amom2  =  ((qsq-p['MK_{0}'.format(fit)]**2-p['MH_{0}_m{1}'.format(fit,mass)]**2)**2/(4*p['MH_{0}_m{1}'.format(fit,mass)]**2)-p['MK_{0}'.format(fit)]**2)
    if p['a_{0}'.format(fit)] == 0:
        amom2 = 0
    f0 = 0
    logs = make_logs(p,mass,Fit)
    if fit in ['Fs','SFs','UFs']:
        if p['a_{0}'.format(fit)] != 0:
            MHs0 = p['MH_{0}_m{1}'.format(fit.split('s')[0],mass)] + p['a_{0}'.format(fit)]*Del
        else:
            MHs0 = p['MH_{0}_m{1}'.format(fit,mass)] + Del
        z  = make_z(qsq,t_0,p['MH_{0}_m{1}'.format(fit.split('s')[0],mass)],p['MK_{0}'.format(fit.split('s')[0])],p['MH_{0}_m{1}'.format(fit,mass)],p['MK_{0}'.format(fit)] )
    else:        
        if p['a_{0}'.format(fit)] != 0:
            MHs0 = p['MH_{0}_m{1}'.format(fit,mass)] + p['a_{0}'.format(fit)]*Del
        else: 
            MHs0 = p['MH_{0}_m{1}'.format(fit,mass)] + Del
        z  = make_z(qsq,t_0,p['MH_{0}_m{1}'.format(fit,mass)],p['MK_{0}'.format(fit)] )
    z0 = make_z(0,t_0,p['MH_{0}_m{1}'.format(fit,mass)],p['MK_{0}'.format(fit)]) # take this for correctly tuned quarks. Does not have to be the case. May be better to not do this? 

    pole = 1-(qsq/MHs0**2)
    f0pole0 = 1.0 
    if newdata:
        g = p['ginf'] + p['c1'] * (LQCD/MBphysav) + p['c2'] * (LQCD/MBphysav)**2
        logsBsEtas = 1 - ( (9/8) * g**2 * 1/10 * ( gv.log(1/10) ) )
        z = make_z(qsq,t_0,MBphysav,MKphysav,M_parent=p['MBsphys'])
    if const:
        #g = p['ginf'] + p['c1'] * (LQCD/p['MBphysav']) + p['c2'] * (LQCD/p['MBphysav'])**2
        #logsBK = 1 - ( (9/8) * g**2 * 1/(10*p['slratio']) * ( gv.log(1/(10*p['slratio'])) ) )
        z0const = make_z(0,t_0,MBphysav,MKphysav )
    for n in range(Npow):
        if const:
            an = make_an_BK(n,Nijk,Nm,addrho,p,'0',Fit,mass,amh,fpf0same,const=const)
            f0 += (1/f0pole0) * an * z0const**n
        elif newdata:
            an = make_an_BK(n,Nijk,Nm,addrho,p,'0',Fit,mass,amh,fpf0same,newdata=newdata)
            f0 += logsBsEtas/(1-qsq/((MBphysav+Del)**2)) * an * z**n
        elif newdata == False and const == False:
            an = make_an_BK(n,Nijk,Nm,addrho,p,'0',Fit,mass,amh,fpf0same)
            f0 += (logs/pole) * an * z**n #* (1 + p['0mom'][n]* amom2/(np.pi**2))
        else:
            print('Error in make_f0_BK(): newdata = {0}'.format(newdata))
    return(f0)

########################################################################################################

def make_fp_BK(Nijk,Npow,Nm,addrho,p,Fit,qsq,t_0,mass,fpf0same,amh,newdata=False,const=False,const2=False,optional_z=False,pole=True):
    MKphysav = (p['MKphys0'] + p['MKphysp'])/2
    MBphysav = (p['MBphys0'] + p['MBphysp'])/2
    fit = Fit['conf']
    amom2  =  ((qsq-p['MK_{0}'.format(fit)]**2-p['MH_{0}_m{1}'.format(fit,mass)]**2)**2/(4*p['MH_{0}_m{1}'.format(fit,mass)]**2)-p['MK_{0}'.format(fit)]**2)
    if p['a_{0}'.format(fit)] == 0:
        amom2 = 0
    fp = 0
    fp0 = 0
    f00 = 0
    logs = make_logs(p,mass,Fit)
    if fit in ['Fs','SFs','UFs']:
        MHsstar = make_MHsstar(p['MH_{0}_m{1}'.format(fit.split('s')[0],mass)],p,p['a_{0}'.format(fit)])
        z = make_z(qsq,t_0,p['MH_{0}_m{1}'.format(fit.split('s')[0],mass)],p['MK_{0}'.format(fit.split('s')[0])],p['MH_{0}_m{1}'.format(fit,mass)],p['MK_{0}'.format(fit)] )
    else:
        MHsstar = make_MHsstar(p['MH_{0}_m{1}'.format(fit,mass)],p,p['a_{0}'.format(fit)])
        z = make_z(qsq,t_0,p['MH_{0}_m{1}'.format(fit,mass)],p['MK_{0}'.format(fit)] )
    z0 = make_z(0,t_0,p['MH_{0}_m{1}'.format(fit,mass)],p['MK_{0}'.format(fit)]) # take at tuned mass, may be best not to
    pole = 1-(qsq/MHsstar**2)
    fppole0 = 1.0
    f0pole0 = 1.0
    if optional_z != False:
        z = optional_z
    if pole == False:
        pole = 1.0 
    if newdata:
        g = p['ginf'] + p['c1'] * (LQCD/MBphysav) + p['c2'] * (LQCD/MBphysav)**2
        logsBsEtas = 1 - ( (9/8) * g**2 * 1/10 * ( gv.log(1/10) ) )
        z = make_z(qsq,t_0,MBphysav,MKphysav,M_parent=p['MBsphys'])
    if const2:
        print('Need to feed g proper mass here')
        logsBK = 1 - ( (9/8) * p['g']**2 * 1/(10*p['slratio']) * ( gv.log(1/(10*p['slratio'])) ) )
    if const:
        #g = p['ginf'] + p['c1'] * (LQCD/p['MBphysav']) + p['c2'] * (LQCD/p['MBphysav'])**2
        #logsBK = 1 - ( (9/8) * g**2 * 1/(10*p['slratio']) * ( gv.log(1/(10*p['slratio'])) ) )
        z0const = make_z(0,t_0,MBphysav,MKphysav )
    for n in range(Npow):
        if const==True and const2 ==False and newdata ==False:
            an = make_an_BK(n,Nijk,Nm,addrho,p,'p',Fit,mass,amh,fpf0same,const=const)
            fp += (1/fppole0) * an  * (z0const**n - (n/Npow) * (-1)**(n-Npow) *  z0const**Npow)
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
            fp += (logs/pole) * an  * (z**n - (n/Npow) * (-1)**(n-Npow) *  z**Npow) #* (1 + p['pmom'][n]* amom2/(np.pi**2))
        else:
            print('Error in make_fp_BK(): newdata = {0} const = {1} const2 = {2}'.format(newdata,const,const2))
    fp = fp + f00 - fp0
    #print(fp0/f00)
    return(fp)

#########################################################################################################

def make_fT_BK(Nijk,Npow,Nm,addrho,p,Fit,qsq,t_0,mass,fpf0same,amh,newdata=False):
    fit = Fit['conf']
    amom2  =  ((qsq-p['MK_{0}'.format(fit)]**2-p['MH_{0}_m{1}'.format(fit,mass)]**2)**2/(4*p['MH_{0}_m{1}'.format(fit,mass)]**2)-p['MK_{0}'.format(fit)]**2)
    if p['a_{0}'.format(fit)] == 0:
        amom2 = 0
    fT = 0
    logs = make_logs(p,mass,Fit)
    if fit in ['Fs','SFs','UFs']:
        MHsstar = make_MHsstar(p['MH_{0}_m{1}'.format(fit.split('s')[0],mass)],p,p['a_{0}'.format(fit)])
        z = make_z(qsq,t_0,p['MH_{0}_m{1}'.format(fit.split('s')[0],mass)],p['MK_{0}'.format(fit.split('s')[0])],p['MH_{0}_m{1}'.format(fit,mass)],p['MK_{0}'.format(fit)] )
    else:
        MHsstar = make_MHsstar(p['MH_{0}_m{1}'.format(fit,mass)],p,p['a_{0}'.format(fit)])
        z = make_z(qsq,t_0,p['MH_{0}_m{1}'.format(fit,mass)],p['MK_{0}'.format(fit)] )
    pole = 1-(qsq/MHsstar**2)
    for n in range(Npow):
        if newdata:
            print('Error,trying to input new data fT')
            an = make_an_BK(n,Nijk,Nm,addrho,p,'T',Fit,mass,amh,fpf0same,newdata=newdata)
            fT += logs/(1-qsq/(p['MBsstarphys']**2)) * an  * (z**n - (n/Npow) * (-1)**(n-Npow) *  z**Npow)
        elif newdata == False:
            an = make_an_BK(n,Nijk,Nm,addrho,p,'T',Fit,mass,amh,fpf0same)
            fT += (logs/pole) * an  * (z**n - (n/Npow) * (-1)**(n-Npow) *  z**Npow)#*(1 + p['Tmom'][n]* amom2/(np.pi**2))
        else:
            print('Error in make_fT_BK(): newdata = {0}'.format(newdata))
    return(fT)

#######################################################################################################

def do_fit_BK(fs_data,adddata,Fits,f,Nijk,Npow,Nm,t_0,addrho,noise,prior,fpf0same,rhopri,dpri,cpri,cvalpri,d0000npri,di0000pri,di100npri,d000lnpri,constraint,const2):
    def fcn(p):
        MDphysav = (p['MDphys0'] + p['MDphysp'])/2
        MBphysav = (p['MBphys0'] + p['MBphysp'])/2
        models = gv.BufferDict()
        if 'constraint' in f:
            if const2:
                print('Error, const and const2')
            models['constraint'] = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,t_0,Fits[0]['masses'][0],fpf0same,0,const=True)- make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,t_0,Fits[0]['masses'][0],fpf0same,0,const=True)
            #pconst = make_p_physical_point_BK(p,Fits)
            #models['constraint'] = make_fp_BK(Nijk,Npow,Nm,addrho,pconst,Fits[0],0,t_0,Fits[0]['masses'][0],fpf0same,0) - make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,t_0,Fits[0]['masses'][0],fpf0same,0)
        if 'f0_qsq{0}'.format(qsqmaxphys) in f:
            models['f0_qsq{0}'.format(qsqmaxphys)] = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],p['qsq_qsq{0}'.format(qsqmaxphys)],t_0,Fits[0]['masses'][0],fpf0same,0,newdata=True)
      #  if 'f0_qsq{0}2'.format(qsqmaxphys) in f:
      #      models['f0_qsq{0}2'.format(qsqmaxphys)] = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,p['qsq_qsq{0}'.format(qsqmaxphys)],p['z_qsq{0}'.format(qsqmaxphys)],Fits[0]['masses'][0],fpf0same,0,newdata=True)
        if 'fp_qsq{0}'.format(qsqmaxphys) in f:
            models['fp_qsq{0}'.format(qsqmaxphys)] = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],p['qsq_qsq{0}'.format(qsqmaxphys)],t_0,Fits[0]['masses'][0],fpf0same,0,newdata=True)
      #      print('WARNING HAVE NOT MADE THIS NEW DATA WORK WITH CONST2 YET')
        if 'f0_qsq{0}'.format(0) in f:
            models['f0_qsq{0}'.format(0)] = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,t_0,Fits[0]['masses'][0],fpf0same,0,newdata=True)
            #models['f0_qsq{0}'.format(0)] = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,t_0,Fits[0]['masses'][0],fpf0same,0,newdata=True)
        models['gB'] = p['ginf'] + p['c1'] * LQCD/MBphysav + p['c2'] * (LQCD/MBphysav)**2
        models['gD'] = p['ginf'] + p['c1'] * LQCD/MDphysav + p['c2'] * (LQCD/MDphysav)**2
        for Fit in Fits:
            for mass in Fit['masses']:
                for twist in Fit['twists']:
                    tag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)
                    if 'S_{0}'.format(tag) in f:
                        models['S_{0}'.format(tag)] = make_function(p,Fit,Nijk,Npow,Nm,addrho,t_0,mass,twist,fpf0same,const2,element='S')
                    if 'V_{0}'.format(tag) in f:
                        models['V_{0}'.format(tag)] = make_function(p,Fit,Nijk,Npow,Nm,addrho,t_0,mass,twist,fpf0same,const2,element='V')
                    if 'X_{0}'.format(tag) in f:
                        models['X_{0}'.format(tag)] = make_function(p,Fit,Nijk,Npow,Nm,addrho,t_0,mass,twist,fpf0same,const2,element='X')
                    if 'T_{0}'.format(tag) in f:
                        models['T_{0}'.format(tag)] = make_function(p,Fit,Nijk,Npow,Nm,addrho,t_0,mass,twist,fpf0same,const2,element='T')
                    #if 'f0_{0}'.format(tag) in f:
                    #    models['f0_{0}'.format(tag)] = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fit,p['qsq_{0}'.format(tag)],t_0,mass,fpf0same,float(mass)) #second mass is amh
                    #if 'fp_{0}'.format(tag) in f:
                    #    models['fp_{0}'.format(tag)] = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fit,p['qsq_{0}'.format(tag)],t_0,mass,fpf0same,float(mass),const2=const2) #second mass is amh
                    #if 'fT_{0}'.format(tag) in f:
                    #    models['fT_{0}'.format(tag)] = make_fT_BK(Nijk,Npow,Nm,addrho,p,Fit,p['qsq_{0}'.format(tag)],t_0,mass,fpf0same,float(mass)) #second mass is amh
                    
                        
        return(models)
    #################################
    def fitargs(w):
        prior,f = make_prior_BK(fs_data,Fits,addrho,t_0,Npow,Nijk,Nm,rhopri,dpri,cpri,cvalpri,d0000npri,di0000pri,di100npri,d000lnpri,adddata,constraint,w=w)
        return dict(data=f, fcn=fcn, prior=prior)
    ##################################################
    p0 = None
    if os.path.isfile('Fits/pmeanBK{0}{1}{2}{3}{4}.pickle'.format(addrho,Npow,Nijk,Nm,t_0)):
        p0 = gv.load('Fits/pmeanBK{0}{1}{2}{3}{4}.pickle'.format(addrho,Npow,Nijk,Nm,t_0))
    #p0 = None
    fit = lsqfit.nonlinear_fit(data=f, prior=prior, p0=p0, fcn=fcn,svdcut=1e-4,noise=noise,  maxit=500, tol=(1e-6,0.0,0.0),fitter='gsl_multifit', alg='subspace2D', solver='cholesky',debug=True ) #svdcut =1e-4
    gv.dump(fit.pmean,'Fits/pmeanBK{0}{1}{2}{3}{4}.pickle'.format(addrho,Npow,Nijk,Nm,t_0))
    print(fit.format(maxline=True))
    print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.2f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
    #fit2, w = lsqfit.empbayes_fit(1.1, fitargs)
    #print(fit2.format(True))
    #print("w = ",w)
    return(fit.p)

#####################################################################################################
def make_p_physical_point_BK(pfit,Fits,B=False):
    #only need to evaluate at one Fit one mass but change all anyway
    # everything should now be in GeV
    # form factors made with Bav Dav and Kav, this is all in prior. Can then choose which using B='p'/'0' for other purposes  
    p = gv.BufferDict()
    MDphysav = (pfit['MDphys0'] + pfit['MDphysp'])/2
    MKphysav = (pfit['MKphys0'] + pfit['MKphysp'])/2
    MBphysav = (pfit['MBphys0'] + pfit['MBphysp'])/2
    for Fit in Fits:
        fit = Fit['conf']
        p['a_{0}'.format(fit)] = 0
        if B == 'p':
            p['MK_{0}'.format(fit)] =  MKphysav # only appears in ff
            p['MKphys'] = pfit['MKphysp']
        elif B == '0':
            p['MK_{0}'.format(fit)] =  MKphysav
            p['MKphys'] = pfit['MKphys0']
        elif B == False:
            p['MK_{0}'.format(fit)] = MKphysav
            p['MKphys'] = MKphysav
        p['LQCD_{0}'.format(fit)] = LQCD
        p['Metac_{0}'.format(fit)] = pfit['Metacphys']
        p['Metas_{0}'.format(fit)] = pfit['Metasphys']
        p['deltas_{0}'.format(fit)] = 0
        p['deltal_{0}'.format(fit)] = 0
        p['deltasval_{0}'.format(fit)] = 0
        p['deltalval_{0}'.format(fit)] = 0
        p['ml10ms_{0}'.format(fit)] = 1/(mlmsfac*pfit['slratio'])
        p['deltaFV_{0}'.format(fit)] = 0
        if B == 'p':
            p['MD_{0}'.format(fit)] = MDphysav # only appears in ff
            p['MDphys'] = pfit['MDphysp']
            p['MBphys'] = pfit['MBphysp']
        elif B == '0':
            p['MD_{0}'.format(fit)] = MDphysav
            p['MDphys'] = pfit['MDphys0']
            p['MBphys'] = pfit['MBphys0']
        elif B == False:
            p['MD_{0}'.format(fit)] = MDphysav
            p['MDphys'] = MDphysav
            p['MBphys'] = MBphysav
            
        for mass in Fit['masses']:
            p['MH_{0}_m{1}'.format(fit,mass)] =  MBphysav# only appears in ff
            
    for key in pfit:
        if key not in p:
            p[key] = pfit[key]
    return(p)

######################################################################################################

def make_p_Mh_BK(pfit,Fits,MH):#just takes average massed for D and K 
    p = gv.BufferDict()
    MDphysav = (pfit['MDphys0'] + pfit['MDphysp'])/2
    MKphysav = (pfit['MKphys0'] + pfit['MKphysp'])/2
    MBphysav = (pfit['MBphys0'] + pfit['MBphysp'])/2
    for Fit in Fits:
        fit = Fit['conf']
        p['a_{0}'.format(fit)] = 0
        p['MK_{0}'.format(fit)] = MKphysav
        p['LQCD_{0}'.format(fit)] = LQCD
        p['Metac_{0}'.format(fit)] = pfit['Metacphys']
        p['Metas_{0}'.format(fit)] = Metasphys
        p['deltas_{0}'.format(fit)] = 0
        p['deltal_{0}'.format(fit)] = 0
        p['deltasval_{0}'.format(fit)] = 0
        p['deltalval_{0}'.format(fit)] = 0
        p['ml10ms_{0}'.format(fit)] = 1/(mlmsfac*pfit['slratio'])
        p['deltaFV_{0}'.format(fit)] = 0
        p['MD_{0}'.format(fit)] = MDphysav
        p['MDphys'] = MDphysav
        p['MKphys'] = MKphysav
        p['MBphys'] = MBphysav
        for mass in Fit['masses']:
            p['MH_{0}_m{1}'.format(fit,mass)] = MH
                
    for key in pfit:
        if key not in p:
            p[key] = pfit[key]
    return(p)


######################################################################################################
def errs_per_ens(res,dat):
    output = ''
    temp = []
    tempk = []
    for key in dat:
        temp.append(res.partialsdev(dat[key]))
        tempk.append(key)
    temp2 = np.sort(temp)[::-1]
    for i in temp2:
        key = tempk[temp.index(i)]
        output = '{0} {1}:{2:.4f}'.format(output,key,i)
    return(output)
######################################################################################################

def fs_at_lims_BK(prior,f,pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    dict_for_plot = collections.OrderedDict()
    data_for_err = collections.OrderedDict()
    for Fit in Fits:
        data_for_err[Fit['conf']] = []
        for key in f:
            if '_{0}_'.format(Fit['conf']) in key:
                data_for_err[Fit['conf']].append(f[key])
    
    As = []
    confs = []
    for fit in Fits:
        As.append(fit['a'])
        confs.append(fit['conf'])
    #print(As)
    #print(confs)
    p = make_p_physical_point_BK(pfit,Fits)
    qsq0 = 0
    f00 = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq0,t_0,Fits[0]['masses'][0],fpf0same,0)
    #inputs = dict(prior=prior,data=f)
    #outputs =dict(f00 = f00)
    #print(gv.fmt_errorbudget(inputs=inputs, outputs=outputs))
    fp0 = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq0,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
    fT0 = make_fT_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq0,t_0,Fits[0]['masses'][0],fpf0same,0)
    qsq = qsqmaxphysBK.mean
    f0max = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0)
    fpmax = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
    fTmax = make_fT_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0)
    print('f_+(0)/f_0(0) = {0}'.format(fp0/f00))
    print('f_0(0) = {0}  error: {1:.2%}'.format(f00,f00.sdev/f00.mean))
    print('f_+(0) = {0}  error: {1:.2%}'.format(fp0,fp0.sdev/fp0.mean))
    print('f_T(0) = {0}  error: {1:.2%}'.format(fT0,fT0.sdev/fT0.mean))
    print('f_0(max) = {0}  error: {1:.2%}'.format(f0max,f0max.sdev/f0max.mean))
    print('f_+(max) = {0}  error: {1:.2%}'.format(fpmax,fpmax.sdev/fpmax.mean))
    print('f_T(max) = {0}  error: {1:.2%}'.format(fTmax,fTmax.sdev/fTmax.mean))
    print(errs_per_ens(f00,data_for_err))
    print(errs_per_ens(fp0,data_for_err))
    print(errs_per_ens(fT0,data_for_err))
    print(errs_per_ens(f0max,data_for_err))
    print(errs_per_ens(fpmax,data_for_err))
    print(errs_per_ens(fTmax,data_for_err))
    
    dict_for_plot['BKf00'] = errs_per_ens(f00,data_for_err)
    dict_for_plot['BKfp0'] = errs_per_ens(fp0,data_for_err)
    dict_for_plot['BKfT0'] = errs_per_ens(fT0,data_for_err)
    dict_for_plot['BKf0max'] = errs_per_ens(f0max,data_for_err)
    dict_for_plot['BKfpmax'] = errs_per_ens(fpmax,data_for_err)
    dict_for_plot['BKfTmax'] = errs_per_ens(fTmax,data_for_err)
    constl = 1/(2*p['MBphys']) * ((1 + (p['MBphys']**2-p['MKphys']**2)/qsqmaxphysBK)*fpmax - f0max* (p['MBphys']**2 - p['MKphys']**2)/qsqmaxphysBK)
    constr = fTmax /(p['MBphys']+p['MKphys'])
    print('Hill const:',constr-constl)
    a0 = []
    ap = []
    aT = []
    for n in range(Npow):
        a0.append(make_an_BK(n,Nijk,Nm,addrho,p,'0',Fits[0],Fits[0]['masses'][0],0,fpf0same))
        ap.append(make_an_BK(n,Nijk,Nm,addrho,p,'p',Fits[0],Fits[0]['masses'][0],0,fpf0same))
        aT.append(make_an_BK(n,Nijk,Nm,addrho,p,'T',Fits[0],Fits[0]['masses'][0],0,fpf0same))
    print('logs',make_logs(p,Fits[0]['masses'][0],Fits[0]))
    print('a_0',a0)
    print('a_+',ap)
    print('a_T',aT)
    print('fT_(0)/fp_(0) = {0}'.format(fT0/fp0))
    print('############## D to K ##############################################')
    p = make_p_Mh_BK(pfit,Fits,(pfit['MDphysp']+pfit['MDphys0'])/2)
    qsq0 = 0
    f00 = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq0,t_0,Fits[0]['masses'][0],fpf0same,0)
    fp0 = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq0,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
    fT0 = Z_T_running*make_fT_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq0,t_0,Fits[0]['masses'][0],fpf0same,0)
    qsq = qsqmaxphysDK.mean
    f0max = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0)
    fpmax = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
    fTmax = Z_T_running*make_fT_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0)
    print('f_+(0)/f_0(0) = {0}'.format(fp0/f00))
    print('f_0(0) = {0}  error: {1:.2%}'.format(f00,f00.sdev/f00.mean))
    #print('Error from as: ',f00.partialsdev(tuple(As)))
    print('f_+(0) = {0}  error: {1:.2%}'.format(fp0,fp0.sdev/fp0.mean))
    print('f_T(0)(2GeV) = {0}  error: {1:.2%}'.format(fT0,fT0.sdev/fT0.mean))
    print('f_0(max) = {0}  error: {1:.2%}'.format(f0max,f0max.sdev/f0max.mean))
    #print('Error from as: ',f0max.partialsdev(tuple(As)))
    print('f_+(max) = {0}  error: {1:.2%}'.format(fpmax,fpmax.sdev/fpmax.mean))
    print('f_T(max)(2GeV) = {0}  error: {1:.2%}'.format(fTmax,fTmax.sdev/fTmax.mean))
    print(errs_per_ens(f00,data_for_err))
    print(errs_per_ens(fp0,data_for_err))
    print(errs_per_ens(fT0,data_for_err))
    print(errs_per_ens(f0max,data_for_err))
    print(errs_per_ens(fpmax,data_for_err))
    print(errs_per_ens(fTmax,data_for_err))
    dict_for_plot['DKf00'] = errs_per_ens(f00,data_for_err)
    dict_for_plot['DKfp0'] = errs_per_ens(fp0,data_for_err)
    dict_for_plot['DKfT0'] = errs_per_ens(fT0,data_for_err)
    dict_for_plot['DKf0max'] = errs_per_ens(f0max,data_for_err)
    dict_for_plot['DKfpmax'] = errs_per_ens(fpmax,data_for_err)
    dict_for_plot['DKfTmax'] = errs_per_ens(fTmax,data_for_err)
    gv.dump(dict_for_plot,'Fits/error_breakdown_data.pickle')
    a0 = []
    ap = []
    aT = []
    for n in range(Npow):
        a0.append(make_an_BK(n,Nijk,Nm,addrho,p,'0',Fits[0],Fits[0]['masses'][0],0,fpf0same))
        ap.append(make_an_BK(n,Nijk,Nm,addrho,p,'p',Fits[0],Fits[0]['masses'][0],0,fpf0same))
        aT.append(Z_T_running*make_an_BK(n,Nijk,Nm,addrho,p,'T',Fits[0],Fits[0]['masses'][0],0,fpf0same))
    print('logs',make_logs(p,Fits[0]['masses'][0],Fits[0]))
    print('a_0',a0)
    print('a_+',ap)
    print('a_T(2GeV)',aT)
    print('fT_(0)(2GeV)/fp_(0) = {0}'.format(fT0/fp0))
    print('########################################################################')
    HFLAV= gv.gvar('0.7180(33)')# from #1909.12524 p318
    Vcsq20 = HFLAV/fp0
    terr = Vcsq20.partialsdev(fp0)
    eerr = Vcsq20.partialsdev(HFLAV)
    print('Vcs at q^2=0 = {0}'.format(Vcsq20))
    print('Theory error = {0:.4f} Exp error = {1:.4f} Total = {2:.4f}'.format(terr,eerr,np.sqrt(eerr**2+terr**2)))
    return()

##############################################################################################################


def make_beta_delta_BK(Fits,t_0,Nijk,Npow,Nm,addrho,p,fpf0same,MH,const2):
    if const2 == True:
        print('ERROR Alpha, beta delat not compatable with const2')
        return()
    t0 = make_t_0(t_0,MH,p['MKphys'],MH,p['MKphys'])
    t_plus = make_t_plus(MH,p['MKphys'])
    logs  = make_logs(p,Fits[0]['masses'][0],Fits[0])
    MHs0 = MH+Del
    MHsstar = make_MHsstar(MH,p)
    zMHsstar = make_z((MHsstar**2).mean,t_0,MH,p['MKphys'])
    fp0 = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,t_0,Fits[0]['masses'][0],fpf0same,0)
    alpharat = 0
    zprime0 = -1/(2*(t_plus+gv.sqrt(t_plus*(t_plus-t0))))
    z0 = make_z(0,t_0,MH,p['MKphys'])
    f0prime0 = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,t_0,Fits[0]['masses'][0],fpf0same,0)/(MHs0**2)
    fpprime0 = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,t_0,Fits[0]['masses'][0],fpf0same,0)/(MHsstar**2)
    for n in range(Npow):
        alpharat += 1/fp0 * make_an_BK(n,Nijk,Nm,addrho,p,'p',Fits[0],Fits[0]['masses'][0],0,fpf0same)* ( zMHsstar**(n) - (n/Npow) * (-1)**(n-Npow) * zMHsstar**(Npow))
        if n !=0:
            f0prime0 += logs * make_an_BK(n,Nijk,Nm,addrho,p,'0',Fits[0],Fits[0]['masses'][0],0,fpf0same) * n * zprime0 * z0**(n-1)
            fpprime0 += logs * make_an_BK(n,Nijk,Nm,addrho,p,'p',Fits[0],Fits[0]['masses'][0],0,fpf0same) * ( n * zprime0 * z0**(n-1) - n * (-1)**(n-Npow) * zprime0 * z0**(Npow-1))
    alpha = 1-1/alpharat
    delta = 1 - ((MH**2-p['MKphys']**2)/fp0) * (fpprime0-f0prime0)
    invbeta = ((MH**2-p['MKphys']**2)/fp0) * f0prime0
    if MH == p['MBphys'].mean:
        print('delta at MB = ',delta)
        print('alpha at MB = ',alpha)
        print('beta at MB = ',1/invbeta)
    if MH == p['MDphys'].mean:
        print('delta at MD = ',delta)
        print('alpha at MD = ',alpha)
        print('beta at MD = ',1/invbeta)
    return(alpha,delta,invbeta)


##############################################################################################################

def make_h(qsq,m):
    x = 4 * m**2/qsq
    if x > 1:
        hR = -4/9 * (gv.log(x/4) - 2/3 - x) - 4/9 * (2 + x) * ( gv.sqrt(x-1) * gv.arctan(1/(gv.sqrt(x-1))))
        hI = 0
    elif x == 0:
        h = 4/9 * (2/3 + 1j*np.pi)
        hR =  h.real
        hI = h.imag
    elif x <= 1:
        #h = -4/9 * (gv.log(x/4) - 2/3 - x) - 4/9 * (2 + x) * ( gv.sqrt(1-x) * (gv.log( (1+gv.sqrt(1-x))/gv.sqrt(x)) - 1j*np.pi/2 ))
        hR = -4/9 * (gv.log(x/4) - 2/3 - x) - 4/9 * (2 + x) * ( gv.sqrt(1-x) * (gv.log( (1+gv.sqrt(1-x))/gv.sqrt(x))  ))
        hI =  4/9 * (2 + x) *  gv.sqrt(1-x) *  np.pi/2 
    return(hR,hI)


##############################################################################################################
alphaEW = 1/gv.gvar('127.952(9)')#PDG  #1/gv.gvar('128.957(20)')  #1/0.027515±0.000149 0807.4206 at M_Z**2
#alphae = 1/gv.gvar('127.952(9)') #PDG is this correct? At M_Z
VtbVts = gv.gvar('0.04086(76)')#CKMfitter
m_c = gv.gvar('1.27(2)')#pdg
m_b = gv.gvar('4.18(3)')#pdg
C1 = gv.gvar('-0.294(9)')#gv.gvar('-0.257(5)') new ones from 1606.00916
C2 = gv.gvar('1.017(1)')#gv.gvar('1.009(20)')
C3 = gv.gvar('-0.0059(2)')#  gv.gvar('-0.0050(1)')
C4 = gv.gvar('-0.087(1)')# gv.gvar('-0.078(2)')
C5 = 0.0004#0
C6 = gv.gvar('0.0011(1)')#0.001  # all the same up to C6 
C7eff = gv.gvar('-0.2957(5)')#gv.gvar('-0.304(6)')
C10eff = gv.gvar('-4.193(33)')#gv.gvar('-4.103(82)')
C9effbase = gv.gvar('4.114(14)')#gv.gvar('4.211(84)') #note that uncertainty not inlcuded. 
#############################################################################################################

def make_al_cl(p,qsq,m_l,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    # h is complex -> C9 eff complex, and thus FV
    hR0,hI0 = make_h(qsq,0)
    hRc,hIc = make_h(qsq,m_c)
    hRb,hIb = make_h(qsq,m_b)
    YR = 4/3 * C3 + 64/9 * C5 + 64/27 * C6 - hR0/2 * (C3 + 4/3 * C4 + 16 * C5 + 64/3 * C6) + hRc * (4/3 * C1 + C2 + 6 * C3 + 60 * C5) - hRb/2 * (7 * C3 + 4/3 * C4 + 76 * C5 + 64/3 * C6)
    YI = 4/3 * C3 + 64/9 * C5 + 64/27 * C6 - hI0/2 * (C3 + 4/3 * C4 + 16 * C5 + 64/3 * C6) + hIc * (4/3 * C1 + C2 + 6 * C3 + 60 * C5) - hIb/2 * (7 * C3 + 4/3 * C4 + 76 * C5 + 64/3 * C6)

    C9effR = C9effbase + YR 
    C9effI = YI
    
    f0 = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0)
    fp = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
    fT = make_fT_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0)
    FA = C10eff*fp
    FVR = C9effR*fp + 2*m_b*C7eff*fT/(p['MBphys']+p['MKphys'])
    FVI = C9effI*fp + 2*m_b*C7eff*fT/(p['MBphys']+p['MKphys'])    
    FP = -m_l*C10eff*(fp - (p['MBphys']**2-p['MKphys']**2)*(f0-fp)/qsq)
    #print('FA',FA,FVR,FVI,FP)
    ######################################################
    betal = gv.sqrt(1-4*m_l**2/qsq)
    lam = qsq**2 + p['MBphys']**4 + p['MKphys']**4 - 2*(p['MBphys']**2*p['MKphys']**2 + qsq*p['MBphys']**2 + qsq*p['MKphys']**2)
    if lam.mean == 0:
        lam = 0
    C = GF**2 * alphaEW**2 * VtbVts**2 * betal * gv.sqrt(lam) / (2**9 * np.pi**5 * p['MBphys']**3)

    al = C*( qsq*FP**2 + lam/4 * (FA**2 + FVR**2+FVI**2) + 4 * m_l**2 * p['MBphys']**2 * FA**2 + 2 * m_l * (p['MBphys']**2 - p['MKphys']**2 + qsq) * (FP * FA) )
    cl = -C*lam*betal**2/4 * (FA**2 + FVR**2+FVI**2)
    #print(betal,lam,C,al,cl)
    return(al,cl)

#####################################################################
def check_gaps(qsq,gaps):
    gap1 = [8.68,10.11]
    gap2 = [12.86,14.18]
    if gaps:
        zero = False
        if gap1[0] <= qsq < gap1[1]:
            zero = True
        if gap2[0] <= qsq < gap2[1]:
            zero = True
    elif gaps == False:
        zero = False
    return(zero)
    

def integrate_Gamma(p,qsq_min,qsq_max,m_l,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=25,gaps=False,qmax=False):
    iters = iters
    def integrand(qsq):
        al,cl = make_al_cl(p,qsq,m_l,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        integrand = 2*al + 2*cl/3
        if np.isnan(integrand.mean):
            integrand = 0
        if check_gaps(qsq,gaps):
            integrand = 0
        return(integrand)
    del_qsq =  (qsq_max-qsq_min) /iters
    if qmax:
        funcs = integrand(qsq_min)
    else:
        funcs = integrand(qsq_min) + integrand(qsq_max)
    for i in range(1,iters):
        funcs += 2*integrand(qsq_min+del_qsq*i)
    result = del_qsq*funcs/2
    return(result)

#########################################################################################

def integrate_FH(p,qsq_min,qsq_max,m_l,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=25):
    iters = iters
    def integrand_top(qsq):
        al,cl = make_al_cl(p,qsq,m_l,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        integrand = al + cl
        return(integrand)
    def integrand_bot(qsq):
        al,cl = make_al_cl(p,qsq,m_l,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        integrand = al + cl/3
        return(integrand)
    
    del_qsq =  (qsq_max-qsq_min) /iters
    funcs_top = integrand_top(qsq_min) + integrand_top(qsq_max)
    funcs_bot = integrand_bot(qsq_min) + integrand_bot(qsq_max)
    for i in range(1,iters):
        funcs_top += 2*integrand_top(qsq_min+del_qsq*i)
        funcs_bot += 2*integrand_bot(qsq_min+del_qsq*i)
    result = funcs_top/funcs_bot
    return(result)

#####################################################################
Xt = gv.gvar('1.469(17)')#1009.0947 check for update
sinthw2 = gv.gvar('0.23121(4)')# PDG
########################### PDG says Vub =3.82(24)*1e-3 Vus = 0.2245(8) 
VubVusfK = gv.gvar('3.72(16)*1e-3') * gv.gvar('35.090(57)*1e-3') #GeV
fB = gv.gvar('189.4(1.4)*1e-3')#GeV B+ from 1702.09262   #gv.gvar('190.5(4.2)*1e-3') #GeV
Gammatau = (1/gv.gvar('290.3(5)*1e-3')) * 6.582119569*1e-13  # ps converted to GeV

def integrate_fp_B(p,qsq_min,qsq_max,pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2,iters=25,qmax=False):
    def integrand(qsq):
        p3 = ((qsq-p['MKphys']**2-p['MBphys']**2)**2/(4*p['MBphys']**2)-p['MKphys']**2)**(3/2)
        if math.isnan(p3.mean):
            p3 = 0
        fp = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
        integrand = p3 * fp**2
        return(integrand)
    iters = iters
    del_qsq =  (qsq_max-qsq_min) /iters
    if qmax:
        funcs = integrand(qsq_min)
    else:
        funcs = integrand(qsq_min) + integrand(qsq_max)
    for i in range(1,iters):
        funcs += 2*integrand(qsq_min+del_qsq*i)
    result = del_qsq*funcs/2
    return(result)
    
def neutrio_branching(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    iters = 200
    qsq_min = 0
    ######## B0 SD ###########
    qsq_max = qsqmaxphysBK0.mean
    p = make_p_physical_point_BK(pfit,Fits,B='0')
    result = integrate_fp_B(p,qsq_min,qsq_max,pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2,iters=150,qmax=True)
    BB0 = (tauB0GeV * VtbVts**2 * GF**2 * alphaEW**2 * Xt**2 * result) /(32 * (np.pi)**5 * sinthw2**2)
    print('B^0 -> K^0 nu nu (SD) {0} x 10^-6'.format(BB0/1e-6))
    ######### Bp SD #####################
    qsq_max = qsqmaxphysBKp.mean
    p = make_p_physical_point_BK(pfit,Fits,B='p')
    result = integrate_fp_B(p,qsq_min,qsq_max,pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2,iters=150,qmax=True)
    BBp = (tauBpmGeV * VtbVts**2 * GF**2 * alphaEW**2 * Xt**2 * result) /(32 * (np.pi)**5 * sinthw2**2) + gv.gvar('6.22(60)*1e-7')
    print('B^+ -> K^+ nu nu (SD+LD) {0} x 10^-6'.format(BBp/1e-6))
    ######## Bp LD ######################################################
    GammaB = 1/tauBpmGeV
    BBpLD = ( (GF**2 * VubVusfK * fB )**2 * 2 * np.pi * m_tau * (p['MBphys']**2 - m_tau**2)**2 * (p['MKphys']**2 - m_tau**2)**2 ) / (256 * np.pi**3 * p['MBphys']**3 * Gammatau * GammaB ) 
    print('B^+ -> K^+ nu nu (LD) {0} x 10^-7'.format(BBpLD/1e-7))
    return()

#####################################################################

def test_stuff(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,B=False):
    p = make_p_physical_point_BK(pfit,Fits,B=B)
    qsq_max = qsqmaxphysBK.mean
    #for qsq in np.linspace(((2*m_l)**2).mean,qsqmaxphysBK.mean,100):
    ########## B ########################################
    for m_l in [m_e,m_mu,m_tau]:
        qsq_min = ((2*m_l)**2)
        if m_l == m_tau:
            qsq_min = 14.18
        result = integrate_Gamma(p,qsq_min,qsq_max,m_l,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2) 


        print('Branching 0 = ', result*tauB0GeV)
        print('Branching pm = ', result*tauBpmGeV)
    ################   R  ################################
    qsq_min = ((2*m_mu)**2)
    R_mu_e =  integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)/integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
    qsq_min = 14.18
    R_tau_e =  integrate_Gamma(p,qsq_min,qsq_max,m_tau,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)/integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
    R_tau_mu =  integrate_Gamma(p,qsq_min,qsq_max,m_tau,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)/integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
    B_LHCb = integrate_Gamma(p,1.1,6,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
    print('LHCb 1903.09252 comp=', B_LHCb*tauBpmGeV/4.9)
    print('R_mu_e',R_mu_e)
    print('R_tau_e',R_tau_e)
    print('R_tau_mu',R_tau_mu)
    return()

##########################################################################

def comp_by_bin(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,B=False):
    p = make_p_physical_point_BK(pfit,Fits,B=B)
    bin_starts = [1,4.3,10.09,14.18,16,16]
    bin_ends = [6,8.68,12.86,16,18,qsqmaxphysBK.mean]
    B = []
    Rmue = []
    Fe = []
    Fmu = []
    #for qsq in np.linspace(((2*m_l)**2).mean,qsqmaxphysBK.mean,100):
    ########## B ########################################
    for b in range(len(bin_starts)):
        qsq_min = bin_starts[b]
        qsq_max = bin_ends[b]
        result = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2) 
        R_mu_e =  integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)/integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        F_H_e = integrate_FH(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        F_H_mu = integrate_FH(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        print('({0},{1}) e Branching 0 = {2}'.format(qsq_min,qsq_max,result*tauB0GeV))
        print('({0},{1}) e Branching pm = {2}'.format(qsq_min,qsq_max,result*tauBpmGeV))
        print('({0},{1}) R_mu_e -1 = {2}'.format(qsq_min,qsq_max,R_mu_e-1))
        print('({0},{1}) F_H_e = {2}'.format(qsq_min,qsq_max,F_H_e))
        print('({0},{1}) F_H_mu = {2}'.format(qsq_min,qsq_max,F_H_mu))
        B.append(1e7*result*tauB0GeV)
        Rmue.append(1e3*(R_mu_e-1))
        Fe.append(1e6*F_H_e)
        Fmu.append(1e2*F_H_mu)
    return(B,Rmue,Fe,Fmu)

########################################################################################################################


def comp_by_bin2(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,B=False):
    #results from 1403.8044 
    p = make_p_physical_point_BK(pfit,Fits,B=B) 
    bin_starts_p = [0.1,1.1,2,3,4,5,6,7,11,11.8,15,16,17,18,19,20,21,1.1,15]
    bin_ends_p =   [0.98,2,3,4,5,6,7,8,11.8,12.5,16,17,18,19,20,21,22,6,22]
    bin_starts_0 = [0.1,2,4,6,11,15,17,1.1,15]
    bin_ends_0 =   [2,4,6,8,12.5,17,22,6,22]
    Bp = []
    B0 = []
    #for qsq in np.linspace(((2*m_l)**2).mean,qsqmaxphysBK.mean,100):
    ########## B ########################################
    for b in range(len(bin_starts_p)):
        qsq_min = bin_starts_p[b]
        qsq_max = bin_ends_p[b]
        result = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)/(qsq_max-qsq_min) 
        Bp.append(1e9*result*tauBpmGeV)
    for b in range(len(bin_starts_0)):
        qsq_min = bin_starts_0[b]
        qsq_max = bin_ends_0[b]
        result = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)/(qsq_max-qsq_min)
        B0.append(1e9*result*tauB0GeV)
    return(Bp,B0)

#####################################################################################################################

def comp_by_bin3(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,B=False):
    p = make_p_physical_point_BK(pfit,Fits,B=B)
    bin_starts = [14.18,14.18,16,16]
    bin_ends = [qsqmaxphysBK.mean,16,18,qsqmaxphysBK.mean]
    Rtaumu = []
    Rtaue = []
    Ftau = []
    Btau = []
    #for qsq in np.linspace(((2*m_l)**2).mean,qsqmaxphysBK.mean,100):
    ########## B ########################################
    for b in range(len(bin_starts)):
        qsq_min = bin_starts[b]
        qsq_max = bin_ends[b]
        result = integrate_Gamma(p,qsq_min,qsq_max,m_tau,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2) 
        R_tau_mu =  integrate_Gamma(p,qsq_min,qsq_max,m_tau,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)/integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        R_tau_e =  integrate_Gamma(p,qsq_min,qsq_max,m_tau,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)/integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        F_H_tau = integrate_FH(p,qsq_min,qsq_max,m_tau,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Btau.append(1e7*result*tauB0GeV)
        Rtaumu.append(R_tau_mu)
        Rtaue.append(R_tau_e)
        Ftau.append(F_H_tau)
    return(Btau,Rtaumu,Rtaue,Ftau)

######################################################################################################################



def make_x_from_bins(bin_starts,bin_ends):
    x = []
    xerr = []
    for i in range(len(bin_starts)):
        delta = bin_ends[i] - bin_starts[i]
        x.append(bin_starts[i]+delta/2)
        xerr.append(delta/2)
    return(x,xerr)

def make_y_from_bins(mean,upp,low,bin_starts,bin_ends):
    y = []
    yupp = []
    ylow = []
    for i in range(len(bin_starts)):
        delta = bin_ends[i] - bin_starts[i]
        y.append(100*mean[i]/delta)
        yupp.append(100*upp[i]/delta)
        ylow.append(100*low[i]/delta)
    return(y,yupp,ylow)

######################################################################################################


def output_error_BK(pfit,prior,Fits,Nijk,Npow,Nm,f,qsqs,t_0,addrho,fpf0same,const2):
    p = make_p_physical_point_BK(pfit,Fits)
    f0dict = collections.OrderedDict()
    fpdict = collections.OrderedDict()
    fTdict = collections.OrderedDict()
    for i in range(1,6):
        f0dict[i] = []
        fpdict[i] = []
        fTdict[i] = []
    disclist = []
    qmislist = []
    heavylist = []
    dat = []
    extinputs = [prior['Metacphys'],prior['MDphys0'],prior['MKphys0'],prior['MDphysp'],prior['MKphysp'],prior['MDsstarphys'],prior['MBphys0'],prior['MBphysp'],prior['MBsphys'],prior['MDsphys'],prior['MBsstarphys'],prior['slratio'],prior['Metasphys']]  
    logs  = [prior['c1'],prior['c2'],prior['ginf'],f['gB'],f['gD']]
    #for Fit in Fits:
    #    fit = Fit['conf']
    #    logs.append(prior['ml10ms_{0}'.format(fit)])
    #    logs.append(prior['Metas_{0}'.format(fit)])
    #    logs.append(prior['deltaFV_{0}'.format(fit)])
    #    logs.append(prior['LQCD_{0}'.format(fit)])
    qmislist=logs
    for Fit in Fits:
        extinputs.append(prior['Metac_{0}'.format(Fit['conf'])])
    for n in range(Npow):
        if addrho:
            heavylist.append(prior['0rho'][n])
        qmislist.append(prior['0cl'][n])
        qmislist.append(prior['0cs'][n])
        qmislist.append(prior['0cc'][n])
        qmislist.append(prior['0csval'][n])
        #for nm in range(Nm):
        #    qmislist.append(prior['0clval'][nm][n])
        if addrho:
            heavylist.append(prior['prho'][n])
        qmislist.append(prior['pcl'][n])
        qmislist.append(prior['pcs'][n])
        qmislist.append(prior['pcc'][n])
        qmislist.append(prior['pcsval'][n])
        #for nm in range(Nm):
        #    qmislist.append(prior['pclval'][nm][n])
        if addrho:
            heavylist.append(prior['Trho'][n])
        qmislist.append(prior['Tcl'][n])
        qmislist.append(prior['Tcs'][n])
        qmislist.append(prior['Tcc'][n])
        qmislist.append(prior['Tcsval'][n])
        #for nm in range(Nm):
        #    qmislist.append(prior['Tclval'][nm][n])
        
        for i in range(Nijk[0]):
            for j in range(Nijk[1]):
                for k in range(Nijk[2]):
                    for l in range(Nijk[3]):
                        if j != 0 or k != 0:
                            if l == 0:
                                disclist.append(prior['0d'][i][j][k][l][n])
                                disclist.append(prior['pd'][i][j][k][l][n])
                                disclist.append(prior['Td'][i][j][k][l][n])
                            else:
                                qmislist.append(prior['0d'][i][j][k][l][n])
                                qmislist.append(prior['pd'][i][j][k][l][n])
                                qmislist.append(prior['Td'][i][j][k][l][n])
                        else:
                            heavylist.append(prior['0d'][i][j][k][l][n])
                            heavylist.append(prior['pd'][i][j][k][l][n])
                            heavylist.append(prior['Td'][i][j][k][l][n])
    for key in prior:
        if not isinstance(prior[key],(list,tuple,np.ndarray)):
            if prior[key] not in disclist + qmislist  + extinputs + heavylist:
                dat.append(prior[key])
    for key in f:
        if key != 'gB' and key != 'gD':
            dat.append(f[key])
    Fit = Fits[0]
    mass = Fit['masses'][0]
    fit = Fit['conf']
    for qsq in qsqs:
        f0 = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fit,qsq,t_0,mass,fpf0same,0)
        fp = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fit,qsq,t_0,mass,fpf0same,0,const2=const2)
        fT = make_fT_BK(Nijk,Npow,Nm,addrho,p,Fit,qsq,t_0,mass,fpf0same,0)
        var1 = (100*(f0.partialsdev(tuple(extinputs)))/f0.mean)**2
        var2 = (100*(f0.partialsdev(tuple(qmislist)))/f0.mean)**2
        var3 = (100*(f0.partialsdev(tuple(dat)))/f0.mean)**2
        var4 = (100*(f0.partialsdev(tuple(heavylist)))/f0.mean)**2
        var5 = (100*(f0.partialsdev(tuple(disclist)))/f0.mean)**2
        f0dict[1].append(var1)
        f0dict[2].append(var1+var2)
        f0dict[3].append(var1+var2+var3)
        f0dict[4].append(var1+var2+var3+var4)
        f0dict[5].append(var1+var2+var3+var4+var5)
        var1 = (100*(fp.partialsdev(tuple(extinputs)))/fp.mean)**2
        var2 = (100*(fp.partialsdev(tuple(qmislist)))/fp.mean)**2
        var3 = (100*(fp.partialsdev(tuple(dat)))/fp.mean)**2
        var4 = (100*(fp.partialsdev(tuple(heavylist)))/fp.mean)**2
        var5 = (100*(fp.partialsdev(tuple(disclist)))/fp.mean)**2
        fpdict[1].append(var1)
        fpdict[2].append(var1+var2)
        fpdict[3].append(var1+var2+var3)
        fpdict[4].append(var1+var2+var3+var4)
        fpdict[5].append(var1+var2+var3+var4+var5)
        var1 = (100*(fT.partialsdev(tuple(extinputs)))/fT.mean)**2
        var2 = (100*(fT.partialsdev(tuple(qmislist)))/fT.mean)**2
        var3 = (100*(fT.partialsdev(tuple(dat)))/fT.mean)**2
        var4 = (100*(fT.partialsdev(tuple(heavylist)))/fT.mean)**2
        var5 = (100*(fT.partialsdev(tuple(disclist)))/fT.mean)**2
        fTdict[1].append(var1)
        fTdict[2].append(var1+var2)
        fTdict[3].append(var1+var2+var3)
        fTdict[4].append(var1+var2+var3+var4)
        fTdict[5].append(var1+var2+var3+var4+var5)
    return(f0dict,fpdict,fTdict)

#######################################################################################################

# D to K

######################################################################################################

def integrate_fp(qsq_min,qsq_max,pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2):
    p = make_p_Mh_BK(pfit,Fits,pfit['MDphys'])
    def integrand(qsq):
        p3 = ((qsq-p['MKphys']**2-p['MDphys']**2)**2/(4*p['MDphys']**2)-p['MKphys']**2)**(3/2)
        if math.isnan(p3.mean):
            p3 = 0
        fp = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
        integrand = p3 * fp**2
        return(integrand)
    iters = 100
    del_qsq =  (qsq_max-qsq_min) /iters
    funcs = integrand(qsq_min) + integrand(qsq_max)
    for i in range(1,iters):
        funcs += 2*integrand(qsq_min+del_qsq*i)
    result = del_qsq*funcs/2
    return(result)

#################################################################################################

def comp_cleo(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2):
    #bins 0,0.2,0.4...1.6,inf
    p3integrals = []
    bins = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,qsqmaxphysDK.mean]
    for i in range(len(bins)-1):
        p3integrals.append(integrate_fp(bins[i],bins[i+1],pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2))
    partials = [17.82,15.83,13.91,11.69,9.36,7.08,5.34,3.09,1.28,17.79,15.62,14.02,12.28,8.92,8.17,4.96,2.67,1.19] ## D^0 to K^- followed by D^+ to K^0
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
        V2 = ( 24 * np.pi**3 * partials[i] * 6.582119569*1e-16 /(GF**2 * p3integrals[i]))# factor converts partials from ps to GeV
        Vcss21.append(V2)
        terr1.append(V2.partialsdev(p3integrals[i]))
        eerr1.append(V2.partialsdev(partials[i]))
        ints1.append(p3integrals[i])
    for i in range(int(len(partials)/2),len(partials)):
        V2 = ( 24 * np.pi**3 * partials[i] * 6.582119569*1e-16 /(GF**2 * p3integrals[i-int(len(partials)/2)]))
        Vcss22.append(V2)
        terr2.append(V2.partialsdev(p3integrals[i-int(len(partials)/2)]))
        eerr2.append(V2.partialsdev( partials[i]))
        ints2.append(p3integrals[i-int(len(partials)/2)])
        
    return(Vcss21,terr1,eerr1,Vcss22,terr2,eerr2,bins,ints1,ints2,partials)

############################################################################################

def comp_BES(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2):
    #bins 0,0.2,0.4...1.6,inf
    p3integrals = []
    bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,qsqmaxphysDK.mean]
    for i in range(len(bins)-1):
        p3integrals.append(integrate_fp(bins[i],bins[i+1],pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2))
    partials = [8.812,8.743,8.295,7.567,7.486,6.446,6.200,5.519,5.028,4.525,3.972,3.326,2.828,2.288,1.737,1.314,0.858,0.379] ## D^0 to K^-
    
    cov_mat = gv.load('covarience_matricies/BES.pickle')
    partials  = gv.gvar(partials,cov_mat)
    Vcss2 = []
    terr = []
    eerr = []
    ints = []
    for i in range(len(partials)):
        V2 = ( 24 * np.pi**3 * partials[i] * 6.582119569*1e-16 /(GF**2 * p3integrals[i]))
        Vcss2.append(V2)
        terr.append(V2.partialsdev(p3integrals[i]))
        eerr.append(V2.partialsdev(partials[i]))
        ints.append(p3integrals[i])
    return(Vcss2,terr,eerr,bins,ints,partials)

#######################BaBar #####################################################

def comp_BaBar(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2):
    #bins 0,0.2,0.4...1.6,inf
    p3integrals = []
    bins = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,qsqmaxphysDK.mean]
    for i in range(len(bins)-1):
        p3integrals.append(integrate_fp(bins[i],bins[i+1],pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2))
    partials = [18.15, 16.63, 14.75, 12.67, 10.14, 7.90, 5.44, 3.32, 1.320, 0.0633] ## D^0 to K^-
    
    cov_mat = gv.load('covarience_matricies/BaBar.pickle')
    #print(cov_mat)
    partials  = gv.gvar(partials,cov_mat)
    #print(partials)
    Vcss2 = []
    terr = []
    eerr = []
    ints = []
    for i in range(len(partials)):
        V2 = ( 24 * np.pi**3 * partials[i] * 6.582119569*1e-16 /(GF**2 * p3integrals[i]))
        Vcss2.append(V2)
        terr.append(V2.partialsdev(p3integrals[i]))
        eerr.append(V2.partialsdev(partials[i]))
        ints.append(p3integrals[i])
    return(Vcss2,terr,eerr,bins,ints,partials)

##################################################################################################

def total(Cleo1V2,Cleo2V2,BESV2,BaBarV2,C1ints,C2ints,BESints,BaBarints,Cpars,BESpars,BaBarpars):
    total = []
    total.extend(Cleo1V2)
    total.extend(Cleo2V2)
    total.extend(BESV2)
    total.extend(BaBarV2)
    av = gv.sqrt(lsqfit.wavg(total))
    print('Average V_cs = ', av)
    ints = []
    pars = []
    ints.extend(C1ints)
    ints.extend(C2ints)
    ints.extend(BESints)
    ints.extend(BaBarints)
    pars.extend(Cpars)
    pars.extend(BESpars)
    pars.extend(BaBarpars)
    terr = av.partialsdev(tuple(ints))
    eerr = av.partialsdev(tuple(pars))
    print('Theory error = {0:.4f} Exp error = {1:.4f} Total = {2:.4f}'.format(terr,eerr,np.sqrt(eerr**2+terr**2)))
    print('#####################################################################')
    return(av)

##################################################################################################


def comp(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2):
    Cleo1V2,Cleo1terr,Cleo1eerr,Cleo2V2,Cleo2terr,Cleo2eerr,Cleobins,C1ints,C2ints,Cpars = comp_cleo(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2)
    BESV2,BESterr,BESeerr,BESbins,BESints,BESpars = comp_BES(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2)
    BaBarV2,BaBarterr,BaBareerr,BaBarbins,BaBarints,BaBarpars = comp_BaBar(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2)
    Cleo1av = lsqfit.wavg(Cleo1V2)
    Cleo2av = lsqfit.wavg(Cleo2V2)
    BESav = lsqfit.wavg(BESV2)
    BaBarav = lsqfit.wavg(BaBarV2)
    #print('Cleo |V_cs|^2 D^0 to K^- sqrt(weighted average) = ',gv.sqrt(Cleo1av))
    #print('Cleo |V_cs|^2 D^+ to K^0 sqrt(weighted average) = ',gv.sqrt(Cleo2av))
    #print('BES |V_cs|^2 D^0 to K^- sqrt(weighted average) = ',gv.sqrt(BESav))
    #print('BaBar |V_cs|^2 D^0 to K^- sqrt(weighted average) = ',gv.sqrt(BaBarav))
    av = total(Cleo1V2,Cleo2V2,BESV2,BaBarV2,C1ints,C2ints,BESints,BaBarints,Cpars,BESpars,BaBarpars)    
    return()

##################################################################################################































































######################################################################################################
###########################Do itstuff below here check stuff is for BK and t_0*a etc etc#######################################################
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

#####################################################################################################

def output_error_BsEtas(pfit,prior,Fits,Nijk,Npow,f,qsqs,t_0,addrho,fpf0same):
    p = make_p_physical_point_BsEtas(pfit,Fits)
    f0dict = collections.OrderedDict()
    fpdict = collections.OrderedDict()
    for i in range(1,6):
        f0dict[i] = []
        fpdict[i] = []
    disclist = []
    qmislist = []
    heavylist = []
    dat = []
    extinputs = [MBsphys,MBsphys+Del,MDsphys,MBsstarphys,prior['Metacphys']]
    for Fit in Fits:
        extinputs.append(prior['Metac_{0}'.format(Fit['conf'])])
    for n in range(Npow):
        if addrho:
            heavylist.append(prior['0rho'][n])
        qmislist.append(prior['0cl'][n])
        qmislist.append(prior['0cs'][n])
        qmislist.append(prior['0cc'][n])
        qmislist.append(prior['0csval'][n])
        if addrho:
            heavylist.append(prior['prho'][n])
        qmislist.append(prior['pcl'][n])
        qmislist.append(prior['pcs'][n])
        qmislist.append(prior['pcc'][n])
        qmislist.append(prior['pcsval'][n])
        
        for i in range(Nijk):
            for j in range(Nijk):
                for k in range(Nijk):
                    if j != 0 or k != 0:
                        disclist.append(prior['0d'][i][j][k][n])
                        disclist.append(prior['pd'][i][j][k][n])
                    else:
                        heavylist.append(prior['0d'][i][j][k][n])
                        heavylist.append(prior['pd'][i][j][k][n])
    for key in prior:
        if not isinstance(prior[key],(list,tuple,np.ndarray)):
            if prior[key] not in disclist + qmislist + heavylist + extinputs:
                dat.append(prior[key])
    for key in f:
        dat.append(f[key])
    Fit = Fits[0]
    mass = Fit['masses'][0]
    fit = Fit['conf']
    for qsq in qsqs:
        z = make_z(qsq,t_0,MBsphys,Metasphys).mean
        f0 = make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,0,qsq,z,mass,fpf0same,0)
        fp = make_fp_BsEtas(Nijk,Npow,addrho,p,Fit,0,qsq,z,mass,fpf0same,0)
        var1 = (100*(f0.partialsdev(tuple(extinputs)))/f0.mean)**2
        var2 = (100*(f0.partialsdev(tuple(qmislist)))/f0.mean)**2
        var3 = (100*(f0.partialsdev(tuple(dat)))/f0.mean)**2
        var4 = (100*(f0.partialsdev(tuple(heavylist)))/f0.mean)**2
        var5 = (100*(f0.partialsdev(tuple(disclist)))/f0.mean)**2
        f0dict[1].append(var1)
        f0dict[2].append(var1+var2)
        f0dict[3].append(var1+var2+var3)
        f0dict[4].append(var1+var2+var3+var4)
        f0dict[5].append(var1+var2+var3+var4+var5)
        var1 = (100*(fp.partialsdev(tuple(extinputs)))/fp.mean)**2
        var2 = (100*(fp.partialsdev(tuple(qmislist)))/fp.mean)**2
        var3 = (100*(fp.partialsdev(tuple(dat)))/fp.mean)**2
        var4 = (100*(fp.partialsdev(tuple(heavylist)))/fp.mean)**2
        var5 = (100*(fp.partialsdev(tuple(disclist)))/fp.mean)**2
        fpdict[1].append(var1)
        fpdict[2].append(var1+var2)
        fpdict[3].append(var1+var2+var3)
        fpdict[4].append(var1+var2+var3+var4)
        fpdict[5].append(var1+var2+var3+var4+var5)
    return(f0dict,fpdict)

#######################################################################################################
