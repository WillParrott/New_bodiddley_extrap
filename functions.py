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


####################################################################################################
# Global parameters
####################################################################################################

Metasphys = gv.gvar('0.6885(22)')   # 1303.1670
Metacphys = gv.gvar('2.98390(50)')  # PDG says 2.9839(5) previously had '2.9863(27)' not sure where from
MBsphys = gv.gvar('5.36688(17)') # PDG
MDsphys = gv.gvar('1.968340(70)')  #PDG 
MBsstarphys = gv.gvar('5.4158(15)') #PDG
tensornorm = gv.gvar('1.09024(56)') # from Dan
w0 = gv.gvar('0.1715(9)')  #fm
hbar = gv.gvar('6.58211928(15)') # x 10^-25 GeV s
clight = 2.99792458 #*10^23 fm/s
slratio = gv.gvar('27.18(10)')   
MetacF = gv.gvar('1.367014(40)')        #lattice units
MetacSF = gv.gvar('0.896806(48)')       #where are these from? 
MetacUF = gv.gvar('0.666754(39)')       #All from Mclean 1906.00701
x =  MBsphys*(MBsstarphys-MBsphys) # gv.gvar('0.2474(81)') #GeV new way
LQCD = 0.5

####################################################################################################

def make_params_BsEtas():
    Fit['momenta'] = []
    for Fit in Fits:
        Fit['a'] = w0/(hbar*clight*0.01)*Fit['w0/a'] #in GeV^-1
        j = 0
        for i in range(len(Fit['masses'])):
            if i not in Masses[Fit['conf']]:
                del Fit['masses'][i-j]
                del Fit['Zdisc'][i-j]
                j += 1
        j = 0
        for i in range(len(Fit['twists'])):
            if i not in Twists[Fit['conf']]:
                del Fit['twists'][i-j]
                del Fit['daughter-Tag'][i-j]
                j += 1
        for twist in Fit['twists']:
            Fit['momenta'].append(np.sqrt(3)*np.pi*float(twist)/Fit['L'])
    return()

####################################################################################################

def make_params_BK():
    Fit['momenta'] = []
    daughters = []
    for Fit in Fits:
        Fit['a'] = w0/(hbar*c*0.01)*Fit['w0/a']
        j = 0
        for i in range(len(Fit['masses'])):
            if i not in Masses[Fit['conf']]:
                del Fit['masses'][i-j]
                del Fit['Zdisc'][i-j]
                j += 1
        j = 0
        for i in range(len(Fit['twists'])):
            if i not in Twists[Fit['conf']]:
                del Fit['twists'][i-j]
                j += 1
        for twist in Fit['twists']:
            Fit['momenta'].append(np.sqrt(3)*np.pi*float(twist)/Fit['L'])
            daughters.append(Fit['daughter-Tag'].format(twist))
        Fit['daughter-Tag'] = daughters
    return()    

####################################################################################################

def get_results(Fit):
    p = gv.gload(Fit['filename'],method='pickle')
    # We should only need goldstone masses and energies here
    for mass in Fit['masses']:
        Fit['M_parent_m{0}'.format(mass)] = p['dE:{0}'.format(Fit['parent-Tag'].format(Fit['m_s'],mass))][0]
    Fit['M_daughter'] = p['dE:{0}'.format(Fit['daghter-Tag'][0])][0]
    for t,twist in enumerate(Fit['twists']):
        #Fit is the actual measured value, theory is obtained from the momentum
        Fit['E_daughter_tw{0}_fit'.format(twist)] = p['dE:{0}'.format(Fit['daughter-Tag'][t])][0]
        Fit['E_daughter_tw{0}_theory'.format(twist)] = gv.sqrt(Fit['M_daughter']**2+Fit['momenta'][t]**2)
        for m, mass in enumerate(Fit['masses']):
            for 3pt in 3pts[Fit['conf']]:
                if twist != '0' or 3pt != 'T':
                    Fit['{0}_m{1}_tw{2}'.format(3pt,mass,twist)] = 2 * 2 * Fit['Zdisc'][m] * gv.sqrt(Fit['M_parent_m{0}'.format(mass)]*Fit['E_daughter_tw{0}_theory'.format(twist)]) * p['{0}Vnn_m{0}_tw{1}'][0][0]
                    #check zdisc is correctly implemented here
    return()
    
####################################################################################################

def make_fs(Fit,fs):
    for m,mass in enumerate(Fit['masses']):
        Z_v = (float(mass) - float(Fit['ms']))*Fit['S_m{0}_tw0'.format(mass)]/((Fit['M_parent_m{0}'.format(mass)] - Fit['M_daughter']) * Fit ['V_m{0}_tw0'.format(mass)])
        fs['Z_v_m{0}'.format(mass)] = Z_v
        for t,twist in enumerate(Fit['twists']):
            delta = (float(mass) - float(Fit['m_s']))*(Fit['M_parent_m{0}'.format(mass)]-Fit['E_daughter_tw{0}_theory'.format(twist)])
            qsq = (Fit['M_parent_m{0}'.format(mass)]-Fit['E_daughter_tw{0}_theory'].format(twist))**2 - Fit['momenta'][t]**2
            f0 = ((float(mass) - float(Fit['m_s']))*(1/(Fit['M_parent_m{0}'.format(mass)]**2 - Fit['M_daughter']**2))*Fit['S_m{0}_tw{1}'.format(mass,twist)])
            
            A = Fit['M_parent_m{0}'.format(mass)] + Fit['E_daughter_tw{0}'.format(twist)]
            B = (Fit['M_parent_m{0}'.format(mass)]**2 - Fit['M_daughter']**2)*(Fit['M_parent_m{0}'.format(mass)] - Fit['E_daughter_tw{0}'.format(twist)])/qsq
            fp = None
            fT = None
            if twist != '0':
                fp = (1/(A-B))*(Z_v*Fit['Vm{0}_tw{1}'.format(mass,twist)] - B*f0)
                if 'T' in 3pts[Fit['conf']]:
                    fT =  tensornorm*Fit['T_m{0}_tw{1}'.format(mass,twist)]*(Fit['M_parent_m{0}'.format(mass)]+Fit['M_daughter'])/(2*Fit['M_parent_m{0}'.format(mass)]*Fit['momenta'][t]])
            fs['qsq_m{0}_tw{0}'.format(mass,twist)] = qsq
            fs['f0_m{0}_tw{0}'.format(mass,twist)] = f0
            fs['fp_m{0}_tw{0}'.format(mass,twist)] = fp
            fs['fT_m{0}_tw{0}'.format(mass,twist)] = fT
    return()
#######################################################################################################    

def make_z(qsq,t_0,M_parent,M_daughter):
    t_plus = (M_parent + M_daughter)**2
    z = (gv.sqrt(t_plus - qsq) - gv.sqrt(t_plus - t_0)) / (gv.sqrt(t_plus - qsq) + gv.sqrt(t_plus - t_0))
    return(z)

#########################################################################################################

def make_prior_BsEtas(fs_data,Fits,Del,addrho,t_0,Npow,Nijk,rhopri,dpri,cpri,di000pri,di10npri):
    prior = gv.BufferDict()
    f = gv.BufferDict()
    for Fit in Fits:
        fit = Fit['conf']
        ms0 = Fit['m_ssea']
        ml0 = Fit['m_lsea']
        ms0val = float(Fit['m_s']) # valence untuned s mass
        Metas = Fit['M_daughter']/Fit['a'] # in GeV
        prior['Metac_{0}'.format(fit)] = globals()['Metac{0}'.format(fit)]/Fit['a'] #in GeV
        prior['mstuned_{0}'.format(fit)] = ms0val*(Metasphys/Metas)**2
        mltuned = prior['mstuned_{0}'.format(Fit['conf'])]/slratio 
        prior['MDs_{0}'.format(fit)] = Fit['M_parent_m{0}'.format(Fit['m_c'])] #lat units
        prior['deltas_{0}'.format(fit)] = ms0-prior['mstuned_{0}'.format(Fit['conf'])]     
        prior['deltasval_{0}'.format(fit)] = ms0val-prior['mstuned_{0}'.format(Fit['conf'])]
        prior['deltal_{0}'.format(fit)] = ml0-mltuned
        for mass in Fit['masses']:
            prior['MHs_{0}_m{1}'.format(fit,mass)] = Fit['M_parent_m{0}'.format(mass)]
            prior['MHs0_{0}_m{1}'.format(fit,mass)] = prior['MHs_{0}_m{1}'.format(fit,mass)] + Fit['a']*Del
            prior['MHsstar_{0}_m{1}'.format(fit,mass)] = prior['MHs_{0}_m{1}'.format(fit,mass)] + x*Fit['a']/prior['MHs_{0}_m{1}'.format(fit,mass)]
            for twist in Fit['twists']:
                tag = '{0}_m{1}_tw{2}'.format(fit,mass,twist)
                qsq = fs_data[fit]['qsq_m{0}_tw{0}'.format(mass,twist)]
                prior['z_{0}'.format(tag)] = make_z(qsq,t_0,prior['MHs_{0}_m{1}'.format(fit,mass)],Fit['M_daughter'])    # x values go in prior
                prior['qsq_{0}'.format(tag)] = qsq
                f['f0_{0}'.format(tag)] = fs_data[fit]['f0_m{0}_tw{0}'.format(mass,twist)]   # y values go in f   
                f['fp_{0}'.format(tag)] = fs_data[fit]['fp_m{0}_tw{0}'.format(mass,twist)]
                for key in f:
                    if f[key] == None:
                        del f[key]   #removes vector tw 0 etc
    if addrho:
        prior['0rho'] =gv.gvar(Npow*[rhopri])
        prior['prho'] =gv.gvar(Npow*[rhopri]) 
    prior['0d'] = gv.gvar(Nijk*[Nijk*[Nijk*[Npow*[dpri]]]])
    prior['0cl'] = gv.gvar(Npow*[cpri])
    prior['0cs'] = gv.gvar(Npow*[cpri])
    prior['0cc'] = gv.gvar(Npow*[cpri])
    prior['0csval'] = gv.gvar(Npow*[cvalpri])
    prior['pd'] = gv.gvar(Nijk*[Nijk*[Nijk*[Npow*[dpri]]]])
    for i in range(Nijk):
        if i != 0:
            prior['0d'][i][0][0][0] = gv.gvar(di000pri)
            for n in range(Npow):
                prior['0d'][i][1][0][n] = gv.gvar(di10npri)
                prior['plusd'][i][1][0][n] = gv.gvar(di10npri)
    prior['pcl'] = gv.gvar(Npow*[cpri])
    prior['pcs'] = gv.gvar(Npow*[cpri])
    prior['pcc'] = gv.gvar(Npow*[cpri])
    prior['pcsval'] = gv.gvar(Npow*[cvalpri])
    return(prior,f)
                
##########################################################################################################
    
def make_an_BsEtas(n,Nijk,addrho,p,tag,Fit,alat,mass): # tag is 0,p,T in this way, we can set fp(0)=f0(0) by just putting 0 for n=0 a is lattice spacing so we can use this to evaluate at different lattice spacings p is dict containing all values (prior or posterior or anything)
    fit = Fit['conf']
    an = 0
    for i in range(Nijk):
        for j in range(Nijk):
            for k in range(Nijk):
                if addrho:
                    an += (1 + p['{0}rho'.format(tag)][n]*gv.log(p['MHs_{0}_m{1}'.format(fit,mass)]/p['MDs_{0}'.format(fit)])) * (1 + (p['{0}csval'.format(tag)][n]*p['deltasval_{0}'.format(fit)] + p['{0}cs'.format(tag)][n]*p['deltas_{0}'.format(fit)] + 2*p['{0}cl'.format(tag)][n]*p['deltal_{0}'.format(fit)])/(10*p['mstuned_{0}'.format(fit)]) + p['{0}cc'.format(tag)][n]*((p['Metac_{0}'.format(fit)] - p['Metacphys'])/p['Metacphys'])) * p['{0}d'.format(tag)][i][j][k][n] * (LQCD*alat/p['MHh_{0}_m{1}'.format(fit,mass)])**int(i) * (float(mass)/np.pi)**int(2*j) * (LQCD*alat/np.pi)**int(2*k)
                else:
                    an += (1 + (p['{0}csval'.format(tag)][n]*p['deltasval_{0}'.format(fit)] + p['{0}cs'.format(tag)][n]*p['deltas_{0}'.format(fit)] + 2*p['{0}cl'.format(tag)][n]*p['deltal_{0}'.format(fit)])/(10*p['mstuned_{0}'.format(fit)]) + p['{0}cc'.format(tag)][n]*((p['Metac_{0}'.format(fit)] - p['Metacphys'])/p['Metacphys'])) * p['{0}d'.format(tag)][i][j][k][n] * (LQCD*alat/p['MHh_{0}_m{1}'.format(fit,mass)])**int(i) * (float(mass)/np.pi)**int(2*j) * (LQCD*alat/np.pi)**int(2*k)
    return(an)

##########################################################################################################

def make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass):
    tag = '0'
    f0 = 0
    for n in range(Npow):
        f0 += 1/(1-qsq/p['MHs0_{0}_m{1}'.format(Fit['conf'],mass)]**2) * make_an_BsEtas(n,Nijk,addrho,p,tag,Fit,alat) * z**n
    return(f0)

###########################################################################################################

def make_fp_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,fpf0same):
    fp = 0
    for n in range(Npow):
        if n == 0 and fpf0same:
            tag = '0'
        else:
            tag = 'p'
        fp += 1/(1-qsq/p['MHsstar_{0}_m{1}'.format(Fit['conf'],mass)]**2) * make_an_BsEtas(n,Nijk,addrho,p,tag,Fit,alat) * (z**n - (n/Npow) * (-1)**(n-N) *  z**N)
    return(f0)

############################################################################################################

def do_fit_BsEtas(Fits,f,Nijk,Npow,addrho,svdnoise,priornoise,prior,f):
    # have to define function in here so it only takes p as an argument (I think)
    ###############################
    def fit_function_BsEtas(p):
    models = {}
    for Fit in Fits:
        for mass in Fit['masses']:
            for twist in Fit['twists']:
                tag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)
                if 'f0_{0}'.format(tag) in f:
                    models['f0_{0}'.format(tag)] = make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,Fit['a'],p['qsq_{0}'.format(tag)],p['z_{0}'.format(tag)],mass)
                if 'fp_{0}'.format(tag) in f:
                    models['fp_{0}'.format(tag)] = make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,Fit['a'],p['qsq_{0}'.format(tag)],p['z_{0}'.format(tag)],mass)
    return(models)
    #################################
    
    p0 = None
    fit = lsqfit.nonlinear_fit(data=f, prior=prior, p0=p0, fcn=fit_function_BsEtas, svdcut=1e-15 ,add_svdnoise=svdnoise, add_priornoise=priornoise,fitter='gsl_multifit', alg='subspace2D', solver='cholesky', maxit=5000, tol=(1e-6,0.0,0.0) )
    gv.dump(fit.p,'Fits/{0}{1}chi{2:.3f}'.format(AddRho,Npow,fit.chi2/fit.dof))
    print(fit.format(maxline=True))
    return(fit.p)

#############################################################################################################
