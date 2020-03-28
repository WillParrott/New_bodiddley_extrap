from functions import *
import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

plt.rc("font",**{"size":18})
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
import collections
import copy
import os.path
import pickle
from collections import defaultdict

################# plots ######################
# speed of light (we use theory value but worth checking this) ###done
# data and f0/fp with and without pole in z and qsq individually
# f0/fp together in qsq and z
# error in qsq
# f0(0) fp(max) and f0(max) in Mh
# ratio in E compared with expectation
# ratio in qsq compared with HQET
# beta and delta
# data at different lattice spacings
################## table outputs ############################
# aMh (aq)^2 aEeta S V N f0 fp on each lattice spacing
# ans and pole masses
# dict of differnt lattice spacings
################### global variables ##########################################
figsca = 14  #size for saving figs
figsize = ((figsca,2*figsca/(1+np.sqrt(5))))
nopts = 200 #number of points on plot
ms = 20 #markersize
alpha = 0.4
fontsizeleg = 18 #legend
fontsizelab = 30 #legend
cols = ['b','r','g','c'] #for each mass
symbs = ['o','^','*']    # for each conf
lines = ['-','--','-.'] # for each conf
major = 15
minor = 8 

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
    aGev = gv.gvar(a)/(gv.gvar(hbar)*clight*1e-2) #bar in x10^-25 GeV seconds so hence 1e-2 
    return(aGev)

####################################################################################################

def speed_of_light(Fits):
    plt.figure(1,figsize=figsize)
    points = ['ko','r^','b*']
    i=0
    for Fit in Fits:
        x = []
        y = []
        for tw,twist in enumerate(Fit['twists']):
            if twist != '0':
                x.append(Fit['momenta'][tw]**2)
                y.append((Fit['E_daughter_tw{0}_fit'.format(twist)]**2 - Fit['M_daughter']**2)/Fit['momenta'][tw]**2)
        y,yerr = unmake_gvar_vec(y)
        plt.errorbar(x,y,yerr=yerr,fmt=points[i],label=Fit['label'],ms=ms,mfc='none')
        i += 1
    plt.plot([0,0.5],[1,1],'k--',lw=3)
    plt.xlim((0,0.22))
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(loc='lower left',handles=handles,labels=labels,ncol=2,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$|ap_{\eta_s}|^2$',fontsize=fontsizelab)
    plt.ylabel('$(E_{\eta_s}^2-M_{\eta_s}^2)/p_{\eta_s}^2$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.02))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.02))
    plt.tight_layout()
    plt.savefig('Plots/speedoflight.pdf')
    plt.show()
    return()

#####################################################################################################

def f0_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho):
    i = 0
    for Fit in Fits:
        j = 0
        for mass in Fit['masses']:
            qsq = []
            z = []
            y = []
            for twist in Fit['twists']:
                q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                z.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))#all lat units
                y.append(fs_data[Fit['conf']]['f0_m{0}_tw{1}'.format(mass,twist)])
            qsq,qsqerr = unmake_gvar_vec(qsq)
            z,zerr = unmake_gvar_vec(z)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(2,figsize=figsize)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            plt.figure(3,figsize=figsize)
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            j += 1
        i += 1
    qsq = []
    z = []
    y = []
    p = make_p_physical_point_BsEtas(pfit,Fits,Del)
    for q2 in np.linspace(0,qsqmaxphys.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        zed = make_z(q2,t_0,MBsphys,Metasphys) #all GeV dimensions
        z.append(zed.mean)
        #        make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,amh)
        y.append(make_f0_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,q2,zed.mean,Fits[0]['masses'][0],0)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(2,figsize=figsize)
    plt.plot(qsq, ymean, color='b')
    plt.fill_between(qsq,ylow,yupp, color='b',alpha=alpha)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$f_0(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.04))
    plt.tight_layout()
    plt.savefig('Plots/f0poleinqsq.pdf')
    plt.close()
    
    plt.figure(3,figsize=figsize)
    plt.plot(z,ymean, color='b')
    plt.fill_between(z,ylow,yupp, color='b',alpha=alpha)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$z$',fontsize=fontsizelab)
    plt.ylabel(r'$f_0(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.04))
    plt.tight_layout()
    plt.savefig('Plots/f0poleinz.pdf')
    plt.close()
    return()

################################################################################################

def fp_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same):
    i = 0
    for Fit in Fits:
        j = 0
        for mass in Fit['masses']:
            qsq = []
            z = []
            y = []
            for twist in Fit['twists']:
                if fs_data[Fit['conf']]['fp_m{0}_tw{1}'.format(mass,twist)] != None:
                    q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                    qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                    z.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))#all lat units
                    y.append(fs_data[Fit['conf']]['fp_m{0}_tw{1}'.format(mass,twist)])
            qsq,qsqerr = unmake_gvar_vec(qsq)
            z,zerr = unmake_gvar_vec(z)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(4,figsize=figsize)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            plt.figure(5,figsize=figsize)
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            j += 1
        i += 1
    qsq = []
    z = []
    y = []
    p = make_p_physical_point_BsEtas(pfit,Fits,Del)
    for q2 in np.linspace(0,qsqmaxphys.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        zed = make_z(q2,t_0,MBsphys,Metasphys) #all GeV dimensions
        z.append(zed.mean)
        #        make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,amh)
        y.append(make_fp_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,q2,zed.mean,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(4,figsize=figsize)
    plt.plot(qsq, ymean, color='b')
    plt.fill_between(qsq,ylow,yupp, color='r',alpha=alpha)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$f_p(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.04))
    plt.tight_layout()
    plt.savefig('Plots/fppoleinqsq.pdf')
    plt.close()
    
    plt.figure(5,figsize=figsize)
    plt.plot(z,ymean, color='b')
    plt.fill_between(z,ylow,yupp, color='r',alpha=alpha)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$z$',fontsize=fontsizelab)
    plt.ylabel(r'$f_p(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.04))
    plt.tight_layout()
    plt.savefig('Plots/fppoleinz.pdf')
    plt.close()
    return()

################################################################################################

def f0_no_pole_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho):
    i = 0
    for Fit in Fits:
        j = 0
        for mass in Fit['masses']:
            qsq = []
            z = []
            y = []
            for twist in Fit['twists']:
                q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                z.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))#all lat units
                pole = 1-q2/pfit['MHs0_{0}_m{1}'.format(Fit['conf'],mass)]**2
                y.append(pole * fs_data[Fit['conf']]['f0_m{0}_tw{1}'.format(mass,twist)])
            qsq,qsqerr = unmake_gvar_vec(qsq)
            z,zerr = unmake_gvar_vec(z)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(6,figsize=figsize)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            plt.figure(7,figsize=figsize)
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            j += 1
        i += 1
    qsq = []
    z = []
    y = []
    p = make_p_physical_point_BsEtas(pfit,Fits,Del)
    for q2 in np.linspace(0,qsqmaxphys.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        pole = 1 - q2/MBs0phys**2
        zed = make_z(q2,t_0,MBsphys,Metasphys) #all GeV dimensions
        z.append(zed.mean)
        #        make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,amh)
        y.append(pole*make_f0_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,q2,zed.mean,Fits[0]['masses'][0],0)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(6,figsize=figsize)
    plt.plot(qsq, ymean, color='b')
    plt.fill_between(qsq,ylow,yupp, color='b',alpha=alpha)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$\left(1-\frac{q^2}{M^2_{H_{s}^0}} \right)f_0(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.04))
    plt.tight_layout()
    plt.savefig('Plots/f0nopoleinqsq.pdf')
    plt.close()
    
    plt.figure(7,figsize=figsize)
    plt.plot(z,ymean, color='b')
    plt.fill_between(z,ylow,yupp, color='b',alpha=alpha)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$z$',fontsize=fontsizelab)
    plt.ylabel(r'$\left(1-\frac{q^2}{M^2_{H_{s}^0}} \right)f_0(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.04))
    plt.tight_layout()
    plt.savefig('Plots/f0nopoleinz.pdf')
    plt.close()
    return()

################################################################################################

def fp_no_pole_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same):
    i = 0
    for Fit in Fits:
        j = 0
        for mass in Fit['masses']:
            qsq = []
            z = []
            y = []
            for twist in Fit['twists']:
                if fs_data[Fit['conf']]['fp_m{0}_tw{1}'.format(mass,twist)] != None:
                    q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                    qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                    z.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))#all lat units
                    pole = 1-q2/pfit['MHsstar_{0}_m{1}'.format(Fit['conf'],mass)]**2
                    y.append(pole * fs_data[Fit['conf']]['fp_m{0}_tw{1}'.format(mass,twist)])
            qsq,qsqerr = unmake_gvar_vec(qsq)
            z,zerr = unmake_gvar_vec(z)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(8,figsize=figsize)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            plt.figure(9,figsize=figsize)
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            j += 1
        i += 1
    qsq = []
    z = []
    y = []
    p = make_p_physical_point_BsEtas(pfit,Fits,Del)
    for q2 in np.linspace(0,qsqmaxphys.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        pole = 1 - q2/MBsstarphys**2
        zed = make_z(q2,t_0,MBsphys,Metasphys) #all GeV dimensions
        z.append(zed.mean)
        #        make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,amh)
        y.append(pole*make_fp_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,q2,zed.mean,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(8,figsize=figsize)
    plt.plot(qsq, ymean, color='r')
    plt.fill_between(qsq,ylow,yupp, color='r',alpha=alpha)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$\left(1-\frac{q^2}{M^2_{H_{s}^*}} \right)f_+(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.04))
    plt.tight_layout()
    plt.savefig('Plots/fpnopoleinqsq.pdf')
    plt.close()
    
    plt.figure(9,figsize=figsize)
    plt.plot(z,ymean, color='r')
    plt.fill_between(z,ylow,yupp, color='r',alpha=alpha)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$z$',fontsize=fontsizelab)
    plt.ylabel(r'$\left(1-\frac{q^2}{M^2_{H_{s}^*}} \right)f_+(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.04))
    plt.tight_layout()
    plt.savefig('Plots/fpnopoleinz.pdf')
    plt.close()
    return()

################################################################################################
