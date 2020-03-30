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
# data and f0/fp with and without pole in z and qsq individually ###done
# f0/fp together in qsq ### done 
# error in qsq #done 
# f0(0) fp(max) and f0(max) in Mh   #done
# ratio in E compared with expectation #done 
# ratio in qsq compared with HQET #done 
# beta and delta   # done 
# f0 fp with data at different lattice spacings #done 
################## table outputs ############################
# aMh (aq)^2 aEeta S V N f0 fp on each lattice spacing
# ans and pole masses
# dict of different lattice spacings #done 
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

def f0_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same):
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
        #        make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,fpf0sameamh)
        y.append(make_f0_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,q2,zed.mean,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(2,figsize=figsize)
    plt.plot(qsq, ymean, color='b')
    plt.fill_between(qsq,ylow,yupp, color='b',alpha=alpha)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2,loc='lower right')
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
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2,loc='lower left')
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
        #        make_fp_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,fpf0same,amh)
        y.append(make_fp_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,q2,zed.mean,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(4,figsize=figsize)
    plt.plot(qsq, ymean, color='r')
    plt.fill_between(qsq,ylow,yupp, color='r',alpha=alpha)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$f_+(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(1.0))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.2))
    plt.tight_layout()
    plt.savefig('Plots/fppoleinqsq.pdf')
    plt.close()
    
    plt.figure(5,figsize=figsize)
    plt.plot(z,ymean, color='r')
    plt.fill_between(z,ylow,yupp, color='r',alpha=alpha)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$z$',fontsize=fontsizelab)
    plt.ylabel(r'$f_+(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(1.0))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.2))
    plt.tight_layout()
    plt.savefig('Plots/fppoleinz.pdf')
    plt.close()
    return()

################################################################################################

def f0_no_pole_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same):
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
        pole = 1 - q2/(MBsphys+Del)**2
        zed = make_z(q2,t_0,MBsphys,Metasphys) #all GeV dimensions
        z.append(zed.mean)
        #        make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,fpf0same,amh)
        y.append(pole*make_f0_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,q2,zed.mean,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
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
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2)
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
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2)
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

def f0_fp_in_qsq(pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same):
    qsq = []
    y0 = []
    yp = []
    p = make_p_physical_point_BsEtas(pfit,Fits,Del)
    for q2 in np.linspace(0,qsqmaxphys.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        zed = make_z(q2,t_0,MBsphys,Metasphys) #all GeV dimensions
        #        make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,amh)
        y0.append(make_f0_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,q2,zed.mean,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
        yp.append(make_fp_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,q2,zed.mean,Fits[0]['masses'][0],fpf0same,0))
    y0mean,y0err = unmake_gvar_vec(y0)
    y0upp,y0low = make_upp_low(y0)
    ypmean,yperr = unmake_gvar_vec(yp)
    ypupp,yplow = make_upp_low(yp)
    plt.figure(10,figsize=figsize)
    plt.plot(qsq, y0mean, color='b')
    plt.fill_between(qsq,y0low,y0upp, color='b',alpha=alpha)
    plt.plot(qsq, ypmean, color='r')
    plt.fill_between(qsq,yplow,ypupp, color='r',alpha=alpha)
    plt.errorbar(0,0.297, yerr=0.047,fmt='k*',ms=ms,mfc='none',label = 'arXiv:1406.2279')
    plt.errorbar(qsqmaxphys.mean,0.816, yerr=0.035,fmt='k*',ms=ms,mfc='none')
    plt.errorbar(qsqmaxphys.mean,2.293, yerr=0.091,fmt='k*',ms=ms,mfc='none')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,loc='upper left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.text(10,1.0,'$f_+(q^2)$',fontsize=fontsizelab)
    plt.text(18.5,0.9,'$f_0(q^2)$',fontsize=fontsizelab)
    plt.tight_layout()
    plt.savefig('Plots/f0fpinqsq.pdf')
    plt.close()
    return()


################################################################################################

def f0_f0_fp_in_Mh(pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same):
    MHs = []
    f00 = []
    f0max = []
    fpmax = []
    for Mh in np.linspace(MDsphys.mean,MBsphys.mean,nopts): #q2 now in GeV
        p = make_p_Mh_BsEtas(pfit,Fits,Del,Mh)
        qsqmax = (Mh-Metasphys)**2
        zmax = make_z(qsqmax,t_0,Mh,Metasphys)
        MHs.append(Mh)
        #        make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,amh)
        f00.append(make_f0_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,0,0,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
        f0max.append(make_f0_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,qsqmax.mean,zmax,Fits[0]['masses'][0],,fpf0same,0))
        fpmax.append(make_fp_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,qsqmax.mean,zmax.mean,Fits[0]['masses'][0],fpf0same,0))
    f00mean,f00err = unmake_gvar_vec(f00)
    f0maxmean,f0maxerr = unmake_gvar_vec(f0max)
    fpmaxmean,fpmaxerr = unmake_gvar_vec(fpmax)
    f00upp,f00low = make_upp_low(f00)
    f0maxupp,f0maxlow = make_upp_low(f0max)
    fpmaxupp,fpmaxlow = make_upp_low(fpmax)
    plt.figure(11,figsize=figsize)
    plt.plot(MHs, f00mean, color='k')
    plt.fill_between(MHs,f00low,f00upp, color='k',alpha=alpha)
    plt.plot(MHs, f0maxmean, color='b')
    plt.fill_between(MHs,f0maxlow,f0maxupp, color='b',alpha=alpha)
    plt.plot(MHs, fpmaxmean, color='r')
    plt.fill_between(MHs,fpmaxlow,fpmaxupp, color='r',alpha=alpha)
    plt.xlabel('$M_{H_s}[\mathrm{GeV}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.plot([MDsphys.mean,MDsphys.mean],[-10,10],'k--',lw=1)
    plt.text(MDsphys.mean+0.02,0.05,'$M_{D_s}$',fontsize=fontsizelab)
    plt.plot([MBsphys.mean,MBsphys.mean],[-10,10],'k--',lw=1)
    plt.text(MBsphys.mean-0.02,0.05,'$M_{B_s}$',fontsize=fontsizelab,horizontalalignment='right')
    plt.text(3.1,2.0,'$f_+(q^2_{\mathrm{max}})$',fontsize=fontsizelab)
    plt.text(2.5,0.3,'$f_{0,+}(0)$',fontsize=fontsizelab)
    plt.text(4.5,1.0,'$f_0(q^2_{\mathrm{max}})$',fontsize=fontsizelab)
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().set_ylim([0,2.8])
    plt.tight_layout()
    plt.savefig('Plots/f0f0fpinmh.pdf')
    plt.close()
    return()

#####################################################################################################

def beta_delta_in_Mh(pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same):
    MHs = []
    delta = []
    invbeta = []
    for Mh in np.linspace(MDsphys.mean,MBsphys.mean,nopts): #q2 now in GeV
        p = make_p_Mh_BsEtas(pfit,Fits,Del,Mh)
        MHs.append(Mh)
        d,invb = make_beta_delta_BsEtas(Fits,t_0,Nijk,Npow,addrho,p,fpf0same,Del,Mh)
        delta.append(d) 
        invbeta.append(invb)
    deltamean,deltaerr = unmake_gvar_vec(delta)
    invbetamean,invbetaerr = unmake_gvar_vec(invbeta)
    deltaupp,deltalow = make_upp_low(delta)
    invbetaupp,invbetalow = make_upp_low(invbeta)
    plt.figure(12,figsize=figsize)
    plt.plot(MHs, invbetamean, color='b')
    plt.fill_between(MHs,invbetalow,invbetaupp, color='b',alpha=alpha)
    plt.plot(MHs, deltamean, color='r')
    plt.fill_between(MHs,deltalow, deltaupp, color='r',alpha=alpha)
    plt.xlabel('$M_{H_s}[\mathrm{GeV}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.plot([MDsphys.mean,MDsphys.mean],[-10,10],'k--',lw=1)
    plt.text(MDsphys.mean+0.02,-0.45,'$M_{D_s}$',fontsize=fontsizelab)
    plt.plot([MBsphys.mean,MBsphys.mean],[-10,10],'k--',lw=1)
    plt.text(MBsphys.mean-0.02,-0.45,'$M_{B_s}$',fontsize=fontsizelab,horizontalalignment='right')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.text(4.5,0.3,r'$\delta$',fontsize=fontsizelab)
    plt.text(2.7,0.7,r'$\beta^{-1}$',fontsize=fontsizelab)
    plt.axes().set_ylim([-0.5,1.0])
    plt.tight_layout()
    plt.savefig('Plots/betadeltainmh.pdf')
    plt.close()
    return()

#####################################################################################################

def HQET_ratio_in_qsq(pfit,Fits,Del,Nijk,Npow,addrho,fpf0same,t_0):
    theory = gv.gvar('1.87(27)')
    qsq = []
    z = []
    rat = []
    p = make_p_physical_point_BsEtas(pfit,Fits,Del)
    for q2 in np.linspace(0,(MBsphys**2).mean,nopts): #q2 now in GeV
        qsq.append(q2)
        z = make_z(q2,t_0,MBsphys,Metasphys) #all GeV dimensions
        #        make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,amh)
        f0 = make_f0_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,q2,z.mean,Fits[0]['masses'][0],fpf0same,0)
        fp = make_fp_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,q2,z.mean,Fits[0]['masses'][0],fpf0same,0)
        rat.append((f0/fp)*(1/(1-q2/MBsstarphys**2))) #only need one fit
    ratmean,raterr = unmake_gvar_vec(rat)
    ratupp,ratlow = make_upp_low(rat)
    plt.figure(13,figsize=figsize)
    plt.plot(qsq, ratmean, color='b')
    plt.fill_between(qsq,ratlow,ratupp, color='b',alpha=alpha)
    plt.plot([0,(MBsphys**2).mean],[theory.mean,theory.mean],color='r')
    plt.fill_between([0,(MBsphys**2).mean],[theory.mean-theory.sdev,theory.mean-theory.sdev],[theory.mean+theory.sdev,theory.mean+theory.sdev],color='r',alpha=alpha)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$\frac{f_0(q^2)}{f_+(q^2)} \left(1-\frac{q^2}{M^2_{B^*_s}}\right)^{-1}$',fontsize=fontsizelab)
    plt.plot([qsqmaxphys.mean,qsqmaxphys.mean],[-10,10],'k--',lw=1)
    plt.text(qsqmaxphys.mean-0.02,2.4,'$q^2_{\mathrm{max}}$',fontsize=fontsizelab,horizontalalignment='right')
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().set_ylim([0.9,2.9])
    plt.axes().set_xlim([0,(MBsphys**2).mean])
    plt.tight_layout()
    plt.savefig('Plots/HQETrat.pdf')
    plt.close()
    return()

####################################################################################################

def Hill_ratios_in_E(pfit,Fits,Del,t_0,Nijk,Npow,addrho,fpf0same):
    Es = []
    rat0 = []
    ratp = []
    Emax = (MDsphys**2 + Metasphys**2)/MDsphys
    theory = gv.sqrt(MBsphys/MDsphys)
    for E in np.linspace(Metasphys.mean,Emax.mean,nopts): #q2 now in GeV
        pD = make_p_Mh_BsEtas(pfit,Fits,Del,MDsphys.mean)
        pB = make_p_physical_point_BsEtas(pfit,Fits,Del)
        qsqD = MDsphys**2 + Metasphys**2 - 2*MDsphys*E
        qsqB =  MBsphys**2 + Metasphys**2 - 2*MBsphys*E
        zD = make_z(qsqD,t_0,MDsphys,Metasphys)
        zB = make_z(qsqB,t_0,MBsphys,Metasphys)
        Es.append(E)
        f0D = make_f0_BsEtas(Nijk,Npow,addrho,pD,Fits[0],0,qsqD,zD,Fits[0]['masses'][0],fpf0same,0)
        fpD = make_fp_BsEtas(Nijk,Npow,addrho,pD,Fits[0],0,qsqD,zD,Fits[0]['masses'][0],fpf0same,0)
        f0B = make_f0_BsEtas(Nijk,Npow,addrho,pB,Fits[0],0,qsqB,zB,Fits[0]['masses'][0],fpf0same,0)
        fpB = make_fp_BsEtas(Nijk,Npow,addrho,pB,Fits[0],0,qsqB,zB,Fits[0]['masses'][0],fpf0same,0)
        ratp.append(fpB/fpD)
        rat0.append(f0D/f0B)
    rat0mean,rat0err = unmake_gvar_vec(rat0)
    ratpmean,ratperr = unmake_gvar_vec(ratp)
    rat0upp,rat0low = make_upp_low(rat0)
    ratpupp,ratplow = make_upp_low(ratp)
    plt.figure(14,figsize=figsize)
    plt.plot(Es, rat0mean, color='b')
    plt.fill_between(Es,rat0low,rat0upp, color='b',alpha=alpha)
    plt.plot(Es, ratpmean, color='r')
    plt.fill_between(Es,ratplow, ratpupp, color='r',alpha=alpha)
    plt.plot([Metasphys.mean,Emax.mean],[theory.mean,theory.mean],color='k')
    plt.fill_between([Metasphys.mean,Emax.mean],[theory.mean-theory.sdev,theory.mean-theory.sdev],[theory.mean+theory.sdev,theory.mean+theory.sdev],color='k',alpha=alpha/2)
    plt.xlabel('$E[\mathrm{GeV}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    #plt.text(4.5,0.3,r'$\delta$',fontsize=fontsizelab)
    #plt.text(2.7,0.7,r'$\beta^{-1}$',fontsize=fontsizelab)
    #plt.axes().set_ylim([-0.5,1.0])
    plt.axes().set_xlim([Metasphys.mean,Emax.mean])
    plt.tight_layout()
    plt.savefig('Plots/HillratinE.pdf')
    plt.close()
    return()

#####################################################################################################

def Hill_ratios_in_lowE(pfit,Fits,Del,t_0,Nijk,Npow,addrho,fpf0same):
    #use E as light mass and make unphysically low
    Es = []
    rat0 = []
    ratp = []
    Emax = Metasphys
    Mpiphys = gv.gvar('0.1349770(5)')
    Emin = Mpiphys
    theory = gv.sqrt(MBsphys/MDsphys)
    for E in np.linspace(Mpiphys.mean,Emax.mean,nopts): #q2 now in GeV
        pD = make_p_Mh_BsEtas(pfit,Fits,Del,MDsphys.mean)
        pB = make_p_physical_point_BsEtas(pfit,Fits,Del)
        qsqD = (MDsphys-E)**2
        qsqB = (MBsphys-E)**2
        zD = make_z(qsqD,t_0,MDsphys,E)
        zB = make_z(qsqB,t_0,MBsphys,E)
        Es.append(E)
        f0D = make_f0_BsEtas(Nijk,Npow,addrho,pD,Fits[0],0,qsqD,zD,Fits[0]['masses'][0],fpf0same,0)
        fpD = make_fp_BsEtas(Nijk,Npow,addrho,pD,Fits[0],0,qsqD,zD,Fits[0]['masses'][0],fpf0same,0)
        f0B = make_f0_BsEtas(Nijk,Npow,addrho,pB,Fits[0],0,qsqB,zB,Fits[0]['masses'][0],fpf0same,0)
        fpB = make_fp_BsEtas(Nijk,Npow,addrho,pB,Fits[0],0,qsqB,zB,Fits[0]['masses'][0],fpf0same,0)
        ratp.append(fpB/fpD)
        rat0.append(f0D/f0B)
    rat0mean,rat0err = unmake_gvar_vec(rat0)
    ratpmean,ratperr = unmake_gvar_vec(ratp)
    rat0upp,rat0low = make_upp_low(rat0)
    ratpupp,ratplow = make_upp_low(ratp)
    plt.figure(14,figsize=figsize)
    plt.plot(Es, rat0mean, color='b')
    plt.fill_between(Es,rat0low,rat0upp, color='b',alpha=alpha)
    plt.plot(Es, ratpmean, color='r')
    plt.fill_between(Es,ratplow, ratpupp, color='r',alpha=alpha)
    plt.plot([Emin.mean,Emax.mean],[theory.mean,theory.mean],color='k')
    plt.fill_between([Emin.mean,Emax.mean],[theory.mean-theory.sdev,theory.mean-theory.sdev],[theory.mean+theory.sdev,theory.mean+theory.sdev],color='k',alpha=alpha/2)
    plt.xlabel('$E[\mathrm{GeV}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    #plt.text(4.5,0.3,r'$\delta$',fontsize=fontsizelab)
    #plt.text(2.7,0.7,r'$\beta^{-1}$',fontsize=fontsizelab)
    #plt.axes().set_ylim([-0.5,1.0])
    plt.axes().set_xlim([Emin.mean,Emax.mean])
    plt.tight_layout()
    plt.savefig('Plots/HillratinlowE.pdf')
    plt.close()
    return()

#####################################################################################################

def f0_different_a_in_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same,afm):
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
            z,zerr = unmake_gvar_vec(z)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(15,figsize=figsize)
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
        y.append(make_f0_BsEtas(Nijk,Npow,addrho,p,Fits[0],convert_Gev(afm).mean,q2,zed.mean,Fits[0]['masses'][0],fpf0same,convert_Gev(afm).mean*mbphys)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(15,figsize=figsize)
    plt.plot(z,ymean, color='b')
    plt.fill_between(z,ylow,yupp, color='b',alpha=alpha)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2,loc='lower left')
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
    plt.savefig('Plots/f0poleinza{0}.pdf'.format(afm))
    plt.close()
    return()

###################################################################################################
    
def fp_different_a_in_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same,afm):
    #takes lattice spacing a in fm
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
            z,zerr = unmake_gvar_vec(z)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(16,figsize=figsize)
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
        y.append(make_fp_BsEtas(Nijk,Npow,addrho,p,Fits[0],convert_Gev(afm).mean,q2,zed.mean,Fits[0]['masses'][0],fpf0same,convert_Gev(afm).mean*mbphys)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
       
    plt.figure(16,figsize=figsize)
    plt.plot(z,ymean, color='r')
    plt.fill_between(z,ylow,yupp, color='r',alpha=alpha)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$z$',fontsize=fontsizelab)
    plt.ylabel(r'$f_+(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(1.0))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.2))
    plt.tight_layout()
    plt.savefig('Plots/fppoleinza{0}.pdf'.format(afm))
    plt.close()
    return()

###################################################################################################

def error_plot(pfit,prior,Fits,Nijk,Npow,f,t_0,Del,addrho,fpf0same):
    qsqs = np.linspace(0,qsqmaxphys.mean,nopts)
    f0,fp = output_error_BsEtas(pfit,prior,Fits,Nijk,Npow,f,qsqs,t_0,Del,addrho,fpf0same)

    plt.figure(17,figsize=figsize)
    ax1 = plt.subplot(211)
    ax1b = ax1.twinx()
    ax1.plot(qsqs,f0[1],color='r', ls='--',lw=2,label='Inputs')
    ax1.plot(qsqs,f0[2],color='purple',ls=':',lw=2,label='q mistunings')
    ax1.plot(qsqs,f0[3],color='g',ls='-',lw=2,label='Statistics')
    ax1.plot(qsqs,f0[4],color='b',ls='-.',lw=2,label='HQET')
    ax1.plot(qsqs,f0[5],color='k',ls='-',lw=4,label='Discretisation')
    ax1.set_ylabel('$(f_0(q^2)~\% \mathrm{err})^2 $ ',fontsize=fontsizelab)
    ax1.tick_params(width=2,labelsize=fontsizelab)
    ax1.tick_params(which='major',length=major)
    ax1.tick_params(which='minor',length=minor)
    #plt.gca().yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('none')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.yaxis.set_major_locator(MultipleLocator(10))
    ax1.yaxis.set_minor_locator(MultipleLocator(2))
    ax1.set_xlim([0,qsqmaxphys.mean])
    ####################################### right hand y axis ###
    ax1b.plot(qsqs,f0[1],color='r', ls='--',lw=2,label='Inputs')
    ax1b.plot(qsqs,f0[2],color='purple',ls=':',lw=2,label='q mistunings')
    ax1b.plot(qsqs,f0[3],color='g',ls='-',lw=2,label='Statistics')
    ax1b.plot(qsqs,f0[4],color='b',ls='-.',lw=2,label='HQET')
    ax1b.plot(qsqs,f0[5],color='k',ls='-',lw=4,label='Discretisation')
    ax1b.set_ylabel('$f_0(q^2)~\% \mathrm{err}$ ',fontsize=fontsizelab)
    ax1b.tick_params(width=2,labelsize=fontsizelab)
    ax1b.tick_params(which='major',length=major)
    ax1b.tick_params(which='minor',length=minor)
    ax1b.set_yticks([4,9,16,25,36,49,64])
    ax1b.set_yticklabels(['2','','4','','6','','8'])
    plt.legend(loc='upper right',ncol=2,fontsize=fontsizeleg)

    ax2 = plt.subplot(212,sharex=ax1)
    ax2b = ax2.twinx()
    ax2.plot(qsqs,fp[1],color='r', ls='--',lw=2)
    ax2.plot(qsqs,fp[2],color='purple',ls=':',lw=2)
    ax2.plot(qsqs,fp[3],color='g',ls='-',lw=2)
    ax2.plot(qsqs,fp[4],color='b',ls='-.',lw=2)
    ax2.plot(qsqs,fp[5],color='k',ls='-',lw=4)
    ax2.set_xlabel(r'$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    ax2.set_ylabel('$(f_+(q^2)~\%\mathrm{err})^2$',fontsize=fontsizelab)
    ax2.tick_params(width=2,labelsize=fontsizelab)
    ax2.tick_params(which='major',length=major)
    ax2.tick_params(which='minor',length=minor)
    ax2.xaxis.set_major_locator(MultipleLocator(5))
    ax2.xaxis.set_minor_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(20))
    ax2.yaxis.set_minor_locator(MultipleLocator(5))
    ax2.set_xlim([0,qsqmaxphys.mean])
    #plt.axes().set_ylim([-0.8,2.5])
    #plt.axes().set_xlim([lower-0.22,upper+0.22])
    ######## right hand axis ##
    ax2b.plot(qsqs,fp[1],color='r', ls='--',lw=2)
    ax2b.plot(qsqs,fp[2],color='purple',ls=':',lw=2)
    ax2b.plot(qsqs,fp[3],color='g',ls='-',lw=2)
    ax2b.plot(qsqs,fp[4],color='b',ls='-.',lw=2)
    ax2b.plot(qsqs,fp[5],color='k',ls='-',lw=4)
    ax2b.set_xlabel(r'$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    ax2b.set_ylabel('$f_+(q^2)~\%\mathrm{err}$',fontsize=fontsizelab)
    ax2b.tick_params(width=2,labelsize=fontsizelab)
    ax2b.tick_params(which='major',length=major)
    ax2b.tick_params(which='minor',length=minor)
    ax2b.set_yticks([4,9,16,25,36,49,64,81,100])
    ax2b.set_yticklabels(['2','','4','','6','','8','','10'])
    plt.tight_layout()
    plt.savefig('Plots/f0fpluserr.pdf')
    plt.close()
    return()

###################################################################################################

def table_of_as(Fits,pfit,Nijk,Npow,fpf0same,addrho):
    list0 = []
    listp = []
    Fit = Fits[0]
    mass = Fit['masses'][0]
    fit = Fit['conf']
    p = make_p_physical_point_BsEtas(pfit,Fits,Del)
    atab = open('tabls/tablesofas.txt','w')
    for n in range Npow:
        if n == 0:
            atab.write('      {0}&'.format(make_an_BsEtas(n,Nijk,addrho,p,'0',Fit,0,mass,0,fpf0same)))
        else:
            atab.write('{0}&'.format(make_an_BsEtas(n,Nijk,addrho,p,'0',Fit,0,mass,0,fpf0same)))
        list0.append(make_an_BsEtas(n,Nijk,addrho,p,'0',Fit,0,mass,0,fpf0same))
        listp.append(make_an_BsEtas(n,Nijk,addrho,p,'p',Fit,0,mass,0,fpf0same))
    for n in range Npow:           
        atab.write('{0}&'.format(make_an_BsEtas(n,Nijk,addrho,p,'p',Fit,0,mass,0,fpf0same)))
    atab.write('{0}&{1}\\\\ [1ex]\n'.format(p['MHs0_{0}_m{1}'.format(fit,mass)],p['MHsstar_{0}_m{1}'.format(fit,mass)]))
    atab.write('      \hline \n')
    list0.extend(listp)
    list0.append(p['MHs0_{0}_m{1}'.format(fit,mass)])
    list0.append(p['MHsstar_{0}_m{1}'.format(fit,mass)])
    covar = gv.evalcorr(list0)
    for i in range(2*Npow+2):
            atab.write('\n      ')
            for k in range(i):
                atab.write('&')
            for j in range(i,2*Npow+2):
                #print(covar[i][j])
                atab.write('{0:.5f}'.format(covar[i][j]))
                if j != 2*Npow+1:
                    atab.write('&')
                else:
                    atab.write('\\\\ [0.5ex]')
    atab.close()
    return()

##################################################################################################

def results_tables(fs_data,Fit):
    table = open('Tables/{0}table.txt'.format(Fit['conf']),'w')
    for mass in Fit['masses']:
        table.write('      \hline \n')
        table.write('      &{0}'.format(mass))
        for tw,twist in enumerate(twists):
            if tw = 0:
                table.write('&{1}&{0}&{2}&{3}&{4}'.format(fs_data['qsq_m{0}_tw{1}'.format(mass,twist)],Fit['M_parent_m{0}'.format(mass)],Fit['E_daughter_tw{0}_theory'.format(twist)],Fit['S_m{0}_tw{1}'.format(mass,twist)],Fit['V_m{0}_tw{1}'.format(mass,twist)]))
            else:
                table.write('      &&&{0}&{1}&{2}&{3}'.format(fs_data['qsq_m{0}_tw{1}'.format(mass,twist)],Fit['E_daughter_tw{0}_theory'.format(twist)],Fit['S_m{0}_tw{1}'.format(mass,twist)],Fit['V_m{0}_tw{1}'.format(mass,twist)]))
            if fs['fp_m{0}_tw{1}'.format(mass,twist)] != None:
                table.write('&{0}&{1}\\\\ [1ex]\n'.format(fs_data['f0_m{0}_tw{1}'.format(mass,twist)],fs_data['f0_m{0}_tw{1}'.format(mass,twist)]))
            else:
                table.write('&{0}&\\\\ [1ex]\n'.format(fs_data['f0_m{0}_tw{1}'.format(mass,twist)]))

        
    table.close()
    return()
###################################################################################################
