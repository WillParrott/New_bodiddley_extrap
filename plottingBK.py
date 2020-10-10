from functionsBK import *
import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rc("font",**{"size":20})
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
factor = 0.5 #multiplies everything to make smaller for big plots etc usually 1
figsca = 14  #size for saving figs
figsize = ((figsca,2*figsca/(1+np.sqrt(5))))
lw =2*factor
nopts = 200 #number of points on plot
ms = 25*factor #markersize
alpha = 0.4
fontsizeleg = 25*factor #legend
fontsizelab = 35*factor #legend
cols = ['b','r','g','c'] #for each mass
symbs = ['o','^','*','D','d','s','p','>','<']    # for each conf
lines = ['-','--','-.','-',':','-','--','-.'] # for each conf
major = 15*factor
minor = 8*factor 
capsize = 10*factor

####################################################################################################

def speed_of_light(Fits):
    plt.figure(1,figsize=figsize)
    points = ['bo','b^','b*','ko','k^','k*','kD','ks']
    i=0
    plotfits = []
    for Fit in Fits:
        if Fit['conf'] not in ['Fs','SFs','UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
        x = []
        y = []
        for tw,twist in enumerate(Fit['twists']):
            if twist != '0':
                x.append(Fit['momenta'][tw]**2)
                y.append((Fit['E_daughter_tw{0}_fit'.format(twist)]**2 - Fit['M_daughter']**2)/Fit['momenta'][tw]**2)
        y,yerr = unmake_gvar_vec(y)
        plt.errorbar(x,y,yerr=yerr,fmt=points[i],label=Fit['label'],ms=ms,mfc='none')
        i += 1
    plt.plot([0,0.6],[1,1],'k--',lw=3)
    plt.xlim((0,0.5))
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(loc='upper right',handles=handles,labels=labels,ncol=2,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$|ap_{K}|^2$',fontsize=fontsizelab)
    plt.ylabel('$(E_{K}^2-M_{K}^2)/p_{K}^2$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.02))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.02))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    #plt.xlim([0,0.2])
    #plt.ylim([0.9,1.2])
    plt.tight_layout()
    plt.savefig('Plots/speedoflight.pdf')
    #plt.show()
    return()

#####################################################################################################

def Z_V_plots(Fits,fs_data):
    plt.figure(19,figsize=figsize)
    i = 0
    plotfits = []
    for Fit in Fits:
        if Fit['conf'] not in ['UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
        x = []
        y = []
        for mass in Fit['masses']:
            x.append(float(mass)**2)
            Z_V = fs_data[Fit['conf']]['Z_v_m{0}'.format(mass)]
            y.append(Z_V)
        y,yerr = unmake_gvar_vec(y)
        if Fit['conf'][-1] == 's':
            plt.errorbar(x,y,yerr=yerr,fmt=symbs[i],color='r',label=Fit['label'],ms=ms,mfc='none')
        elif Fit['conf'][-1] == 'p':
            plt.errorbar(x,y,yerr=yerr,fmt=symbs[i],color='b',label=Fit['label'],ms=ms,mfc='none')
        else:
            plt.errorbar(x,y,yerr=yerr,fmt=symbs[i],color='k',label=Fit['label'],ms=ms,mfc='none')
        i+=1
    plt.plot([-0.1,1.1],[1,1],'k--')
    plt.xlim([0,0.9])
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=3,loc='upper left')
    plt.xlabel('$(am_h)^2$',fontsize=fontsizelab)
    plt.ylabel('$Z_V$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.05))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.02))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    plt.tight_layout()
    plt.savefig('Plots/Z_Vinamhsq.pdf')            
    return()

#####################################################################################################

def f0_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata):
    i = 0
    plotfits = []
    for Fit in Fits:
        if Fit['conf'] not in ['Fs','SFs','UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
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
    p = make_p_physical_point_BK(pfit,Fits)
    for q2 in np.linspace(0,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        zed = make_z(q2,t_0,p['MBphys'],p['MKphys']) #all GeV dimensions
        if zed == 0:
            z.append(zed)
        else:
            z.append(zed.mean)
        #        make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,fpf0sameamh)
        y.append(make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(2,figsize=figsize)
    plt.plot(qsq, ymean, color='b')
    plt.fill_between(qsq,ylow,yupp, color='b',alpha=alpha)
    #if dataf0maxBK != None and adddata:
    #    plt.errorbar(qsqmaxphysBK.mean, dataf0maxBK.mean, xerr=qsqmaxphysBK.sdev, yerr=dataf0maxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$\mathrm{arXiv:} 1510.07446$')
   # if dataf00BK != None and adddata:
   #     plt.errorbar(0, dataf00BK.mean, yerr=dataf00BK.sdev, color='k', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
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
    #if dataf0maxBK != None and adddata:
    #    plt.errorbar( make_z(qsqmaxphysBK,t_0,MBphys,MKphys).mean, dataf0maxBK.mean, xerr=make_z(qsqmaxphysBK,t_0,MBphys,MKphys).sdev, yerr=dataf0maxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$\mathrm{arXiv:} 1510.07446$')
    #if dataf00BK != None and adddata:
    #    plt.errorbar(make_z(0,t_0,MBphys,MKphys).mean, dataf00BK.mean,xerr = make_z(0,t_0,MBphys,MKphys).sdev ,yerr=dataf00BK.sdev, color='k', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
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

def fp_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,const2):
    i = 0
    plotfits = []
    for Fit in Fits:
        if Fit['conf'] not in ['Fs','SFs','UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
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
    p = make_p_physical_point_BK(pfit,Fits)
    for q2 in np.linspace(0,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        zed = make_z(q2,t_0,p['MBphys'],p['MKphys']) #all GeV dimensions
        if zed ==0:
            z.append(zed)
        else:
            z.append(zed.mean)
        y.append(make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(4,figsize=figsize)
    plt.plot(qsq, ymean, color='r')
    plt.fill_between(qsq,ylow,yupp, color='r',alpha=alpha)
    #if datafpmaxBK != None and adddata:
    #    plt.errorbar(qsqmaxphysBK.mean, datafpmaxBK.mean, xerr=qsqmaxphysBK.sdev, yerr=datafpmaxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
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
   # if datafpmaxBK != None and adddata:
   #     plt.errorbar( make_z(qsqmaxphysBK,t_0,MBphys,MKphys).mean, datafpmaxBK.mean, xerr=make_z(qsqmaxphysBK,t_0,MBphys,MKphys).sdev, yerr=datafpmaxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
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

def fT_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata):
    i = 0
    plotfits = []
    for Fit in Fits:
        if Fit['conf'] not in ['VCp','Cp','Fs','SFs','UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
        j = 0
        for mass in Fit['masses']:
            qsq = []
            z = []
            y = []
            for twist in Fit['twists']:
                if fs_data[Fit['conf']]['fT_m{0}_tw{1}'.format(mass,twist)] != None:
                    q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                    qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                    z.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))#all lat units
                    y.append(fs_data[Fit['conf']]['fT_m{0}_tw{1}'.format(mass,twist)])
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
    p = make_p_physical_point_BK(pfit,Fits)
    for q2 in np.linspace(0,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        zed = make_z(q2,t_0,p['MBphys'],p['MKphys']) #all GeV dimensions
        if zed ==0:
            z.append(zed)
        else:
            z.append(zed.mean)
        #        make_fp_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,fpf0same,amh)
        y.append(make_fT_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(4,figsize=figsize)
    plt.plot(qsq, ymean, color='g')
    plt.fill_between(qsq,ylow,yupp, color='g',alpha=alpha)
    #if datafTmaxBK != None and adddata:
    #    plt.errorbar(qsqmaxphysBK.mean, datafTmaxBK.mean, xerr=qsqmaxphysBK.sdev, yerr=datafTmaxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$f_T(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(1.0))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.2))
    plt.tight_layout()
    plt.savefig('Plots/fTpoleinqsq.pdf')
    plt.close()
    
    plt.figure(5,figsize=figsize)
    plt.plot(z,ymean, color='g')
    plt.fill_between(z,ylow,yupp, color='g',alpha=alpha)
    #if datafTmaxBK != None and adddata:
    #    plt.errorbar( make_z(qsqmaxphysBK,t_0,MBphys,MKphys).mean, datafTmaxBK.mean, xerr=make_z(qsqmaxphysBK,t_0,MBphys,MKphys).sdev, yerr=datafTmaxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$z$',fontsize=fontsizelab)
    plt.ylabel(r'$f_T(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(1.0))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.2))
    plt.tight_layout()
    plt.savefig('Plots/fTpoleinz.pdf')
    plt.close()
    return()

################################################################################################

def f0_no_pole_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata):
    i = 0
    plotfits = []
    for Fit in Fits:
        if Fit['conf'] not in ['Fs','SFs','UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
        fit = Fit['conf']
        j = 0
        for mass in Fit['masses']:
            qsq = []
            z = []
            y = []
            for twist in Fit['twists']:
                q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                z.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))#all lat units
                MHs0 = pfit['MH_{0}_m{1}'.format(fit,mass)] + pfit['a_{0}'.format(fit)] * Del
                pole = 1-(q2/MHs0**2)
                y.append(pole * fs_data[fit]['f0_m{0}_tw{1}'.format(mass,twist)])
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
    p = make_p_physical_point_BK(pfit,Fits)
    for q2 in np.linspace(0,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        pole = 1 - q2/(p['MBphys']+Del)**2
        zed = make_z(q2,t_0,p['MBphys'],p['MKphys']) #all GeV dimensions
        if zed ==0:
            z.append(zed)
        else:
            z.append(zed.mean)
        #        make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,fpf0same,amh)
        y.append(pole*make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(6,figsize=figsize)
    plt.plot(qsq, ymean, color='b')
    plt.fill_between(qsq,ylow,yupp, color='b',alpha=alpha)
    #if dataf0maxBsEtas != None and adddata:
   #     pole = 1 - qsqmaxphysBK/(MBsphys+Del)**2
   #     plt.errorbar(qsqmaxphysBK.mean, (pole*dataf0maxBsEtas).mean, xerr=qsqmaxphysBK.sdev, yerr=(pole*dataf0maxBsEtas).sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$\mathrm{arXiv:} 1510.07446$')
    #if dataf00BsEtas != None and adddata:
   #     plt.errorbar(0, dataf00BsEtas.mean, yerr=dataf00BsEtas.sdev, color='k', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
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
   # if dataf0maxBsEtas != None and adddata:
   #     pole = 1 - qsqmaxphysBK/(MBsphys+Del)**2
   #     plt.errorbar( make_z(qsqmaxphysBK,t_0,MBsphys,Metasphys).mean, (pole*dataf0maxBsEtas).mean, xerr=make_z(qsqmaxphysBK,t_0,MBsphys,Metasphys).sdev, yerr=(pole*dataf0maxBsEtas).sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$\mathrm{arXiv:} 1510.07446$')
   # if dataf00BsEtas != None and adddata:
  #      plt.errorbar(make_z(0,t_0,MBsphys,Metasphys).mean, dataf00BsEtas.mean,xerr = make_z(0,t_0,MBsphys,Metasphys).sdev ,yerr=dataf00BsEtas.sdev, color='k', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper left')
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

def fp_no_pole_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,const2):
    i = 0
    plotfits = []
    for Fit in Fits:
        if Fit['conf'] not in ['Fs','SFs','UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
        fit = Fit['conf']
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
                    MHsstar = make_MHsstar(pfit['MH_{0}_m{1}'.format(fit,mass)],pfit,pfit['a_{0}'.format(fit)])
                    pole = 1-(q2/MHsstar**2)
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
    p = make_p_physical_point_BK(pfit,Fits)
    for q2 in np.linspace(0,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        pole = 1 - q2/p['MBsstarphys']**2
        zed = make_z(q2,t_0,p['MBphys'],p['MKphys']) #all GeV dimensions
        if zed ==0:
            z.append(zed)
        else:
            z.append(zed.mean)
        #        make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,amh)
        y.append(pole*make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(8,figsize=figsize)
    plt.plot(qsq, ymean, color='r')
    plt.fill_between(qsq,ylow,yupp, color='r',alpha=alpha)
    #if datafpmaxBsEtas != None and adddata:
    #    pole = 1 - qsqmaxphysBK/MBsstarphys**2
    #    plt.errorbar(qsqmaxphysBK.mean, (pole*datafpmaxBsEtas).mean, xerr=qsqmaxphysBK.sdev, yerr=(pole*datafpmaxBsEtas).sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
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
    #if datafpmaxBsEtas != None and adddata:
    #    plt.errorbar( make_z(qsqmaxphysBK,t_0,MBsphys,Metasphys).mean, datafpmaxBsEtas.mean, xerr=make_z(qsqmaxphysBK,t_0,MBsphys,Metasphys).sdev, yerr=datafpmaxBsEtas.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')    
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=3,loc='upper right')
    plt.xlabel('$z$',fontsize=fontsizelab)
    plt.ylabel(r'$\left(1-\frac{q^2}{M^2_{H_{s}^*}} \right)f_+(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig('Plots/fpnopoleinz.pdf')
    plt.close()
    return()

################################################################################################

def f0_fp_fT_in_qsq(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,const2):
    qsq = []
    y0 = []
    yp = []
    yT = []
    p = make_p_physical_point_BK(pfit,Fits)
    for q2 in np.linspace(0,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        y0.append(make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
        yp.append(make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2))
        yT.append(make_fT_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0))
    y0mean,y0err = unmake_gvar_vec(y0)
    y0upp,y0low = make_upp_low(y0)
    ypmean,yperr = unmake_gvar_vec(yp)
    ypupp,yplow = make_upp_low(yp)
    yTmean,yTerr = unmake_gvar_vec(yT)
    yTupp,yTlow = make_upp_low(yT)
    plt.figure(10,figsize=figsize)
    plt.plot(qsq, y0mean, color='b',linestyle='-',label='$f_0(q^2)$')
    plt.fill_between(qsq,y0low,y0upp, color='b',alpha=alpha)
    plt.plot(qsq, ypmean, color='r',linestyle='-',label='$f_+(q^2)$')
    plt.fill_between(qsq,yplow,ypupp, color='r',alpha=alpha)
    plt.plot(qsq, yTmean, color='g',linestyle='-',label='$f_T(q^2)$')
    plt.fill_between(qsq,yTlow,yTupp, color='g',alpha=alpha)
    plt.plot
    plt.errorbar(0,0.319, yerr=0.066,fmt='r*',ms=ms,mfc='none')#,label = r'arXiv:1306.2384',lw=lw)
    plt.errorbar(qsqmaxphysBK.mean,0.861, yerr=0.048,fmt='b*',ms=ms,mfc='none',label = r'arXiv:1306.2384',lw=lw)
    plt.errorbar(qsqmaxphysBK.mean,2.63, yerr=0.13,fmt='r*',ms=ms,mfc='none',label = r'arXiv:1306.2384',lw=lw)
    plt.errorbar(0,0.270, yerr=0.095,fmt='g*',ms=ms,mfc='none')#,label = r'arXiv:1306.2384',lw=lw)
    plt.errorbar(qsqmaxphysBK.mean,2.39, yerr=0.17,fmt='g*',ms=ms,mfc='none',label = r'arXiv:1306.2384',lw=lw)
    handles, labels = plt.gca().get_legend_handles_labels()
    #handles = [h[0] for h in handles]
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
    #plt.text(10,1.0,'$f_+(q^2)$',fontsize=fontsizelab)
    #plt.text(18.5,0.9,'$f_0(q^2)$',fontsize=fontsizelab)
    plt.tight_layout()
    plt.savefig('Plots/f0fpfTinqsq.pdf')
    plt.close()
    return()
################################################################################################

def Hill_eq_19_20(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,const2):
    E = []
    e19 = []
    e20 = []
    p = make_p_physical_point_BK(pfit,Fits)
    for q2 in np.linspace(0.01,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        Ek = -(q2-p['MBphys']**2-p['MKphys']**2)/(2*p['MBphys'])
        E.append(Ek.mean)
        fT = make_fT_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0)
        fp = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
        f0 = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0)
        e19.append((fp-f0)/(q2*fT/p['MBphys']**2))
        e20.append(((1-Ek/p['MBphys'])*fp-0.5*f0)/(q2*fT/p['MBphys']**2))
    expectation = p['MBphys']/(p['MBphys']+p['MKphys'])
    e19mean,e19err = unmake_gvar_vec(e19)
    e19upp,e19low = make_upp_low(e19)
    e20mean,e20err = unmake_gvar_vec(e20)
    e20upp,e20low = make_upp_low(e20)
    plt.figure(10,figsize=figsize)
    plt.plot(E, e19mean, color='b',linestyle='-',label='$eq19$')
    plt.fill_between(E,e19low,e19upp, color='b',alpha=alpha)
    plt.plot(E, e20mean, color='r',linestyle='-',label='$eq20$')
    plt.fill_between(E,e20low,e20upp, color='r',alpha=alpha)
    plt.plot([0,3],[expectation.mean,expectation.mean],label='expectation',color='k')
    #print('expectation',expectation)
    handles, labels = plt.gca().get_legend_handles_labels()
    #handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,loc='upper right')
    plt.xlabel('$E_K[\mathrm{GeV}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.05))
    #plt.text(10,1.0,'$f_+(q^2)$',fontsize=fontsizelab)
    #plt.text(18.5,0.9,'$f_0(q^2)$',fontsize=fontsizelab)
    plt.ylim([0.3,1.3])
    plt.xlim([E[nopts-1],E[0]])
    plt.tight_layout()
    plt.savefig('Plots/Hill1920.pdf')
    plt.close()
    return()

################################################################################################

def f0_fp_fT_in_Mh(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,const2):
    MHs = []
    f00 = []
    f0max = []
    fpmax = []
    fT0 = []
    fTmax = []
    for Mh in np.linspace(MDphys.mean,MBphys.mean,nopts): #q2 now in GeV
        p = make_p_Mh_BK(pfit,Fits,Mh)
        qsqmax = (Mh-p['MKphys'])**2
        MHs.append(Mh)
        f00.append(make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,t_0,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
        f0max.append(make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsqmax,t_0,Fits[0]['masses'][0],fpf0same,0))
        fpmax.append(make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsqmax,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2))
        fT0.append(make_fT_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,t_0,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
        fTmax.append(make_fT_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsqmax,t_0,Fits[0]['masses'][0],fpf0same,0))
    f00mean,f00err = unmake_gvar_vec(f00)
    f0maxmean,f0maxerr = unmake_gvar_vec(f0max)
    fpmaxmean,fpmaxerr = unmake_gvar_vec(fpmax)
    fT0mean,fT0err = unmake_gvar_vec(fT0)
    fTmaxmean,fTmaxerr = unmake_gvar_vec(fTmax)
    f00upp,f00low = make_upp_low(f00)
    f0maxupp,f0maxlow = make_upp_low(f0max)
    fpmaxupp,fpmaxlow = make_upp_low(fpmax)
    fT0upp,fT0low = make_upp_low(fT0)
    fTmaxupp,fTmaxlow = make_upp_low(fTmax)
    plt.figure(11,figsize=figsize)
    plt.plot(MHs, f00mean, color='k',label=r'$f_{0/+}(0)$')
    plt.fill_between(MHs,f00low,f00upp, color='k',alpha=alpha)
    plt.plot(MHs, f0maxmean, color='b',label=r'$f_{0}(q^2_{\mathrm{max}})$')
    plt.fill_between(MHs,f0maxlow,f0maxupp, color='b',alpha=alpha)
    plt.plot(MHs, fpmaxmean, color='r',label=r'$f_{+}(q^2_{\mathrm{max}})$')
    plt.fill_between(MHs,fpmaxlow,fpmaxupp, color='r',alpha=alpha)
    plt.plot(MHs, fT0mean, color='g',label=r'$f_{T}(0)$')
    plt.fill_between(MHs,fT0low,fT0upp, color='g',alpha=alpha)
    plt.plot(MHs, fTmaxmean, color='purple',label=r'$f_{T}(q^2_{\mathrm{max}})$')
    plt.fill_between(MHs,fTmaxlow,fTmaxupp, color='purple',alpha=alpha)
    plt.xlabel('$M_{H}[\mathrm{GeV}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.plot([MDphys.mean,MDphys.mean],[-10,10],'k--',lw=lw/2,alpha=alpha)
    plt.text(MDphys.mean,-0.30,'$M_{D}$',fontsize=fontsizelab,horizontalalignment='center')
    plt.plot([MBphys.mean,MBphys.mean],[-10,10],'k--',lw=lw/2,alpha=alpha)
    plt.text(MBphys.mean,-0.30,'$M_{B}$',fontsize=fontsizelab,horizontalalignment='center')
    #plt.text(4.0,2.5,'$f_+(q^2_{\mathrm{max}})$',fontsize=fontsizelab)
    #plt.text(2.5,0.3,'$f_{0,+}(0)$',fontsize=fontsizelab)
    #plt.text(4.5,1.0,'$f_0(q^2_{\mathrm{max}})$',fontsize=fontsizelab)
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    ############ add data ############
    plt.errorbar(MBsphys.mean,0.323, yerr=0.063,fmt='k*',ms=ms,mfc='none',label = r'$B_s\to{}K$ arXiv:1406.2279',lw=lw)#,capsize=capsize)
    plt.errorbar(MBsphys.mean,0.819, yerr=0.021,fmt='b*',ms=ms,mfc='none',label = r'$B_s\to{}K$ arXiv:1406.2279',lw=lw)#,capsize=capsize)
    plt.errorbar(MBsphys.mean,3.27, yerr=0.15,fmt='r*',ms=ms,mfc='none',label = r'$B_s\to{}K$ arXiv:1406.2279',lw=lw)#,capsize=capsize)
    plt.errorbar(MBphys.mean-0.01,0.20, yerr=0.14,fmt='k^',ms=ms,mfc='none',label = r'$B\to{}\pi$ arXiv:1503.07839',lw=lw)#,capsize=capsize)
    plt.errorbar(MBphys.mean,1.024, yerr=0.025,fmt='b^',ms=ms,mfc='none',label = r'$B\to{}\pi$ arXiv:1503.07839',lw=lw)#,capsize=capsize)
    plt.errorbar(MBphys.mean-0.01,2.82, yerr=0.14,fmt='r^',ms=ms,mfc='none',label = r'$B\to{}\pi$ arXiv:1503.07839',lw=lw)#,capsize=capsize)
    plt.errorbar(MBphys.mean+0.01,0.319, yerr=0.066,fmt='ko',ms=ms,mfc='none',label = r'$B\to{}K$ arXiv:1306.2384',lw=lw)#,capsize=capsize)
    plt.errorbar(MBphys.mean,0.861, yerr=0.048,fmt='bo',ms=ms,mfc='none',label = r'$B\to{}K$ arXiv:1306.2384',lw=lw)#,capsize=capsize)
    plt.errorbar(MBphys.mean+0.01,2.63, yerr=0.13,fmt='ro',ms=ms,mfc='none',label = r'$B\to{}K$ arXiv:1306.2384',lw=lw)#,capsize=capsize)
    plt.errorbar(MDphys.mean,0.612, yerr=0.035,fmt='kd',ms=ms,mfc='none',label = r'$D\to{}\pi$ arXiv:1706.03017',lw=lw)#,capsize=capsize)
    plt.errorbar(MDphys.mean,0.765, yerr=0.031,fmt='ks',ms=ms,mfc='none',label = r'$D\to{}K$ arXiv:1706.03017',lw=lw)#,capsize=capsize)
    plt.errorbar(MDphys.mean,1.134, yerr=0.049,fmt='bd',ms=ms,mfc='none',label = r'$D\to{}\pi$ arXiv:1706.03017',lw=lw)#,capsize=capsize)
    plt.errorbar(MDphys.mean,2.130, yerr=0.096,fmt='rd',ms=ms,mfc='none',label = r'$D\to{}\pi$ arXiv:1706.03017',lw=lw)#,capsize=capsize)
    plt.errorbar(MDphys.mean,0.979, yerr=0.019,fmt='bs',ms=ms,mfc='none',label = r'$D\to{}K$ arXiv:1706.03017',lw=lw)#,capsize=capsize)
    plt.errorbar(MDphys.mean,1.336, yerr=0.054,fmt='rs',ms=ms,mfc='none',label = r'$D\to{}K$ arXiv:1706.03017',lw=lw)#,capsize=capsize)
    #handles, labels = plt.gca().get_legend_handles_labels()
    #handles = [h[0] for h in handles]
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper left')#handles=handles,labels=labels)
    ##################################
    plt.axes().set_ylim([0,4.0])
    plt.tight_layout()
    plt.savefig('Plots/f0fpfTinmh.pdf')
    plt.close()
    return()

#####################################################################################################

def beta_delta_in_Mh(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,const2):
    MHs = []
    delta = []
    invbeta = []
    alp =[]
    for MH in np.linspace(MDphys.mean,MBphys.mean,nopts): #q2 now in GeV
        p = make_p_Mh_BK(pfit,Fits,MH)
        MHs.append(MH)
        al,d,invb = make_beta_delta_BK(Fits,t_0,Nijk,Npow,Nm,addrho,p,fpf0same,MH,const2)
        delta.append(d) 
        invbeta.append(invb)
        alp.append(al)
    alphamean,alphaerr = unmake_gvar_vec(alp)
    deltamean,deltaerr = unmake_gvar_vec(delta)
    invbetamean,invbetaerr = unmake_gvar_vec(invbeta)
    alphaupp,alphalow = make_upp_low(alp)
    deltaupp,deltalow = make_upp_low(delta)
    invbetaupp,invbetalow = make_upp_low(invbeta)
    plt.figure(12,figsize=figsize)
    plt.plot(MHs, alphamean, color='k')
    plt.fill_between(MHs,alphalow,alphaupp, color='k',alpha=alpha)
    plt.plot(MHs, invbetamean, color='b')
    plt.fill_between(MHs,invbetalow,invbetaupp, color='b',alpha=alpha)
    plt.plot(MHs, deltamean, color='r')
    plt.fill_between(MHs,deltalow, deltaupp, color='r',alpha=alpha)
    plt.xlabel('$M_{H}[\mathrm{GeV}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.plot([MDphys.mean,MDphys.mean],[-10,10],'k--',lw=lw/2,alpha=alpha)
    plt.text(MDphys.mean+0.05,-0.05,'$M_{D}$',fontsize=fontsizelab)
    plt.plot([MBphys.mean,MBphys.mean],[-10,10],'k--',lw=lw/2,alpha=alpha)
    plt.text(MBphys.mean-0.05,-0.05,'$M_{B}$',fontsize=fontsizelab,horizontalalignment='right')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.text(3.5,0.1,r'$\delta$',fontsize=fontsizelab,color='r')
    plt.text(2.7,0.6,r'$\beta^{-1}$',fontsize=fontsizelab,color='b')
    plt.text(4.3,0.5,r'$\alpha$',fontsize=fontsizelab,color='k')
    plt.axes().set_ylim([-0.1,1.0])
    ############ add data ############
    # data from hep-lat/0409116
    plt.errorbar(MBphys.mean,0.63,yerr=0.05,fmt='k*',ms=ms,mfc='none',label = r'$\alpha^{B\to\pi} \mathrm{hep-lat/0409116}$',lw=lw)#,capsize=capsize)
    plt.errorbar(MBphys.mean,0.847, yerr=0.036,fmt='b*',ms=ms,mfc='none',label = r'$1/\beta^{B\to\pi} \mathrm{hep-lat/0409116}$',lw=lw)#,capsize=capsize)
    plt.errorbar(MDphys.mean,0.44,yerr=0.04,fmt='k*',ms=ms,mfc='none',label = r'$\alpha^{D\to\pi} \mathrm{hep-lat/0409116}$',lw=lw)#,capsize=capsize)
    plt.errorbar(MDphys.mean,0.50,yerr=0.04,fmt='ko',ms=ms,mfc='none',label = r'$\alpha^{D\to K} \mathrm{hep-lat/0409116}$',lw=lw)#,capsize=capsize)
    plt.errorbar(MDphys.mean,0.709, yerr=0.030,fmt='b*',ms=ms,mfc='none',label = r'$1/\beta^{D\to\pi} \mathrm{hep-lat/0409116}$',lw=lw)#,capsize=capsize)
    plt.errorbar(MDphys.mean,0.763, yerr=0.041,fmt='bo',ms=ms,mfc='none',label = r'$1/\beta^{D\to K} \mathrm{hep-lat/0409116}$',lw=lw)#,capsize=capsize)

    
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,loc='upper center',ncol=2)
    ##################################
    plt.tight_layout()
    plt.savefig('Plots/betadeltainmh.pdf')
    plt.close()
    return()

#####################################################################################################



def B_by_bin(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    B,Rmue,Fe,Fmu = comp_by_bin(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
    #################### Exp ################################################################
    Bmean,Berr = unmake_gvar_vec(B)
    BABARmean = [1.36,0.94,0.90,0.49,-1,0.67]
    BABARupp = [0.27,0.2,0.2,0.15,0,0.24]
    BABARlow = [0.24,0.19,0.19,0.14,0,0.22]
    Bellemean = [1.36,1.00,0.55,0.38,-1,0.98]
    Belleupp = [0.24,0.2,0.16,0.19,0,0.21]
    Bellelow = [0.22,0.19,0.14,0.12,0,0.19]
    CDFmean = [1.29,1.05,0.48,0.52,-1,0.38]
    CDFupp = [0.2,0.18,0.1,0.095,0,0.092]
    CDFlow = [0.2,0.18,0.1,0.095,0,0.092]
    LHCb0mean = [0.65,1.22,0.5,0.2,-1,0.35]
    LHCb0upp = [0.45,0.31,0.22,0.13,0,0.21]
    LHCb0low = [0.35,0.31,0.19,0.09,0,0.14]
    LHCbpmean = [1.21,1.00,0.57,0.38,0.35,-1]
    LHCbpupp = [0.11,0.081,0.054,0.045,0.045,0]
    LHCbplow = [0.11,0.081,0.054,0.045,0.045,0]
    x = [1,2,3,4,5,6]#np.arange(len(B))
    plt.figure(figsize=figsize)
    plt.xticks(x, ['(1,6)', '(4.3,8.68)', '(10.09,12.86)', '(14.18,16)', '(16,18)','(16,$q^2_{\mathrm{max}}$)'],fontsize=5,rotation=40)
    x = [0.75,1.75,2.75,3.75,4.75,5.75]
    plt.errorbar(x, CDFmean, yerr=[CDFlow,CDFupp], color='g', fmt='*',ms=ms, mfc='none',label=('CDF'),capsize=capsize)
    x = [0.85,1.85,2.85,3.85,4.85,5.85]
    plt.errorbar(x, BABARmean, yerr=[BABARlow,BABARupp], color='r', fmt='D',ms=ms, mfc='none',label=('BABAR'),capsize=capsize)
    x = [0.95,1.95,2.95,3.95,4.95,5.95]
    plt.errorbar(x, LHCbpmean, yerr=[LHCbplow,LHCbpupp], color='purple', fmt='o',ms=ms, mfc='none',label=('LHCb+'),capsize=capsize)
    
    x = [1.05,2.05,3.05,4.05,5.05,6.05]
    plt.errorbar(x, Bmean, yerr=Berr, color='k', fmt='d',ms=ms, mfc='k',label=('This work'),capsize=capsize)
    
    x = [1.15,2.15,3.15,4.15,5.15,6.15]
    plt.errorbar(x, LHCb0mean, yerr=[LHCb0low,LHCb0upp], color='purple', fmt='^',ms=ms, mfc='none',label=('LHCb0'),capsize=capsize)
    x = [1.25,2.25,3.25,4.25,5.25,6.25]
    plt.errorbar(x, Bellemean, yerr=[Bellelow,Belleupp], color='b', fmt='s',ms=ms, mfc='none',label=('Belle'),capsize=capsize)
    plt.xlabel('Bins $[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel('$10^7\mathcal{B}_{\ell}$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper right')
    plt.axes().set_ylim([0,1.7])
    plt.tight_layout()
    plt.savefig('Plots/Bbybinexp.pdf')
    plt.close()
    ##############################################Theor ###########################################################
    # 4 = 1206.0273 5 = 1212.2321 8 = 1211.0234
    ref1306mean = [1.81,1.65,0.87,0.442,0.391,0.797]
    ref1306err = [0.61,0.42,0.13,0.051,0.042,0.082]
    ref1206mean = [1.29,-1,-1,0.43,-1,0.86]
    ref1206err = [0.30,0,0,0.1,0,0.2]
    ref1212mean = [1.63,1.38,-1,0.34,0.309,0.634]
    ref1212upp = [0.56,0.51,0,0.179,0.176,0.382]
    ref1212low = [0.27,0.25,0,0.083,0.081,0.175]
    ref1211mean = [1.76,1.39,-1,-1,-1,-1]
    ref1211upp = [0.6,0.53,0,0,0,0]
    ref1211low = [0.23,0.22,0,0,0,0]

    x = [1,2,3,4,5,6]#np.arange(len(B))
    plt.figure(figsize=figsize)
    plt.xticks(x, ['(1,6)', '(4.3,8.68)', '(10.09,12.86)', '(14.18,16)', '(16,18)','(16,$q^2_{\mathrm{max}}$)'],fontsize=5,rotation=40)
    x = [0.8,1.8,2.8,3.8,4.8,5.8]
    plt.errorbar(x, ref1306mean, yerr=ref1306err, color='r', fmt='*',ms=ms, mfc='none',label=('arXiv: 1306.0434'),capsize=capsize)
    x = [0.9,1.9,2.9,3.9,4.9,5.9]
    plt.errorbar(x, ref1206mean, yerr=ref1206err, color='b', fmt='o',ms=ms, mfc='none',label=('arXiv: 1206.0273'),capsize=capsize)
    x = [1,2,3,4,5,6]
    plt.errorbar(x, Bmean, yerr=Berr, color='k', fmt='d',ms=ms, mfc='k',label=('This work'),capsize=capsize)
    x = [1.1,2.1,3.1,4.1,5.1,6.1]
    plt.errorbar(x, ref1212mean, yerr=[ref1212low,ref1212upp], color='purple', fmt='^',ms=ms, mfc='none',label=('arXiv: 1212.2321'),capsize=capsize)
    x = [1.2,2.2,3.2,4.2,5.2,6.2]
    plt.errorbar(x, ref1211mean, yerr=[ref1211low,ref1211upp], color='g', fmt='s',ms=ms, mfc='none',label=('arXiv: 1211.0234'),capsize=capsize)

    plt.xlabel('Bins $[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel('$10^7\mathcal{B}_{\ell}$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper right')
    plt.axes().set_ylim([0,2.5])
    plt.tight_layout()
    plt.savefig('Plots/Bbybintheory.pdf')
    plt.close()
    ############################################ other stuff #################
    x = [1,2,3,4,5,6]#np.arange(len(B))
    Rmuemean,Rmueerr = unmake_gvar_vec(Rmue)
    Femean,Feerr = unmake_gvar_vec(Fe)
    Femean,Feerr = unmake_gvar_vec(Fe)
    Fmumean,Fmuerr = unmake_gvar_vec(Fmu)
    R1306mean = [0.74,0.89,1.35,1.98,2.56,3.86]
    R1306err = [0.35,0.25,0.23,0.22,0.23,0.29]
    Fe1306mean = [0.577,0.2722,0.1694,0.1506,0.1525,0.1766]
    Fe1306err = [0.01,0.0054,0.0053,0.0052,0.0055,0.0068]
    Fmu1306mean = [2.441,1.158,0.722,0.642,0.649,0.751]
    Fmu1306err = [0.043,0.023,0.022,0.022,0.023,0.029]
    Fmu1212mean = [2.54,1.24,-1,0.704,0.318,0.775]
    Fmu1212upp = [0.2,0.12,0,0.147,0.201,0.210]
    Fmu1212low = [0.36,0.2,0,0.196,0.092,0.254]
    
    
    plt.figure(figsize=figsize)
    

    ax1 = plt.subplot(311)
    plt.errorbar(x,Rmuemean,yerr=Rmueerr,fmt='kd',ms=ms,mfc='none',capsize=capsize,label=('This work'))
    plt.errorbar(x, R1306mean, yerr=R1306err, color='r', fmt='s',ms=ms, mfc='none',label=('arXiv: 1306.0434'),capsize=capsize)
    plt.errorbar(1,0.31,yerr=[[0.07],[0.1]],color='b', fmt='o',ms=ms, mfc='none',label=('arXiv: 0709.4174'),capsize=capsize)
    plt.setp(ax1.get_xticklabels(), fontsize=fontsizelab)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.gca().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.gca().tick_params(which='major',length=major)
    plt.gca().tick_params(which='minor',length=minor)
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))
    plt.ylabel('$10^3(R^{\mu}_e-1)$',fontsize=fontsizelab)
    xlabs = ['(1,6)', '(4.3,8.68)', '(10.09,12.86)', '(14.18,16)', '(16,18)','(16,$q^2_{\mathrm{max}}$)']
    for i in range(len(xlabs)):
        plt.text(i+1,plt.ylim()[1],xlabs[i],fontsize=fontsizeleg, verticalalignment='bottom',horizontalalignment='center',rotation =40)
    ax2 = plt.subplot(312, sharex=ax1)
    plt.errorbar(x,Femean,yerr=Feerr,fmt='kd',ms=ms,mfc='none',capsize=capsize)
    plt.errorbar(x, Fe1306mean, yerr=Fe1306err, color='r', fmt='s',ms=ms, mfc='none',label=('arXiv: 1306.0434'),capsize=capsize)
    # make these tick labels invisible
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.gca().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.gca().tick_params(which='major',length=major)
    plt.gca().tick_params(which='minor',length=minor)
    plt.ylabel('$10^6F_H^e$',fontsize=fontsizelab)
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().set_ylim([0,0.7])
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.2)) 
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))


    ax3 = plt.subplot(313, sharex=ax1)
    plt.errorbar(x,Fmumean,yerr=Fmuerr,fmt='kd',ms=ms,mfc='none',capsize=capsize)
    plt.errorbar(x, Fmu1306mean, yerr=Fmu1306err, color='r', fmt='s',ms=ms, mfc='none',label=('arXiv: 1306.0434'),capsize=capsize)
    plt.errorbar(x, Fmu1212mean, yerr=[Fmu1212low,Fmu1212upp], color='purple', fmt='^',ms=ms, mfc='none',label=('arXiv: 1212.2321'),capsize=capsize)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.gca().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.gca().tick_params(which='major',length=major)
    plt.gca().tick_params(which='minor',length=minor)
    plt.ylabel('$10^2F_H^{\mu}$',fontsize=fontsizelab)
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().set_ylim([-0.2,3.5])
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5)) 
    plt.gca().xaxis.set_ticks_position('none')
    legend_elements = [Line2D([0], [0],color='k',linestyle='None', marker='d',ms=ms, mfc='none',label=('This work')),Line2D([0], [0],color='b',linestyle='None', marker='o',ms=ms, mfc='none',label=('arXiv: 0709.4174')),Line2D([0], [0],color='r',linestyle='None', marker='s',ms=ms, mfc='none',label=('arXiv: 1306.0434')),Line2D([0], [0],color='purple',linestyle='None', marker='^',ms=ms, mfc='none',label=('arXiv: 1212.2321'))]
    plt.legend(handles=legend_elements,fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper right')
    plt.xlabel('Bins $[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.tight_layout()
    plt.savefig('Plots/RandFbybin.pdf')
    return()

####################################### tau stuff ##########################################################

def tau_by_bin(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    Btau,Rtaumu,Rtaue,Ftau = comp_by_bin3(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
    
    x = [1,2,3,4]#np.arange(len(B))
    Btaumean,Btauerr = unmake_gvar_vec(Btau) 
    Rtaumumean,Rtaumuerr = unmake_gvar_vec(Rtaumu)
    Rtauemean,Rtaueerr = unmake_gvar_vec(Rtaue)
    Ftaumean,Ftauerr = unmake_gvar_vec(Ftau)
    Bmean = [1.44,0.349,0.413,1.09]
    Berr = [0.15,0.04,0.044,0.11]
    Rmumean = [1.158,0.790,1.055,1.361]
    Rmuerr = [0.039,0.025,0.033,0.046]
    #Rtauemean = [1.161,0.792,1.058,1.367]
    #Rtaueerr = [0.040,0.025,0.034,0.047]
    Fmean = [0.8856,0.9176,0.8784,0.8753]
    Ferr = [0.0037,0.0026,0.0038,0.0042]
    plt.figure(figsize=figsize)
    

    ax1 = plt.subplot(311)
    plt.errorbar(x,Rtaumumean,yerr=Rtaumuerr,fmt='kd',ms=ms,mfc='none',capsize=capsize,label=('This work'))
    plt.errorbar(x, Rmumean, yerr=Rmuerr, color='r', fmt='s',ms=ms, mfc='none',label=('arXiv: 1306.0434'),capsize=capsize)
    plt.setp(ax1.get_xticklabels(), fontsize=fontsizelab)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.gca().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.gca().tick_params(which='major',length=major)
    plt.gca().tick_params(which='minor',length=minor)
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.ylabel('$R^{\tau}_{\mu}$',fontsize=fontsizelab)
    xlabs = ['$(14.18,q^2_{\mathrm{max}})$', '(14.18,16)','(16,18)', '(16,18)','(16,$q^2_{\mathrm{max}}$)']
    for i in range(len(xlabs)):
        plt.text(i+1,plt.ylim()[1],xlabs[i],fontsize=fontsizeleg, verticalalignment='bottom',horizontalalignment='center',rotation =40)
    ax2 = plt.subplot(312, sharex=ax1)
    plt.errorbar(x,Btaumean,yerr=Btauerr,fmt='kd',ms=ms,mfc='none',capsize=capsize)
    plt.errorbar(x, Bmean, yerr=Berr, color='r', fmt='s',ms=ms, mfc='none',label=('arXiv: 1306.0434'),capsize=capsize)
    plt.errorbar(1, 1.26, yerr=[[0.23],[0.41]], color='b', fmt='o',ms=ms, mfc='none',label=('arXiv: 1111.2558'),capsize=capsize)
    # make these tick labels invisible
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.gca().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.gca().tick_params(which='major',length=major)
    plt.gca().tick_params(which='minor',length=minor)
    plt.ylabel('$10^7\mathcal{B}_{\tau}$',fontsize=fontsizelab)
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('none')
    #plt.gca().set_ylim([0,0.7])
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.5)) 
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))


    ax3 = plt.subplot(313, sharex=ax1)
    plt.errorbar(x,Ftaumean,yerr=Ftauerr,fmt='kd',ms=ms,mfc='none',capsize=capsize)
    plt.errorbar(x, Fmean, yerr=Ferr, color='r', fmt='s',ms=ms, mfc='none',label=('arXiv: 1306.0434'),capsize=capsize)
    plt.errorbar(1, 0.89, yerr=[[0.045],[0.033]], color='b', fmt='o',ms=ms, mfc='none',label=('arXiv: 1111.2558'),capsize=capsize)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.gca().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.gca().tick_params(which='major',length=major)
    plt.gca().tick_params(which='minor',length=minor)
    plt.ylabel('$F_H^{\tau}$',fontsize=fontsizelab)
    plt.gca().yaxis.set_ticks_position('both')
    #plt.gca().set_ylim([-0.2,3.5])
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.05)) 
    plt.gca().xaxis.set_ticks_position('none')
    #legend_elements = [Line2D([0], [0],color='k',linestyle='None', marker='d',ms=ms, mfc='none',label=('This work')),Line2D([0], [0],color='b',linestyle='None', marker='o',ms=ms, mfc='none',label=('arXiv: 0709.4174')),Line2D([0], [0],color='r',linestyle='None', marker='s',ms=ms, mfc='none',label=('arXiv: 1306.0434')),Line2D([0], [0],color='purple',linestyle='None', marker='^',ms=ms, mfc='none',label=('arXiv: 1212.2321'))]
    #plt.legend(handles=legend_elements,fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper right')
    plt.xlabel('Bins $[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.tight_layout()
    plt.savefig('Plots/Taustuffbybin.pdf')
    return()

################################################################################################################


def dBdq2_by_bin(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2): # results from 1403.8044
    Bp,B0 = comp_by_bin2(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
    Bpmean,Bperr = unmake_gvar_vec(Bp)
    B0mean,B0err = unmake_gvar_vec(B0)
    pmean = [33.2,23.3,28.2,25.4,22.1,23.1,24.5,23.1,17.7,19.3,16.1,16.4,20.6,13.7,7.4,5.9,4.3,24.2,12.1]
    pupp = [2.5,1.9,2.1,2.0,1.8,1.8,1.8,1.8,1.6,1.6,1.3,1.3,1.5,1.2,0.9,0.8,0.7,1.4,0.7]
    plow = pupp
    b0mean = [12.2,18.7,17.3,27.0,12.7,14.3,7.8,18.7,9.5]
    b0upp = [5.9,5.6,5.4,6.0,4.5,3.6,1.7,3.6,1.7]
    b0low = [5.2,5.0,4.9,5.5,4.0,3.3,1.6,3.3,1.6]
    x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    plt.figure(figsize=figsize)
    plt.xticks(x, ['(0.1,0.98)', '(1.1,2.0)', '(2.0,3.0)', '(3.0,4.0)', '(4.0,5.0)','(5.0,6.0)','(6.0,7.0)','(7.0,8.0)','(11.0,11.8)','(11.8,12.5)','(15.0,16.0)','(16.0,17.0)','(17.0,18.0)','(18.0,19.0)','(19.0,20.0)','(20.0,21.0)','(21.0,22.0)','(1.1,6.0)','(15.0,22.0)'],fontsize=5,rotation=40)
    
    plt.errorbar(x, pmean, yerr=[plow,pupp], color='r', fmt='o',ms=ms, mfc='none',label=('arXiv: 1403.8044 (+)'),capsize=capsize)
    plt.errorbar(x, Bpmean, yerr=Bperr, color='k', fmt='d',ms=ms, mfc='k',label=('This work'),capsize=capsize)
    plt.xlabel('Bins $[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel('$10^9 d\mathcal{B}_{\mu}/dq^2$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    #plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper right')
    #plt.axes().set_ylim([0,1.7])
    plt.tight_layout()
    plt.savefig('Plots/dBdq2bybinp.pdf')
    plt.close()

    plt.figure(figsize=figsize)
    x = [1,2,3,4,5,6,7,8,9]
    plt.xticks(x, ['(0.1,2.0)', '(2.0,4.0)', '(4.0,6.0)', '(6.0,8.0)', '(11.0,12.5)','(15.0,17.0)','(17.0,22.0)','(1.1,6.0)','(15.0,22.0)'],fontsize=5,rotation=40)
 
    plt.errorbar(x, b0mean, yerr=[b0low,b0upp], color='r', fmt='o',ms=ms, mfc='none',label=('arXiv: 1403.8044 (0)'),capsize=capsize)
    plt.errorbar(x, B0mean, yerr=B0err, color='k', fmt='d',ms=ms, mfc='k',label=('This work'),capsize=capsize)
    plt.xlabel('Bins $[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel('$10^9 d\mathcal{B}_{\mu}/dq^2$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    #plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper right')
    #plt.axes().set_ylim([0,1.7])
    plt.tight_layout()
    plt.savefig('Plots/dBdq2bybin0.pdf')
    plt.close()
    
    return()

####################################################################################################################


def dBdq2_plots(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    p = make_p_physical_point_BK(pfit,Fits)
    m_mu = 105.6583755 * 1e-3
    tauB0 = gv.gvar('1.519(4)') #1e-12s # HFLAV 
    tauBpm = gv.gvar('1.638(4)') #1e-12s # HFLAV 
    tauB0GeV = tauB0/(6.582119569*1e-13)
    tauBpmGeV = tauBpm/(6.582119569*1e-13)
    pmean = [33.2,23.3,28.2,25.4,22.1,23.1,24.5,23.1,17.7,19.3,16.1,16.4,20.6,13.7,7.4,5.9,4.3,24.2,12.1]
    pupp = [2.5,1.9,2.1,2.0,1.8,1.8,1.8,1.8,1.6,1.6,1.3,1.3,1.5,1.2,0.9,0.8,0.7,1.4,0.7]
    plow = pupp
    b0mean = [12.2,18.7,17.3,27.0,12.7,14.3,7.8,18.7,9.5]
    b0upp = [5.9,5.6,5.4,6.0,4.5,3.6,1.7,3.6,1.7]
    b0low = [5.2,5.0,4.9,5.5,4.0,3.3,1.6,3.3,1.6]
    BABARmean = [1.36,0.94,0.90,0.49,0.67]
    BABARupp = [0.27,0.2,0.2,0.15,0.24]
    BABARlow = [0.24,0.19,0.19,0.14,0.22]
    Bellemean = [1.36,1.00,0.55,0.38,0.98]
    Belleupp = [0.24,0.2,0.16,0.19,0.21]
    Bellelow = [0.22,0.19,0.14,0.12,0.19]
    CDFmean = [1.29,1.05,0.48,0.52,0.38]
    CDFupp = [0.2,0.18,0.1,0.095,0.092]
    CDFlow = [0.2,0.18,0.1,0.095,0.092]
    bin_starts_p = [0.1,1.1,2,3,4,5,6,7,11,11.8,15,16,17,18,19,20,21,1.1,15]
    bin_ends_p =   [0.98,2,3,4,5,6,7,8,11.8,12.5,16,17,18,19,20,21,22,6,22]
    bin_starts_0 = [0.1,2,4,6,11,15,17,1.1,15]
    bin_ends_0 =   [2,4,6,8,12.5,17,22,6,22]
    bin_starts_BABAR = [1,4.3,10.11,14.18,16]
    bin_ends_BABAR = [6,8.12,12.89,16,qsqmaxphysBK.mean]
    bin_starts_other = [1,4.3,10.09,14.18,16]
    bin_ends_other = [6,8.68,12.86,16,qsqmaxphysBK.mean]
    B = []
    qsq = []
    for q2 in np.linspace(0.1,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        al,cl = make_al_cl(p,q2,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        B.append((2*al + 2*cl/3)*tauB0GeV*1e9) # we're using B0

    Bmean,Berr = unmake_gvar_vec(B)
    Bupp,Blow = make_upp_low(B)
    plt.figure(figsize=figsize)
    plt.plot(qsq, Bmean, color='b')
    plt.fill_between(qsq,Blow,Bupp, color='b',alpha=alpha)
    
    x,xerr = make_x_from_bins(bin_starts_0,bin_ends_0)
    plt.errorbar(x, b0mean,xerr=xerr, yerr=[b0low,b0upp], color='r', fmt='o',ms=ms, mfc='none',label=('LHCb (0)'),capsize=capsize)
    x,xerr = make_x_from_bins(bin_starts_p,bin_ends_p)
    plt.errorbar(x, pmean,xerr=xerr, yerr=[plow,pupp], color='b', fmt='o',ms=ms, mfc='none',label=('LHCb (+)'),capsize=capsize)
    
    x,xerr = make_x_from_bins(bin_starts_BABAR,bin_ends_BABAR)
    y,yupp,ylow = make_y_from_bins(BABARmean,BABARupp,BABARlow,bin_starts_BABAR,bin_ends_BABAR)
    plt.errorbar(x, y,xerr=xerr, yerr=[ylow,yupp], color='g', fmt='s',ms=ms, mfc='none',label=('BABAR'),capsize=capsize)
    
    x,xerr = make_x_from_bins(bin_starts_other,bin_ends_other)
    y,yupp,ylow = make_y_from_bins(Bellemean,Belleupp,Bellelow,bin_starts_other,bin_ends_other)
    plt.errorbar(x, y,xerr=xerr, yerr=[ylow,yupp], color='k', fmt='d',ms=ms, mfc='none',label=('Belle'),capsize=capsize)
    y,yupp,ylow = make_y_from_bins(CDFmean,CDFupp,CDFlow,bin_starts_other,bin_ends_other)
    plt.errorbar(x, y,xerr=xerr, yerr=[ylow,yupp], color='purple', fmt='*',ms=ms, mfc='none',label=('CDF'),capsize=capsize)
    
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel('$10^9 d\mathcal{B}_{\mu}/dq^2$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    #plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper right')
    #plt.axes().set_ylim([0,1.7])
    plt.tight_layout()
    plt.savefig('Plots/dBdq2.pdf')
    plt.close()
    
    return()
##############################################################################################################








































































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
    plt.plot([qsqmaxphysBK.mean,qsqmaxphysBK.mean],[-10,10],'k--',lw=1)
    plt.text(qsqmaxphysBK.mean-0.02,2.4,'$q^2_{\mathrm{max}}$',fontsize=fontsizelab,horizontalalignment='right')
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().set_ylim([0.8,2.8])
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
    Emin = Metasphys
    thmean = gv.sqrt(MBsphys/MDsphys)
    therr = thmean*LQCD**2*(1/MDsphys**2 - 1/MBsphys**2)
    theory = gv.gvar('{0}({1})'.format(thmean.mean,therr.mean))
    for E in np.linspace(Emin.mean,Emax.mean,nopts): #q2 now in GeV
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
    plt.plot(Es, rat0mean, color='b',label=r'$\frac{f_0^{D_s}(E)}{f_0^{B_s}(E)}$')
    plt.fill_between(Es,rat0low,rat0upp, color='b',alpha=alpha)
    plt.plot(Es, ratpmean, color='r',label=r'$\frac{f_+^{B_s}(E)}{f_+^{D_s}(E)}$')
    plt.fill_between(Es,ratplow, ratpupp, color='r',alpha=alpha)
    plt.plot([Emin.mean,Emax.mean],[theory.mean,theory.mean],color='k',label =r'$\sqrt{\frac{M_{B_s}}{M_{D_s}}}$')
    plt.fill_between([Emin.mean,Emax.mean],[theory.mean-theory.sdev,theory.mean-theory.sdev],[theory.mean+theory.sdev,theory.mean+theory.sdev],color='k',alpha=alpha/2)
    plt.xlabel('$E[\mathrm{GeV}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.legend(fontsize=fontsizeleg,frameon=False,loc='upper right',ncol=3)
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().set_xlim([Emin.mean,Emax.mean])
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
    Emin = Mpiphys
    theory = gv.gvar('{0}({1})'.format(gv.sqrt(MBsphys/MDsphys).mean,((LQCD/MDsphys)**2).mean))
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
    plt.axes().set_xlim([Emin.mean,Emax.mean])
    plt.tight_layout()
    plt.savefig('Plots/HillratinlowE.pdf')
    plt.close()
    return()

####################################################################################################

def Hill_ratios_in_mh(pfit,Fits,Del,t_0,Nijk,Npow,addrho,fpf0same):
    Emax = (MDsphys**2 + Metasphys**2)/MDsphys
    Emin = Metasphys
    E = Emax
    M_h = []
    rat0 = []
    ratp = []
    theory = []
    for Mh in np.linspace(MDsphys.mean,MBsphys.mean,nopts): #q2 now in GeV
        ph = make_p_Mh_BsEtas(pfit,Fits,Del,Mh)
        pB = make_p_physical_point_BsEtas(pfit,Fits,Del)
        M_h.append(Mh)
        qsqh = Mh**2 + Metasphys**2 - 2*Mh*E
        qsqB =  MBsphys**2 + Metasphys**2 - 2*MBsphys*E
        zh = make_z(qsqh,t_0,Mh,Metasphys)
        zB = make_z(qsqB,t_0,MBsphys,Metasphys)
        f0h = make_f0_BsEtas(Nijk,Npow,addrho,ph,Fits[0],0,qsqh,zh,Fits[0]['masses'][0],fpf0same,0)
        fph = make_fp_BsEtas(Nijk,Npow,addrho,ph,Fits[0],0,qsqh,zh,Fits[0]['masses'][0],fpf0same,0)
        f0B = make_f0_BsEtas(Nijk,Npow,addrho,pB,Fits[0],0,qsqB,zB,Fits[0]['masses'][0],fpf0same,0)
        fpB = make_fp_BsEtas(Nijk,Npow,addrho,pB,Fits[0],0,qsqB,zB,Fits[0]['masses'][0],fpf0same,0)
        ratp.append(fpB/fph)
        rat0.append(f0h/f0B)
        thmean = gv.sqrt(MBsphys/Mh)
        therr = thmean * LQCD**2 *(1/Mh**2-1/MBsphys**2)
        theory.append(gv.gvar('{0:.7f}({1:.7f})'.format(thmean.mean,therr.mean)))
    rat0mean,rat0err = unmake_gvar_vec(rat0)
    ratpmean,ratperr = unmake_gvar_vec(ratp)
    theorymean,theoryerr = unmake_gvar_vec(theory)
    rat0upp,rat0low = make_upp_low(rat0)
    ratpupp,ratplow = make_upp_low(ratp)
    theoryupp,theorylow = make_upp_low(theory)
    plt.figure(14,figsize=figsize)
    plt.plot(M_h, rat0mean, color='b',label=r'$\frac{f_0^{H_s}(E)}{f_0^{B_s}(E)}$')
    plt.fill_between(M_h,rat0low,rat0upp, color='b',alpha=alpha)
    plt.plot(M_h, ratpmean, color='r',label=r'$\frac{f_+^{B_s}(E)}{f_+^{H_s}(E)}$')
    plt.fill_between(M_h,ratplow, ratpupp, color='r',alpha=alpha)
    plt.plot(M_h,theorymean,color='k',label =r'$\sqrt{\frac{M_{B_s}}{M_{H_s}}}$')
    plt.fill_between(M_h,theorylow,theoryupp,color='k',alpha=alpha)
    plt.xlabel('$M_{H_s}[\mathrm{GeV}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.legend(fontsize=fontsizeleg,frameon=False)
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().set_xlim([MDsphys.mean,MBsphys.mean])
    plt.tight_layout()
    plt.savefig('Plots/Hillratinmh_E{0}.pdf'.format(E))
    plt.close()
    return()

#####################################################################################################

def Hill_ratios_in_inv_mh(pfit,Fits,Del,t_0,Nijk,Npow,addrho,fpf0same):
    Emax = (MDsphys**2 + Metasphys**2)/MDsphys
    Emin = Metasphys
    E = Emax
    M_h = []
    rat0 = []
    ratp = []
    theory = []
    for Mh in np.linspace(MDsphys.mean,50,nopts*2): #q2 now in GeV
        ph = make_p_Mh_BsEtas(pfit,Fits,Del,Mh)
        pB = make_p_physical_point_BsEtas(pfit,Fits,Del)
        M_h.append(1/Mh)
        qsqh = Mh**2 + Metasphys**2 - 2*Mh*E
        qsqB =  MBsphys**2 + Metasphys**2 - 2*MBsphys*E
        zh = make_z(qsqh,t_0,Mh,Metasphys)
        zB = make_z(qsqB,t_0,MBsphys,Metasphys)
        f0h = make_f0_BsEtas(Nijk,Npow,addrho,ph,Fits[0],0,qsqh,zh,Fits[0]['masses'][0],fpf0same,0)
        fph = make_fp_BsEtas(Nijk,Npow,addrho,ph,Fits[0],0,qsqh,zh,Fits[0]['masses'][0],fpf0same,0)
        f0B = make_f0_BsEtas(Nijk,Npow,addrho,pB,Fits[0],0,qsqB,zB,Fits[0]['masses'][0],fpf0same,0)
        fpB = make_fp_BsEtas(Nijk,Npow,addrho,pB,Fits[0],0,qsqB,zB,Fits[0]['masses'][0],fpf0same,0)
        ratp.append(fpB/fph)
        rat0.append(f0h/f0B)
        thmean = gv.sqrt(MBsphys/Mh)
        therr = thmean * LQCD**2 * abs((1/MBsphys**2).mean -1/Mh**2)
        theory.append(gv.gvar('{0:.7f}({1:.7f})'.format(thmean.mean,therr.mean)))
    rat0mean,rat0err = unmake_gvar_vec(rat0)
    ratpmean,ratperr = unmake_gvar_vec(ratp)
    theorymean,theoryerr = unmake_gvar_vec(theory)
    rat0upp,rat0low = make_upp_low(rat0)
    ratpupp,ratplow = make_upp_low(ratp)
    theoryupp,theorylow = make_upp_low(theory)
    plt.figure(15,figsize=figsize)
    plt.plot(M_h, rat0mean, color='b',label=r'$\frac{f_0^{H_s}(E)}{f_0^{B_s}(E)}$')
    plt.fill_between(M_h,rat0low,rat0upp, color='b',alpha=alpha)
    plt.plot(M_h, ratpmean, color='r',label=r'$\frac{f_+^{B_s}(E)}{f_+^{H_s}(E)}$')
    plt.fill_between(M_h,ratplow, ratpupp, color='r',alpha=alpha)
    plt.plot(M_h,theorymean,color='k',label =r'$\sqrt{\frac{M_{B_s}}{M_{H_s}}}$')
    plt.fill_between(M_h,theorylow,theoryupp,color='k',alpha=alpha)
    plt.xlabel(r'$M_{H_s}^{-1}[\mathrm{GeV}^{-1}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.legend(fontsize=fontsizeleg,frameon=False,loc='lower right')
    #plt.plot([MDsphys.mean,MDsphys.mean],[-10,10],'k--',lw=1)
    #plt.text(MDsphys.mean+0.05,-0.45,'$M_{D_s}$',fontsize=fontsizelab)
    plt.plot([1/MBsphys.mean,1/MBsphys.mean],[-10,10],'k--',lw=1)
    plt.text(1/MBsphys.mean,0,r'$M_{B_s}^{-1}$',fontsize=fontsizelab,horizontalalignment='left')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.02))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().set_xlim([0,1/MDsphys.mean])
    plt.axes().set_ylim([0,2.2])
    plt.tight_layout()
    plt.savefig('Plots/Hillratininvmh_E{0}.pdf'.format(E))
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
            
            plt.figure(16,figsize=figsize)
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            j += 1
        i += 1
    qsq = []
    z = []
    y = []
    p = make_p_physical_point_BsEtas(pfit,Fits,Del)
    for q2 in np.linspace(0,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        zed = make_z(q2,t_0,MBsphys,Metasphys) #all GeV dimensions
        z.append(zed.mean)
        #        make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,amh)
        y.append(make_f0_BsEtas(Nijk,Npow,addrho,p,Fits[0],convert_Gev(afm).mean,q2,zed.mean,Fits[0]['masses'][0],fpf0same,convert_Gev(afm).mean*mbphys)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(16,figsize=figsize)
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
            
            plt.figure(17,figsize=figsize)
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            j += 1
        i += 1
    qsq = []
    z = []
    y = []
    p = make_p_physical_point_BsEtas(pfit,Fits,Del)
    for q2 in np.linspace(0,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        zed = make_z(q2,t_0,MBsphys,Metasphys) #all GeV dimensions
        z.append(zed.mean)
        #        make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,amh)
        y.append(make_fp_BsEtas(Nijk,Npow,addrho,p,Fits[0],convert_Gev(afm).mean,q2,zed.mean,Fits[0]['masses'][0],fpf0same,convert_Gev(afm).mean*mbphys)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
       
    plt.figure(17,figsize=figsize)
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
    qsqs = np.linspace(0,qsqmaxphysBK.mean,nopts)
    f0,fp = output_error_BsEtas(pfit,prior,Fits,Nijk,Npow,f,qsqs,t_0,Del,addrho,fpf0same)

    plt.figure(18,figsize=figsize)
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
    ax1.yaxis.set_major_locator(MultipleLocator(20))
    ax1.yaxis.set_minor_locator(MultipleLocator(5))
    ax1.set_xlim([0,qsqmaxphysBK.mean])
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
    low,upp = ax1.get_ylim()
    ax1b.set_ylim([low,upp])
    points = []
    rootpoints = []
    i = 2
    while i != 'stop':
        if i**2 < upp:
            points.append(i**2)
            if i%2 == 0:
                rootpoints.append('{0}'.format(i))
            else:
                rootpoints.append('')
            i+=1
        else:
            i ='stop'
    ax1b.set_yticks(points)
    ax1b.set_yticklabels(rootpoints)
    
    plt.legend(loc='upper right',ncol=2,fontsize=fontsizeleg,frameon=False)

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
    ax2.set_xlim([0,qsqmaxphysBK.mean])
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
    low,upp = ax2.get_ylim()
    ax2b.set_ylim([low,upp])
    points = []
    rootpoints = []
    i = 2
    while i != 'stop':
        if i**2 < upp:
            points.append(i**2)
            if i%2 == 0:
                rootpoints.append('{0}'.format(i))
            else:
                rootpoints.append('')
            i+=1
        else:
            i ='stop'
    ax2b.set_yticks(points)
    ax2b.set_yticklabels(rootpoints)
    plt.tight_layout()
    plt.savefig('Plots/f0fpluserr.pdf')
    plt.close()
    return()

###################################################################################################

def table_of_as(Fits,pfit,Nijk,Npow,fpf0same,addrho,Del):
    list0 = []
    listp = []
    Fit = Fits[0]
    mass = Fit['masses'][0]
    fit = Fit['conf']
    p = make_p_physical_point_BsEtas(pfit,Fits,Del)
    atab = open('Tables/tablesofas.txt','w')
    for n in range(Npow):
        if n == 0:
            atab.write('      {0}&'.format(make_an_BsEtas(n,Nijk,addrho,p,'0',Fit,0,mass,0,fpf0same)))
        else:
            atab.write('{0}&'.format(make_an_BsEtas(n,Nijk,addrho,p,'0',Fit,0,mass,0,fpf0same)))
        list0.append(make_an_BsEtas(n,Nijk,addrho,p,'0',Fit,0,mass,0,fpf0same))
        listp.append(make_an_BsEtas(n,Nijk,addrho,p,'p',Fit,0,mass,0,fpf0same))
    for n in range(Npow):           
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
        for tw,twist in enumerate(Fit['twists']):
            if tw == 0:
                table.write('&{1}&{0}&{2}&{3}&{4}'.format(fs_data['qsq_m{0}_tw{1}'.format(mass,twist)],Fit['M_parent_m{0}'.format(mass)],Fit['E_daughter_tw{0}_theory'.format(twist)],Fit['S_m{0}_tw{1}'.format(mass,twist)],Fit['V_m{0}_tw{1}'.format(mass,twist)]))
            else:
                table.write('      &&&{0}&{1}&{2}&{3}'.format(fs_data['qsq_m{0}_tw{1}'.format(mass,twist)],Fit['E_daughter_tw{0}_theory'.format(twist)],Fit['S_m{0}_tw{1}'.format(mass,twist)],Fit['V_m{0}_tw{1}'.format(mass,twist)]))
            if fs_data['fp_m{0}_tw{1}'.format(mass,twist)] != None:
                table.write('&{0}&{1}\\\\ [1ex]\n'.format(fs_data['f0_m{0}_tw{1}'.format(mass,twist)],fs_data['fp_m{0}_tw{1}'.format(mass,twist)]))
            else:
                table.write('&{0}&\\\\ [1ex]\n'.format(fs_data['f0_m{0}_tw{1}'.format(mass,twist)]))

        
    table.close()
    return()
###################################################################################################
