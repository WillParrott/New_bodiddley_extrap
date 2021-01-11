from spline_functionsDK import *
import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
#from matplotlib import Scatter

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
factor = 1.0 #multiplies everything to make smaller for big plots etc usually 1
figsca = 14  #size for saving figs
figsize = ((figsca,2*figsca/(1+np.sqrt(5))))
lw =2*factor
nopts = 200 #number of points on plot
ms = 25*factor #markersize
alpha = 0.4
fontsizeleg = 25*factor #legend
fontsizelab = 35*factor #legend
cols = ['k','r','b','c'] #for each mass
symbs = ['o','^','*','D','d','s','p','>','<']    # for each conf
lines = ['-','--','-.','-',':','-','--','-.'] # for each conf
major = 15*factor
minor = 8*factor 
capsize = 10*factor
####################
firstknot = -3.25
lastknot = 2.0
####################################################################################################

def plot_gold_non_split(Fits):
    plt.figure(figsize=figsize)
    i = 0
    for Fit in Fits:
        for mass in Fit['masses']:
            y = 1000*(Fit['GNGsplit_m{0}'.format(mass)]/Fit['a']).mean
            yerr = 1000*(Fit['GNGsplit_m{0}'.format(mass)]/Fit['a']).sdev
            x = float(mass)**2
            if Fit['conf'][-1] == 'p':
                plt.errorbar(x,y,yerr=yerr,fmt=symbs[i],color='k',label=Fit['conf'],ms=ms,mfc='none')
            else:
                plt.errorbar(x,y,yerr=yerr,fmt=symbs[i],color='r',label=Fit['conf'],ms=ms,mfc='none')
            i+=1
   
    plt.plot([-0.1,0.9],[0,0],'k--')
    plt.xlim([0,0.85])
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=3,loc='upper left')
    plt.xlabel('$(am_c)^2$',fontsize=fontsizelab)
    plt.ylabel('$(M_{D_{\mathrm{non-gold}}}-M_{D_{\mathrm{gold}}})[\mathrm{MeV}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.05))
    plt.axes().yaxis.set_major_locator(MultipleLocator(1))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(5))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/gold-non-split.pdf')            
    return()


####################################################################################################
def speed_of_light(Fits):
    plt.figure(1,figsize=figsize)
    points = ['ko','k^','k*','ro','r^','r*','rD','r>']
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
    plt.xlim((-0.02,0.5))
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(loc='upper right',handles=handles,labels=labels,ncol=2,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$|ap_{K}|^2$',fontsize=fontsizelab)
    plt.ylabel('$(E_{K}^2-M_{K}^2)/p_{K}^2$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.005))
    #plt.xlim([0,0.2])
    #plt.ylim([0.9,1.2])
    plt.tight_layout()
    plt.savefig('DK_spline_plots/speedoflight.pdf')
    #plt.show()
    return()

#####################################################################################################

def Z_V_plots(Fits,fs_data):
    plt.figure(19,figsize=figsize)
    i = 0
    plotfits =[]
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
            plt.errorbar(x,y,yerr=yerr,fmt=symbs[i],color='b',label=Fit['label'],ms=ms,mfc='none')
        elif Fit['conf'][-1] == 'p':
            plt.errorbar(x,y,yerr=yerr,fmt=symbs[i],color='k',label=Fit['label'],ms=ms,mfc='none')
        else:
            plt.errorbar(x,y,yerr=yerr,fmt=symbs[i],color='r',label=Fit['label'],ms=ms,mfc='none')
        i+=1
    plt.plot([-0.1,0.9],[1,1],'k--')
    plt.xlim([0,0.85])
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=3,loc='upper left')
    plt.xlabel('$(am_c)^2$',fontsize=fontsizelab)
    plt.ylabel('$Z_V$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.05))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.005))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/Z_Vinamhsq.pdf')            
    return()

#####################################################################################################

def f0_in_qsq(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata):
    i = 0
    plotfits = []
    for Fit in Fits:
        if Fit['conf'] not in ['Fs','SFs','UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
        j = 0
        for mass in Fit['masses']:
            qsq = []
            y = []
            for twist in Fit['twists']:
                q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                y.append(fs_data[Fit['conf']]['f0_m{0}_tw{1}'.format(mass,twist)])
            qsq,qsqerr = unmake_gvar_vec(qsq)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(2,figsize=figsize)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)),alpha=alpha)
            
            
            j += 1
        i += 1
    qsq = []
    y = []
    p = make_p_physical_point_DK(pfit,Fits)
    for q2 in np.linspace(firstknot,lastknot,nopts): #q2 now in GeV
        qsq.append(q2)
        f0,fp=make_spline_f0_fp_physpoint(p,Fits[0],q2) #only need one fit
        y.append(f0)
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(2,figsize=figsize)
    plt.plot(qsq, ymean, color='b')
    plt.fill_between(qsq,ylow,yupp, color='b',alpha=alpha)
    #plt.errorbar(0,0.765, yerr=0.031,fmt='s',color='purple',ms=ms,mfc='none',label = r'$D\to{}K$ arXiv:1706.03017',lw=lw)#,capsize=capsize)
    #plt.errorbar(qsqmaxphysDK.mean,0.979, yerr=0.019,fmt='s',color='purple',ms=ms,mfc='none',label = r'$D\to{}K$ arXiv:1706.03017',lw=lw)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$f_0(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.04))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/f0poleinqsq.pdf')
    plt.close()

    qsq = []
    y = []
    f0orig = gv.load('Fits/f0_orig.pickle')
    for q2 in np.linspace(0,qsqmaxphysDK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        f0,fp=make_spline_f0_fp_physpoint(p,Fits[0],q2) #only need one fit
        y.append(f0)
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    origmean,origerr = unmake_gvar_vec(f0orig)
    origupp,origlow = make_upp_low(f0orig)
    
    plt.figure(figsize=figsize)
    plt.plot(qsq, ymean, color='k')
    plt.fill_between(qsq,ylow,yupp, facecolor='none', edgecolor='k', hatch='X',alpha=alpha)
    plt.plot(qsq, origmean, color='b')
    plt.fill_between(qsq,origlow,origupp, color='b',alpha=alpha)
    
    #plt.errorbar(0,0.765, yerr=0.031,fmt='s',color='purple',ms=ms,mfc='none',label = r'$D\to{}K$ arXiv:1706.03017',lw=lw)#,capsize=capsize)
    #plt.errorbar(qsqmaxphysDK.mean,0.979, yerr=0.019,fmt='s',color='purple',ms=ms,mfc='none',label = r'$D\to{}K$ arXiv:1706.03017',lw=lw)
    #handles, labels = plt.gca().get_legend_handles_labels()
    #handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$f_0(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.04))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/f0compinqsq.pdf')
    plt.close()
    
    return()

################################################################################################

def fp_in_qsq(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,const2):
    i = 0
    plotfits = []
    for Fit in Fits:
        if Fit['conf'] not in ['Fs','SFs','UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
        j = 0
        for mass in Fit['masses']:
            qsq = []
            y = []
            for twist in Fit['twists']:
                if fs_data[Fit['conf']]['fp_m{0}_tw{1}'.format(mass,twist)] != None:
                    q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                    qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                    y.append(fs_data[Fit['conf']]['fp_m{0}_tw{1}'.format(mass,twist)])
            qsq,qsqerr = unmake_gvar_vec(qsq)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(4,figsize=figsize)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)),alpha=alpha)
            

            
            j += 1
        i += 1
    qsq = []
    y = []
    p = make_p_physical_point_DK(pfit,Fits)
    for q2 in np.linspace(firstknot,lastknot,nopts): #q2 now in GeV
        qsq.append(q2)
        f0,fp=make_spline_f0_fp_physpoint(p,Fits[0],q2)
        y.append(fp) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(4,figsize=figsize)
    plt.plot(qsq, ymean, color='r')
    plt.fill_between(qsq,ylow,yupp, color='r',alpha=alpha)
#    if datafpmaxBK != None and adddata:
#        plt.errorbar(qsqmaxphysBK.mean, datafpmaxBK.mean, xerr=qsqmaxphysBK.sdev, yerr=datafpmaxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    #plt.errorbar(0,0.765, yerr=0.031,fmt='s',color='purple',ms=ms,mfc='none',label = r'$D\to{}K$ arXiv:1706.03017',lw=lw)
    #plt.errorbar(0,0.745, yerr=0.011,fmt='bs',ms=ms,mfc='none',label = r'$D\to{}K$ arXiv:1305.1462',lw=lw)
    #plt.errorbar(qsqmaxphysDK.mean,1.336, yerr=0.054,fmt='s',color='purple',ms=ms,mfc='none',label = r'$D\to{}K$ arXiv:1706.03017',lw=lw)#
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$f_+(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/fppoleinqsq.pdf')
    plt.close()

    
    qsq = []
    y = []
    fporig = gv.load('Fits/fp_orig.pickle')
    for q2 in np.linspace(0,qsqmaxphysDK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        f0,fp=make_spline_f0_fp_physpoint(p,Fits[0],q2)
        y.append(fp) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    origmean,origerr = unmake_gvar_vec(fporig)
    origupp,origlow = make_upp_low(fporig)
    plt.figure(figsize=figsize)
    plt.plot(qsq, ymean, color='k')
    plt.fill_between(qsq,ylow,yupp, facecolor='none', edgecolor='k', hatch='X',alpha=alpha)
    plt.plot(qsq, origmean, color='r')
    plt.fill_between(qsq,origlow,origupp, color='r',alpha=alpha)
    
    #handles, labels = plt.gca().get_legend_handles_labels()
    #handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$f_+(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/fpcompinqsq.pdf')
    plt.close()
    return()
###############################################################################################

def f0fp_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,const2):
    qsq = []
    z = []
    y0 = []
    yp = []
    p = make_p_physical_point_DK(pfit,Fits)
    for q2 in np.linspace(0,qsqmaxphysDK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        zed = make_z(q2,t_0,p['MDphys'],p['MKphys']) #all GeV dimensions
        if zed ==0:
            z.append(zed)
        else:
            z.append(zed.mean)
        #        make_fp_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,fpf0same,amh)
        yp.append(make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)) #only need one fit
        y0.append(make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
    y0mean,y0err = unmake_gvar_vec(y0)
    y0upp,y0low = make_upp_low(y0)
    ypmean,yperr = unmake_gvar_vec(yp)
    ypupp,yplow = make_upp_low(yp)
    plt.figure(4,figsize=figsize)
    plt.plot(qsq, y0mean, color='r',label='$f_0$')
    plt.fill_between(qsq,y0low,y0upp, color='r',alpha=alpha)
    plt.plot(qsq, ypmean, color='b',label='$f_+$')
    plt.fill_between(qsq,yplow,ypupp, color='b',alpha=alpha)

    #plt.errorbar(0,0.765, yerr=0.031,fmt='s',color='purple',ms=ms,mfc='none',label = r'$D\to{}K$ arXiv:1706.03017',lw=lw)
    #plt.errorbar(0,0.745, yerr=0.011,fmt='bs',ms=ms,mfc='none',label = r'$D\to{}K$ arXiv:1305.1462',lw=lw)
    #plt.errorbar(qsqmaxphysDK.mean,1.336, yerr=0.054,fmt='s',color='purple',ms=ms,mfc='none',label = r'$D\to{}K$ arXiv:1706.03017',lw=lw)#
    #plt.errorbar(0,0.765, yerr=0.031,fmt='s',color='purple',ms=ms,mfc='none',label = r'arXiv:1706.03017',lw=lw)
    #plt.errorbar(qsqmaxphysDK.mean,0.979, yerr=0.019,fmt='s',color='r',ms=ms,mfc='none',label = r'arXiv:1706.03017',lw=lw)
    #plt.errorbar(qsqmaxphysDK.mean,1.336, yerr=0.054,fmt='s',color='b',ms=ms,mfc='none',label = r'arXiv:1706.03017',lw=lw)
    plt.legend(fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    #plt.ylabel(r'$f_+(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/f0fpinqsq.pdf')
    plt.close()
    
    plt.figure(5,figsize=figsize)
    plt.plot(z,y0mean, color='r',label='$f_0$')
    plt.fill_between(z,y0low,y0upp, color='r',alpha=alpha)
    plt.plot(z,ypmean, color='b',label='$f_+$')
    plt.fill_between(z,yplow,ypupp, color='b',alpha=alpha)
    plt.legend(fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$z$',fontsize=fontsizelab)
    #plt.ylabel(r'$f_+(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/f0fpinz.pdf')
    plt.close()
    return()
##########################################################################################################################

def f0fp_data_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,const2):
    i = 0
    plotfits = []
    legend_elements = []
    #legend_elements = [Line2D([0], [0], color='b', lw=4, label='Line'), Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='g', markersize=15), Patch(facecolor='orange', edgecolor='r',label='Color Patch')]
    for Fit in Fits:
        if Fit['conf'] not in ['Fs','SFs','UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
        j = 0
        for mass in Fit['masses']:
            qsq = []
            z = []
            y0 = []
            yp = []
            for twist in Fit['twists']:
                if fs_data[Fit['conf']]['fp_m{0}_tw{1}'.format(mass,twist)] != None:
                    q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                    qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                    z.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))#all lat units
                    yp.append(fs_data[Fit['conf']]['fp_m{0}_tw{1}'.format(mass,twist)])
                    y0.append(fs_data[Fit['conf']]['f0_m{0}_tw{1}'.format(mass,twist)])
            qsq,qsqerr = unmake_gvar_vec(qsq)
            z,zerr = unmake_gvar_vec(z)
            y0,y0err = unmake_gvar_vec(y0)
            yp,yperr = unmake_gvar_vec(yp)            
            plt.figure(108,figsize=figsize)
            plt.errorbar(qsq, y0, xerr=qsqerr, yerr=y0err, color='r', mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(qsq, y0, xerr=qsqerr, yerr=y0err, color='r', fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)),alpha=alpha)
            legend_elements.append(Line2D([0], [0],color='k',linestyle='None', marker=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass))))
            plt.errorbar(qsq, yp, xerr=qsqerr, yerr=yperr, color='b', mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(qsq, yp, xerr=qsqerr, yerr=yperr, color='b', fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)),alpha=alpha)
            
            plt.figure(109,figsize=figsize)
            plt.errorbar(z, y0, xerr=zerr, yerr=y0err, color='r', mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(z, y0, xerr=zerr, yerr=y0err, color='r', fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)),alpha=alpha)
            plt.errorbar(z, yp, xerr=zerr, yerr=yperr, color='b', mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(z, yp, xerr=zerr, yerr=yperr, color='b', fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)),alpha=alpha)
            
            j += 1
        i += 1
    qsq = []
    z = []
    y0 = []
    yp = []
    p = make_p_physical_point_DK(pfit,Fits)
    for q2 in np.linspace(0,qsqmaxphysDK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        zed = make_z(q2,t_0,p['MDphys'],p['MKphys']) #all GeV dimensions
        if zed ==0:
            z.append(zed)
        else:
            z.append(zed.mean)
        #        make_fp_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,fpf0same,amh)
        yp.append(make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)) #only need one fit
        y0.append(make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
    y0mean,y0err = unmake_gvar_vec(y0)
    y0upp,y0low = make_upp_low(y0)
    ypmean,yperr = unmake_gvar_vec(yp)
    ypupp,yplow = make_upp_low(yp)
    plt.figure(108,figsize=figsize)
    plt.plot(qsq, y0mean, color='r',label='$f_0$')
    plt.fill_between(qsq,y0low,y0upp, color='r',alpha=alpha)
    plt.plot(qsq, ypmean, color='b',label='$f_+$')
    plt.fill_between(qsq,yplow,ypupp, color='b',alpha=alpha)
    legend_elements.append(Line2D([0], [0], color='r', lw=4, label='$f_0$'))
    legend_elements.append(Line2D([0], [0], color='b', lw=4, label='$f_+$'))
    #plt.errorbar(0,0.765, yerr=0.031,fmt='s',color='purple',ms=ms,mfc='none',label = r'$D\to{}K$ arXiv:1706.03017',lw=lw)
    #plt.errorbar(0,0.745, yerr=0.011,fmt='bs',ms=ms,mfc='none',label = r'$D\to{}K$ arXiv:1305.1462',lw=lw)
    #plt.errorbar(qsqmaxphysDK.mean,1.336, yerr=0.054,fmt='s',color='purple',ms=ms,mfc='none',label = r'$D\to{}K$ arXiv:1706.03017',lw=lw)#

    #plt.legend(fontsize=fontsizeleg,frameon=False,ncol=2)
    
    
    plt.legend(handles=legend_elements,fontsize=fontsizeleg,frameon=False,ncol=2)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    #plt.ylabel(r'$f_+(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/f0fpdatainqsq.pdf')
    plt.close()
    
    plt.figure(109,figsize=figsize)
    plt.plot(z,y0mean, color='r',label='$f_0$')
    plt.fill_between(z,y0low,y0upp, color='r',alpha=alpha)
    plt.plot(z,ypmean, color='b',label='$f_+$')
    plt.fill_between(z,yplow,ypupp, color='b',alpha=alpha)
    #plt.legend(fontsize=fontsizeleg,frameon=False,ncol=2)
    plt.legend(handles=legend_elements,fontsize=fontsizeleg,frameon=False,ncol=2)
    plt.xlabel('$z$',fontsize=fontsizelab)
    #plt.ylabel(r'$f_+(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/f0fpdatainz.pdf')
    plt.close()
    return()

################################################################################################

def f0_no_pole_in_qsq(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata):
    i = 0
    for Fit in Fits:
        fit = Fit['conf']
        j = 0
        for mass in Fit['masses']:
            qsq = []
            y = []
            for twist in Fit['twists']:
                q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                MHs0 = pfit['MH_{0}_m{1}'.format(fit,mass)] + pfit['a_{0}'.format(fit)] * (pfit['MDs0phys']- pfit['MDphys'])
                pole = 1-(q2/MHs0**2)
                y.append(pole * fs_data[Fit['conf']]['f0_m{0}_tw{1}'.format(mass,twist)])
            qsq,qsqerr = unmake_gvar_vec(qsq)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(6,figsize=figsize)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)),alpha=alpha)
            
            
            j += 1
        i += 1
    qsq = []
    y = []
    p = make_p_physical_point_DK(pfit,Fits)
    for q2 in np.linspace(firstknot,lastknot,nopts): #q2 now in GeV
        qsq.append(q2)
        pole = 1 - q2/(p['MDs0phys'])**2
        f0,fp=make_spline_f0_fp_physpoint(p,Fits[0],q2)
        y.append(pole*f0) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(6,figsize=figsize)
    plt.plot(qsq, ymean, color='b')
    plt.fill_between(qsq,ylow,yupp, color='b',alpha=alpha)
    #if dataf0maxBsEtas != None and adddata:
    #    pole = 1 - qsqmaxphysBK/(MBsphys+Del)**2
    #    plt.errorbar(qsqmaxphysBK.mean, (pole*dataf0maxBsEtas).mean, xerr=qsqmaxphysBK.sdev, yerr=(pole*dataf0maxBsEtas).sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$\mathrm{arXiv:} 1510.07446$')
    #if dataf00BsEtas != None and adddata:
    #    plt.errorbar(0, dataf00BsEtas.mean, yerr=dataf00BsEtas.sdev, color='k', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2,loc='lower right')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$\left(1-\frac{q^2}{M^2_{D_{s}^0}} \right)f_0(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/f0nopoleinqsq.pdf')
    plt.close()
    
    return()

################################################################################################

def fp_no_pole_in_qsq(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,const2):
    i = 0
    for Fit in Fits:
        fit = Fit['conf']
        j = 0
        for mass in Fit['masses']:
            qsq = []
            y = []
            for twist in Fit['twists']:
                if fs_data[Fit['conf']]['fp_m{0}_tw{1}'.format(mass,twist)] != None:
                    q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                    qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                    MHsstar = make_MHsstar(pfit['MH_{0}_m{1}'.format(fit,mass)],pfit,pfit['a_{0}'.format(fit)])
                    pole = 1-(q2/MHsstar**2)
                    y.append(pole * fs_data[Fit['conf']]['fp_m{0}_tw{1}'.format(mass,twist)])
            qsq,qsqerr = unmake_gvar_vec(qsq)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(8,figsize=figsize)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)),alpha=alpha)
            
            
            j += 1
        i += 1
    qsq = []
    y = []
    p = make_p_physical_point_DK(pfit,Fits)
    for q2 in np.linspace(firstknot,lastknot,nopts): #q2 now in GeV
        qsq.append(q2)
        pole = 1 - (q2/p['MDsstarphys']**2)
        f0,fp=make_spline_f0_fp_physpoint(p,Fits[0],q2)
        y.append(pole*fp) #only need one fit
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
    plt.ylabel(r'$\left(1-\frac{q^2}{M^2_{D_{s}^*}} \right)f_+(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/fpnopoleinqsq.pdf')
    plt.close()
    return()

##########################################################################################################################

def plot_Vcs_by_bin(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2):
    plt.figure(figsize=figsize)
    d = comp(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2)
    x = []
    y = []
    for i in range(len(d['Cleo1V2'])):
        x.append((d['Cleobins'][i]+d['Cleobins'][i+1])/2)
        y.append(gv.sqrt(d['Cleo1V2'][i]))
    ymean,yerr = unmake_gvar_vec(y)
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='ko' ,mfc='none',label='CLEO0',alpha=0.8)
    x = []
    y = []
    for i in range(len(d['Cleo2V2'])):
        x.append((d['Cleobins'][i]+d['Cleobins'][i+1])/2)
        y.append(gv.sqrt(d['Cleo2V2'][i]))
    ymean,yerr = unmake_gvar_vec(y)
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='rs' ,mfc='none',label='CLEO+',alpha=0.8)
    x = []
    y = []
    for i in range(len(d['BESV2'])):
        x.append((d['BESbins'][i]+d['BESbins'][i+1])/2)
        y.append(gv.sqrt(d['BESV2'][i]))
    ymean,yerr = unmake_gvar_vec(y)
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='b^' ,mfc='none',label='BES',alpha=0.8)
    x = []
    y = []
    for i in range(len(d['BaBarV2'])):
        x.append((d['BaBarbins'][i]+d['BaBarbins'][i+1])/2)
        y.append(gv.sqrt(d['BaBarV2'][i]))
    ymean,yerr = unmake_gvar_vec(y)
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='gd' ,mfc='none',label='BaBar',alpha=0.8)
    av = (d['average'])
    plt.plot([-1,2],[av.mean,av.mean],color='purple',label='Average')
    plt.fill_between([-1,2],[av.mean-av.sdev,av.mean-av.sdev],[av.mean+av.sdev,av.mean+av.sdev], linestyle ='-',color='purple',alpha=alpha/2)
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=3,loc='upper left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$V_{cs}$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.xlim([0,qsqmaxphysDK.mean])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/Vcsbybin.pdf')
    plt.close()
    ##################################################################################################################
    plt.figure(figsize=figsize)
    x = []
    y = []
    terr = []
    eerr = []
    for i in range(len(d['Cleo1V2'])):
        x.append((d['Cleobins'][i]+d['Cleobins'][i+1])/2)
        y.append((d['Cleo1V2'][i]))
        terr.append(d['Cleo1terr'][i])
        eerr.append(d['Cleo1eerr'][i])
    ymean,yerr = unmake_gvar_vec(y)
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='ko' ,mfc='k',label='ClEO0',alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=terr,ms=ms,fmt='none',color='r' ,mfc='none',label='Theory',alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=eerr,ms=ms,fmt='none',color='b' ,mfc='none',label='Exp',alpha=0.8,capsize=capsize,lw=lw)
    
    av = d['Cleo1av']
    plt.plot([-1,2],[av.mean,av.mean],color='purple',label='Average')
    plt.fill_between([-1,2],[av.mean-av.sdev,av.mean-av.sdev],[av.mean+av.sdev,av.mean+av.sdev], color='purple',alpha=alpha/2)
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$V_{cs}^2$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.xlim([0,qsqmaxphysDK.mean])
    plt.ylim([0.77,1.21])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/Cleo1Vcsbybin.pdf')
    plt.close()
    ############################################################################
    plt.figure(figsize=figsize)
    x = []
    y = []
    terr = []
    eerr = []
    for i in range(len(d['Cleo2V2'])):
        x.append((d['Cleobins'][i]+d['Cleobins'][i+1])/2)
        y.append((d['Cleo2V2'][i]))
        terr.append(d['Cleo2terr'][i])
        eerr.append(d['Cleo2eerr'][i])
    ymean,yerr = unmake_gvar_vec(y)
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='ks' ,mfc='k',label='CLEO+',alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=terr,ms=ms,fmt='none',color='r' ,mfc='none',label='Theory',alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=eerr,ms=ms,fmt='none',color='b' ,mfc='none',label='Exp',alpha=0.8,capsize=capsize,lw=lw)
    av = d['Cleo2av']
    plt.plot([-1,2],[av.mean,av.mean],color='purple',label='Average')
    plt.fill_between([-1,2],[av.mean-av.sdev,av.mean-av.sdev],[av.mean+av.sdev,av.mean+av.sdev], color='purple',alpha=alpha/2)
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$V_{cs}^2$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.xlim([0,qsqmaxphysDK.mean])
    plt.ylim([0.77,1.21])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/Cleo2Vcsbybin.pdf')
    plt.close()
    # ###########################################################################################
    plt.figure(figsize=figsize)
    x = []
    y = []
    terr = []
    eerr = []
    for i in range(len(d['BESV2'])):
        x.append((d['BESbins'][i]+d['BESbins'][i+1])/2)
        y.append((d['BESV2'][i]))
        terr.append(d['BESterr'][i])
        eerr.append(d['BESeerr'][i])
    ymean,yerr = unmake_gvar_vec(y)
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='k^' ,mfc='k',label='BES',alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=terr,ms=ms,fmt='none',color='r' ,mfc='none',label='Theory',alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=eerr,ms=ms,fmt='none',color='b' ,mfc='none',label='Exp',alpha=0.8,capsize=capsize,lw=lw)
    av = d['BESav']
    plt.plot([-1,2],[av.mean,av.mean],color='purple',label='Average')
    plt.fill_between([-1,2],[av.mean-av.sdev,av.mean-av.sdev],[av.mean+av.sdev,av.mean+av.sdev], color='purple',alpha=alpha/2)
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$V_{cs}^2$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.xlim([0,qsqmaxphysDK.mean])
    plt.ylim([0.77,1.21])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/BESVcsbybin.pdf')
    plt.close()
    ###########################################################################################
    plt.figure(figsize=figsize)
    x = []
    y = []
    terr = []
    eerr = []
    for i in range(len(d['BaBarV2'])):
        x.append((d['BaBarbins'][i]+d['BaBarbins'][i+1])/2)
        y.append((d['BaBarV2'][i]))
        terr.append(d['BaBarterr'][i])
        eerr.append(d['BaBareerr'][i])
    ymean,yerr = unmake_gvar_vec(y)
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='kd' ,mfc='k',label='BaBar',alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=terr,ms=ms,fmt='none',color='r' ,mfc='none',label='Theory',alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=eerr,ms=ms,fmt='none',color='b' ,mfc='none',label='Exp',alpha=0.8,capsize=capsize,lw=lw)
    av = d['BaBarav']
    plt.plot([-1,2],[av.mean,av.mean],color='purple',label='Average')
    plt.fill_between([-1,2],[av.mean-av.sdev,av.mean-av.sdev],[av.mean+av.sdev,av.mean+av.sdev], color='purple',alpha=alpha/2)
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$V_{cs}^2$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.xlim([0,qsqmaxphysDK.mean])
    plt.ylim([0.77,1.21])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/BaBarVcsbybin.pdf')
    plt.close()
    ############################################################################################
    ##compare V_cs
    plt.figure(figsize=figsize)
    p = make_p_physical_point_DK(pfit,Fits)
    #fp0 = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
    f00,fp0 = make_spline_f0_fp_physpoint(p,Fits[0],0)
    HFLAV= gv.gvar('0.7180(33)')# from #1909.12524 p318
    Vcsq20 = HFLAV/fp0
    av = (d['average'])
    ETMC1 = gv.gvar('0.945(38)')#1706.03017
    ETMC2 = gv.gvar('0.970(33)')#1706.03657
    HPQCD1 = gv.gvar('0.961(26)')#1008.4562
    HPQCD2 = gv.gvar('0.963(15)')#1305.1462
    plt.errorbar(av.mean, 6, xerr=av.sdev,ms=ms,fmt='kd' ,mfc='none',capsize=capsize,lw=lw)
    plt.fill_between([av.mean-av.sdev,av.mean+av.sdev],[0,0],[7,7], color='k',alpha=alpha/2)
    plt.errorbar(Vcsq20.mean, 5, xerr=Vcsq20.sdev,ms=ms,fmt='rd' ,mfc='none',capsize=capsize,lw=lw)
    plt.errorbar(ETMC2.mean, 4, xerr=ETMC2.sdev,ms=ms,fmt='k*' ,mfc='none',capsize=capsize,lw=lw)
    plt.errorbar(ETMC1.mean, 3, xerr=ETMC1.sdev,ms=ms,fmt='r*' ,mfc='none',capsize=capsize,lw=lw)
    plt.plot([0.8,1.1],[2.5,2.5],color='k')
    plt.errorbar(HPQCD2.mean, 2, xerr=HPQCD2.sdev,ms=ms,fmt='ks' ,mfc='none',capsize=capsize,lw=lw)
    plt.errorbar(HPQCD1.mean, 1, xerr=HPQCD1.sdev,ms=ms,fmt='ro' ,mfc='none',capsize=capsize,lw=lw)
    plt.text(0.91,1.5,'$N_f=2+1$',fontsize=fontsizelab)
    plt.text(0.91,4.5,'$N_f=2+1+1$',fontsize=fontsizelab)
    plt.xlabel('$|V_{cs}|$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=False,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('none')
    plt.xlim([0.90,1.01])
    plt.ylim([0.5,6.5])
    plt.gca().set_yticks([6,5,4,3,2,1])
    plt.gca().set_yticklabels(['This work','This work','ETMC','ETMC','HPQCD','HPQCD'])#'arXiv:1706.03657','arXiv:1706.03017','arXiv:1305.1462','arXiv:1008.4562'])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/Vcscomp.pdf')
    plt.close()
    return()


####################################################################################################


def plot_Ht_H0(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,const2):
    p = make_p_physical_point_DK(pfit,Fits)
    H0 = []
    Ht = []
    qsq = []
    for q2 in np.linspace(0.05,qsqmaxphysDK.mean,nopts): #q2 now in GeV
        pV = gv.sqrt((q2-(p['MDphys']+p['MKphys'])**2)*(q2-(p['MDphys']-p['MKphys'])**2)/(4*p['MDphys']**2))
        qsq.append(q2)
        Ht.append(make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0)/gv.sqrt(q2)) #only need one fit
        H0.append(2*p['MDphys']*pV*make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)/gv.sqrt(q2))
    H0mean,H0err = unmake_gvar_vec(H0)
    H0upp,H0low = make_upp_low(H0)
    Htmean,Hterr = unmake_gvar_vec(Ht)
    Htupp,Htlow = make_upp_low(Ht)
    plt.figure(figsize=figsize)
    plt.plot(qsq, H0mean, color='b',label ='$H_0$')
    plt.fill_between(qsq,H0low,H0upp, color='b',alpha=alpha)
    plt.plot(qsq, Htmean, color='r',label ='$H_t$')
    plt.fill_between(qsq,Htlow,Htupp, color='b',alpha=alpha)
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_major_locator(MultipleLocator(1.0))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.5))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/HtH0.pdf')
    plt.close()
    return()

####################################################################################################

def plot_re_fit_fp(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,svdnoise,priornoise,const2):
    fitp,refitp = re_fit_fp(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,svdnoise,priornoise,const2)

    qsq = []
    orfit = []
    refit = []
    rat = []
    for q2 in np.linspace(0,qsqmaxphysDK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        orfit.append(make_fp_BK(Nijk,Npow,Nm,addrho,fitp,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2))
        refit.append(make_refit_fp(q2,refitp,Npow))
        rat.append(make_fp_BK(Nijk,Npow,Nm,addrho,fitp,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)/make_refit_fp(q2,refitp,Npow))
    orfitmean,orfiterr = unmake_gvar_vec(orfit)
    orfitupp,orfitlow = make_upp_low(orfit)
    refitmean,refiterr = unmake_gvar_vec(refit)
    refitupp,refitlow = make_upp_low(refit)
    ratmean,raterr = unmake_gvar_vec(rat)
    ratupp,ratlow = make_upp_low(rat)
    plt.figure(figsize=figsize)
    plt.plot(qsq, orfitmean, color='r')
    plt.fill_between(qsq,orfitlow,orfitupp, color='r',alpha=alpha,label=r'$f_+(q^2)_{\mathrm{orig}}$')
    plt.plot(qsq, refitmean, color='b')
    plt.fill_between(qsq,refitlow,refitupp, color='b',alpha=alpha,label=r'$f_+(q^2)_{\mathrm{refit}}$')
    plt.plot(qsq, ratmean, color='k')
    plt.fill_between(qsq,ratlow,ratupp, color='k',alpha=alpha,label=r'$\frac{f_+(q^2)_{\mathrm{orig}}}{f_+(q^2)_{\mathrm{refit}}}$')

    #handles, labels = plt.gca().get_legend_handles_labels()
    #handles = [h[0] for h in handles]
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=1)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    #plt.ylabel(r'$\frac{f_+(q^2)_{\mathrm{orig}}}{f_+(q^2)_{\mathrm{refit}}}$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/fitvsrefit.pdf')
    plt.close()


###########################################################################################################################

def error_plot(pfit,prior,Fits,Nijk,Npow,Nm,f,t_0,addrho,fpf0same,const2):
    qsqs = np.linspace(0,qsqmaxphysDK.mean,nopts)
    f0,fp = output_error_DK(pfit,prior,Fits,Nijk,Npow,Nm,f,qsqs,t_0,addrho,fpf0same,const2)
    plt.figure(18,figsize=figsize)
    ax1 = plt.subplot(211)
    ax1b = ax1.twinx()
    ax1.plot(qsqs,f0[1],color='r', ls='--',lw=2,label='Inputs')
    ax1.plot(qsqs,f0[2],color='purple',ls=':',lw=2,label='+ q mistunings')
    ax1.plot(qsqs,f0[3],color='b',ls='-',lw=2,label='+ Statistics')
    #ax1.plot(qsqs,f0[4],color='b',ls='-.',lw=2,label='HQET')
    ax1.plot(qsqs,f0[4],color='k',ls='-',lw=4,label='+ Discretisation')
    ax1.set_ylabel('$(f_0(q^2)~\% \mathrm{err})^2 $ ',fontsize=fontsizelab)
    ax1.tick_params(width=2,labelsize=fontsizelab)
    ax1.tick_params(which='major',length=major)
    ax1.tick_params(which='minor',length=minor)
    #plt.gca().yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('none')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.yaxis.set_major_locator(MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax1.set_xlim([0,qsqmaxphysDK.mean])
    ####################################### right hand y axis ###
    ax1b.plot(qsqs,f0[1],color='r', ls='--',lw=2,label='Inputs')
    ax1b.plot(qsqs,f0[2],color='purple',ls=':',lw=2,label='+ q mistunings')
    ax1b.plot(qsqs,f0[3],color='b',ls='-',lw=2,label='+ Statistics')
    #ax1b.plot(qsqs,f0[4],color='b',ls='-.',lw=2,label='HQET')
    ax1b.plot(qsqs,f0[4],color='k',ls='-',lw=4,label='+ Discretisation')
    ax1b.set_ylabel('$f_0(q^2)~\% \mathrm{err}$ ',fontsize=fontsizelab)
    ax1b.tick_params(width=2,labelsize=fontsizelab)
    ax1b.tick_params(which='major',length=major)
    ax1b.tick_params(which='minor',length=minor)
    low,upp = ax1.get_ylim()
    ax1b.set_ylim([low,upp])
    points = []
    rootpoints = []
    i = 0.1
    while i != 'stop':
        if i**2 < upp:
            points.append(i**2)
            if int(10*i)%2 == 0:
                rootpoints.append('{0}'.format(i))
            else:
                rootpoints.append('')
            i+=0.1
        else:
            i ='stop'
    ax1b.set_yticks(points)
    ax1b.set_yticklabels(rootpoints)
    
    plt.legend(loc='upper right',ncol=2,fontsize=fontsizeleg,frameon=False)

    ax2 = plt.subplot(212,sharex=ax1)
    ax2b = ax2.twinx()
    ax2.plot(qsqs,fp[1],color='r', ls='--',lw=2)
    ax2.plot(qsqs,fp[2],color='purple',ls=':',lw=2)
    ax2.plot(qsqs,fp[3],color='b',ls='-',lw=2)
    #ax2.plot(qsqs,fp[4],color='b',ls='-.',lw=2)
    ax2.plot(qsqs,fp[4],color='k',ls='-',lw=4)
    ax2.set_xlabel(r'$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    ax2.set_ylabel('$(f_+(q^2)~\%\mathrm{err})^2$',fontsize=fontsizelab)
    ax2.tick_params(width=2,labelsize=fontsizelab)
    ax2.tick_params(which='major',length=major)
    ax2.tick_params(which='minor',length=minor)
    ax2.xaxis.set_major_locator(MultipleLocator(0.5))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.set_xlim([0,qsqmaxphysDK.mean])
    #plt.axes().set_ylim([-0.8,2.5])
    #plt.axes().set_xlim([lower-0.22,upper+0.22])
    ######## right hand axis ##
    ax2b.plot(qsqs,fp[1],color='r', ls='--',lw=2)
    ax2b.plot(qsqs,fp[2],color='purple',ls=':',lw=2)
    ax2b.plot(qsqs,fp[3],color='b',ls='-',lw=2)
    #ax2b.plot(qsqs,fp[4],color='b',ls='-.',lw=2)
    ax2b.plot(qsqs,fp[4],color='k',ls='-',lw=4)
    ax2b.set_xlabel(r'$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    ax2b.set_ylabel('$f_+(q^2)~\%\mathrm{err}$',fontsize=fontsizelab)
    ax2b.tick_params(width=2,labelsize=fontsizelab)
    ax2b.tick_params(which='major',length=major)
    ax2b.tick_params(which='minor',length=minor)
    low,upp = ax2.get_ylim()
    ax2b.set_ylim([low,upp])
    points = []
    rootpoints = []
    i = 0.2
    while i != 'stop':
        if i**2 < upp:
            points.append(i**2)
            if int(10*i)%4 == 0:
                rootpoints.append('{0}'.format(i))
            else:
                rootpoints.append('')
            i+=0.2
        else:
            i ='stop'
    ax2b.set_yticks(points)
    ax2b.set_yticklabels(rootpoints)
    plt.tight_layout()
    plt.savefig('DK_spline_plots/f0fpluserr.pdf')
    plt.close()
    return()

###################################################################################################

def table_of_as(Fits,pfit,Nijk,Npow,Nm,fpf0same,addrho):
    list0 = []
    listp = []
    Fit = Fits[0]
    mass = Fit['masses'][0]
    fit = Fit['conf']
    p = make_p_physical_point_DK(pfit,Fits)
    logs = make_logs(p,Fit)
    atab = open('DKTables/tablesofas.txt','w')
    for n in range(Npow):
        if n == 0:
            atab.write('      {0}&'.format(make_an_BK(n,Nijk,Nm,addrho,p,'0',Fit,mass,0,fpf0same)))
        else:
            atab.write('{0}&'.format(make_an_BK(n,Nijk,Nm,addrho,p,'0',Fit,mass,0,fpf0same)))
        list0.append(make_an_BK(n,Nijk,Nm,addrho,p,'0',Fit,mass,0,fpf0same))
        listp.append(make_an_BK(n,Nijk,Nm,addrho,p,'p',Fit,mass,0,fpf0same))
    MDsstar = p['MDsstarphys']
    MDs0 = p['MDs0phys']
    for n in range(Npow):           
        atab.write('{0}&'.format(make_an_BK(n,Nijk,Nm,addrho,p,'p',Fit,mass,0,fpf0same)))
    atab.write('{0}&{1}&{2}\\\\ [1ex]\n'.format(MDs0,MDsstar,logs))
    atab.write('      \hline \n')
    list0.extend(listp)
    list0.append(MDs0)
    list0.append(MDsstar)
    list0.append(logs)
    covar = gv.evalcorr(list0)
    for i in range(2*Npow+3):
            atab.write('\n      ')
            for k in range(i):
                atab.write('&')
            for j in range(i,2*Npow+3):
                #print(covar[i][j])
                atab.write('{0:.5f}'.format(covar[i][j]))
                if j != 2*Npow+2:
                    atab.write('&')
                else:
                    atab.write('\\\\ [0.5ex]')
    atab.close()
    return()

################################################################################################################
def results_tables(fs_data,Fit):
    table = open('DKTables/{0}table.txt'.format(Fit['conf']),'w')
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
####################################################################################################

def plot_spline_fit_fp(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,svdnoise,priornoise,const2):
    fitp,fitf0,fitfp = spline_fit(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,svdnoise,priornoise,const2)
    qsq = []
    orfitfp = []
    spfitfp = []
    ratfp = []
    orfitf0 = []
    spfitf0 = []
    ratf0 = []
    for q2 in np.linspace(0,qsqmaxphysDK.mean,nopts): #q2 now in GeV
        f0pole = 1 #- q2/fitp['MDsstarphys']**2
        fppole = 1 #- q2/fitp['MDs0phys']**2
        qsq.append(q2)
        orfitfp.append(make_fp_BK(Nijk,Npow,Nm,addrho,fitp,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2))
        spfitfp.append(fitfp(q2)/fppole)
        ratfp.append(fppole*make_fp_BK(Nijk,Npow,Nm,addrho,fitp,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)/fitfp(q2))
        orfitf0.append(make_f0_BK(Nijk,Npow,Nm,addrho,fitp,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0))
        spfitf0.append(fitf0(q2)/f0pole)
        ratf0.append(f0pole*make_f0_BK(Nijk,Npow,Nm,addrho,fitp,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0)/fitf0(q2))
    orfitfpmean,orfitfperr = unmake_gvar_vec(orfitfp)
    orfitfpupp,orfitfplow = make_upp_low(orfitfp)
    spfitfpmean,spfitfperr = unmake_gvar_vec(spfitfp)
    spfitfpupp,spfitfplow = make_upp_low(spfitfp)
    ratfpmean,ratfperr = unmake_gvar_vec(ratfp)
    ratfpupp,ratfplow = make_upp_low(ratfp)
    plt.figure(figsize=figsize)
    plt.plot(qsq, orfitfpmean, color='r')
    plt.fill_between(qsq,orfitfplow,orfitfpupp, color='r',alpha=alpha,label=r'$f_+(q^2)_{\mathrm{orig}}$')
    plt.plot(qsq, spfitfpmean, color='b')
    plt.fill_between(qsq,spfitfplow,spfitfpupp, color='b',alpha=alpha,label=r'$f_+(q^2)_{\mathrm{spline}}$')
    plt.plot(qsq, ratfpmean, color='k')
    plt.fill_between(qsq,ratfplow,ratfpupp, color='k',alpha=alpha,label=r'$\frac{f_+(q^2)_{\mathrm{orig}}}{f_+(q^2)_{\mathrm{spline}}}$')
    plt.plot([0,qsqmaxphysDK.mean],[1.0,1.0],'k--', alpha=0.25)


    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=1)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/fpfitvsspline.pdf')

    orfitf0mean,orfitf0err = unmake_gvar_vec(orfitf0)
    orfitf0upp,orfitf0low = make_upp_low(orfitf0)
    spfitf0mean,spfitf0err = unmake_gvar_vec(spfitf0)
    spfitf0upp,spfitf0low = make_upp_low(spfitf0)
    ratf0mean,ratf0err = unmake_gvar_vec(ratf0)
    ratf0upp,ratf0low = make_upp_low(ratf0)
    plt.figure(figsize=figsize)
    plt.plot(qsq, orfitf0mean, color='r')
    plt.fill_between(qsq,orfitf0low,orfitf0upp, color='r',alpha=alpha,label=r'$f_0(q^2)_{\mathrm{orig}}$')
    plt.plot(qsq, spfitf0mean, color='b')
    plt.fill_between(qsq,spfitf0low,spfitf0upp, color='b',alpha=alpha,label=r'$f_0(q^2)_{\mathrm{spline}}$')
    plt.plot(qsq, ratf0mean, color='k')
    plt.fill_between(qsq,ratf0low,ratf0upp, color='k',alpha=alpha,label=r'$\frac{f_0(q^2)_{\mathrm{orig}}}{f_0(q^2)_{\mathrm{spline}}}$')
    plt.plot([0,qsqmaxphysDK.mean],[1.0,1.0],'k--',alpha=0.25)


    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=1)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig('DK_spline_plots/f0fitvsspline.pdf')
    plt.close()

#########################################################################################################































































































































#######################################################################################################################
#Work below here



#########################################################################################################################


def fT_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,Del,addrho,fpf0same,adddata):
    i = 0
    plotfits = []
    for Fit in Fits:
        if Fit['conf'] not in ['Fp','VCp','Cp','Fs','SFs','UFs']:
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
    p = make_p_physical_point_DK(pfit,Fits,Del,t_0)
    for q2 in np.linspace(0,qsqmaxphysDK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        zed = make_z(q2,t_0,p['MDphys'],MKphys) #all GeV dimensions
        z.append(zed.mean)
        #        make_fp_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,fpf0same,amh)
        y.append(make_fT_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,zed.mean,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(4,figsize=figsize)
    plt.plot(qsq, ymean, color='g')
    plt.fill_between(qsq,ylow,yupp, color='g',alpha=alpha)
    if datafTmaxBK != None and adddata:
        plt.errorbar(qsqmaxphysBK.mean, datafTmaxBK.mean, xerr=qsqmaxphysBK.sdev, yerr=datafTmaxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
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
    plt.savefig('DK_spline_plots/fTpoleinqsq.pdf')
    plt.close()
    
    plt.figure(5,figsize=figsize)
    plt.plot(z,ymean, color='g')
    plt.fill_between(z,ylow,yupp, color='g',alpha=alpha)
    if datafTmaxBK != None and adddata:
        plt.errorbar( make_z(qsqmaxphysBK,t_0,MBphys,MKphys).mean, datafTmaxBK.mean, xerr=make_z(qsqmaxphysBK,t_0,MBphys,MKphys).sdev, yerr=datafTmaxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
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
    plt.savefig('DK_spline_plots/fTpoleinz.pdf')
    plt.close()
    return()
################################################################################################
#Not changed below here
################################################################################################






###############################################################################################

def f0_fp_fT_in_qsq(pfit,Fits,t_0,Nijk,Npow,Nm,Del,addrho,fpf0same):
    qsq = []
    y0 = []
    yp = []
    yT = []
    p = make_p_physical_point_BK(pfit,Fits,Del)
    for q2 in np.linspace(0,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        zed = make_z(q2,t_0,MBphys,MKphys) #all GeV dimensions
        #        make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,amh)
        y0.append(make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,q2,zed.mean,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
        yp.append(make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,q2,zed.mean,Fits[0]['masses'][0],fpf0same,0))
        yT.append(make_fT_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,q2,zed.mean,Fits[0]['masses'][0],fpf0same,0))
    y0mean,y0err = unmake_gvar_vec(y0)
    y0upp,y0low = make_upp_low(y0)
    ypmean,yperr = unmake_gvar_vec(yp)
    ypupp,yplow = make_upp_low(yp)
    yTmean,yTerr = unmake_gvar_vec(yT)
    yTupp,yTlow = make_upp_low(yT)
    plt.figure(10,figsize=figsize)
    plt.plot(qsq, y0mean, color='b',label='$f_0(q^2)$')
    plt.fill_between(qsq,y0low,y0upp, color='b',alpha=alpha)
    plt.plot(qsq, ypmean, color='r',label='$f_+(q^2)$')
    plt.fill_between(qsq,yplow,ypupp, color='r',alpha=alpha)
    plt.plot(qsq, yTmean, color='g',label='$f_T(q^2)$')
    plt.fill_between(qsq,yTlow,yTupp, color='g',alpha=alpha)
    plt.errorbar(0,0.319, yerr=0.066,fmt='r*',ms=ms,mfc='none')#,label = r'arXiv:1306.2384',lw=lw)
    #plt.errorbar(qsqmaxphysBK.mean,0.861, yerr=0.048,fmt='b*',ms=ms,mfc='none',label = r'arXiv:1306.2384',lw=lw)
    #plt.errorbar(qsqmaxphysBK.mean,2.63, yerr=0.13,fmt='r*',ms=ms,mfc='none',label = r'arXiv:1306.2384',lw=lw)
    #plt.errorbar(0,0.270, yerr=0.095,fmt='g*',ms=ms,mfc='none')#,label = r'arXiv:1306.2384',lw=lw)
    #plt.errorbar(qsqmaxphysBK.mean,2.39, yerr=0.17,fmt='g*',ms=ms,mfc='none',label = r'arXiv:1306.2384',lw=lw)
    plt.legend(fontsize=fontsizeleg,frameon=False,loc='upper left')
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
    plt.savefig('DK_spline_plots/f0fpfTinqsq.pdf')
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
        f0max.append(make_f0_BsEtas(Nijk,Npow,addrho,p,Fits[0],0,qsqmax.mean,zmax,Fits[0]['masses'][0],fpf0same,0))
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
    plt.plot([MDsphys.mean,MDsphys.mean],[-10,10],'k--',lw=lw/2,alpha=alpha)
    plt.text(MDsphys.mean,-0.30,'$M_{D_s}$',fontsize=fontsizelab,horizontalalignment='center')
    plt.plot([MBsphys.mean,MBsphys.mean],[-10,10],'k--',lw=lw/2,alpha=alpha)
    plt.text(MBsphys.mean,-0.30,'$M_{B_s}$',fontsize=fontsizelab,horizontalalignment='center')
    plt.text(4.0,2.5,'$f_+(q^2_{\mathrm{max}})$',fontsize=fontsizelab)
    plt.text(2.5,0.3,'$f_{0,+}(0)$',fontsize=fontsizelab)
    plt.text(4.5,1.0,'$f_0(q^2_{\mathrm{max}})$',fontsize=fontsizelab)
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
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper left')
    ##################################
    plt.axes().set_ylim([0,3.45])
    plt.tight_layout()
    plt.savefig('DK_spline_plots/f0f0fpinmh.pdf')
    plt.close()
    return()

#####################################################################################################

def beta_delta_in_Mh(pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same):
    MHs = []
    delta = []
    invbeta = []
    alp =[]
    for Mh in np.linspace(MDsphys.mean,MBsphys.mean,nopts): #q2 now in GeV
        p = make_p_Mh_BsEtas(pfit,Fits,Del,Mh)
        MHs.append(Mh)
        al,d,invb = make_beta_delta_BsEtas(Fits,t_0,Nijk,Npow,addrho,p,fpf0same,Del,Mh)
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
    plt.xlabel('$M_{H_s}[\mathrm{GeV}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.plot([MDsphys.mean,MDsphys.mean],[-10,10],'k--',lw=lw/2,alpha=alpha)
    plt.text(MDsphys.mean+0.05,-0.45,'$M_{D_s}$',fontsize=fontsizelab)
    plt.plot([MBsphys.mean,MBsphys.mean],[-10,10],'k--',lw=lw/2,alpha=alpha)
    plt.text(MBsphys.mean-0.05,-0.45,'$M_{B_s}$',fontsize=fontsizelab,horizontalalignment='right')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.text(3.5,-0.1,r'$\delta$',fontsize=fontsizelab)
    plt.text(2.7,0.7,r'$\beta^{-1}$',fontsize=fontsizelab)
    plt.text(4.3,0.4,r'$\alpha$',fontsize=fontsizelab)
    plt.axes().set_ylim([-0.5,1.32])
    ############ add data ############
    plt.errorbar(MBphys.mean+0.01,0.80,yerr=[[0.2],[0.5]],fmt='k*',ms=ms,mfc='none',label = r'$\alpha^{B\to\pi}(\beta^{-1}=0.833)$',lw=lw)#,capsize=capsize)
    plt.errorbar(MBphys.mean-0.01,0.60, yerr=[[0.7],[0.3]],fmt='r*',ms=ms,mfc='none',label = r'$\delta^{B\to\pi}(\beta^{-1}=0.833)$',lw=lw)#,capsize=capsize)
    plt.errorbar(MDphys.mean+0.01,1+1/1.6-1.1,yerr=[[0.2],[0.6]],fmt='r^',ms=ms,mfc='none',label = r'$\delta^{D\to\pi}(\beta^{-1}=0.625)$',lw=lw)#,capsize=capsize)
    plt.errorbar(MDphys.mean-0.01,1+1/1.8-0.91, yerr=[[0.05],[0.12]],fmt='ro',ms=ms,mfc='none',label = r'$\delta^{D\to{}K}(\beta^{-1}=0.556)$',lw=lw)#,capsize=capsize)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,loc='upper center',ncol=2)
    ##################################
    plt.tight_layout()
    plt.savefig('DK_spline_plots/betadeltainmh.pdf')
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
    plt.savefig('DK_spline_plots/HQETrat.pdf')
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
    plt.savefig('DK_spline_plots/HillratinE.pdf')
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
    plt.savefig('DK_spline_plots/HillratinlowE.pdf')
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
    plt.savefig('DK_spline_plots/Hillratinmh_E{0}.pdf'.format(E))
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
    plt.savefig('DK_spline_plots/Hillratininvmh_E{0}.pdf'.format(E))
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
    plt.savefig('DK_spline_plots/f0poleinza{0}.pdf'.format(afm))
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
    plt.savefig('DK_spline_plots/fppoleinza{0}.pdf'.format(afm))
    plt.close()
    return()



##################################################################################################


