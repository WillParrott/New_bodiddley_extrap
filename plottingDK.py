from functionsDK import *
import numpy as np
import gvar as gv
import lsqfit
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
#from matplotlib import Scatter
matplotlib.use('Agg')
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

####################################################################################################
def sort_for_bar(data,spacing):
    y = []
    x = []
    i = 0
    while i+spacing <= 1:
        y.append(len(list(x for x in data if i <= x < i + spacing)))
        x.append(i+spacing/2)
        i += spacing
    return(x,y)


def twist_corr_plots(Fit,fs,thpts):
    corrs = collections.OrderedDict()
    corrs['f0'] = []
    corrs['fp'] = []
    corrs['S'] = []
    corrs['V'] = []
    for i,twist in enumerate(Fit['twists']):
        for j in range(i+1,len(Fit['twists'])):
            twist2 = Fit['twists'][j]
            for k,mass in enumerate(Fit['masses']):
                for l in range(len(Fit['masses'])):
                    mass2 = Fit['masses'][l]
                    f0 = fs['f0_m{0}_tw{1}'.format(mass,twist)]
                    fp = fs['fp_m{0}_tw{1}'.format(mass,twist)]
                    
                    f02 = fs['f0_m{0}_tw{1}'.format(mass2,twist2)]
                    fp2 = fs['fp_m{0}_tw{1}'.format(mass2,twist2)]
                    corrs['f0'].append(abs(gv.evalcorr([f0,f02])[0][1]))
                    if fp != None and fp2!= None:
                        corrs['fp'].append(abs(gv.evalcorr([fp,fp2])[0][1]))
                    for thpt in thpts[Fit['conf']]:
                        if twist != '0' or thpt != 'T':
                            V = Fit['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)]
                            V2 = Fit['{0}Vnn_m{1}_tw{2}'.format(thpt,mass2,twist2)]
                            correlation = abs(gv.evalcorr([V,V2])[0][1])
                            #if correlation > 0.5:
                            #print(Fit['conf'],thpt,mass,twist,mass2,twist2, correlation)
                            corrs[thpt].append(correlation)
    for tag in ['f0','fp','S','V']:
        data = corrs[tag]
        x,y = sort_for_bar(data,0.025)
        plt.figure(figsize=figsize)
        plt.bar(x,y,width=0.025)
        plt.xlabel('correlation',fontsize=fontsizelab)
        plt.ylabel(r'Frequency',fontsize=fontsizelab)
        plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
        plt.axes().tick_params(which='major',length=major)
        plt.axes().tick_params(which='minor',length=minor)
        plt.axes().yaxis.set_ticks_position('both')
        #plt.axes().xaxis.set_major_locator(MultipleLocator(0.05))
        #plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
        #plt.axes().yaxis.set_major_locator(MultipleLocator(0.1))
        #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.05))
        plt.tight_layout()
        plt.savefig('TwistCorrs/DK{0}corrs{1}_tw.pdf'.format(Fit['conf'],tag))
        plt.close()
    for twist in Fit['twists'][1:]:
        f0 = fs['f0_m{0}_tw{1}'.format(mass,twist)]
        fp = fs['fp_m{0}_tw{1}'.format(mass,twist)]
        print(Fit['conf'],'tw = ', twist, 'f0 fp corr = ', gv.evalcorr([f0,fp])[0][1])
    return()



######################################################################################################
def plot_gold_non_split(Fits):
    plt.figure(figsize=figsize)
    i = 0
    for Fit in Fits:
        for mass in Fit['masses']:
            y = 1000*(Fit['GNGsplit_m{0}'.format(mass)]/Fit['a']).mean
            yerr = 1000*(Fit['GNGsplit_m{0}'.format(mass)]/Fit['a']).sdev
            x = float(mass)**2
            if Fit['conf'][-1] == 'p':
                plt.errorbar(x,y,yerr=yerr,fmt=symbs[i],color='k',label=Fit['label'],ms=ms,mfc='none')
            else:
                plt.errorbar(x,y,yerr=yerr,fmt=symbs[i],color='r',label=Fit['label'],ms=ms,mfc='none')
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
    plt.savefig('DKPlots/gold-non-split.pdf')
    plt.close()
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
    plt.savefig('DKPlots/speedoflight.pdf')
    #plt.show()
    plt.close()
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
    plt.savefig('DKPlots/Z_Vinamhsq.pdf')
    plt.close()
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
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label'])),alpha=alpha)
            
            plt.figure(3,figsize=figsize)
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label'])),alpha=alpha)
            
            j += 1
        i += 1
    qsq = []
    z = []
    y = []
    p = make_p_physical_point_DK(pfit,Fits)
    for q2 in np.linspace(0,qsqmaxphysDK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        zed = make_z(q2,t_0,p['MDphys'],p['MKphys']) #all GeV dimensions
        if zed == 0:
            z.append(zed)
        else:
            z.append(zed.mean)
            
        #        make_f0_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,fpf0sameamh)
        y.append(make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
    gv.dump(y,'Fits/f0_orig.pickle')
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(2,figsize=figsize)
    plt.plot(qsq, ymean, color='b')
    plt.fill_between(qsq,ylow,yupp, color='b',alpha=alpha)
    #if dataf0maxBK != None and adddata:
    #    plt.errorbar(qsqmaxphysBK.mean, dataf0maxBK.mean, xerr=qsqmaxphysBK.sdev, yerr=dataf0maxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$\mathrm{arXiv:} 1510.07446$')
    #if dataf00BK != None and adddata:
    #    plt.errorbar(0, dataf00BK.mean, yerr=dataf00BK.sdev, color='k', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
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
    plt.savefig('DKPlots/f0poleinqsq.pdf')
    plt.close()
    
    plt.figure(3,figsize=figsize)
    plt.plot(z,ymean, color='b')
    plt.fill_between(z,ylow,yupp, color='b',alpha=alpha)
    if dataf0maxBK != None and adddata:
        plt.errorbar( make_z(qsqmaxphysBK,t_0,MBphys,MKphys).mean, dataf0maxBK.mean, xerr=make_z(qsqmaxphysBK,t_0,MBphys,MKphys).sdev, yerr=dataf0maxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$\mathrm{arXiv:} 1510.07446$')
    if dataf00BK != None and adddata:
        plt.errorbar(make_z(0,t_0,MBphys,MKphys).mean, dataf00BK.mean,xerr = make_z(0,t_0,MBphys,MKphys).sdev ,yerr=dataf00BK.sdev, color='k', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2,loc='lower left')
    plt.xlabel('$z$',fontsize=fontsizelab)
    plt.ylabel(r'$f_0(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.tight_layout()
    plt.savefig('DKPlots/f0poleinz.pdf')
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
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label'])),alpha=alpha)
            
            plt.figure(5,figsize=figsize)
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label'])),alpha=alpha)
            
            j += 1
        i += 1
    qsq = []
    z = []
    y = []
    p = make_p_physical_point_DK(pfit,Fits)
    for q2 in np.linspace(0,qsqmaxphysDK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        zed = make_z(q2,t_0,p['MDphys'],p['MKphys']) #all GeV dimensions
        if zed ==0:
            z.append(zed)
        else:
            z.append(zed.mean)
        #        make_fp_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,fpf0same,amh)
        y.append(make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)) #only need one fit
    gv.dump(y,'Fits/fp_orig.pickle')
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
    plt.savefig('DKPlots/fppoleinqsq.pdf')
    plt.close()
    
    plt.figure(5,figsize=figsize)
    plt.plot(z,ymean, color='r')
    plt.fill_between(z,ylow,yupp, color='r',alpha=alpha)
    if datafpmaxBK != None and adddata:
        plt.errorbar( make_z(qsqmaxphysBK,t_0,MBphys,MKphys).mean, datafpmaxBK.mean, xerr=make_z(qsqmaxphysBK,t_0,MBphys,MKphys).sdev, yerr=datafpmaxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$z$',fontsize=fontsizelab)
    plt.ylabel(r'$f_+(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig('DKPlots/fppoleinz.pdf')
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
    plt.savefig('DKPlots/f0fpinqsq.pdf')
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
    plt.savefig('DKPlots/f0fpinz.pdf')
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
            plt.errorbar(qsq, y0, xerr=qsqerr, yerr=y0err, color='r', fmt=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label'])),alpha=alpha)
            legend_elements.append(Line2D([0], [0],color='k',linestyle='None', marker=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label']))))
            plt.errorbar(qsq, yp, xerr=qsqerr, yerr=yperr, color='b', mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(qsq, yp, xerr=qsqerr, yerr=yperr, color='b', fmt=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label'])),alpha=alpha)
            
            plt.figure(109,figsize=figsize)
            plt.errorbar(z, y0, xerr=zerr, yerr=y0err, color='r', mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(z, y0, xerr=zerr, yerr=y0err, color='r', fmt=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label'])),alpha=alpha)
            plt.errorbar(z, yp, xerr=zerr, yerr=yperr, color='b', mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(z, yp, xerr=zerr, yerr=yperr, color='b', fmt=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label'])),alpha=alpha)
            
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
    plt.savefig('DKPlots/f0fpdatainqsq.pdf')
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
    plt.savefig('DKPlots/f0fpdatainz.pdf')
    plt.close()
    return()

################################################################################################

def f0_no_pole_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata):
    i = 0
    for Fit in Fits:
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
                MHs0 = pfit['MH_{0}_m{1}'.format(fit,mass)] + pfit['a_{0}'.format(fit)] * (pfit['MDs0phys']- pfit['MDphys'])
                pole = 1-(q2/MHs0**2)
                y.append(pole * fs_data[Fit['conf']]['f0_m{0}_tw{1}'.format(mass,twist)])
            qsq,qsqerr = unmake_gvar_vec(qsq)
            z,zerr = unmake_gvar_vec(z)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(6,figsize=figsize)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label'])),alpha=alpha)
            
            plt.figure(7,figsize=figsize)
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label'])),alpha=alpha)
            
            j += 1
        i += 1
    qsq = []
    z = []
    y = []
    p = make_p_physical_point_DK(pfit,Fits)
    for q2 in np.linspace(0,qsqmaxphysDK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        pole = 1 - q2/(p['MDs0phys'])**2
        zed = make_z(q2,t_0,p['MDphys'],p['MKphys']) #all GeV dimensions
        if zed == 0:
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
    plt.savefig('DKPlots/f0nopoleinqsq.pdf')
    plt.close()
    
    plt.figure(7,figsize=figsize)
    plt.plot(z,ymean, color='b')
    plt.fill_between(z,ylow,yupp, color='b',alpha=alpha)
    #if dataf0maxBsEtas != None and adddata:
    #    pole = 1 - qsqmaxphysBK/(MBsphys+Del)**2
    #    plt.errorbar( make_z(qsqmaxphysBK,t_0,MBsphys,Metasphys).mean, (pole*dataf0maxBsEtas).mean, xerr=make_z(qsqmaxphysBK,t_0,MBsphys,Metasphys).sdev, yerr=(pole*dataf0maxBsEtas).sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$\mathrm{arXiv:} 1510.07446$')
    #if dataf00BsEtas != None and adddata:
    #    plt.errorbar(make_z(0,t_0,MBsphys,Metasphys).mean, dataf00BsEtas.mean,xerr = make_z(0,t_0,MBsphys,Metasphys).sdev ,yerr=dataf00BsEtas.sdev, color='k', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper left')
    plt.xlabel('$z$',fontsize=fontsizelab)
    plt.ylabel(r'$\left(1-\frac{q^2}{M^2_{H_{s}^0}} \right)f_0(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.tight_layout()
    plt.savefig('DKPlots/f0nopoleinz.pdf')
    plt.close()
    return()

################################################################################################

def fp_no_pole_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,const2):
    i = 0
    for Fit in Fits:
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
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label'])),alpha=alpha)
            
            plt.figure(9,figsize=figsize)
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i],alpha=alpha)
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label'])),alpha=alpha)
            
            j += 1
        i += 1
    qsq = []
    z = []
    y = []
    p = make_p_physical_point_DK(pfit,Fits)
    for q2 in np.linspace(0,qsqmaxphysDK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        pole = 1 - (q2/p['MDsstarphys']**2)
        zed = make_z(q2,t_0,p['MDphys'],p['MKphys']) #all GeV dimensions
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
    plt.savefig('DKPlots/fpnopoleinqsq.pdf')
    plt.close()
    
    plt.figure(9,figsize=figsize)
    plt.plot(z,ymean, color='r')
    plt.fill_between(z,ylow,yupp, color='r',alpha=alpha)
    #if datafpmaxBsEtas != None and adddata:
    #    plt.errorbar( make_z(qsqmaxphysBK,t_0,MBsphys,Metasphys).mean, datafpmaxBsEtas.mean, xerr=make_z(qsqmaxphysBK,t_0,MBsphys,Metasphys).sdev, yerr=datafpmaxBsEtas.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')    
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=3,loc='lower left')
    plt.xlabel('$z$',fontsize=fontsizelab)
    plt.ylabel(r'$\left(1-\frac{q^2}{M^2_{D_{s}^*}} \right)f_+(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.tight_layout()
    plt.savefig('DKPlots/fpnopoleinz.pdf')
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
        y.append(gv.sqrt(d['Cleo1V2'][i]/cor2))
    ymean,yerr = unmake_gvar_vec(y)
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='ko' ,mfc='none',label="CLEO '09 ($D^0$)",alpha=0.8)
    x = []
    y = []
    for i in range(len(d['Cleo2V2'])):
        x.append((d['Cleobins'][i]+d['Cleobins'][i+1])/2)
        y.append(gv.sqrt(d['Cleo2V2'][i]/cor2))
    ymean,yerr = unmake_gvar_vec(y)
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='rs' ,mfc='none',label="CLEO '09 ($D^+$)",alpha=0.8)
    x = []
    y = []
    for i in range(len(d['BESV2'])):
        x.append((d['BESbins'][i]+d['BESbins'][i+1])/2)
        y.append(gv.sqrt(d['BESV2'][i]/cor2))
    ymean,yerr = unmake_gvar_vec(y)
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='b^' ,mfc='none',label="BES '15A",alpha=0.8)
    x = []
    y = []
    for i in range(len(d['BaBarV2'])):
        x.append((d['BaBarbins'][i]+d['BaBarbins'][i+1])/2)
        y.append(gv.sqrt(d['BaBarV2'][i]*Banorm/cor2))
    ymean,yerr = unmake_gvar_vec(y)
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='gd' ,mfc='none',label="BaBar '06",alpha=0.8)
    av = d['average']
    plt.plot([-1,2],[av.mean,av.mean],color='purple',label='Average')
    plt.fill_between([-1,2],[av.mean-av.sdev,av.mean-av.sdev],[av.mean+av.sdev,av.mean+av.sdev], linestyle ='-',color='purple',alpha=alpha/2)
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=3,loc='upper left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$|V_{cs}|$',fontsize=fontsizelab)
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
    plt.savefig('DKPlots/Vcsbybin.pdf')
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
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='ko' ,mfc='k',label="CLEO '09 ($D^0$)",alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=terr,ms=ms,fmt='none',color='r' ,mfc='none',label='Theory',alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=eerr,ms=ms,fmt='none',color='b' ,mfc='none',label='Exp',alpha=0.8,capsize=capsize,lw=lw)
    
    av = d['Cleo1av']
    plt.plot([-1,2],[av.mean,av.mean],color='purple',label='Average')
    plt.fill_between([-1,2],[av.mean-av.sdev,av.mean-av.sdev],[av.mean+av.sdev,av.mean+av.sdev], color='purple',alpha=alpha/2)
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$\eta^2_{\mathrm{EW}}(1+\delta_{\mathrm{EM}})|V_{cs}|^2$',fontsize=fontsizelab)#r'$\eta^2_{\mathrm{EW}}(1+\delta_{\mathrm{EM}})|V_{cs}|^2$'
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.xlim([0,qsqmaxphysDK.mean])
    plt.ylim([0.74,1.23])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    plt.tight_layout()
    plt.savefig('DKPlots/Cleo1Vcsbybin.pdf')
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
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='ks' ,mfc='k',label="CLEO '09 ($D^+$)",alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=terr,ms=ms,fmt='none',color='r' ,mfc='none',label='Theory',alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=eerr,ms=ms,fmt='none',color='b' ,mfc='none',label='Exp',alpha=0.8,capsize=capsize,lw=lw)
    av = d['Cleo2av']
    plt.plot([-1,2],[av.mean,av.mean],color='purple',label='Average')
    plt.fill_between([-1,2],[av.mean-av.sdev,av.mean-av.sdev],[av.mean+av.sdev,av.mean+av.sdev], color='purple',alpha=alpha/2)
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$\eta^2_{\mathrm{EW}}(1+\delta_{\mathrm{EM}})|V_{cs}|^2$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.xlim([0,qsqmaxphysDK.mean])
    plt.ylim([0.74,1.23])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    plt.tight_layout()
    plt.savefig('DKPlots/Cleo2Vcsbybin.pdf')
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
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='k^' ,mfc='k',label="BES '15A",alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=terr,ms=ms,fmt='none',color='r' ,mfc='none',label='Theory',alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=eerr,ms=ms,fmt='none',color='b' ,mfc='none',label='Exp',alpha=0.8,capsize=capsize,lw=lw)
    av = d['BESav']
    plt.plot([-1,2],[av.mean,av.mean],color='purple',label='Average')
    plt.fill_between([-1,2],[av.mean-av.sdev,av.mean-av.sdev],[av.mean+av.sdev,av.mean+av.sdev], color='purple',alpha=alpha/2)
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$\eta^2_{\mathrm{EW}}(1+\delta_{\mathrm{EM}})|V_{cs}|^2$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.xlim([0,qsqmaxphysDK.mean])
    plt.ylim([0.74,1.23])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    plt.tight_layout()
    plt.savefig('DKPlots/BESVcsbybin.pdf')
    plt.close()
    #############################################################################################
    plt.figure(figsize=figsize)
    x = []
    y = []
    terr = []
    eerr = []
    for i in range(len(d['BESDpV2'])):
        x.append((d['BESDpbins'][i]+d['BESDpbins'][i+1])/2)
        y.append((d['BESDpV2'][i]))
        terr.append(d['BESDpterr'][i])
        eerr.append(d['BESDpeerr'][i])
    ymean,yerr = unmake_gvar_vec(y)
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='k^' ,mfc='k',label="BES '17",alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=terr,ms=ms,fmt='none',color='r' ,mfc='none',label='Theory',alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=eerr,ms=ms,fmt='none',color='b' ,mfc='none',label='Exp',alpha=0.8,capsize=capsize,lw=lw)
    av = d['BESDpav']
    plt.plot([-1,2],[av.mean,av.mean],color='purple',label='Average')
    plt.fill_between([-1,2],[av.mean-av.sdev,av.mean-av.sdev],[av.mean+av.sdev,av.mean+av.sdev], color='purple',alpha=alpha/2)
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$\eta^2_{\mathrm{EW}}(1+\delta_{\mathrm{EM}})|V_{cs}|^2$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.xlim([0,qsqmaxphysDK.mean])
    plt.ylim([0.74,1.23])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    plt.tight_layout()
    plt.savefig('DKPlots/BESDpVcsbybin.pdf')
    plt.close()
    ###########################################################################################
    plt.figure(figsize=figsize)
    x = []
    y = []
    terr = []
    eerr = []
    for i in range(len(d['BaBarV2'])):
        x.append((d['BaBarbins'][i]+d['BaBarbins'][i+1])/2)
        y.append((d['BaBarV2'][i]*Banorm))
        terr.append(d['BaBarterr'][i])
        eerr.append(d['BaBareerr'][i])
    ymean,yerr = unmake_gvar_vec(y)
    plt.errorbar(x, ymean, yerr=yerr,ms=ms,fmt='kd' ,mfc='k',label="BaBar '06",alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=terr,ms=ms,fmt='none',color='r' ,mfc='none',label='Theory',alpha=0.8,capsize=capsize,lw=lw)
    plt.errorbar(x, ymean, yerr=eerr,ms=ms,fmt='none',color='b' ,mfc='none',label='Exp',alpha=0.8,capsize=capsize,lw=lw)
    av = d['BaBarav']
    plt.plot([-1,2],[av.mean,av.mean],color='purple',label='Average')
    plt.fill_between([-1,2],[av.mean-av.sdev,av.mean-av.sdev],[av.mean+av.sdev,av.mean+av.sdev], color='purple',alpha=alpha/2)
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=2,loc='upper left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$\eta^2_{\mathrm{EW}}(1+\delta_{\mathrm{EM}})|V_{cs}|^2$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.xlim([0,qsqmaxphysDK.mean])
    plt.ylim([0.74,1.23])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    plt.tight_layout()
    plt.savefig('DKPlots/BaBarVcsbybin.pdf')
    plt.close()
    ############################################################################################
    ##compare V_cs
    plt.figure(figsize=figsize)
    VcsB = plot_Vcs_from_B(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2)
    p = make_p_physical_point_DK(pfit,Fits)
    fp0 = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
    HFLAV= gv.gvar('0.7180(33)')# from #1909.12524 p318
    Vcsq20 = HFLAV/(fp0*gv.sqrt(cor2))
    av = d['average']
    ETMC1 = gv.gvar('0.945(38)')#1706.03017
    ETMC2 = gv.gvar('0.970(33)')#1706.03657
    HPQCD1 = gv.gvar('0.961(26)')#1008.4562
    HPQCD2 = gv.gvar('0.963(15)')#1305.1462
    FERMI = gv.gvar('0.969(105)')#0408306
    Vcssl = gv.gvar('0.939(38)') #Current PDG sl value
    plt.errorbar(av.mean, 9, xerr=av.sdev,ms=ms,fmt='bd' ,capsize=capsize,lw=lw)
    plt.fill_between([av.mean-av.sdev,av.mean+av.sdev],[0,0],[10,10], color='b',alpha=alpha/2)
    plt.errorbar(Vcsq20.mean, 8, xerr=Vcsq20.sdev,ms=ms,fmt='rd' ,capsize=capsize,lw=lw)
    plt.errorbar(VcsB.mean, 7, xerr=VcsB.sdev,ms=ms,fmt='gd' ,capsize=capsize,lw=lw)
    plt.errorbar(ETMC2.mean, 6, xerr=ETMC2.sdev,ms=ms,fmt='b*' ,capsize=capsize,lw=lw)
    plt.errorbar(ETMC1.mean, 5, xerr=ETMC1.sdev,ms=ms,fmt='r*' ,capsize=capsize,lw=lw)
    plt.plot([0.8,1.1],[4.5,4.5],color='k')
    plt.errorbar(HPQCD2.mean, 4, xerr=HPQCD2.sdev,ms=ms,fmt='bs' ,capsize=capsize,lw=lw)
    plt.errorbar(HPQCD1.mean, 3, xerr=HPQCD1.sdev,ms=ms,fmt='ro' ,capsize=capsize,lw=lw)
    plt.errorbar(FERMI.mean, 2, xerr=FERMI.sdev,ms=ms,fmt='r^' ,capsize=capsize,lw=lw)
    plt.plot([0.8,1.1],[1.5,1.5],color='k')
    plt.errorbar(Vcssl.mean, 1, xerr=Vcssl.sdev,ms=ms,fmt='kh' ,capsize=capsize,lw=lw)
    plt.text(0.87,3.0,'$N_f=2+1$',fontsize=fontsizelab, va='center')
    plt.text(0.87,7.0,'$N_f=2+1+1$',fontsize=fontsizelab, va='center')
    plt.xlabel('$|V_{cs}|$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=False,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('none')
    plt.xlim([0.85,1.085])
    plt.ylim([0.5,9.5])
    plt.gca().set_yticks([9,8,7,6,5,4,3,2,1])
    plt.gca().set_yticklabels(["HPQCD '21","HPQCD '21","HPQCD '21","ETMC '17","ETMC '17","HPQCD '13","HPQCD '10","Fermilab/MILC '04","PDG '20 SL av."])#'arXiv:1706.03657','arXiv:1706.03017','arXiv:1305.1462','arXiv:1008.4562'])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    handles = [ Patch(facecolor='b', edgecolor='b',label=r'$\frac{d\Gamma}{dq^2}$'),Patch(facecolor='r', edgecolor='r',label=r'$|V_{cs}|f_+(0)$'),Patch(facecolor='g', edgecolor='g',label=r'$\Gamma$')]
    plt.legend(handles=handles,fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper right')
    plt.tight_layout()
    plt.savefig('DKPlots/Vcscomp.pdf')
    plt.close()

############################################################################################

    plt.figure(figsize=figsize)
    Vcsl = gv.gvar('0.983(18)') #PDG on decay constants 0.992(12) is other PDG value
    Vusl = gv.gvar('0.2252(5)') #PDG
    Vussl = gv.gvar('0.2231(7)') #PDG
    Vts = gv.gvar('0.04189(93)') #1907.01025
    Vud = gv.gvar('0.97370(14)') #PDG 
    stuff1 = [Vcsl]#,Vcssl]
    stuff2 = [Vusl,Vussl]
    cols = ['r','b']
    unitx = []
    unity = []
    #hatches = ['*','\\','x'] # patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    
    for i in range(len(stuff1)):
        x = stuff1[i]
        plt.fill_between([0.1,0.3],[x.mean-x.sdev,x.mean-x.sdev],[x.mean+x.sdev,x.mean+x.sdev], color=cols[i],alpha=alpha)
    for i in range(len(stuff2)):
        x = stuff2[i]
        plt.fill_between([x.mean-x.sdev,x.mean+x.sdev],[0.8,0.8],[1.05,1.05], color=cols[i],alpha=alpha)
    x = av
    matplotlib.rcParams['hatch.linewidth'] = 2
    plt.fill_between([0.1,0.3],[x.mean-x.sdev,x.mean-x.sdev],[x.mean+x.sdev,x.mean+x.sdev],linewidth=3, fc='none' ,edgecolor='b',hatch='x')
    for i in np.linspace(0.1,0.3,nopts):
        unitx.append(i)
        unity.append(gv.sqrt(1-i**2-Vts**2))
    unitymean,unitysdev = unmake_gvar_vec(unity)
    plt.plot([0.1,0.3],[Vud.mean,Vud.mean],color='g',lw=5)
    plt.plot(unitx, unitymean, color='k',linestyle='--',lw=5)
    plt.xlabel('$|V_{us}|$',fontsize=fontsizelab)
    plt.ylabel(r'$|V_{cs}|$',fontsize=fontsizelab)#r'$\eta^2_{\mathrm{EW}}(1+\delta_{\mathrm{EM}})|V_{cs}|^2$'
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.ylim([0.89,1.01])
    #plt.xlim([0.17,0.29])
    perc = 6
    meany  = (av.mean+Vcsl.mean)/2 # +Vcssl)/3
    meanx = (Vusl.mean+Vussl.mean)/2
    plt.ylim([meany*(1-perc/100),meany*(1+perc/100)])
    plt.xlim([meanx*(1-perc/100),meanx*(1+perc/100)])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.005))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.001))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.02))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    handles = [ Patch(facecolor='b', edgecolor='b',label=r'SL',alpha=alpha),Patch(facecolor='r', edgecolor='r',label=r'L',alpha=alpha),Line2D([0], [0], color='g', lw=5, label=r'$V_{ud}$'),Patch(facecolor='none', edgecolor='b',hatch='x',label=r"HPQCD '21"),Line2D([0], [0], color='k',linestyle='--', lw=5, label='Unitarity')]
    plt.legend(handles=handles,fontsize=fontsizeleg*0.9,frameon=False,ncol=2,loc='lower right')
    plt.tight_layout()
    plt.savefig('DKPlots/Vcsus.pdf')
    plt.close()

    plt.figure(figsize=figsize)
    Vcdl = gv.gvar(0.2173,np.sqrt(0.0051**2+0.0007**2)) #PDG
    Vcdsl = gv.gvar(0.2330,np.sqrt(0.0029**2+0.0133**2)) #PDG
    Vcb = gv.gvar('0.0410(14)') #PDG
    Vud = gv.gvar('0.97370(14)') #PDG 
    stuff1 = [Vcsl]#,Vcssl]
    stuff2 = [Vcdl,Vcdsl]
    cols = ['r','b']
    unitx = []
    unity = []
    #hatches = ['*','\\','x'] # patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    
    for i in range(len(stuff1)):
        x = stuff1[i]
        plt.fill_between([0.1,0.3],[x.mean-x.sdev,x.mean-x.sdev],[x.mean+x.sdev,x.mean+x.sdev], color=cols[i],alpha=alpha)
    for i in range(len(stuff2)):
        x = stuff2[i]
        plt.fill_between([x.mean-x.sdev,x.mean+x.sdev],[0.5,0.5],[1.5,1.5], color=cols[i],alpha=alpha)
    x = av
    matplotlib.rcParams['hatch.linewidth'] = 2
    plt.fill_between([0.1,0.3],[x.mean-x.sdev,x.mean-x.sdev],[x.mean+x.sdev,x.mean+x.sdev],linewidth=3, fc='none' ,edgecolor='b',hatch='x')
    for i in np.linspace(0.1,0.3,nopts):
        unitx.append(i)
        unity.append(gv.sqrt(1-i**2-Vcb**2))
    unitymean,unitysdev = unmake_gvar_vec(unity)
    plt.plot([0.1,0.3],[Vud.mean,Vud.mean],color='g',lw=5)
    plt.plot(unitx, unitymean, color='k',linestyle='--',lw=5)
    plt.xlabel('$|V_{cd}|$',fontsize=fontsizelab)
    plt.ylabel(r'$|V_{cs}|$',fontsize=fontsizelab)#r'$\eta^2_{\mathrm{EW}}(1+\delta_{\mathrm{EM}})|V_{cs}|^2$'
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.ylim([0.89,1.01])
    #plt.xlim([0.17,0.29])
    perc = 15
    meany  = (av.mean+Vcsl.mean)/2 # +Vcssl)/3
    meanx = (Vcdl.mean+Vcdsl.mean)/2
    plt.ylim([meany*(1-perc/100),meany*(1+perc/100)])
    plt.xlim([meanx*(1-perc/100),meanx*(1+perc/100)])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.01))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.005))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    handles = [ Patch(facecolor='b', edgecolor='b',label=r'SL',alpha=alpha),Patch(facecolor='r', edgecolor='r',label=r'L',alpha=alpha),Line2D([0], [0], color='g', lw=5, label=r'$V_{ud}$'),Patch(facecolor='none', edgecolor='b',hatch='x',label=r"HPQCD '21"),Line2D([0], [0], color='k',linestyle='--', lw=5, label='Unitarity')]
    plt.legend(handles=handles,fontsize=fontsizeleg*0.9,frameon=False,ncol=1,loc='lower left')
    plt.tight_layout()
    plt.savefig('DKPlots/Vcscd.pdf')
    plt.close()

############################################################################################
    ##compare exps
    plt.figure(figsize=figsize)
    #p = make_p_physical_point_DK(pfit,Fits)
    #f00,fp0 = make_spline_f0_fp_physpoint(p,Fits[0],0)
    
    CLEO0 =  gv.sqrt(d['Cleo1av']/cor2)
    CLEOp =  gv.sqrt(d['Cleo2av']/cor2)
    BES =   gv.sqrt(d['BESav']/cor2)
    BESDp =   gv.sqrt(d['BESDpav']/cor2)
    BABAR =  gv.sqrt(d['BaBarav']/cor2)
    #av2 = gv.sqrt(lsqfit.wavg([d['Cleo1av'],d['Cleo2av'],d['BESav'],d['BaBarav']])/cor2)
    plt.errorbar(BESDp.mean, 5, xerr=BESDp.sdev,ms=ms,fmt='ro',mfc='none' ,capsize=capsize,lw=lw)
    plt.plot([0.9275,1.1],[4.5,4.5],color='k',linestyle='--')
    plt.plot([0.8,0.8825],[4.5,4.5],color='k',linestyle='--')
    plt.errorbar(CLEOp.mean, 4, xerr=CLEOp.sdev,ms=ms,fmt='r*' ,capsize=capsize,lw=lw)
    plt.plot([0.8,1.1],[3.5,3.5],color='k')
    plt.fill_between([av.mean-av.sdev,av.mean+av.sdev],[0,0],[4.5,4.5], color='purple',alpha=alpha/2)
    #plt.fill_between([av2.mean-av2.sdev,av2.mean+av2.sdev],[0,0],[8,8], color='g',alpha=alpha/2)
    plt.errorbar(BES.mean, 3, xerr=BES.sdev,ms=ms,fmt='bo' ,capsize=capsize,lw=lw)
    plt.errorbar(CLEO0.mean, 2, xerr=CLEO0.sdev,ms=ms,fmt='b*' ,capsize=capsize,lw=lw)    
    plt.errorbar(BABAR.mean, 1, xerr=BABAR.sdev,ms=ms,fmt='b^' ,capsize=capsize,lw=lw)
    plt.text(0.905,2.0,r'$D^0\to{}K^-e^+\nu_e$',fontsize=fontsizelab, va='center',ha='center')
    plt.text(0.905,4.5,r'$D^+\to\bar{K}^0e^+\nu_e$',fontsize=fontsizelab, va='center',ha='center')
    plt.text(av.mean,1.5,r'Weighted av.',fontsize=fontsizelab,va='center',ha='center')
    plt.xlabel('$|V_{cs}|$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=False,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('none')
    plt.xlim([0.875,1.01])
    plt.ylim([0.5,5.5])
    plt.gca().set_yticks([5,4,3,2,1])
    plt.gca().set_yticklabels(["BES '17","CLEO '09","BES '15A","CLEO '09","BaBar '06"])#'arXiv:1706.03657','arXiv:1706.03017','arXiv:1305.1462','arXiv:1008.4562'])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.02))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.text(1.01,5.5,r'$\frac{d\Gamma}{dq^2}$',fontsize=fontsizelab*1.9,va='top',ha='right')
    plt.plot([0.998,1.1],[4.5,4.5],color='k')
    plt.plot([0.998,0.998],[4.5,5.5],color='k')
    plt.tight_layout()
    plt.savefig('DKPlots/Vcsexpcomp.pdf')
    plt.close()
############################################################################################
    ##compare exps f+(0)
    plt.figure(figsize=figsize)
    #p = make_p_physical_point_DK(pfit,Fits)
    #f00,fp0 = make_spline_f0_fp_physpoint(p,Fits[0],0)
    av = Vcsq20
    # all from HFLAV 19
    BELLE6 = 1/fp0 * gv.gvar(0.6762,np.sqrt(0.0068**2+0.0214**2))/gv.sqrt(cor2) #D0 0604049 
    BABAR6 = 1/fp0 * gv.gvar(0.7211,np.sqrt(0.0069**2+0.0085**2))/gv.sqrt(cor2) #D0 0704.0020
    CLEOC9 = 1/fp0 * gv.gvar(0.7189,np.sqrt(0.0064**2+0.0048**2))/gv.sqrt(cor2) #both combined 0906.2983
    BES15A = 1/fp0 * gv.gvar(0.7195,np.sqrt(0.0035**2+0.0041**2))/gv.sqrt(cor2) #D0 1508.07560
    BES19  = 1/fp0 * gv.gvar(0.7133,np.sqrt(0.0038**2+0.0030**2))/gv.sqrt(cor2) #D0 muon 1810.03127
    p = make_p_physical_point_DK(pfit,Fits,Dplus=True)
    fp0 = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],0,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
    BES15B = 1/fp0 * gv.gvar(0.7370,np.sqrt(0.0060**2+0.0090**2))/gv.sqrt(cor2) #Dp 1510.00308 
    BES17  = 1/fp0 * gv.gvar(0.6983,np.sqrt(0.0056**2+0.0112**2))/gv.sqrt(cor2) #Dp 1703.09084
    plt.errorbar(BES17.mean, 7, xerr=BES17.sdev,ms=ms,fmt='r*' ,capsize=capsize,lw=lw)
    plt.errorbar(BES15B.mean, 6, xerr=BES15B.sdev,ms=ms,fmt='r*' ,capsize=capsize,lw=lw)
    plt.plot([0.8,1.1],[5.5,5.5],color='k')
    plt.errorbar(BES19.mean, 5, xerr=BES19.sdev,ms=ms,fmt='g*' ,capsize=capsize,lw=lw)
    plt.plot([0.8,1.1],[4.5,4.5],color='k')
    plt.errorbar(BES15A.mean, 4, xerr=BES15A.sdev,ms=ms,fmt='b*' ,capsize=capsize,lw=lw)
    plt.errorbar(CLEOC9.mean, 3, xerr=CLEOC9.sdev,ms=ms,fmt='bo' ,capsize=capsize,lw=lw)    
    plt.errorbar(BABAR6.mean, 2, xerr=BABAR6.sdev,ms=ms,fmt='bs' ,capsize=capsize,lw=lw)
    plt.errorbar(BELLE6.mean, 1, xerr=BELLE6.sdev,ms=ms,fmt='b^' ,capsize=capsize,lw=lw)
    plt.fill_between([av.mean-av.sdev,av.mean+av.sdev],[0,0],[8,8], color='purple',alpha=alpha/2)
    plt.text(0.88,2.5,r'$D^0\to{}K^-e^+\nu_e$',fontsize=fontsizelab, va='center')
    plt.text(0.88,5,r'$D^0\to{}K^-\mu^+\nu_{\mu}$',fontsize=fontsizelab, va='center')
    plt.text(0.88,6.5,r'$D^+\to\bar{K}^0e^+\nu_e$',fontsize=fontsizelab, va='center')
    plt.text(av.mean,1.5,r'Weighted av.',fontsize=fontsizelab, va='center',ha='center')
    #plt.text(0.95,1.5,r'Wt. average',fontsize=fontsizelab)
    plt.xlabel('$|V_{cs}|$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=False,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('none')
    plt.xlim([0.875,1.01])
    plt.ylim([0.5,7.5])
    plt.gca().set_yticks([7,6,5,4,3,2,1])
    plt.gca().set_yticklabels(["BES '17","BES '15B","BES '19","BES '15A","CLEO '09","BaBar '06","BELLE '06"])#'arXiv:1706.03657','arXiv:1706.03017','arXiv:1305.1462','arXiv:1008.4562'])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.02))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.text(1.01,7.5,r'$|V_{cs}|f_+(0)$',fontsize=fontsizelab*1.25,va='top',ha='right')
    plt.plot([0.979,1.1],[6.7,6.7],color='k')
    plt.plot([0.979,0.979],[6.7,7.5],color='k')
    plt.tight_layout()
    plt.savefig('DKPlots/Vcsexpf0comp.pdf')
    plt.close()
    return()

###################################################################################################

def find_bin_centres(bins):
    centres = []
    for i in range(len(bins)-1):
        centre =  (bins[i+1]+bins[i])/2
        err = (bins[i+1]-bins[i])/2
        centres.append(gv.gvar(centre,err))
    return(centres)

###################################################################################################

def plot_BES_R_mu_e(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2):
    p = make_p_physical_point_DK(pfit,Fits)
    y = []
    qsq = []
    bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,qsqmaxphysDK.mean]
    Rs = gv.gvar(['0.796(27)','0.973(26)','0.959(26)','0.974(37)','0.979(29)','1.057(34)','0.990(31)','1.012(39)','1.060(34)','1.026(35)','0.963(36)','1.076(68)','1.004(47)','1.065(57)','1.008(64)','0.979(67)','0.940(86)','1.318(217)'])#only stat error
    A = 3/8 * p['MDphys']**2 * (1 - (p['MKphys']**2/p['MDphys']**2) )**2 
    def ratio(qsq):
        epse = M_e**2/qsq
        epsmu = M_mu**2/qsq
        p2 = (qsq-p['MKphys']**2-p['MDphys']**2)**2/(4*p['MDphys']**2)-p['MKphys']**2
        if p2.mean < 0:
            p2 = 0
        p3 = (p2)**(3/2)
        p1 = p2**(1/2)
        fp = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
        f0 = make_f0_BK(Nijk,Npow,Nm,addrho,p,Fits[0],qsq,t_0,Fits[0]['masses'][0],fpf0same,0)
        integrand = (1-epsmu)**2 * (p3 * fp**2 * (1 + epsmu/2) + p1 * f0**2 * epsmu * A) /((1-epse)**2 * (p3 * fp**2 * (1 + epse/2) + p1 * f0**2 * epse * A))
        if qsq == qsqmaxphysDK.mean:
            integrand = epsmu*(1-epsmu)**2/(epse*(1-epse)**2)
        return(integrand)
    
    for q2 in np.linspace((M_mu**2).mean,qsqmaxphysDK.mean-0.001,nopts):
        y.append(ratio(q2))
        qsq.append(q2)
    ymean,yerr = unmake_gvar_vec(y)
    Rmean,Rerr = unmake_gvar_vec(Rs)
    yupp,ylow = make_upp_low(y)
    binmean,binerr = unmake_gvar_vec(find_bin_centres(bins))  
    plt.figure(figsize=figsize)
    plt.plot(qsq, ymean, color='k',label=r'$R_{\mu/e}$')
    plt.fill_between(qsq,ylow,yupp, color='k',alpha=alpha)

    plt.errorbar(binmean,Rmean, xerr=binerr, yerr=Rerr,fmt='s',color='purple',ms=ms,mfc='none',label = r"BES '19",lw=lw)
    plt.legend(fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    #plt.ylabel(r'$f_+(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    #plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.ylim([0.7,1.6])
    plt.tight_layout()
    plt.savefig('DKPlots/Rmue.pdf')
    plt.close()
        
    return()
####################################################################################################

def partially_correlate_errors(data):
    # takes a list of uncorrelated data points where we wish to 100% correlate some of the error
    # each data point should take the form [mean,uncorr_err, corr_err] and will be returned as a list in the same order with    the error corr part correlated.
    #We must add the covarience matricies. The uncorrelated matrix is just the error^2 along the diagonal. The correlated one is    sigma_i * sigma_j * 1 in all cases because the correlation matrix is just 1 everywhere  
    result = []
    correrr = []
    uncorrerr = []
    for element in data:
        result.append(element[0])
        uncorrerr.append(element[1])
        correrr.append(element[2])
    dim = len(result)
    cov_mat = np.zeros(shape=(dim,dim))
    for i in range(dim):
        for j in range(dim):
            if i == j:
                cov_mat[i][j] = uncorrerr[i]*uncorrerr[j] + correrr[i]*correrr[j]
            else:
                cov_mat[i][j] = correrr[i]*correrr[j]
    resultcorr = gv.gvar(result,cov_mat)
    return(resultcorr)

####################################################################################################

def plot_Vcs_from_B(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2):
    p = make_p_physical_point_DK(pfit,Fits)
    integrale0 = integrate_fp(0,qsqmaxphysDK.mean,pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2,M_e,iters=500)
    integralep = integrate_fp(0,qsqmaxphysDK.mean,pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2,M_e,iters=500,Dplus=True)
    integralmu = integrate_fp(0,qsqmaxphysDK.mean,pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2,M_mu,iters=500)
    BbyVcsD0mu = GF**2/(24*np.pi**3*6.582119569*1e-16)*integralmu*tauD0
    BbyVcsD0e = GF**2/(24*np.pi**3*6.582119569*1e-16)*integrale0*tauD0
    BbyVcsDpme = GF**2/(24*np.pi**3*6.582119569*1e-16)*integralep*tauDpm
    plt.figure(figsize=figsize)
    # [mean, stat, syst]
    Cleo0 = [0.0350,0.0003,0.0004]#gv.gvar(0.0350,np.sqrt(0.0003**2+0.0004**2)) # 0906.2983 CLEO '09
    Cleop = [0.0883,0.0010,0.0020]#gv.gvar(0.0883,np.sqrt(0.0010**2+0.0020**2)) # 0906.2983 CLEO '09
    [Cleo0,Cleop] = partially_correlate_errors([Cleo0,Cleop])
    VCleo0 = gv.sqrt(Cleo0/(cor2*BbyVcsD0e))
    VCleop = gv.sqrt(Cleop/(cor2*BbyVcsDpme))
    
    BESe = [0.03505,0.00014,0.00033]#gv.gvar(0.03505,np.sqrt(0.00014**2+0.00033**2)) # 1508.07560 BES 15'A
    BESmu = [0.03413,0.00019,0.00035]#gv.gvar(0.03413,np.sqrt(0.00019**2+0.00035**2)) #1810.03127 BES '19
    BES17 = [0.0860,0.0006,0.0015]#gv.gvar(0.0860,np.sqrt(0.0006**2+0.0015**2))#Dp 1703.09084 BES '17
    [BESe,BESmu,BES17] = partially_correlate_errors([BESe,BESmu,BES17])
    VBESe = gv.sqrt(BESe/(cor2*BbyVcsD0e))
    VBESmu = gv.sqrt(BESmu/(cor2*BbyVcsD0mu))
    VBES17 = gv.sqrt(BES17/(cor2*BbyVcsDpme))

    BELLEe = [0.0345,0.0010,0.0019]#gv.gvar(0.0345,np.sqrt(0.0010**2+0.0019**2)) # hep-ex/0604049 Belle '06
    BELLEmu = [0.0345,0.0010,0.0021] #gv.gvar(0.0345,np.sqrt(0.0010**2+0.0021**2)) #hep-ex/0604049 Belle '06
    [BELLEe,BELLEmu] = partially_correlate_errors([BELLEe,BELLEmu])
    VBELLEe = gv.sqrt(BELLEe/(cor2*BbyVcsD0e))
    VBELLEmu = gv.sqrt(BELLEmu/(cor2*BbyVcsD0mu))
    
    BaBar = gv.gvar(0.927,np.sqrt(0.007**2+0.012**2)) * gv.gvar(0.03999,0.00044) #  0704.0020 BaBar '06
    VBaBar = gv.sqrt(BaBar/(cor2*BbyVcsD0e))

    
    #BES15B = gv.gvar(0.04481,np.sqrt(0.00027**2+0.00103**2))#Dp 1510.00308 K_L ?? BES '15B
    #VBES15B = gv.sqrt(BES15B/BbyVcsDpme)

    av = gv.sqrt(lsqfit.wavg([Cleo0/BbyVcsD0e,Cleop/BbyVcsDpme,BESe/BbyVcsD0e,BESmu/BbyVcsD0mu,BES17/BbyVcsDpme,BELLEe/BbyVcsD0e,BELLEmu/BbyVcsD0mu,BaBar/BbyVcsD0e])/cor2)
    terr = av.partialsdev(tuple([integrale0,integralep,integralmu]))
    eerr = av.partialsdev(tuple([Cleo0,Cleop,BESe,BESmu,BES17,BELLEe,BELLEmu,BaBar,tauD0,tauDpm]))
    cerr = av.partialsdev(cor2)
    print('C) Average V_cs from branching = ', av)
    print('Theory error = {0:.4f} Exp error = {1:.4f} Correction error = {2:.4f} Total = {3:.4f}'.format(terr,eerr,cerr,np.sqrt(eerr**2+terr**2+cerr**2)))
    plt.errorbar(VBES17.mean, 8, xerr=VBES17.sdev,ms=ms,fmt='r*' ,capsize=capsize,lw=lw)
    plt.errorbar(VCleop.mean, 7, xerr=VCleop.sdev,ms=ms,fmt='ro' ,capsize=capsize,lw=lw)
    plt.plot([0.8,1.1],[6.5,6.5],color='k')
    plt.errorbar(VBESmu.mean, 6, xerr=VBESmu.sdev,ms=ms,fmt='g*' ,capsize=capsize,lw=lw)
    plt.errorbar(VBELLEmu.mean, 5, xerr=VBELLEmu.sdev,ms=ms,fmt='g^' ,capsize=capsize,lw=lw)
    plt.plot([0.8,1.1],[4.5,4.5],color='k')
    plt.errorbar(VBESe.mean, 4, xerr=VBESe.sdev,ms=ms,fmt='b*' ,capsize=capsize,lw=lw)
    plt.errorbar(VCleo0.mean, 3, xerr=VCleo0.sdev,ms=ms,fmt='bo' ,capsize=capsize,lw=lw)
    plt.errorbar(VBaBar.mean, 2, xerr=VBaBar.sdev,ms=ms,fmt='bs' ,capsize=capsize,lw=lw)
    plt.errorbar(VBELLEe.mean, 1, xerr=VBELLEe.sdev,ms=ms,fmt='b^' ,capsize=capsize,lw=lw)
    #plt.errorbar(VBES15B.mean, 8, xerr=VBES15B.sdev,ms=ms,fmt='r*' ,capsize=capsize,lw=lw)
    plt.fill_between([av.mean-av.sdev,av.mean+av.sdev],[0,0],[9,9], color='purple',alpha=alpha/2)
    plt.text(0.89,2.5,r'$D^0\to{}K^-e^+\nu_e$',fontsize=fontsizelab, va='center')
    plt.text(0.89,5.5,r'$D^0\to{}K^-\mu^+\nu_{\mu}$',fontsize=fontsizelab, va='center')
    plt.text(0.89,7.5,r'$D^+\to\bar{K}^0e^+\nu_e$',fontsize=fontsizelab, va='center')
    plt.text(av.mean,1.5,r'Weighted av.',fontsize=fontsizelab, va='center',ha='center')
    plt.xlabel('$|V_{cs}|$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=False,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('none')
    plt.xlim([0.875,1.01])
    plt.ylim([0.5,8.5])
    plt.gca().set_yticks([8,7,6,5,4,3,2,1])
    plt.gca().set_yticklabels(["BES '17","CLEO '09","BES '19","BELLE '06","BES '15A","CLEO '09","BaBar '06","BELLE '06"])
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.02))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.text(1.01,8.5,r'$\Gamma$',fontsize=fontsizelab*2,va='top',ha='right')
    plt.plot([1,1.1],[7.5,7.5],color='k')
    plt.plot([1,1],[7.5,8.5],color='k')
    plt.tight_layout()
    plt.savefig('DKPlots/VcsfromB.pdf')
    plt.close()
    return(av)

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
    plt.savefig('DKPlots/HtH0.pdf')
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
    plt.savefig('DKPlots/fitvsrefit.pdf')
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
    plt.savefig('DKPlots/f0fpluserr.pdf')
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
###################################################################################################
































































































































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
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label'])))
            
            plt.figure(5,figsize=figsize)
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label'])))
            
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
    plt.savefig('DKPlots/fTpoleinqsq.pdf')
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
    plt.savefig('DKPlots/fTpoleinz.pdf')
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
    plt.savefig('DKPlots/f0fpfTinqsq.pdf')
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
    plt.savefig('DKPlots/f0f0fpinmh.pdf')
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
    plt.savefig('DKPlots/betadeltainmh.pdf')
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
    plt.savefig('DKPlots/HQETrat.pdf')
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
    plt.savefig('DKPlots/HillratinE.pdf')
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
    plt.savefig('DKPlots/HillratinlowE.pdf')
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
    plt.savefig('DKPlots/Hillratinmh_E{0}.pdf'.format(E))
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
    plt.savefig('DKPlots/Hillratininvmh_E{0}.pdf'.format(E))
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
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label'])))
            
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
    plt.savefig('DKPlots/f0poleinza{0}.pdf'.format(afm))
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
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0}'.format(Fit['label'])))
            
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
    plt.savefig('DKPlots/fppoleinza{0}.pdf'.format(afm))
    plt.close()
    return()



##################################################################################################


