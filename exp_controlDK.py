import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from exp_functionsDK import *
from exp_plottingDK import *
plt.rc("font",**{"size":18})
import collections
import copy
import os.path
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
########################### B to K ######################################################
VCp = collections.OrderedDict()
VCp['conf']='VCp'
VCp['label'] = 'Set 1'
VCp['filename'] = 'Corrfits/VCprelabelled_relabelled_all_VCp_998cfg0.860502.0133.053.969BGBNGKGSV9121518unchained_Nexp6_sfac1.0_pfac1.0_Q1.00_chi0.797_smTrue_Stmin0_Vtmin0.pickle'
VCp['masses'] = ['0.8605']# note m eta_c for slightly wrong ensemble 0.863 = 
VCp['Zdisc'] = [0.99197] 
VCp['twists'] = ['0','2.013','3.05','3.969']
VCp['m_l'] = '0.00235' #ampi = 0.101720(40)
VCp['m_s'] = '0.0678'  
VCp['m_c'] = '0.8605' 
VCp['m_ssea'] = 0.0647
VCp['m_lsea'] = 0.00235
VCp['L'] = 32
VCp['w0/a'] = gv.gvar('1.1367(5)')
VCp['parent-Tag'] = '2pt_D_Gold_vc.ll'
VCp['daughter-Tag'] = 5*['2pt_K_vc_tw{0}.ll']
#########################################################################################
Cp = collections.OrderedDict()
Cp['conf']='Cp'
Cp['label'] = 'Set 2'
Cp['filename'] = 'Corrfits/Cprelabelled_relabelled_DK-phys-coarse-alldata_allqsq-binned-985config0.64302.4053.6414.735BGBNGKGSV12151821unchained_Nexp6_sfac1.0_pfac1.0_Q1.00_chi0.361_smTrue_Stmin0_Vtmin0.pickle'
Cp['masses'] = ['0.643']# note m eta_c is not corect for this
Cp['Zdisc'] = [0.99718]
Cp['twists'] = ['0','2.405','3.641','4.735']
Cp['m_l'] = '0.00184'
Cp['m_s'] = '0.0527'
Cp['m_c'] = '0.643'
Cp['m_ssea'] = 0.0507
Cp['m_lsea'] = 0.00184
Cp['L'] = 48
Cp['w0/a'] = gv.gvar('1.4149(6)')
Cp['parent-Tag'] = '2pt_D_Gold_coarse.ll'
Cp['daughter-Tag'] = 5*['2pt_K_coarse_tw{0}.ll'] 

#########################################################################
Fp = collections.OrderedDict()
Fp['conf']='Fp'
Fp['label'] = 'Set 3'
Fp['filename'] = 'Corrfits/Fprelabelled_binned_584_confs_G5G5_freshfullwall0.43202.3153.5074.563BGBNGKGSV16192225unchained_Nexp6_sfac1.0_pfac1.0_Q1.00_chi0.202_smTrue_Stmin0_Vtmin0.pickle'
Fp['masses'] = ['0.432']
Fp['Zdisc'] = [0.99938]
Fp['twists'] = ['0','2.315','3.507','4.563']
Fp['m_l'] = '0.0012'
Fp['m_s'] = '0.036'
Fp['m_c'] = '0.432'
Fp['m_ssea'] = 0.0363
Fp['m_lsea'] = 0.0012
#F['tp'] = 96
Fp['L'] = 64
Fp['w0/a'] = gv.gvar('1.9518(7)')
Fp['parent-Tag'] = 'D_G5-G5_m0.432'
Fp['daughter-Tag'] = 5*['K_G5-G5_tw{0}']
#######################################VC PARAMETERS ####################
VC = collections.OrderedDict()
VC['conf']='VC'
VC['label'] = 'Set 4'
VC['filename'] = 'Corrfits/VCtest-KDscalarvectortensor_1020cfgs_neg0.88800.36651.0971.828BGBNGKGKNGSTV9121518unchained_Nexp5_sfac1.0_pfac1.0_Q1.00_chi0.399_smTrue_Stmin1_Ttmin1_Vtmin1.pickle'
VC['masses'] = ['0.888']# note m eta_c for slightly wrong ensemble 0.863 = 
VC['Zdisc'] = [0.99105]  
VC['twists'] = ['0','0.3665','1.097','1.828']
VC['m_l'] = '0.013' #ampi = 0.101720(40)
VC['m_s'] = '0.0705' #these are valence  
VC['m_c'] = '0.888' 
VC['m_ssea'] = 0.065
VC['m_lsea'] = 0.013
VC['L'] = 16
VC['w0/a'] = gv.gvar('1.1119(10)')
VC['parent-Tag'] = 'D_G5-G5_m{1}'
VC['daughter-Tag'] = 5*['K_G5-G5_tw{0}']

#################################### C PARAMETERS #####################
C = collections.OrderedDict()
C['conf']='C'
C['label'] = 'Set 5'
C['filename'] = 'Corrfits/Cthreemass-KDscalarvectortensor_1053cfgs_neg0.66400.4411.3232.2052.646BGBNGKGSV12151821unchained_Nexp5_sfac1.0_pfac1.0_Q1.00_chi0.303_smTrue_Stmin1_Vtmin1.pickle'
C['masses'] = ['0.664','0.8','0.9']# note m eta_c is not corect for this
C['Zdisc'] = [0.99683,0.99377,0.99063]
C['twists'] = ['0','0.441','1.323','2.205','2.646']
C['m_l'] = '0.0102'
C['m_s'] = '0.0545'
C['m_c'] = '0.664'
C['m_ssea'] = 0.0509
C['m_lsea'] = 0.0102
C['L'] = 24
C['w0/a'] = gv.gvar('1.3826(11)')
C['parent-Tag'] = 'D_G5-G5_m{1}'
C['daughter-Tag'] = 5*['K_G5-G5_tw{0}']
################################## F PARAMETERS ##########################
F = collections.OrderedDict()
F['conf']='F'
F['label'] = 'Set 6'
F['filename'] = 'Corrfits/Ftest-KBscalarvectortensor_499cfgs_neg0.44900.42811.2822.1412.570BGBNGKGSV141720unchained_Nexp6_sfac1.0_pfac1.0_Q1.00_chi0.286_smTrue_Stmin2_Vtmin2.pickle'
F['masses'] = ['0.449','0.566','0.683','0.8']
F['Zdisc'] = [0.99892,0.99826,0.99648,0.99377]
F['twists'] = ['0','0.4281','1.282','2.141','2.570']
F['m_l'] = '0.0074'
F['m_s'] = '0.0376'
F['m_c'] = '0.449'
F['m_ssea'] = 0.037
F['m_lsea'] = 0.0074
#F['tp'] = 96
F['L'] = 32
F['w0/a'] = gv.gvar('1.9006(20)')
F['parent-Tag'] = 'B_G5-G5_m{1}'
F['daughter-Tag'] = 5*['K_G5-G5_tw{0}'] 

######################## SF PARAMETERS ####################################
SF = collections.OrderedDict()
SF['conf']='SF'
SF['label'] = 'Set 7'
SF['filename'] = 'Corrfits/SFnohimem-KBscalarvectortensor_415cfgs_negscalarvector0.27401.2612.1082.9463.624BGBNGKGSV202530unchained_Nexp6_sfac1.0_pfac1.0_Q1.00_chi0.180_smTrue_Stmin2_Vtmin2.pickle' 
SF['masses'] = ['0.274','0.45','0.6','0.8']
SF['Zdisc'] = [0.99990,0.99928,0.99783,0.99377]
SF['twists'] = ['0','1.261','2.108','2.946','3.624']
SF['m_l'] = '0.0048'
SF['m_s'] = '0.0234'
SF['m_c'] = '0.274'
SF['m_ssea'] = 0.024
SF['m_lsea'] = 0.0048
#SF['tp'] = 144
SF['L'] = 48
SF['w0/a'] = gv.gvar('2.896(6)')
SF['parent-Tag'] = 'B_G5-G5_m{1}'
SF['daughter-Tag'] = 5*['K_G5-G5_tw{0}']


######################## UF PARAMETERS ####################################
UF = collections.OrderedDict()
UF['conf']='UF'
UF['label'] = 'Set 8'
UF['filename'] = '../Fits/UF5_3pts_Q1.00_Nexp2_NMarg6_Stmin2_Vtmin2_svd0.01000_chi0.047_pl1.0_svdfac1.0'
#UF['Hsfilename'] = UFs['Hsfilename']
#UF['Hsparent-Tag'] = 'Bs_G5-G5_m{1}'
UF['masses'] = ['0.194','0.45','0.6','0.8']
UF['Zdisc'] = [0.99997,0.99928,0.99783,0.99377]
UF['twists'] = ['0','0.706','1.529','2.235','4.705']
UF['m_s'] = '0.0165'
UF['m_c'] = '0.194'
UF['m_ssea'] = 0.0158
UF['m_lsea'] = 0.00316
#UF['tp'] = 192
UF['L'] = 64
UF['w0/a'] = gv.gvar('3.892(12)')
UF['parent-Tag'] = 'Bs_G5-G5_m{1}'
UF['daughter-Tag'] = ['etas_G5-G5_tw0','etas_G5-G5_tw0.706','etas_G5-G5_tw1.529','etas_G5-G5_tw2.235','etas_G5-G5_tw4.705']
######################### BsEtas ########################################
################################## F PARAMETERS ##########################
Fs = collections.OrderedDict()
Fs['conf']='Fs'
Fs['label'] = 'Set 9'
Fs['filename'] = '../../H_sToEta_s/Analysis/Fits/F5_3pts_Q1.00_Nexp2_NMarg5_Stmin2_Vtmin1_svd0.00157_chi0.342_pl1.0_svdfac1.0'
Fs['Hlfilename'] = F['filename']  # this is to get the H mass for t_plus etc
Fs['Hltag'] = F['parent-Tag']
Fs['ldaughtertag'] = F['daughter-Tag']
Fs['masses'] = ['0.449','0.566','0.683','0.8']
Fs['Zdisc'] = [0.99892,0.99826,0.99648,0.99377]
Fs['twists'] = ['0','0.4281','1.282','2.141','2.570']
Fs['m_l'] = '0.0376'
Fs['m_s'] = '0.0376'
Fs['m_c'] = '0.449'
Fs['m_ssea'] = 0.037
Fs['m_lsea'] = 0.0074
#F['tp'] = 96
Fs['L'] = 32
Fs['w0/a'] = gv.gvar('1.9006(20)')
Fs['parent-Tag'] = 'meson.m{0}_m{1}'
Fs['daughter-Tag'] = ['etas','etas_p0.0728','etas_p0.218','etas_p0.364','etas_p0.437','etas_p0.509']      

######################## SFs PARAMETERS ####################################
SFs = collections.OrderedDict()
SFs['conf']='SFs'
SFs['label'] = 'Set 10'
SFs['filename'] = '../../H_sToEta_s/Analysis/Fits/SF5_3pts_Q1.00_Nexp3_NMarg6_Stmin2_Vtmin2_svd0.00457_chi0.079_pl1.0_svdfac1.0'
SFs['Hlfilename'] = SF['filename']
SFs['Hltag'] = SF['parent-Tag']
SFs['ldaughtertag'] = SF['daughter-Tag']
SFs['masses'] = ['0.274','0.450','0.6','0.8']
SFs['Zdisc'] = [0.99990,0.99928,0.99783,0.99377]
SFs['twists'] = ['0','1.261','2.108','2.946','3.624']
SFs['m_l'] = '0.0234'  #treating strange as light here
SFs['m_s'] = '0.0234'
SFs['m_c'] = '0.274'
SFs['m_ssea'] = 0.024
SFs['m_lsea'] = 0.0048
#SF['tp'] = 144
SFs['L'] = 48
SFs['w0/a'] = gv.gvar('2.896(6)')
SFs['parent-Tag'] = 'meson.m{0}_m{1}' 
SFs['daughter-Tag'] = ['etas_p0','etas_p0.143','eta_s_tw2.108_m0.0234','etas_p0.334','eta_s_tw3.624_m0.0234']


######################## UFs PARAMETERS ####################################
UFs = collections.OrderedDict()
UFs['conf']='UFs'
UFs['label'] = 'Set 11'
UFs['filename'] = '../../H_sToEta_s/Analysis/Fits/UF5_3pts_Q1.00_Nexp2_NMarg6_Stmin2_Vtmin2_svd0.01000_chi0.047_pl1.0_svdfac1.0'
UFs['Hlfilename'] = UF['filename']
UFs['Hltag'] = UF['parent-Tag']
UFs['ldaughtertag'] = UF['daughter-Tag']
UFs['masses'] = ['0.194','0.45','0.6','0.8']
UFs['Zdisc'] = [0.99997,0.99928,0.99783,0.99377]
UFs['twists'] = ['0','0.706','1.529','2.235','4.705']
UFs['m_l'] = '0.0165'
UFs['m_s'] = '0.0165'
UFs['m_c'] = '0.194'
UFs['m_ssea'] = 0.0158
UFs['m_lsea'] = 0.00316
#UF['tp'] = 192
UFs['L'] = 64
UFs['w0/a'] = gv.gvar('3.892(12)')
UFs['parent-Tag'] = 'Bs_G5-G5_m{1}'
UFs['daughter-Tag'] = ['etas_G5-G5_tw0','etas_G5-G5_tw0.706','etas_G5-G5_tw1.529','etas_G5-G5_tw2.235','etas_G5-G5_tw4.705']


##################### USER INPUTS ##########################################
Masses = collections.OrderedDict()
Twists = collections.OrderedDict()
thpts = collections.OrderedDict()
############################################################################

Fits = [VCp,Cp,Fp,VC,C,F,SF]#,Fs,SFs]#,UFs]#,UF] # choose what to fit
Masses['VCp'] = [0]                                     # Choose which masses to fit
Twists['VCp'] = [0,1,2,3]
thpts['VCp'] = ['S','V']
Masses['Cp'] = [0]                                     # Choose which masses to fit
Twists['Cp'] = [0,1,2,3] 
thpts['Cp'] = ['S','V']
Masses['Fp'] = [0]                                     # Choose which masses to fit
Twists['Fp'] = [0,1,2,3]
thpts['Fp'] = ['S','V']
Masses['VC'] = [0]                                     # Choose which masses to fit
Twists['VC'] = [0,1,2,3]
thpts['VC'] = ['S','V']
Masses['C'] = [0]                                     # Choose which masses to fit
Twists['C'] = [0,1,2,3,4] 
thpts['C'] = ['S','V']
Masses['F'] = [0]                                     # Choose which masses to fit
Twists['F'] = [0,1,2,3,4]
thpts['F'] = ['S','V']
Masses['SF'] = [0]
Twists['SF'] = [0,1,2,3,4]
thpts['SF'] = ['S','V']
Masses['UF'] = [0]
Twists['UF'] = [0,1,2,3,4]
thpts['UF'] = ['S','V']
Masses['Fs'] = [0]                                     # Choose which masses to fit
Twists['Fs'] = [0,1,2,3,4]
thpts['Fs'] = ['S','V']
Masses['SFs'] = [0]
Twists['SFs'] = [0,1,2,3,4]
thpts['SFs'] = ['S','V']
Masses['UFs'] = [0]
Twists['UFs'] = [0,1,2,3,4]
thpts['UFs'] = ['S','V']
addrho = False #like this for DK
fpf0same = False
constraint = False#True #add constraint the f0(0)=fp(0)
constraint2 = True
svdnoise = False
priornoise = False
FitNegQsq = True
dpri = '0.0(1.0)'#1.0
d000npri = '0.0(4.0)'# backbone of a without disc effects
di000pri = '0.0(1.0)'#1.0
di10npri = '0.0(1.0)' # 0.5 as expect to be smaller
cpri = '0.0(0.5)' #0.5
cvalpri ='0.0(1.0)'#1.0
rhopri ='0.0(1.0)'#1.0
DoFit = True
Npow = 3 #3
Nijk = 3 #3
Nm=3
SHOWPLOTS = False
t_0 = 'min' # for z conversion can be '0','rev','min' rev gives t_-
adddata = False #include data in continuum from other papers currently only for f0 Bsetas max
############################################################################
if t_0 != '0':
    print('t_0 != 0, so fpf0same set to False')
    fpf0same = False
############################################################################
fs_data = collections.OrderedDict() #fs from data fs_data[Fit][]

make_params_BK(Fits,Masses,Twists) #change to BK in BK case
Z_T = make_Z_T() # Tensor normalisation
for Fit in Fits:
    fs_data[Fit['conf']] = collections.OrderedDict()
    get_results(Fit,thpts)
    make_fs(Fit,fs_data[Fit['conf']],thpts,Z_T)
    results_tables(fs_data[Fit['conf']],Fit)
#check_poles(Fits) Doesn't work atm 
Z_V_plots(Fits,fs_data)
prior,f = make_prior_BK(fs_data,Fits,addrho,t_0,Npow,Nijk,Nm,rhopri,dpri,cpri,cvalpri,d000npri,di000pri,di10npri,adddata,constraint)

pfit = do_fit_BK(fs_data,adddata,Fits,f,Nijk,Npow,Nm,t_0,addrho,svdnoise,priornoise,prior,fpf0same,rhopri,dpri,cpri,cvalpri,d000npri,di000pri,di10npri,constraint,constraint2)
#print values
fs_at_lims_DK(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#re_fit_fp(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,svdnoise,priornoise,constraint2)
#Now to plot whatever we like, we only need the fit output, pfit, the fs from the data fs_data and Fit

#comp_cleo(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,constraint2)
#comp_BES(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,constraint2)
#comp_BaBar(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,constraint2)
#plot_poles(t_0)
#speed_of_light(Fits)
#f0_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata)
#fp_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,constraint2)
#f0_no_pole_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata)
#fp_no_pole_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,constraint2)
#fT_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,Del,addrho,fpf0same,adddata)

#f0_fp_fT_in_qsq(pfit,Fits,t_0,Nijk,Npow,Nm,Del,addrho,fpf0same)
#f0_f0_fp_in_Mh(pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same)
#beta_delta_in_Mh(pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same)
#HQET_ratio_in_qsq(pfit,Fits,Del,Nijk,Npow,addrho,fpf0same,t_0)
#Hill_ratios_in_E(pfit,Fits,Del,t_0,Nijk,Npow,addrho,fpf0same)
#Hill_ratios_in_mh(pfit,Fits,Del,t_0,Nijk,Npow,addrho,fpf0same)
#Hill_ratios_in_inv_mh(pfit,Fits,Del,t_0,Nijk,Npow,addrho,fpf0same)
#Hill_ratios_in_lowE(pfit,Fits,Del,t_0,Nijk,Npow,addrho,fpf0same)
#eval_at_different_spacings_BsEtas([0,0.09,0.12],pfit,Fits,Del,fpf0same,Npow,Nijk,addrho)
#f0_different_a_in_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same,0)
#fp_different_a_in_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same,0)
#f0_different_a_in_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same,0.06)
#fp_different_a_in_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same,0.06)
#f0_different_a_in_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same,0.09)
#fp_different_a_in_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same,0.09)
#f0_different_a_in_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same,0.12)
#fp_different_a_in_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same,0.12)
#error_plot(pfit,prior,Fits,Nijk,Npow,f,t_0,Del,addrho,fpf0same)
#table_of_as(Fits,pfit,Nijk,Npow,fpf0same,addrho,Del)
