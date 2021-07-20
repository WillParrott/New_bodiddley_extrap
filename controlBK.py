import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from functionsBK import *
from plottingBK import *
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
VCp['filename'] = 'Corrfits/VCpVCp_998cfg0.860502.0133.053.969BGBNGKGSV9121518unchained_Nexp4_sfac1.0_pfac1.0_Q1.00_chi0.506_smTrue_Stmin2_Vtmin2.pickle'
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
VCp['parent-Tag'] = 'D_G5-G5_m{1}'
VCp['daughter-Tag'] = 5*['K_G5-G5_tw{0}']
#########################################################################################
Cp = collections.OrderedDict()
Cp['conf']='Cp'
Cp['label'] = 'Set 2'
Cp['filename'] = 'Corrfits/CpCp-binned-985_cfg0.64302.4053.6414.735BGBNGKGSV12151821unchained_Nexp4_sfac1.0_pfac1.0_Q1.00_chi0.365_smTrue_Stmin1_Vtmin1.pickle'
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
Cp['parent-Tag'] = 'D_G5-G5_m{1}'
Cp['daughter-Tag'] = 5*['K_G5-G5_tw{0}']

#########################################################################
Fp = collections.OrderedDict()
Fp['conf']='Fp'
Fp['label'] = 'Set 3'
Fp['filename'] = 'Corrfits/Fpall_streams_620cfgs0.4330.6830.800.85632.9985.140BGBNGKGKNGSTV141720unchained_Nexp4_sfac1.0_pfac1.0_Q1.00_chi0.197_smTrue_Stmin2_Ttmin2_Vtmin2.pickle'
#Fp['filename'] = 'Corrfits/Fpall_streams_620cfgs0.4330.6830.800.85632.9985.140BGBNGKGKNGSTV141720sep_mass_Nexp4_sfac1.0_pfac1.0_Q1.00_chi0.272_smTrue_Stmin2_Ttmin2_Vtmin2.pickle'
Fp['masses'] = ['0.433','0.683','0.8']
Fp['Zdisc'] = [0.99938,0.99648,0.99377]
Fp['twists'] = ['0','0.8563','2.998','5.140']
Fp['m_l'] = '0.0012'
Fp['m_s'] = '0.036'
Fp['m_c'] = '0.433'
Fp['m_ssea'] = 0.0363
Fp['m_lsea'] = 0.0012
#F['tp'] = 96
Fp['L'] = 64
Fp['w0/a'] = gv.gvar('1.9518(7)')
Fp['parent-Tag'] = 'B_G5-G5_m{1}'
Fp['daughter-Tag'] = 5*['K_G5-G5_tw{0}']
################################################
VC = collections.OrderedDict()
VC['conf']='VC'
VC['label'] = 'Set 4'
VC['filename'] = 'Corrfits/VCtest-KDscalarvectortensor_1020cfgs_neg0.88800.36651.0971.828BGBNGKGKNGSTV9121518unchained_Nexp4_sfac1.0_pfac1.0_Q1.00_chi0.419_smTrue_Stmin2_Ttmin2_Vtmin2.pickle'
VC['masses'] = ['0.888']# note m eta_c for slightly wrong ensemble 0.863 = 
VC['Zdisc'] = [0.9905] 
VC['twists'] =  ['0','0.3665','1.097','1.828']
VC['m_l'] = '0.013' #ampi = 0.101720(40)
VC['m_s'] = '0.0705'  
VC['m_c'] = '0.888' 
VC['m_ssea'] = 0.065
VC['m_lsea'] = 0.013
VC['L'] = 16
VC['w0/a'] = gv.gvar('1.1119(10)')
VC['parent-Tag'] = 'D_G5-G5_m{1}'
VC['daughter-Tag'] = 5*['K_G5-G5_tw{0}']
#########################################################################################
C = collections.OrderedDict()
C['conf']='C'
C['label'] = 'Set 5'
C['filename'] = 'Corrfits/Cthreemass-KDscalarvectortensor_1053cfgs_neg0.6640.80.900.4411.3232.2052.646BGBNGKGKNGSTV12151821unchained_Nexp4_sfac1.0_pfac1.0_Q1.00_chi0.163_smTrue_Stmin1_Ttmin1_Vtmin1.pickle'
#C['filename'] = 'Corrfits/Cthreemass-KDscalarvectortensor_1053cfgs_neg0.6640.80.900.4411.3232.2052.646BGBNGKGKNGSTV12151821sep_mass_Nexp4_sfac1.0_pfac1.0_Q1.00_chi0.274_smTrue_Stmin1_Ttmin1_Vtmin1.pickle'
C['masses'] = ['0.664','0.8','0.9']# note m eta_c is not corect for this
C['Zdisc'] = [0.99683,0.99377,0.99063]
C['twists'] = ['0','0.441','1.323','2.205','2.646'] 
C['m_l'] = '0.0102'
C['m_s'] = '0.0545'
C['m_c'] = '0.664'
C['m_ssea'] = 0.0509
C['m_lsea'] = 0.00102
C['L'] = 24
C['w0/a'] = gv.gvar('1.3826(11)')
C['parent-Tag'] = 'D_G5-G5_m{1}'
C['daughter-Tag'] = 5*['K_G5-G5_tw{0}'] 

################################## F PARAMETERS ##########################
F = collections.OrderedDict()
F['conf']='F'
F['label'] = 'Set 6'
F['filename'] ='Corrfits/Ff5BsandB_499cfgs0.4490.5660.6830.800.42811.2822.1412.570BGBNGBsGBsNGKGKNGSSsTVVsetasG141720sep_mass_Nexp4_sfac1.0_pfac1.0_Q1.00_chi0.181_smTrue_Stmin2_Sstmin2_Ttmin2_Vtmin2_Vstmin2.pickle'
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
#SF['filename'] = 'Corrfits/SFBsandB_413cfg0.2740.450.60.801.2612.1082.9463.624BGBNGBsGBsNGKGKNGSSsTVVsetasG202530sep_mass_Nexp4_sfac0.8_pfac1.0_Q1.00_chi0.129_smTrue_Stmin2_Sstmin2_Ttmin2_Vtmin2_Vstmin2.pickle'
SF['filename'] = 'Corrfits/SFBsandBVx_413cfg0.2740.450.60.801.2612.1082.9463.624BGBNGBsGBsNGKGKNGSSsTVVsXetasG202530sep_mass_Nexp4_sfac0.8_pfac1.0_Q1.00_chi0.129_smTrue_Stmin2_Sstmin2_Ttmin2_Vtmin2_Vstmin2_Xtmin2.pickle'
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
#UF['filename'] = 'Corrfits/UFunbinned-run-KBscalarvectortensor_375cfgs_neg0.1940.450.60.800.7061.5292.2354.705BGBNGKGKNGSTV243340sep_mass_Nexp5_sfac1.0_pfac1.0_Q1.00_chi0.143_smTrue_Stmin2_Ttmin2_Vtmin2.pickle'
UF['filename'] = 'Corrfits/UFnewVx_unbinned0.1940.450.60.800.7061.5292.2354.705BGBNGKGKNGSTVX243340sep_mass_Nexp5_sfac1.0_pfac1.0_Q1.00_chi0.140_smTrue_Stmin2_Ttmin2_Vtmin2_Xtmin2.pickle'
UF['masses'] = ['0.194','0.45','0.6','0.8']
UF['Zdisc'] = [0.99997,0.99928,0.99783,0.99377]
UF['twists'] = ['0','0.706','1.529','2.235','4.705']
UF['m_l'] = '0.00316'
UF['m_s'] = '0.0165'
UF['m_c'] = '0.194'
UF['m_ssea'] = 0.0158
UF['m_lsea'] = 0.00316
#UF['tp'] = 192
UF['L'] = 64
UF['w0/a'] = gv.gvar('3.892(12)')
UF['parent-Tag'] = 'B_G5-G5_m{1}'
UF['daughter-Tag'] = 5*['K_G5-G5_tw{0}']
#UF['XVnn_m0.45_tw0.706'] = gv.gvar('0.159(12)')
#UF['XVnn_m0.45_tw1.529'] = gv.gvar('0.231(21)')
#UF['XVnn_m0.6_tw0.706'] = gv.gvar('0.166(14)')
#UF['XVnn_m0.6_tw1.529'] = gv.gvar('0.241(24)')
#UF['XVnn_m0.8_tw0.706'] = gv.gvar('0.173(15)')
#UF['XVnn_m0.8_tw1.529'] = gv.gvar('0.252(26)')
######################### BsEtas ########################################
################################## F PARAMETERS ##########################
Fs = collections.OrderedDict()
Fs['conf']='Fs'
Fs['label'] = 'Set 9'
#Fs['filename'] = '../../H_sToEta_s/Analysis/Fits/F5_3pts_Q1.00_Nexp2_NMarg5_Stmin2_Vtmin1_svd0.00157_chi0.342_pl1.0_svdfac1.0'
#Fs['filename'] = 'Corrfits/F5_3pts_Q1.00_Nexp2_NMarg5_Stmin2_Vtmin1_svd0.00157_chi0.342_pl1.0_svdfac1.0'
#Fs['Hlfilename'] = F['filename']  # this is to get the H mass for t_plus etc
#Fs['Hltag'] = F['parent-Tag']
#Fs['ldaughtertag'] = F['daughter-Tag']
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
Fs['w0/a'] = F['w0/a']#gv.gvar('1.9006(20)')
Fs['parent-Tag'] = 'Bs_G5-G5_m{1}'
Fs['daughter-Tag'] = 5*['etas_G5-G5_tw{0}']
#Fs['parent-Tag'] = 'meson.m{0}_m{1}'
#Fs['daughter-Tag'] = ['etas','etas_p0.0728','etas_p0.218','etas_p0.364','etas_p0.437','etas_p0.509']      

######################## SFs PARAMETERS ####################################
SFs = collections.OrderedDict()
SFs['conf']='SFs'
SFs['label'] = 'Set 10'
#SFs['filename'] = '../../H_sToEta_s/Analysis/Fits/SF5_3pts_Q1.00_Nexp3_NMarg6_Stmin2_Vtmin2_svd0.00457_chi0.079_pl1.0_svdfac1.0'
#SFs['filename'] = 'Corrfits/relabelled-SF5_3pts_Q1.00_Nexp3_NMarg6_Stmin2_Vtmin2_svd0.00457_chi0.079_pl1.0_svdfac1.0'
#SFs['Hlfilename'] = SF['filename']
#SFs['Hltag'] = SF['parent-Tag']
#SFs['ldaughtertag'] = SF['daughter-Tag']
SFs['masses'] = ['0.274','0.45','0.6','0.8']
SFs['Zdisc'] = [0.99990,0.99928,0.99783,0.99377]
SFs['twists'] = ['0','1.261','2.108','2.946','3.624']
SFs['m_l'] = '0.0234'  #treating strange as light here
SFs['m_s'] = '0.0234'
SFs['m_c'] = '0.274'
SFs['m_ssea'] = 0.024
SFs['m_lsea'] = 0.0048
#SF['tp'] = 144
SFs['L'] = 48
SFs['w0/a'] = SF['w0/a']#gv.gvar('2.896(6)')
SFs['parent-Tag'] = 'Bs_G5-G5_m{1}' 
SFs['daughter-Tag'] = 5*['etas_G5-G5_tw{0}'] #['etas_p0','etas_p0.143','eta_s_tw2.108_m0.0234','etas_p0.334','eta_s_tw3.624_m0.0234']


######################## UFs PARAMETERS ####################################
UFs = collections.OrderedDict()
UFs['conf']='UFs'
UFs['label'] = 'Set 11'
UFs['filename'] = 'Corrfits/UF5_3pts_Q1.00_Nexp2_NMarg6_Stmin2_Vtmin2_svd0.01000_chi0.047_pl1.0_svdfac1.0'
#UFs['Hlfilename'] = UF['filename']
#UFs['Hltag'] = UF['parent-Tag']
#UFs['ldaughtertag'] = UF['daughter-Tag']
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
UFs['w0/a'] = UF['w0/a']#gv.gvar('3.892(12)')
UFs['parent-Tag'] = 'Bs_G5-G5_m{1}'
UFs['daughter-Tag'] = ['etas_G5-G5_tw0','etas_G5-G5_tw0.706','etas_G5-G5_tw1.529','etas_G5-G5_tw2.235','etas_G5-G5_tw4.705']


##################### USER INPUTS ##########################################
Masses = collections.OrderedDict()
Twists = collections.OrderedDict()
thpts = collections.OrderedDict()
############################################################################

Fits = [VCp,Cp,Fp,VC,C,F,SF,UF,Fs,SFs]#,UFs] # choose what to fit make sure s not first
Masses['VCp'] = [0]                                     # Choose which masses to fit
Twists['VCp'] = [0,1,2,3]
thpts['VCp'] = ['S','V']
Masses['Cp'] = [0]                                     # Choose which masses to fit
Twists['Cp'] = [0,1,2,3] 
thpts['Cp'] = ['S','V']
Masses['Fp'] = [0,1,2]                                     # Choose which masses to fit
Twists['Fp'] = [0,1,2,3]
thpts['Fp'] = ['S','V','T']
Masses['VC'] = [0]                                     # Choose which masses to fit
Twists['VC'] = [0,1,2,3]
thpts['VC'] = ['S','V','T']
Masses['C'] = [0,1,2]                                     # Choose which masses to fit
Twists['C'] = [0,1,2,3,4] 
thpts['C'] = ['S','V','T']
Masses['F'] = [0,1,2,3]                                     # Choose which masses to fit
Twists['F'] = [0,1,2,3,4]
thpts['F'] = ['S','V','T']
Masses['SF'] = [0,1,2,3]
Twists['SF'] = [0,1,2,3,4]
thpts['SF'] = ['S','V','T','X'] # x is Vx
Masses['UF'] = [0,1,2,3]
Twists['UF'] = [0,1,2,3,4]
thpts['UF'] = ['S','V','T','X']
Masses['Fs'] = [0,1,2,3]                                     # Choose which masses to fit
Twists['Fs'] = [0,1,2,3,4]
thpts['Fs'] = ['S','V']
Masses['SFs'] = [0,1,2,3]
Twists['SFs'] = [0,1,2,3,4]
thpts['SFs'] = ['S','V']
Masses['UFs'] = [0,1,2,3]
Twists['UFs'] = [0,1,2,3,4]
thpts['UFs'] = ['S','V']
addrho = True
fpf0same = True
constraint = False#add constraint the f0(0)=fp(0)
constraint2 = False # not working
noise = False
FitNegQsq = True
prifac = 1.0
dpri = '0.0({0})'.format(1.0*prifac)
d0000npri = '0.0({0})'.format(1.0*prifac)# backbone of a without disc effects
di0000pri = '0.0({0})'.format(1.0*prifac)# HQET terms
d000lnpri = '0.0({0})'.format(1.0*prifac)# NOTE actually di00ln i.e. all non disc terms
di100npri = '0.0({0})'.format(0.3*prifac) #covers any j=1 or k=1 but not both suppressed as O(a^2)
rhopri ='0.0({0})'.format(1.0*prifac)
cpri = '0.0(0.5)'
cvalpri ='0.0(1.0)'
Kwikfit = False
Npow = 3 #3
Nijk = [3,2,2,3] # i,j,k,l #[3,2,2,3]
Nm = 0
SHOWPLOTS = False
t_0 = '0' # for z conversion can be '0','rev','min' rev gives t_-
adddata = False #This includes the Bs Etas data, should not be used in conjunction with Fs, SFs, UFs Doesn't work with const2
############################################################################
if t_0 != '0':
    print('t_0 = {0} , so fpf0same = False constraint = True'.format(t_0))
    fpf0same = False
    constraint = True
############################################################################

fs_data = collections.OrderedDict() #fs from data fs_data[Fit][]

make_params_BK(Fits,Masses,Twists) #change to BK in BK case
Z_T = make_Z_T() # Tensor normalisation
for Fit in Fits:
    fs_data[Fit['conf']] = collections.OrderedDict()
    if Fit['conf'] == 'F' and Fs in Fits:
        get_results(Fit,thpts,Fit_s=Fs)
    elif Fit['conf'] == 'SF' and SFs in Fits:
        get_results(Fit,thpts,Fit_s=SFs)
    else:
        get_results(Fit,thpts)
        
for Fit in Fits:
    make_fs(Fit,fs_data[Fit['conf']],thpts,Z_T)
    #results_tables(fs_data[Fit['conf']],Fit)
    #mass_corr_plots(Fit,fs_data[Fit['conf']],thpts,F,fs_data['F'])
#check_poles(Fits) Not working atm
#plot_gold_non_split(Fits)
#fp_V0_V1_diff(fs_data,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,constraint2)
#Z_V_plots(Fits,fs_data)
################################ Test average #####################################
average_t_0_cases = gv.BufferDict()
#fpf0same = True
#constraint = False#add constraint the f0(0)=fp(0)
#t_0 = '0'
prior,f = make_prior_BK(fs_data,Fits,addrho,t_0,Npow,Nijk,Nm,rhopri,dpri,cpri,cvalpri,d0000npri,di0000pri,di100npri,d000lnpri,adddata,constraint)
pfit = do_fit_BK(fs_data,adddata,Fits,f,Nijk,Npow,Nm,t_0,addrho,noise,prior,fpf0same,rhopri,dpri,cpri,cvalpri,d0000npri,di0000pri,di100npri,d000lnpri,constraint,constraint2)
#average_t_0_cases = fp_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,constraint2,average_t_0_cases)
#fs_at_lims_BK(f,pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)

#fpf0same = False
#constraint = True#add constraint the f0(0)=fp(0)
#t_0 = 'min'
#prior,f = make_prior_BK(fs_data,Fits,addrho,t_0,Npow,Nijk,Nm,rhopri,dpri,cpri,cvalpri,d0000npri,di0000pri,di100npri,d000lnpri,adddata,constraint)
#pfit = do_fit_BK(fs_data,adddata,Fits,f,Nijk,Npow,Nm,t_0,addrho,noise,prior,fpf0same,rhopri,dpri,cpri,cvalpri,d0000npri,di0000pri,di100npri,d000lnpri,constraint,constraint2)
#average_t_0_cases = fp_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,constraint2,average_t_0_cases)
#fs_at_lims_BK(f,pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)

#fpf0same = False
#constraint = True#add constraint the f0(0)=fp(0)
#t_0 = 'rev'
#prior,f = make_prior_BK(fs_data,Fits,addrho,t_0,Npow,Nijk,Nm,rhopri,dpri,cpri,cvalpri,d0000npri,di0000pri,di100npri,d000lnpri,adddata,constraint)
#pfit = do_fit_BK(fs_data,adddata,Fits,f,Nijk,Npow,Nm,t_0,addrho,noise,prior,fpf0same,rhopri,dpri,cpri,cvalpri,d0000npri,di0000pri,di100npri,d000lnpri,constraint,constraint2)
#average_t_0_cases = fp_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,constraint2,average_t_0_cases)
###################################################################################

fs_at_lims_BK(prior,f,pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#ensemble_error_breakdown()
################
#dBdq2_emup(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#dBdq2_emu0(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#dBdq2_emu(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#dBdq2_the_p(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#dBdq2_the_0(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#dBdq2_the(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#dBdq2_the_tau(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#B_exp_plots(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#B_the_plots(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#Rmue_plots(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#Rbybin_the(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#F_h_plots(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#R_the_plot(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#F_H_the_plots(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#FHbybin_the_plots(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#p_in_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,constraint2)
#neutrio_branching(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#nu_in_qsq(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,constraint2)
#Bemu_results_tables(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#Bnu_results_tables(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#Btau_results_tables(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#Remu_results_tables(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#Rtau_results_tables(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#FHemu_results_tables(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#FHtau_results_tables(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
###############################
#speed_of_light(Fits)
#ff_ratios_qsq_MH(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,constraint2)
#f0_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata)
#fp_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,constraint2,average_t_0_cases)
#fT_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,constraint2)
#f0_no_pole_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata)
#fp_no_pole_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,constraint2)
#fT_no_pole_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata)
#f0_fp_fT_in_qsq(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,constraint2)
#f0_fp_fT_in_Mh(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,constraint2)
#DK_fT_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,constraint2)
#Hill_eq_19_20(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,constraint2)
#beta_delta_in_Mh(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,constraint2)
#table_of_as(Fits,pfit,Nijk,Npow,Nm,fpf0same,addrho,Del)
#DKfT_table_of_as(Fits,pfit,Nijk,Npow,Nm,fpf0same,addrho)
#error_plot(pfit,prior,Fits,Nijk,Npow,Nm,f,t_0,addrho,fpf0same,constraint2)
############################## Redundant?#########
#comp(pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,constraint2)
#test_stuff(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#B_by_bin(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#dBdq2_by_bin(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
#tau_by_bin(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,constraint2)
