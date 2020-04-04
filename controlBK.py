import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from functions import *
from plottingBK import *
plt.rc("font",**{"size":18})
import collections
import copy
import os.path
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
################################## F PARAMETERS ##########################
F = collections.OrderedDict()
F['conf']='F'
F['label'] = 'Set 1'
F['filename'] = 'Corrfits/FKBscalarvectortensor_398cfgs_negFalse0.4490.5660.6830.800.42811.2822.1412.570BGBNGKGKNGSTV141720chained_Nexp3_sfac1.0_pfac1.0_Q1.00_chi0.422_Stmin2_Ttmin2_Vtmin2.pickle'
F['Hsfilename'] = 'Corrfits/F5_Q1.00_Nexp5_Stmin2_Vtmin1_svd0.00013_chi0.341'
F['Hsparent-Tag'] = 'meson.m{0}_m{1}'
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
F['daughter-Tag'] = 'K_G5-G5_tw{0}' 

######################## SF PARAMETERS ####################################
SF = collections.OrderedDict()
SF['conf']='SF'
SF['label'] = 'Set 2'
SF['filename'] = 'Corrfits/SFnohimem-KBscalarvectortensor_158cfgs_negscalarvector0.2740.450.60.801.2612.1082.9463.624BGBNGKGKNGSTV202530chained_Nexp2_sfac1.0_pfac1.0_Q1.00_chi0.772_Stmin2_Ttmin2_Vtmin2.pickle' # in stability plot
#SF['filename'] = 'Corrfits/SFnohimem-KBscalarvectortensor_158cfgs_negscalarvector0.2740.450.60.801.2612.1082.9463.624BGBNGKGKNGSTV202530unchained_Nexp2_sfac1.0_pfac1.0_Q1.00_chi0.549_Stmin2_Ttmin2_Vtmin2.pickle' 
SF['Hsfilename'] = 'Corrfits/SF5_Q1.00_Nexp4_Stmin2_Vtmin2_svd0.00002_chi0.722'
SF['Hsparent-Tag'] = 'meson.m{0}_m{1}'
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
SF['daughter-Tag'] = 'K_G5-G5_tw{0}'


######################## SF PARAMETERS ####################################
UF = collections.OrderedDict()
UF['conf']='UF'
UF['label'] = 'Set 3'
UF['filename'] = '../Fits/UF5_3pts_Q1.00_Nexp2_NMarg6_Stmin2_Vtmin2_svd0.01000_chi0.047_pl1.0_svdfac1.0'
UF['Hsfilename'] = 'UF5_Q1.00_Nexp9_Stmin2_Vtmin2_svd0.01647_chi0.073'
UF['Hsparent-Tag'] = 'Bs_G5-G5_m{1}'
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


##################### USER INPUTS ##########################################
Masses = collections.OrderedDict()
Twists = collections.OrderedDict()
thpts = collections.OrderedDict()
############################################################################

Fits = [F,SF]#,UF]                                         # Choose to fit F, SF or UF
Masses['F'] = [0,1,2,3]                                     # Choose which masses to fit
Twists['F'] = [0,1,2,3,4]
thpts['F'] = ['S','V','T']
Masses['SF'] = [0,1,2,3]
Twists['SF'] = [0,1,2,3,4]
thpts['SF'] = ['S','V','T']
Masses['UF'] = [0,1,2,3]
Twists['UF'] = [0,1,2,3,4]
thpts['UF'] = ['S','V','T']
addrho = True
fpf0same = True
svdnoise = False
priornoise = False
FitNegQsq = True
dpri = '0.0(1.0)'
di000pri = '0.0(1.0)'#'0.0(5.0)' for no rho
di10npri = '0.0(0.5)'
cpri = '0.0(0.3)'
cvalpri ='0.0(1.0)'
rhopri ='0.0(1.0)'
DoFit = True
Npow = 3 #3
Nijk = 3 #3
SHOWPLOTS = False
Del = 0.4 #0.4 change in functions too
t_0 = 0 # for z conversion
adddata = False #include data in continuum from other papers currently only for f0

############################################################################

fs_data = collections.OrderedDict() #fs from data fs_data[Fit][]

make_params_BK(Fits,Masses,Twists) #change to BK in BK case

for Fit in Fits:
    fs_data[Fit['conf']] = collections.OrderedDict()
    get_results(Fit,thpts)
    make_fs(Fit,fs_data[Fit['conf']],thpts)
    results_tables(fs_data[Fit['conf']],Fit)
prior,f = make_prior_BK(fs_data,Fits,Del,addrho,t_0,Npow,Nijk,rhopri,dpri,cpri,cvalpri,di000pri,di10npri,adddata)

pfit = do_fit_BK(Fits,f,Nijk,Npow,addrho,svdnoise,priornoise,prior,fpf0same)
#print values
fs_at_lims_BK(pfit,t_0,Fits,fpf0same,Del,Nijk,Npow,addrho)

#Now to plot whatever we like, we only need the fit output, pfit, the fs from the data fs_data and Fit
speed_of_light(Fits)
f0_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same,adddata)
fp_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same,adddata)
fT_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same,adddata)
#f0_no_pole_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same,adddata)
#fp_no_pole_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same,adddata)
#f0_fp_in_qsq(pfit,Fits,t_0,Nijk,Npow,Del,addrho,fpf0same)
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
