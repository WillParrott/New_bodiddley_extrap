import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from functions import *
from plotting import *
plt.rc("font",**{"size":18})
import collections
import copy
import os.path
import pickle
from collections import defaultdict
################################## F PARAMETERS ##########################
F = collections.OrderedDict()
F['conf']='F'
F['filename'] = 'Fits/F5_3pts_Q1.00_Nexp2_NMarg5_Stmin2_Vtmin1_svd0.00157_chi0.342_pl1.0_svdfac1.0'
F['masses'] = ['0.449','0.566','0.683','0.8']
F['Zdisc'] = [0.99892,0.99826,0.99648,0.99377]
F['twists'] = ['0','0.4281','1.282','2.141','2.570','2.993']
F['m_s'] = '0.0376'
F['m_c'] = '0.449'
F['m_ssea'] = 0.037
F['m_lsea'] = 0.0074
F['Ts'] = [14,17,20]
F['tp'] = 96
F['L'] = 32
F['w0/a'] = gv.gvar('1.9006(20)')
F['parent-Tag'] = 'meson.m{0}_m{1}'
F['daughter-Tag'] = ['etas','etas_p0.0728','etas_p0.218','etas_p0.364','etas_p0.437','etas_p0.509'] 

######################## SF PARAMETERS ####################################
SF = collections.OrderedDict()
SF['conf']='SF'
SF['filename'] = 'Fits/SF5_3pts_Q1.00_Nexp3_NMarg6_Stmin2_Vtmin2_svd0.00457_chi0.079_pl1.0_svdfac1.0' # in stability plot
SF['Masses'] = ['0.274','0.450','0.6','0.8']
SF['Zdisc'] = [0.99990,0.99928,0.99783,0.99377]
SF['Twists'] = ['0','1.261','2.108','2.946','3.624']
SF['m_s'] = '0.0234'
SF['m_c'] = '0.274'
SF['m_ssea'] = 0.024
SF['m_lsea'] = 0.0048
SF['Ts'] = [20,25,30]
SF['tp'] = 144
SF['L'] = 48
SF['w0/a'] = gv.gvar('2.896(6)')
SF['parent-Tag'] = 'meson.m{0}_m{1}'
SF['daughter-Tag'] = ['etas_p0','etas_p0.143','eta_s_tw2.108_m0.0234','etas_p0.334','eta_s_tw3.624_m0.0234']


######################## SF PARAMETERS ####################################
UF = collections.OrderedDict()
UF['conf']='UF'
UF['filename'] = 'Fits/UF5_3pts_Q1.00_Nexp2_NMarg6_Stmin2_Vtmin2_svd0.01000_chi0.047_pl1.0_svdfac1.0' 
UF['Masses'] = ['0.194','0.45','0.6','0.8']
UF['Zdisc'] = [0.99997,0.99928,0.99783,0.99377]
UF['Twists'] = ['0','0.706','1.529','2.235','4.705']
UF['m_s'] = '0.0165'
UF['m_c'] = '0.194'
UF['m_ssea'] = 0.0158
UF['m_lsea'] = 0.00316
UF['Ts'] = [33,40]
UF['tp'] = 192
UF['L'] = 64
UF['w0/a'] = gv.gvar('3.892(12)')
UF['parent-Tag'] = 'Bs_G5-G5_m{1}'
UF['daughter-Tag'] = ['etas_G5-G5_tw0','etas_G5-G5_tw0.706','etas_G5-G5_tw1.529','etas_G5-G5_tw2.235','etas_G5-G5_tw4.705']


##################### USER INPUTS ##########################################
Masses = collections.OrderedDict()
Twists = collections.OrderedDict()
############################################################################

Fits = [F,SF,UF]                                         # Choose to fit F, SF or UF
Masses['F'] = [0,1,2,3]                                     # Choose which masses to fit
Twists['F'] = [0,1,2,3,4]#,5]
3pts['F'] = ['S','V']
Masses['SF'] = [0,1,2,3]
Twists['SF'] = [0,1,2,3,4]
3pts['S'] = ['S','V']
Masses['UF'] = [0,1,2,3]
Twists['UF'] = [0,1,2,3,4]
3pts['UF'] = ['S','V']
addrho = True
fpf0same = True
svdnoise = False
priornoise = False
FitNegQsq = True
dpri = '0.0(1.0)'
di000pri = '0.0(1.0)'
di10npri = '0.0(0.5)'
cpri = '0.0(0.3)'
cvalpri ='0.0(1.0)'
rhopri ='0.0(1.0)'
DoFit = True
N = 3#3
Nijk = 3 #3
SHOWPLOTS = False
Del = 0.4 #0.4
t_0 = 0 # for z conversion

############################################################################
figsca = 14  #size for saving figs
############################################################################

fs_data = collections.OrderedDict() #fs from data fs_data[Fit][]

make_params_BsEtas() #change to BK in BK case

for Fit in Fits:
    fs_data[Fit['conf']] = collections.OrderedDict()
    get_results(Fit)
    make_fs(Fit,fs_data[Fit['conf']],t_0)
    
prior,f = make_prior_BsEtas(fs_data,Fits,Del,addrho,t_0,Npow,Nijk,rhopri,dpri,cpri,di000pri,di10npri)

pfit = do_fit_BsEtas(Fits,f,Nijk,Npow,addrho,svdnoise,priornoise,prior,f)
