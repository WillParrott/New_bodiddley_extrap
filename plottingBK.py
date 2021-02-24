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
factor = 0.5 #multiplies everything to make smaller for big plots (0.5) usually 1
figsca = 14  #size for saving figs
figsize = ((figsca,2*figsca/(1+np.sqrt(5))))
lw =2*factor
nopts = 100 #number of points on plot
ms = 25*factor #markersize
alpha = 0.4
fontsizeleg = 25*factor #legend
fontsizelab = 40*factor #labels
cols = ['b','r','g','c'] #for each mass
symbs = ['o','^','*','D','d','s','p','>','<','h']    # for each conf
lines = ['-','--','-.','-',':','-','--','-.',':','-.'] # for each conf
major = 15*factor
minor = 8*factor 
capsize = 10*factor

################################### EXP data #################################################################

class Exp():
    def __init__(self,label='',arxiv='',Rmue=[],Be=[],Be0=[],Bep=[],Bmu=[],Bmu0=[],Bmup=[],Btau=[],Btaup=[],Bemu=[],Bemu0=[],Bemup=[],bins=[],binBemu=[],binBep=[],binBmu=[],binBmu0=[],binBmup=[],FHmup=[],Rmuep=[],sym=''):
        self.label = label
        self.arxiv = arxiv
        self.Be = Be
        self.Be0 = Be0
        self.Bep = Bep
        self.Bmu = Bmu
        self.Bmu0 = Bmu0
        self.Bmup = Bmup
        self.Btau = Btau
        self.Btaup = Btaup
        self.Bemu = Bemu
        self.Bemu0 = Bemu0
        self.Bemup = Bemup
        self.bins = bins
        self.binBemu = binBemu
        self.binBep = binBep
        self.binBmu = binBmu   #some of these values are B, some are dB/dq^2. i.e we need to divide by the bin width
        self.binBmu0 = binBmu0
        self.binBmup = binBmup
        self.FHmup = FHmup
        self.Rmuep = Rmuep
        self.Rmue = Rmue
        self.sym = sym

    def make_x(self):
        x = []
        xerr = []
        for element in self.bins:
            x.append((element[0]+element[1])/2)
            xerr.append((element[1]-element[0])/2)
        return(x,xerr)

    def make_1_y(self,ys):
        y = []
        yupp = []
        ylow = []
        y.append(ys[0])
        if len(ys) == 4:
            yupp.append(np.sqrt(ys[1]**2+ys[3]**2))
            ylow.append(np.sqrt(ys[2]**2+ys[3]**2))
        elif len(ys) == 5:
            yupp.append(np.sqrt(ys[1]**2+ys[3]**2))
            ylow.append(np.sqrt(ys[2]**2+ys[4]**2))
        yerr =[ylow,yupp]
        return(y,yerr)

        
    def make_y(self,ys):
        y = []
        yupp = []
        ylow = []
        for element in ys:
            y.append(element[0])
            if len(element) == 4:
                yupp.append(np.sqrt(element[1]**2+element[3]**2))
                ylow.append(np.sqrt(element[2]**2+element[3]**2))
            elif len(element) == 5:
                yupp.append(np.sqrt(element[1]**2+element[3]**2))
                ylow.append(np.sqrt(element[2]**2+element[4]**2))
        yerr =[ylow,yupp]
        return(y,yerr)
    
    def fix_B_bins(self,ys,const=1):
        widths = []
        for element in self.bins:
            widths.append(element[1]-element[0])
        for i in range(len(widths)):
            for j in range(len(ys[i])):
                ys[i][j] = ys[i][j]/(const*widths[i])
        return(ys)
    
    def fix_B_bins_const(self,ys,const):
        for i in range(np.shape(ys)[0]):
            for j in range(len(ys[i])):
                ys[i][j] = ys[i][j]/(const)
        return(ys)
                

#bins: emu:2  mu:1 mup:3 mu0:3 ep:1
    
BELLE09 = Exp(label="Belle '09",arxiv='0904.0770',Rmue=[1.03,0.19,0.19,0.06],Be=[4.8,0.8,0.7,0.3],Bmu=[5.0,0.6,0.6,0.3],Bemu=[4.8,0.5,0.4,0.3],Be0=[2.0,1.4,1.0,0.1],   Bep=[5.7,0.9,0.8,0.3],Bmu0=[4.4,1.3,1.1,0.3],Bmup=[5.3,0.8,0.7,0.3],Bemu0=[3.4,0.9,0.8,0.2], Bemup=[5.7,0.9,0.8,0.3]     ,bins=[[0,2],[2,4.3],[4.3,8.68],[10.09,12.86],[14.18,16.00],[16.00,qsqmaxphysBK.mean],[1,6]]                     ,binBemu=[[0.81,0.18,0.16,0.05],[0.46,0.14,0.12,0.03],[1.00,0.19,0.18,0.06],[0.55,0.16,0.14,0.03],[0.38,0.19,0.12,0.02],[0.98,0.2,0.18,0.06],[1.36,0.23,0.21,0.08]],sym='s')
BELLE09.fix_B_bins(BELLE09.binBemu)

BaBar09 = Exp(label="BaBar '08",arxiv='0807.4119',Bemu = [3.94,0.73,0.69,0.20],Rmue=[0.96,0.44,0.34,0.05],Bmu0=[4.9,2.9,2.5,0.3],Bmup=[4.1,1.6,1.5,0.2],Be0=[0.8,1.5,1.2,0.1],Bep=[5.1,1.2,1.1,0.2],Bemu0=[2.1,1.5,1.3,0.2],Bemup=[4.76,0.92,0.86,0.22],Bmu=[4.1,1.3,1.2,0.2],Be=[3.88,0.9,0.83,0.2],sym='d')

BaBar12 = Exp(label="BaBar '12",arxiv='1204.3933',Bemu=[4.7,0.6,0.6,0.2], bins=[[0.1,2],[2,4.3],[4.3,8.12],[10.11,12.89],[14.21,16],[16,qsqmaxphysBK.mean],[1,6]],   binBemu=[[0.71,0.2,0.18,0.02],[0.49,0.15,0.13,0.01],[0.94,0.2,0.19,0.02],[0.9,0.2,0.19,0.04],[0.49,0.15,0.14,0.02],[0.6,0.23,0.21,0.05],[1.36,0.27,0.24,0.03]],sym='d' )
BaBar12.fix_B_bins(BaBar12.binBemu)

BaBar16 = Exp(label="BaBar '16",arxiv='1605.09637',Btaup=[0.00131,0.00066,0.00061,0.00035,0.00025],sym='d')#no *1e-7

CDF11 = Exp(label="CDF '11",arxiv='1107.3753',Bmu=[4.2,0.4,0.4,0.2], Bmu0=[3.2,1,1,0.2],Bmup=[4.6,0.4,0.4,0.2],  bins=[[0,2],[2,4.3],[4.3,8.68],[10.09,12.86],[14.18,16],[16,23],[0,4.3],[1,6]],   binBmu =[[0.33,0.1,0.1,0.02],[0.77,0.14,0.14,0.05],[1.05,0.17,0.17,0.07],[0.48,0.1,0.1,0.03],[0.52,0.09,0.09,0.03],[0.38,0.09,0.09,0.02],[1.07,0.17,0.17,0.07],[1.29,0.18,0.18,0.08]],binBmup =[[0.36,0.11,0.11,0.03],[0.8,0.15,0.15,0.05],[1.18,0.19,0.19,0.09],[0.68,0.12,0.12,0.05],[0.53,0.1,0.1,0.03],[0.48,0.11,0.11,0.03],[1.13,0.19,0.19,0.08],[1.41,0.2,0.2,0.1]],   binBmu0=[[0.312,0.372,0.372,0.024],[0.929,0.485,0.485,0.070],[0.663,0.510,0.510,0.052],[-0.030,0.223,0.223,0.005],[0.726,0.257,0.257,0.055],[0.214,0.182,0.182,0.016],[1.268,0.622,0.622,0.096],[0.980,0.614,0.614,0.076]],sym='^')
CDF11.fix_B_bins(CDF11.binBmup)
CDF11.fix_B_bins(CDF11.binBmu0)
CDF11.fix_B_bins(CDF11.binBmu)

LHCb12A = Exp(label="LHCb '12A",arxiv='1205.3422',Bmu0=[3.1,0.7,0.6,0],bins=[[0.05,2],[2,4.3],[4.3,8.68],[10.09,12.86],[14.18,16],[16,23],[1,6]],  binBmu0=[[1.1,1.4,1.2,0],[0.3,1.1,0.9,0],[2.8,0.7,0.7,0],[1.8,0.8,0.7,0],[1.1,0.7,0.5,0],[0.5,0.3,0.2,0],[1.3,0.9,0.7,0]],sym='o')
LHCb12A.fix_B_bins_const(LHCb12A.binBmu0,10)

LHCb12B = Exp(label="LHCb '12B",arxiv='1209.4284',Bmup=[4.36,0.15,0.15,0.18],bins=[[0.05,2],[2,4.3],[4.3,8.68],[10.09,12.86],[14.18,16],[16,18],[18,22],[1,6]], binBmup=[[2.85,0.27,0.27,0.14],[2.49,0.23,0.23,0.1],[2.29,0.16,0.16,0.09],[2.04,0.18,0.18,0.08],[2.07,0.2,0.2,0.08],[1.77,0.18,0.18,0.09],[0.78,0.10,0.10,0.04],[2.41,0.17,0.17,0.14]], FHmup=[[0,0.12,0,0.06,0],[0.14,0.16,0.1,0.04,0.02],[0.04,0.1,0.04,0.06,0.04],[0.11,0.2,0.08,0.02,0.01],[0.08,0.28,0.08,0.02,0.01],[0.18,0.22,0.14,0.01,0.04],[0.14,0.31,0.14,0.01,0.02],[0.05,0.08,0.05,0.04,0.02]],sym='o')
LHCb12B.fix_B_bins_const(LHCb12B.binBmup,10)

LHCb14A = Exp(label="LHCb '14A",arxiv='1403.8044',Bmup=[4.29,0.07,0.07,0.21],Bmu0=[3.27,0.34,0.34,0.17],bins=[[0.1,0.98],[1.1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[11,11.8],[11.8,12.5],[15,16],[16,17],[17,18],[18,19],[19,20],[20,21],[21,22],[1.1,6],[15,22]],  binBmup=[[0.332,0.018,0.018,0.017],[0.233,0.015,0.015,0.012],[0.282,0.016,0.016,0.014],[0.254,0.015,0.015,0.013],[0.221,0.014,0.014,0.011],[0.231,0.014,0.014,0.012],[0.245,0.014,0.014,0.012],[0.231,0.014,0.014,0.012],[0.177,0.013,0.013,0.009],[0.193,0.012,0.012,0.01],[0.161,0.01,0.01,0.008],[0.164,0.01,0.01,0.008],[0.206,0.011,0.011,0.01],[0.137,0.01,0.01,0.007],[0.074,0.008,0.008,0.004],[0.059,0.007,0.007,0.003],[0.043,0.007,0.007,0.002],[0.242,0.007,0.007,0.012],[0.121,0.004,0.004,0.006]],sym='o')

LHCb14A2 = Exp(label="LHCb '14A",arxiv='1403.8044',bins =[[0.1,2],[2,4],[4,6],[6,8],[11,12.5],[15,17],[17,22],[1.1,6],[15,22]],binBmu0=[[0.122,0.059,0.052,0.006],[0.187,0.055,0.049,0.009],[0.173,0.053,0.048,0.009],[0.27,0.058,0.053,0.014],[0.127,0.045,0.04,0.006],[0.143,0.035,0.032,0.007],[0.078,0.017,0.015,0.004],[0.187,0.035,0.032,0.009],[0.095,0.016,0.015,0.005]],sym='o')

LHCb14B = Exp(label="LHCb '14B",arxiv='1403.8045',sym='o',bins=[[0.1,0.98],[1.1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[11,11.75],[11.75,12.5],[15,16],[16,17],[17,18],[18,19],[19,20],[20,21],[21,22],[1.1,6],[15,22]], FHmup=[[0.01,0.2,0.03],[0,0.21,0.03],[0.05,0.3,0.03],[0,0.04,0.02],[0,0.09,0.03],[0,0.14,0.02],[0,0.08,0.02],[0,0.03,0.03],[0.06,0.23,0.03],[0,0.1,0.02],[0.06,0.2,0.02],[0,0.12,0.02],[0.01,0.16,0.02],[0.05,0.23,0.02],[0,0.1,0.04],[0,0.14,0.04],[0.04,0.41,0.05],[0,0.06,0.02],[0,0.07,0.02]])#contains FHmu intervals, so each give supper and lower value [low,upp,systerr]

LHCb14C = Exp(label="LHCb '14C",arxiv='1406.6482',Rmuep=[0.745,0.09,0.074,0.036],bins=[[1,6]], binBep=[[1.56,0.19,0.15,0.06,0.04]],sym='o')# R, like B, is over range 1<q^2<6
LHCb14C.fix_B_bins(LHCb14C.binBep)
################################################################################################################

class Theory():
    def __init__(self,label='',arxiv='',Be=[],Bemu0=[],Bmu=[],Btau=[],Rmue=[],Rtaumu=[],Rmutau0=[],Rtaue=[],Rtauemu=[],Bmu0=[],Bemum=[],Bmup=[],Btaum=[],Btaup=[],Btau0=[],FHep=[],FHe0=[],FHmup=[],FHmu0=[],FHtaup=[],FHtau0=[],FHmu=[],FHtau=[],bins=[],binBmum=[],binBmu0=[],binBmup=[],binFHmum=[],binFHmu0=[],binRmue0=[],binRmuem=[],binRmuep=[],binBemu=[],binBemu0=[],binBemum=[],binFHemu0=[],binFHemum=[],Rmuep=[],binFHep=[],binFHe0=[],binFHmup=[],binFHtaup=[],binBtaup=[],binBtau0=[],binRmutaup=[],binRmutau0=[],binFHtau0=[],binRmue=[],binFHe=[],binFHmu=[],binRtaumu=[],binRtaue=[],binRtauemu=[],binFHtau=[],binBtau=[],sym=''):
        self.label = label
        self.arxiv = arxiv
        self.Bemum = Bemum
        self.Bmup = Bmup
        self.Bmu0 = Bmu0
        self.Bemu0 = Bemu0
        self.Btaup = Btaup
        self.Btau0 = Btau0
        self.Btaum = Btaum
        self.FHmu = FHmu
        self.FHtau = FHtau
        self.FHep = FHep
        self.FHe0 = FHe0
        self.FHmup = FHmup
        self.FHmu0 = FHmu0
        self.FHtaup = FHtaup
        self.FHtau0 = FHtau0
        self.bins = bins
        self.binBmum = binBmum
        self.binBmu0 = binBmu0
        self.binBmup = binBmup
        self.binBtaup = binBtaup
        self.binBtau0 = binBtau0
        self.binFHmum = binFHmum
        self.binFHmu0 = binFHmu0
        self.binFHmup = binFHmup
        self.binFHtaup = binFHtaup
        self.binFHtau0 = binFHtau0
        self.binFHep = binFHep
        self.binFHe0 = binFHe0
        self.binRmue0 = binRmue0
        self.binRmuem = binRmuem
        self.binRmuep = binRmuep
        self.binRmue = binRmue
        self.binRmutaup = binRmutaup
        self.binRmutau0 = binRmutau0
        self.binBemu = binBemu
        self.binBemu0 = binBemu0
        self.binBemum = binBemum
        self.binFHemu0 = binFHemu0
        self.binFHemum = binFHemum
        self.Rmuep = Rmuep
        self.Rmutau0 = Rmutau0
        self.sym = sym
        self.Be = Be
        self.Bmu = Bmu
        self.Btau = Btau
        self.Rmue = Rmue
        self.Rtaumu = Rtaumu
        self.Rtaue = Rtaue
        self.Rtauemu = Rtauemu
        self.binFHe = binFHe
        self.binFHmu = binFHmu
        self.binRtaumu = binRtaumu
        self.binRtaue = binRtaue
        self.binRtauemu = binRtauemu
        self.binFHtau = binFHtau
        self.binBtau = binBtau
        
    def make_x(self):
        x = []
        xerr = []
        for element in self.bins:
            x.append((element[0]+element[1])/2)
            xerr.append((element[1]-element[0])/2)
        return(x,xerr)

    def add_up_down_errs(self,ys):
        if np.shape(np.shape(ys)) == (1,):
            ys = [ys]
        y = []
        yupp =[]
        ylow = []
        for element in ys:
            tempupp = 0
            templow = 0
            y.append(element[0])
            for i,err in enumerate(element[1:]):
                if i%2 == 0:
                    tempupp += err**2
                if i%2 == 1:
                    templow += err**2
            yupp.append(np.sqrt(tempupp))
            ylow.append(np.sqrt(templow))
        yerr = [ylow,yupp]
        return(y,yerr)
    
    def add_errs(self,ys):
        if np.shape(np.shape(ys)) == (1,):
            ys = [ys]
        y = []
        yerr =[]
        for element in ys:
            temp = 0
            y.append(element[0])
            for err in element[1:]:
                temp += err**2
            yerr.append(np.sqrt(temp))
        return(y,yerr)
    
    def fix_B_bins(self,ys):
        widths = []
        for element in self.bins:
            widths.append(element[1]-element[0])
        for i in range(len(widths)):
            for j in range(len(ys[i])):
                ys[i][j] = ys[i][j]/widths[i]
        return(ys)
    
    def fix_B_bins_const(self,ys,const):
        for i in range(np.shape(ys)[0]):
            for j in range(len(ys[i])):
                ys[i][j] = ys[i][j]/(const)
        return(ys)

Wang12 = Theory(label="Wang et al. '12",arxiv='1207.0265',Bemu0=[5.1,1.5,1.1,0.5,0.5,0.4,0.4], Btau0=[1.2,0.32,0.35,0.07,0.07,0.11,0.1], Bemum=[5.5,1.59,1.18,0.57,0.55,0.42,0.41], Btaum=[1.29,0.35,0.26,0.08,0.08,0.11,0.011],sym='D') #B *10-7 err is upp down upp down
    
Bobeth07 = Theory(label="Bobeth et al. '07",arxiv='0709.4174', bins=[[1,6],[2,6],[1,7],[2,7]], binBmum=[[1.6,0.51,0.46],[1.27,0.40,0.36],[1.91,0.59,0.54],[1.59,0.48,0.44]], binBmu0=[[1.46,0.47,0.43],[1.16,0.37,0.33],[1.74,0.55,0.50],[1.45,0.45,0.41]], binFHmum=[[0.0244,0.0003,0.0003],[0.0188,0.0002,0.0001],[0.0221,0.0003,0.0003],[0.0172,0.0002,0.0002]], binFHmu0=[[0.0243,0.0003,0.0003],[0.187,0.0002,0.0001],[0.0221,0.0003,0.0004],[0.0172,0.0002,0.0002]], binRmuem=[[1.00030,0.0001,0.00007],[1.00037,0.0001,0.00007],[1.00032,0.0001,0.00007],[1.00039,0.00011,0.00007]], binRmue0=[[1.00031,0.0001,0.00007],[1.00038,0.00011,0.00007],[1.00033,0.00011,0.00007],[1.0004,0.00011,0.00007]],sym='o')# RK, FHmu and B exist for bins 1,6 2,6 1,7 2,7 decide which I want FHmu does not change

Bobeth11 = Theory(label="Bobeth '11",arxiv='1111.2558',Bemum=[1.04,0.6,0.27,0.04,0.02,0.03,0.03,0.04,0.07], Btaum=[1.26,0.4,0.21,0.04,0.03,0.02,0.02,0.05,0.09], FHmu=[7.50,2.85,2.61,0.04,0.1,0.1,0.1], FHtau=[0.890,0.033,0.045,0.001,0.002,0.002,0.002],               bins=[[1,6],[14.18,16],[16,qsqmaxphysBK.mean]], binBmu0=[[1.59,0.59,0.35],[0.34,0.18,0.09],[0.63,0.39,0.18]], binBmum=[[1.75,0.64,0.38],[0.37,0.2,0.09],[0.68,0.41,0.19]],sym='o')#B*1e7 FHmu*1e3 up and down errors for all cases, with both F_H from 14.18 upwards err on binBmu0 is just upp, down

Bobeth16 = Theory(label="Bobeth et al. '12",arxiv='1212.2321', bins=[[4*m_mu**2,2],[2,4.3],[4.3,8.68],[1,6],[14.18,16],[16,18],[18,22],[16,qsqmaxphysBK.mean]], binBemu0=[[0.644,0.207,0.106],[0.75,0.256,0.125],[1.38,0.51,0.25],[1.63,0.56,0.27],[0.34,0.179,0.083],[0.309,0.176,0.081],[0.318,0.201,0.092],[0.634,0.382,0.175]], binBemum=[[0.692,0.222,0.113],[0.808,0.275,0.135],[1.48,0.55,0.27],[1.75,0.60,0.29],[0.365,0.192,0.089],[0.331,0.189,0.087],[0.341,0.216,0.098],[0.68,0.41,0.188]], binFHemu0=[[0.103,0.006,0.012],[0.0237,0.0018,0.0033],[0.0124,0.012,0.02],[0.0254,0.002,0.0036],[0.00704,0.00147,0.00196],[0.00693,0.00166,0.00209],[0.00817,0.00243,0.00284],[0.00775,0.0021,0.00254]], binFHemum=[[0.103,0.006,0.012],[0.0237,0.0018,0.0033],[0.0124,0.012,0.02],[0.0255,0.002,0.0036],[0.00704,0.00148,0.00197],[0.00693,0.00166,0.00209],[0.00818,0.00243,0.00284],[0.00775,0.0021,0.00255]],sym='o') #FH does not change

HPQCD14A = Theory(label="HPQCD '13",arxiv='1306.0434',Be=[6.14,1.33],Bmu=[6.12,1.32],Btau=[1.44,0.15],Rmue=[1.00023,0.00063],Rtaumu=[1.158,0.039],Rtaue=[1.161,0.04],Rtauemu=[1.159,0.040], FHtau=[0.8856,0.0037],bins=[[1,6],[4.3,8.68],[10.09,12.86],[14.18,16],[16,18],[16,qsqmaxphysBK.mean]], binBemu=[[1.81,0.61],[1.65,0.42],[0.87,0.13],[0.442,0.051],[0.391,0.042],[0.797,0.082]], binRmue=[[0.74,0.35],[0.89,0.25],[1.35,0.23],[1.98,0.22],[2.56,0.23],[3.86,0.29]], binFHe=[[0.577,0.01],[0.2722,0.0054],[0.1694,0.0053],[0.1506,0.0052],[0.1525,0.0055],[0.1766,0.0068]], binFHmu=[[2.441,0.043],[1.158,0.023],[0.722,0.022],[0.642,0.022],[0.649,0.023],[0.751,0.029]],sym='h')#errors just errors binRs are 10^3*(R-1) FHe 106 FHmu 10^2

HPQCD14B = Theory(label="HPQCD '13",arxiv='1306.0434',bins=[[14.18,16],[16,18],[16,qsqmaxphysBK.mean]], binRtaumu=[[0.790,0.025],[1.055,0.033],[1.361,0.046]], binRtaue=[[0.792,0.025],[1.058,0.034],[1.367,0.047]], binRtauemu=[[0.791,0.025],[1.056,0.033],[1.364,0.046]],binFHtau=[[0.9176,0.0026],[0.8784,0.0038],[0.8753,0.0042]], binBtau=[[0.349,0.04],[0.413,0.044],[1.09,0.11]],sym='h')

Khod =  Theory(label="Khod. et al. '12",arxiv='1211.0234', bins=[[0.05,2],[2,4.3],[4.3,8.68],[1,6]], binBmu0=[[0.71,0.22,0.08],[0.8,0.27,0.11],[1.39,0.53,0.22],[1.76,0.6,0.23]],sym='s')#Khodjamirian

Altmann12 = Theory(label="Altmann. et al. '12", arxiv='1206.0273',bins=[[1,6],[14.18,16],[16,22.9]], binBemu=[[1.28,0.12],[0.41,0.06],[0.49,0.07]],sym='d')#Altmannshofer

Fermi16A = Theory(label="FNAL/MILC '15",arxiv='1510.02349',Bmup=[6.0533,0.3296,0.6514,0.1703], Btaup=[1.6036,0.0873,0.0787,0.0546], Bmu0=[5.5880,0.3043,0.5972,0.1573], Btau0=[1.4745,0.0803,0.0722,0.0502], bins=[[0.1,2],[2,4],[4,6],[6,8],[15,17],[17,19],[19,22],[1,6],[1.1,6],[15,22]], binBmup=[[0.6803,0.037,0.1372,0.0155],[0.7172,0.0391,0.1244,0.0163],[0.7059,0.0384,0.1036,0.0154],[0.6894,0.0375,0.0847,0.0146],[0.4615,0.0251,0.0248,0.0162],[0.3491,0.0190,0.0168,0.0113],[0.2573,0.014,0.0117,0.0086],[1.7835,0.0971,0.2980,0.04],[1.7475,0.0952,0.2907,0.0392],[1.0679,0.0582,0.0521,0.0349]], binBmu0 = [[0.6338,0.0345,0.127,0.0151],[0.6588,0.0359,0.1135,0.0146],[0.6494,0.0354,0.0947,0.0137],[0.6360,0.0346,0.0776,0.0132],[0.4276,0.0233,0.0230,0.0151],[0.3225,0.0176,0.0155,0.0104],[0.2353,0.0128,0.0107,0.0079],[1.6409,0.0894,0.2723,0.0359],[1.6075,0.0875,0.2656,0.0351],[0.9854,0.0537,0.048,0.0322]],sym='^')# neutrinos in this paper too. Need to just add all errors, no up/down # Loads more stuff in this paper


Fermi16B = Theory(label="FNAL/MILC '15",arxiv='1510.02349', FHep=[71.5,6.1,2.1], FHmup=[22.2,1.7,0.7],FHtaup=[0.89,0.,0.03], FHe0=[73.5,6.4,2.2], FHmu0=[22.4,1.7,0.7], FHtau0=[0.89,0,0.03], bins=[[0.1,2],[2,4],[4,6],[6,8],[15,17],[17,19],[19,22],[1,6],[15,22]], binRmuep=[[-3.3,0.02,0.02],[0.5,0.38,0.02],[0.62,0.59,0.02],[0.72,0.86,0.02],[1.79,3.2,0.06],[2.55,4.23,0.09],[5.08,5.95,0.19],[0.5,0.43,0.02],[2.83,4.2,0.1]], binRmue0=[[-4.27,0.22,0.03],[0.44,0.39,0.02],[0.59,0.59,0.02],[0.7,0.85,0.02],[1.78,3.20,0.06],[2.56,4.23,0.09],[5.13,5.94,0.19],[0.43,0.44,0.02],[2.84,4.19,0.1]], binFHep=[[248,2.2,0.5],[55.7,0.7,0],[33.3,0.6,0],[24.3,0.5,0],[14,0.3,0.4],[14.7,0.3,0.5],[19.7,0.4,0.7],[57.8,1.1,0.1],[15.6,0.3,0.5]], binFHe0=[[258.6,2.4,0.4],[55.8,0.7,0],[33.3,0.6,0],[24.3,0.5,0],[14,0.3,0.4],[14.7,0.3,0.5],[19.8,0.4,0.7],[58.0,1.1,0.1],[15.6,0.3,0.5]], binFHmup=[[98.3,0.8,0.2],[23.6,0.3,0],[14.2,0.2,0],[10.3,0.2,0],[6,0.1,0.2],[6.3,0.1,0.2],[8.4,0.2,0.3],[24.5,0.5,0],[6.6,0.1,0.2]], binFHmu0=[[101.8,0.9,0.2],[23.6,0.3,0],[14.2,0.2,0],[10.3,0.2,0],[6,0.1,0.2],[6.3,0.1,0.2],[8.4,0.2,0.3],[24.5,0.5,0],[6.7,0.1,0.2]], sym='^')# Rmue is 10^3(R-1) FHe 10^8 FHmu 10^3

Fermi16C = Theory(label="FNAL/MILC '15",arxiv='1510.02349', bins=[[15,17],[17,19],[19,22],[15,22]], binBtaup=[[0.3992,0.0217,0.0222,0.0140],[0.3931,0.0214,0.0181,0.0133],[0.4323,0.0235,0.018,0.0157],[1.2246,0.0667,0.0563,0.0417]], binBtau0=[[0.3696,0.0201,0.0204,0.013],[0.3634,0.0198,0.0167,0.0123],[0.3974,0.0216,0.0165,0.0144],[1.1305,0.0616,0.0519,0.0385]],binRmutaup=[[0.16,0.02,0.01],[-0.11,0.02,0.01],[-0.40,0.01,0.01],[-0.13,0.02,0.01]], binRmutau0=[[0.16,0.02,0.01],[-0.11,0.02,0.01],[-0.41,0.01,0.01],[-0.13,0.02,0.01]],binFHtaup=[[0.89,0,0.03],[0.86,0,0.02],[0.87,0,0.02],[0.87,0,0.02]],binFHtau0=[[0.89,0,0.03],[0.86,0,0.02],[0.87,0,0.02],[0.87,0,0.02]],sym='^')#Rmutau is (R-1) NOT 1e-3(R-1) Ftau is as is and both charges the same

####################################################################################################

def speed_of_light(Fits):
    plt.figure(figsize=figsize)
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
        plt.errorbar(x,y,yerr=yerr,fmt=points[i],label=Fit['label'],ms=ms,mfc='none',capsize=capsize)
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
    plt.close()
    return()

#####################################################################################################

def Z_V_plots(Fits,fs_data):
    plt.figure(19,figsize=figsize)
    i = 0
    plotfits = []
    for Fit in Fits:
        if Fit['conf'] not in ['Fs','SFs','UFs']:
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
            plt.errorbar(x,y,yerr=yerr,fmt=symbs[i],color='r',label=Fit['label'],ms=ms,mfc='none',capsize=capsize)
        elif Fit['conf'][-1] == 'p':
            plt.errorbar(x,y,yerr=yerr,fmt=symbs[i],color='b',label=Fit['label'],ms=ms,mfc='none',capsize=capsize)
        else:
            plt.errorbar(x,y,yerr=yerr,fmt=symbs[i],color='k',label=Fit['label'],ms=ms,mfc='none',capsize=capsize)
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
    plt.close()
    return()

#####################################################################################################

def f0_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata):
    i = 0
    plotfits = []
    for Fit in Fits:
        if Fit['conf'] not in ['UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
        if Fit['conf' ] == 'F':
            F = Fit
        if Fit['conf' ] == 'SF':
            SF = Fit
        j = 0
        for mass in Fit['masses']:
            qsq = []
            z = []
            y = []
            y2 = []
            qsq2 = []
            z2 = []
            for twist in Fit['twists']:
                q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                if Fit['conf'] == 'Fs':
                    z.append(make_z(q2,t_0,F['M_parent_m{0}'.format(mass)],F['M_daughter'],Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))#all lat units
                elif Fit['conf'] == 'SFs':
                    z.append(make_z(q2,t_0,SF['M_parent_m{0}'.format(mass)],SF['M_daughter'],Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))
                else:
                    z.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))
                y.append(fs_data[Fit['conf']]['f0_m{0}_tw{1}'.format(mass,twist)])
            q2max = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,Fit['twists'][0])]
            q2min = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,Fit['twists'][-1])]
            for q2 in np.linspace(q2min.mean,q2max.mean,nopts):
                y2.append(make_f0_BK(Nijk,Npow,Nm,addrho,pfit,Fit,q2,t_0,mass,fpf0same,float(mass)).mean)
                qsq2.append((q2/Fit['a']**2).mean)
                if Fit['conf'] == 'Fs':
                    z2.append(make_z(q2,t_0,pfit['MH_{0}_m{1}'.format('F',mass)],pfit['MK_{0}'.format('F')],pfit['MH_{0}_m{1}'.format(Fit['conf'],mass)],pfit['MK_{0}'.format(Fit['conf'])]).mean)#all lat units
                elif Fit['conf'] == 'SFs':
                    z2.append(make_z(q2,t_0,pfit['MH_{0}_m{1}'.format('SF',mass)],pfit['MK_{0}'.format('SF')],pfit['MH_{0}_m{1}'.format(Fit['conf'],mass)],pfit['MK_{0}'.format(Fit['conf'])]).mean)
                else:
                    z2.append(make_z(q2,t_0,pfit['MH_{0}_m{1}'.format(Fit['conf'],mass)],pfit['MK_{0}'.format(Fit['conf'])]).mean)
            qsq,qsqerr = unmake_gvar_vec(qsq)
            z,zerr = unmake_gvar_vec(z)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(2,figsize=figsize)
            plt.plot(qsq2, y2, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            plt.figure(3,figsize=figsize)
            plt.errorbar(z2, y2, color=cols[j], mfc='none',linestyle=lines[i])
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
        if Fit['conf'] not in ['UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
        if Fit['conf' ] == 'F':
            F = Fit
        if Fit['conf' ] == 'SF':
            SF = Fit
        j = 0
        for mass in Fit['masses']:
            qsq = []
            z = []
            y = []
            y2 = []
            qsq2 = []
            z2 = []
            for twist in Fit['twists']:
                if fs_data[Fit['conf']]['fp_m{0}_tw{1}'.format(mass,twist)] != None:
                    q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                    qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                    if Fit['conf'] == 'Fs':
                        z.append(make_z(q2,t_0,F['M_parent_m{0}'.format(mass)],F['M_daughter'],Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))#all lat units
                    elif Fit['conf'] == 'SFs':
                        z.append(make_z(q2,t_0,SF['M_parent_m{0}'.format(mass)],SF['M_daughter'],Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))
                    else:
                        z.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))
                    y.append(fs_data[Fit['conf']]['fp_m{0}_tw{1}'.format(mass,twist)])
            q2max = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,Fit['twists'][1])]
            q2min = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,Fit['twists'][-1])]
            for q2 in np.linspace(q2min.mean,q2max.mean,nopts):
                y2.append(make_fp_BK(Nijk,Npow,Nm,addrho,pfit,Fit,q2,t_0,mass,fpf0same,float(mass),const2=const2).mean)
                qsq2.append((q2/Fit['a']**2).mean)
                if Fit['conf'] == 'Fs':
                    z2.append(make_z(q2,t_0,pfit['MH_{0}_m{1}'.format('F',mass)],pfit['MK_{0}'.format('F')],pfit['MH_{0}_m{1}'.format(Fit['conf'],mass)],pfit['MK_{0}'.format(Fit['conf'])]).mean)#all lat units
                elif Fit['conf'] == 'SFs':
                    z2.append(make_z(q2,t_0,pfit['MH_{0}_m{1}'.format('SF',mass)],pfit['MK_{0}'.format('SF')],pfit['MH_{0}_m{1}'.format(Fit['conf'],mass)],pfit['MK_{0}'.format(Fit['conf'])]).mean)
                else:
                    z2.append(make_z(q2,t_0,pfit['MH_{0}_m{1}'.format(Fit['conf'],mass)],pfit['MK_{0}'.format(Fit['conf'])]).mean)
            qsq,qsqerr = unmake_gvar_vec(qsq)
            z,zerr = unmake_gvar_vec(z)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(4,figsize=figsize)
            plt.plot(qsq2, y2, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            plt.figure(5,figsize=figsize)
            plt.errorbar(z2, y2, color=cols[j], mfc='none',linestyle=lines[i])
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
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=3)
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
    plt.ylim([0.2,3.4])
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
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=3)
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
    plt.ylim([0.2,3.4])
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
            y2 = []
            qsq2 = []
            z2 = []
            for twist in Fit['twists']:
                if fs_data[Fit['conf']]['fT_m{0}_tw{1}'.format(mass,twist)] != None:
                    q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                    qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                    z.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))#all lat units
                    y.append(fs_data[Fit['conf']]['fT_m{0}_tw{1}'.format(mass,twist)])
            q2max = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,Fit['twists'][1])]
            q2min = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,Fit['twists'][-1])]
            for q2 in np.linspace(q2min.mean,q2max.mean,nopts):
                y2.append(make_fT_BK(Nijk,Npow,Nm,addrho,pfit,Fit,q2,t_0,mass,fpf0same,float(mass)).mean)
                qsq2.append((q2/Fit['a']**2).mean)
                z2.append(make_z(q2,t_0,pfit['MH_{0}_m{1}'.format(Fit['conf'],mass)],pfit['MK_{0}'.format(Fit['conf'])]).mean)
                
            qsq,qsqerr = unmake_gvar_vec(qsq)
            z,zerr = unmake_gvar_vec(z)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(6,figsize=figsize)
            plt.plot(qsq2, y2, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            plt.figure(7,figsize=figsize)
            plt.plot(z2, y2, color=cols[j], mfc='none',linestyle=lines[i])
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
    plt.figure(6,figsize=figsize)
    plt.plot(qsq, ymean, color='g')
    plt.fill_between(qsq,ylow,yupp, color='g',alpha=alpha)
    #if datafTmaxBK != None and adddata:
    #    plt.errorbar(qsqmaxphysBK.mean, datafTmaxBK.mean, xerr=qsqmaxphysBK.sdev, yerr=datafTmaxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2)
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
    
    plt.figure(7,figsize=figsize)
    plt.plot(z,ymean, color='g')
    plt.fill_between(z,ylow,yupp, color='g',alpha=alpha)
    #if datafTmaxBK != None and adddata:
    #    plt.errorbar( make_z(qsqmaxphysBK,t_0,MBphys,MKphys).mean, datafTmaxBK.mean, xerr=make_z(qsqmaxphysBK,t_0,MBphys,MKphys).sdev, yerr=datafTmaxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=2)
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
        if Fit['conf'] not in ['UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
        if Fit['conf' ] == 'F':
            F = Fit
        if Fit['conf' ] == 'SF':
            SF = Fit
        fit = Fit['conf']
        j = 0
        for mass in Fit['masses']:
            qsq = []
            z = []
            y = []
            y2 = []
            qsq2 = []
            z2 = []
            for twist in Fit['twists']:
                q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                if Fit['conf'] == 'Fs':
                    z.append(make_z(q2,t_0,F['M_parent_m{0}'.format(mass)],F['M_daughter'],Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))#all lat units
                    MHs0 = F['M_parent_m{0}'.format(mass)] + F['a'] * Del
                elif Fit['conf'] == 'SFs':
                    z.append(make_z(q2,t_0,SF['M_parent_m{0}'.format(mass)],SF['M_daughter'],Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))
                    MHs0 = SF['M_parent_m{0}'.format(mass)] + SF['a'] * Del
                else:
                    z.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))
                    MHs0 = Fit['M_parent_m{0}'.format(mass)] + Fit['a'] * Del  
                pole = 1-(q2/MHs0**2)
                y.append(pole * fs_data[fit]['f0_m{0}_tw{1}'.format(mass,twist)])
            q2max = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,Fit['twists'][0])]
            q2min = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,Fit['twists'][-1])]
            if fit == 'Fs':
                MHs0 = pfit['MH_{0}_m{1}'.format('F',mass)] + pfit['a_{0}'.format('F')] * Del
            elif fit == 'SFs':
                MHs0 = pfit['MH_{0}_m{1}'.format('SF',mass)] + pfit['a_{0}'.format('SF')] * Del
            else:
                MHs0 = pfit['MH_{0}_m{1}'.format(fit,mass)] + pfit['a_{0}'.format(fit)] * Del
            for q2 in np.linspace(q2min.mean,q2max.mean,nopts):
                qsq2.append((q2/Fit['a']**2).mean)
                if Fit['conf'] == 'Fs':
                    z2.append(make_z(q2,t_0,pfit['MH_{0}_m{1}'.format('F',mass)],pfit['MK_{0}'.format('F')],pfit['MH_{0}_m{1}'.format(Fit['conf'],mass)],pfit['MK_{0}'.format(Fit['conf'])]).mean)#all lat units
                elif Fit['conf'] == 'SFs':
                    z2.append(make_z(q2,t_0,pfit['MH_{0}_m{1}'.format('SF',mass)],pfit['MK_{0}'.format('SF')],pfit['MH_{0}_m{1}'.format(Fit['conf'],mass)],pfit['MK_{0}'.format(Fit['conf'])]).mean)
                else:
                    z2.append(make_z(q2,t_0,pfit['MH_{0}_m{1}'.format(Fit['conf'],mass)],pfit['MK_{0}'.format(Fit['conf'])]).mean)
                pole = 1-(q2/MHs0**2)
                y2.append((pole*make_f0_BK(Nijk,Npow,Nm,addrho,pfit,Fit,q2,t_0,mass,fpf0same,float(mass))).mean)
            qsq,qsqerr = unmake_gvar_vec(qsq)
            z,zerr = unmake_gvar_vec(z)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(8,figsize=figsize)
            plt.plot(qsq2, y2, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            plt.figure(9,figsize=figsize)
            plt.plot(z2, y2, color=cols[j], mfc='none',linestyle=lines[i])
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
    plt.figure(8,figsize=figsize)
    plt.plot(qsq, ymean, color='b')
    plt.fill_between(qsq,ylow,yupp, color='b',alpha=alpha)
    #if dataf0maxBsEtas != None and adddata:
   #     pole = 1 - qsqmaxphysBK/(MBsphys+Del)**2
   #     plt.errorbar(qsqmaxphysBK.mean, (pole*dataf0maxBsEtas).mean, xerr=qsqmaxphysBK.sdev, yerr=(pole*dataf0maxBsEtas).sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$\mathrm{arXiv:} 1510.07446$')
    #if dataf00BsEtas != None and adddata:
   #     plt.errorbar(0, dataf00BsEtas.mean, yerr=dataf00BsEtas.sdev, color='k', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=3)
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
    
    plt.figure(9,figsize=figsize)
    plt.plot(z,ymean, color='b')
    plt.fill_between(z,ylow,yupp, color='b',alpha=alpha)
   # if dataf0maxBsEtas != None and adddata:
   #     pole = 1 - qsqmaxphysBK/(MBsphys+Del)**2
   #     plt.errorbar( make_z(qsqmaxphysBK,t_0,MBsphys,Metasphys).mean, (pole*dataf0maxBsEtas).mean, xerr=make_z(qsqmaxphysBK,t_0,MBsphys,Metasphys).sdev, yerr=(pole*dataf0maxBsEtas).sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$\mathrm{arXiv:} 1510.07446$')
   # if dataf00BsEtas != None and adddata:
  #      plt.errorbar(make_z(0,t_0,MBsphys,Metasphys).mean, dataf00BsEtas.mean,xerr = make_z(0,t_0,MBsphys,Metasphys).sdev ,yerr=dataf00BsEtas.sdev, color='k', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=3,loc='upper left')
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
        if Fit['conf'] not in ['UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
        if Fit['conf' ] == 'F':
            F = Fit
        if Fit['conf' ] == 'SF':
            SF = Fit
        fit = Fit['conf']
        j = 0
        for mass in Fit['masses']:
            qsq = []
            z = []
            y = []
            y2 = []
            qsq2 = []
            z2 = []
            for twist in Fit['twists']:
                if fs_data[Fit['conf']]['fp_m{0}_tw{1}'.format(mass,twist)] != None:
                    q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                    qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                    if Fit['conf'] == 'Fs':
                        z.append(make_z(q2,t_0,F['M_parent_m{0}'.format(mass)],F['M_daughter'],Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))#all lat units
                        MHsstar = make_MHsstar(F['M_parent_m{0}'.format(mass)],pfit,F['a'])
                    elif Fit['conf'] == 'SFs':
                        z.append(make_z(q2,t_0,SF['M_parent_m{0}'.format(mass)],SF['M_daughter'],Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))
                        MHsstar = make_MHsstar(SF['M_parent_m{0}'.format(mass)],pfit,SF['a'])
                    else:
                        z.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))
                        MHsstar = make_MHsstar(Fit['M_parent_m{0}'.format(mass)],pfit,Fit['a'])
                    pole = 1-(q2/MHsstar**2)
                    y.append(pole * fs_data[Fit['conf']]['fp_m{0}_tw{1}'.format(mass,twist)])
            q2max = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,Fit['twists'][1])]
            q2min = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,Fit['twists'][-1])]
            if fit == 'Fs':
                MHsstar = make_MHsstar(pfit['MH_{0}_m{1}'.format('F',mass)],pfit,pfit['a_{0}'.format('F')])
            elif fit == 'SFs':
                MHsstar = make_MHsstar(pfit['MH_{0}_m{1}'.format('SF',mass)],pfit,pfit['a_{0}'.format('SF')])
            else:
                MHsstar = make_MHsstar(pfit['MH_{0}_m{1}'.format(fit,mass)],pfit,pfit['a_{0}'.format(fit)])
            for q2 in np.linspace(q2min.mean,q2max.mean,nopts):
                qsq2.append((q2/Fit['a']**2).mean)
                if Fit['conf'] == 'Fs':
                    z2.append(make_z(q2,t_0,pfit['MH_{0}_m{1}'.format('F',mass)],pfit['MK_{0}'.format('F')],pfit['MH_{0}_m{1}'.format(Fit['conf'],mass)],pfit['MK_{0}'.format(Fit['conf'])]).mean)#all lat units
                elif Fit['conf'] == 'SFs':
                    z2.append(make_z(q2,t_0,pfit['MH_{0}_m{1}'.format('SF',mass)],pfit['MK_{0}'.format('SF')],pfit['MH_{0}_m{1}'.format(Fit['conf'],mass)],pfit['MK_{0}'.format(Fit['conf'])]).mean)
                else:
                    z2.append(make_z(q2,t_0,pfit['MH_{0}_m{1}'.format(Fit['conf'],mass)],pfit['MK_{0}'.format(Fit['conf'])]).mean)
                pole = 1-(q2/MHsstar**2)
                y2.append((pole*make_fp_BK(Nijk,Npow,Nm,addrho,pfit,Fit,q2,t_0,mass,fpf0same,float(mass),const2=const2)).mean)
            qsq,qsqerr = unmake_gvar_vec(qsq)
            z,zerr = unmake_gvar_vec(z)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(10,figsize=figsize)
            plt.plot(qsq2, y2, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            plt.figure(11,figsize=figsize)
            plt.plot(z2, y2, color=cols[j], mfc='none',linestyle=lines[i])
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
    plt.figure(10,figsize=figsize)
    plt.plot(qsq, ymean, color='r')
    plt.fill_between(qsq,ylow,yupp, color='r',alpha=alpha)
    #if datafpmaxBsEtas != None and adddata:
    #    pole = 1 - qsqmaxphysBK/MBsstarphys**2
    #    plt.errorbar(qsqmaxphysBK.mean, (pole*datafpmaxBsEtas).mean, xerr=qsqmaxphysBK.sdev, yerr=(pole*datafpmaxBsEtas).sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=3)
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
    plt.ylim([0.2,1.2])
    plt.tight_layout()
    plt.savefig('Plots/fpnopoleinqsq.pdf')
    plt.close()
    
    plt.figure(11,figsize=figsize)
    plt.plot(z,ymean, color='r')
    plt.fill_between(z,ylow,yupp, color='r',alpha=alpha)
    #if datafpmaxBsEtas != None and adddata:
    #    plt.errorbar( make_z(qsqmaxphysBK,t_0,MBsphys,Metasphys).mean, datafpmaxBsEtas.mean, xerr=make_z(qsqmaxphysBK,t_0,MBsphys,Metasphys).sdev, yerr=datafpmaxBsEtas.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')    
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=3)#,loc='upper right')
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
    plt.ylim([0.2,1.2])
    plt.tight_layout()
    plt.savefig('Plots/fpnopoleinz.pdf')
    plt.close()
    return()

###############################################################################

def fT_no_pole_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata):
    i = 0
    plotfits = []
    for Fit in Fits:
        if Fit['conf'] not in ['VCp','Cp','Fs','SFs','UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
        fit = Fit['conf']
        j = 0
        for mass in Fit['masses']:
            qsq = []
            z = []
            y = []
            y2 = []
            qsq2 = []
            z2 = []
            MHsstar = make_MHsstar(Fit['M_parent_m{0}'.format(mass)],pfit,Fit['a'])
            for twist in Fit['twists']:
                if fs_data[Fit['conf']]['fT_m{0}_tw{1}'.format(mass,twist)] != None:
                    q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                    qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                    z.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))#all lat units
                    pole = 1-(q2/MHsstar**2)
                    y.append(pole*fs_data[Fit['conf']]['fT_m{0}_tw{1}'.format(mass,twist)])
            q2max = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,Fit['twists'][1])]
            q2min = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,Fit['twists'][-1])]
            MHsstar = make_MHsstar(pfit['MH_{0}_m{1}'.format(fit,mass)],pfit,pfit['a_{0}'.format(fit)])
            for q2 in np.linspace(q2min.mean,q2max.mean,nopts):
                qsq2.append((q2/Fit['a']**2).mean)
                z2.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']).mean)
                pole = 1-(q2/MHsstar**2)
                y2.append((pole*make_fT_BK(Nijk,Npow,Nm,addrho,pfit,Fit,q2,t_0,mass,fpf0same,float(mass))).mean)
            qsq,qsqerr = unmake_gvar_vec(qsq)
            z,zerr = unmake_gvar_vec(z)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(12,figsize=figsize)
            plt.plot(qsq2, y2, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            plt.figure(13,figsize=figsize)
            plt.plot(z2, y2, color=cols[j], mfc='none',linestyle=lines[i])
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
        #        make_fp_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,fpf0same,amh)
        y.append(pole*make_fT_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(12,figsize=figsize)
    plt.plot(qsq, ymean, color='g')
    plt.fill_between(qsq,ylow,yupp, color='g',alpha=alpha)
    #if datafTmaxBK != None and adddata:
    #    plt.errorbar(qsqmaxphysBK.mean, datafTmaxBK.mean, xerr=qsqmaxphysBK.sdev, yerr=datafTmaxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False, ncol=3)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$\left(1-\frac{q^2}{M^2_{H_{s}^*}} \right)f_T(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig('Plots/fTnopoleinqsq.pdf')
    plt.close()
    
    plt.figure(13,figsize=figsize)
    plt.plot(z,ymean, color='g')
    plt.fill_between(z,ylow,yupp, color='g',alpha=alpha)
    #if datafTmaxBK != None and adddata:
    #    plt.errorbar( make_z(qsqmaxphysBK,t_0,MBphys,MKphys).mean, datafTmaxBK.mean, xerr=make_z(qsqmaxphysBK,t_0,MBphys,MKphys).sdev, yerr=datafTmaxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False, ncol=3)
    plt.xlabel('$z$',fontsize=fontsizelab)
    plt.ylabel(r'$\left(1-\frac{q^2}{M^2_{H_{s}^*}} \right)f_T(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig('Plots/fTnopoleinz.pdf')
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
    plt.figure(figsize=figsize)
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
    plt.figure(figsize=figsize)
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
    #plt.errorbar(MBsphys.mean,0.323, yerr=0.063,fmt='k*',ms=ms,mfc='none',label = r'$B_s\to{}K$ arXiv:1406.2279',lw=lw)#,capsize=capsize)
    #plt.errorbar(MBsphys.mean,0.819, yerr=0.021,fmt='b*',ms=ms,mfc='none',label = r'$B_s\to{}K$ arXiv:1406.2279',lw=lw)#,capsize=capsize)
    #plt.errorbar(MBsphys.mean,3.27, yerr=0.15,fmt='r*',ms=ms,mfc='none',label = r'$B_s\to{}K$ arXiv:1406.2279',lw=lw)#,capsize=capsize)
    #plt.errorbar(MBphys.mean-0.01,0.20, yerr=0.14,fmt='k^',ms=ms,mfc='none',label = r'$B\to{}\pi$ arXiv:1503.07839',lw=lw)#,capsize=capsize)
    #plt.errorbar(MBphys.mean,1.024, yerr=0.025,fmt='b^',ms=ms,mfc='none',label = r'$B\to{}\pi$ arXiv:1503.07839',lw=lw)#,capsize=capsize)
    #plt.errorbar(MBphys.mean-0.01,2.82, yerr=0.14,fmt='r^',ms=ms,mfc='none',label = r'$B\to{}\pi$ arXiv:1503.07839',lw=lw)#,capsize=capsize)
    plt.errorbar(MBphys.mean,0.319, yerr=0.066,fmt='ko',ms=ms,mfc='none',lw=lw,capsize=capsize,label='HPQCD')#,label = r'arXiv:1306.2384')
    plt.errorbar(MBphys.mean,0.861, yerr=0.048,fmt='bo',ms=ms,mfc='none',label = r'HPQCD',lw=lw,capsize=capsize)
    plt.errorbar(MBphys.mean,2.63, yerr=0.13,fmt='ro',ms=ms,mfc='none',label = r'HPQCD',lw=lw,capsize=capsize)
    plt.errorbar(MBphys.mean,2.39, yerr=0.17,fmt='o',color='purple',ms=ms,mfc='none',label = r'HPQCD',lw=lw,capsize=capsize)
    plt.errorbar(MBphys.mean,0.270, yerr=0.095,fmt='go',ms=ms,mfc='none',label = r'HPQCD',lw=lw,capsize=capsize)
    #plt.errorbar(MDphys.mean,0.612, yerr=0.035,fmt='kd',ms=ms,mfc='none',label = r'$D\to{}\pi$ arXiv:1706.03017',lw=lw)#,capsize=capsize)
    plt.errorbar(MDphys.mean,0.765, yerr=0.031,fmt='ks',ms=ms,mfc='none',label = r'ETMC',lw=lw,capsize=capsize)
    #plt.errorbar(MDphys.mean,1.134, yerr=0.049,fmt='bd',ms=ms,mfc='none',label = r'$D\to{}\pi$ arXiv:1706.03017',lw=lw)#,capsize=capsize)
    #plt.errorbar(MDphys.mean,2.130, yerr=0.096,fmt='rd',ms=ms,mfc='none',label = r'$D\to{}\pi$ arXiv:1706.03017',lw=lw)#,capsize=capsize)
    plt.errorbar(MDphys.mean,0.979, yerr=0.019,fmt='bs',ms=ms,mfc='none',label = r'ETMC',lw=lw,capsize=capsize)#,label = r'arXiv:1706.03017'
    plt.errorbar(MDphys.mean,1.336, yerr=0.054,fmt='rs',ms=ms,mfc='none',label = r'ETMC',lw=lw,capsize=capsize)
    plt.errorbar(MDphys.mean,0.687, yerr=0.054,fmt='g*',ms=ms,mfc='none',label = r'ETMC',lw=lw,capsize=capsize)#,label = r'arXiv:1803.04807'
    plt.errorbar(MDphys.mean,1.170, yerr=0.056,fmt='*',color='purple',ms=ms,mfc='none',label = r'ETMC',lw=lw,capsize=capsize)
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
    plt.figure(figsize=figsize)
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
    plt.ylabel('$10^9 d\mathcal{B}_{\mu}/dq^2~[\mathrm{GeV}^{-2}]$',fontsize=fontsizelab)
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
    plt.ylabel('$10^9 d\mathcal{B}_{\mu}/dq^2~[\mathrm{GeV}^{-2}]$',fontsize=fontsizelab)
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


def dBdq2_emup(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    exps = [CDF11,LHCb12B,LHCb14A,LHCb14C]
    m_lep = m_mu # use mu as most data muon and make no difference
    p = make_p_physical_point_BK(pfit,Fits)
    B = []
    qsq = []
    for q2 in np.linspace(4*m_lep**2,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        al,cl = make_al_cl(p,q2,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        B.append((2*al + 2*cl/3)*tauBpmGeV*1e7)
    Bmean,Berr = unmake_gvar_vec(B)
    Bupp,Blow = make_upp_low(B)
    plt.figure(figsize=figsize)
    plt.plot(qsq, Bmean, color='b')
    plt.fill_between(qsq,Blow,Bupp, color='b',alpha=alpha)
    cols = ['r','g','purple','m']
    for i,exp in enumerate(exps):
        x,xerr = exp.make_x()
        if exp.label == "LHCb '14C":
            y,yerr = exp.make_y(exp.binBep)
        else:
            y,yerr = exp.make_y(exp.binBmup)
        plt.errorbar(x,y,xerr=xerr, yerr=yerr, color=cols[i], fmt=exp.sym,ms=ms,label=(exp.label),capsize=capsize)
        
    plt.fill_between([8.68,10.11],[-5,-5],[45,45], color='k',alpha=alpha)
    plt.fill_between([12.86,14.18],[-5,-5],[45,45], color='k',alpha=alpha)
    plt.text((10.11+8.68)/2,0.05,r'$J/\Psi$',va='center',ha='center',fontsize=fontsizelab)
    plt.text((12.86+14.18)/2,0.05,r'$\Psi(2S)$',va='center',ha='center',fontsize=fontsizelab)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel('$10^7 d\mathcal{B}^{(+)}/dq^2~[\mathrm{GeV}^{-2}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    #plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper right')
    plt.axes().set_ylim([-0.02,0.42])
    plt.tight_layout()
    plt.savefig('Plots/dBdq2_emup.pdf')
    plt.close()
    
    return()

def dBdq2_emu0(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    exps = [CDF11,LHCb12A,LHCb14A2]
    m_lep = m_mu # use mu as most data muon and make no difference
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    B = []
    qsq = []
    for q2 in np.linspace(4*m_lep**2,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        al,cl = make_al_cl(p,q2,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        B.append((2*al + 2*cl/3)*tauB0GeV*1e7)
    Bmean,Berr = unmake_gvar_vec(B)
    Bupp,Blow = make_upp_low(B)
    plt.figure(figsize=figsize)
    plt.plot(qsq, Bmean, color='b')
    plt.fill_between(qsq,Blow,Bupp, color='b',alpha=alpha)
    cols = ['r','g','purple','m']
    for i,exp in enumerate(exps):
        x,xerr = exp.make_x()
        y,yerr = exp.make_y(exp.binBmu0)
        plt.errorbar(x,y,xerr=xerr, yerr=yerr, color=cols[i], fmt=exp.sym,ms=ms,label=(exp.label),capsize=capsize)
        
    plt.fill_between([8.68,10.11],[-5,-5],[45,45], color='k',alpha=alpha)
    plt.fill_between([12.86,14.18],[-5,-5],[45,45], color='k',alpha=alpha)
    plt.text((10.11+8.68)/2,0.6,r'$J/\Psi$',va='center',ha='center',fontsize=fontsizelab)
    plt.text((12.86+14.18)/2,0.6,r'$\Psi(2S)$',va='center',ha='center',fontsize=fontsizelab)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel('$10^7 d\mathcal{B}^{(0)}/dq^2~[\mathrm{GeV}^{-2}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    #plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper right')
    plt.axes().set_ylim([-0.1,0.65])
    plt.tight_layout()
    plt.savefig('Plots/dBdq2_emu0.pdf')
    plt.close()
    
    return()

def dBdq2_emu(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    exps = [BELLE09,BaBar12,CDF11]
    m_lep = m_mu # use mu as most data muon and make no difference
    B1 = []
    B2 = []
    B = []
    qsq = []
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    for q2 in np.linspace(4*m_lep**2,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        al,cl = make_al_cl(p,q2,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        B1.append((2*al + 2*cl/3)*tauB0GeV*1e7)
        
    p = make_p_physical_point_BK(pfit,Fits)
    for q2 in np.linspace(4*m_lep**2,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        al,cl = make_al_cl(p,q2,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        B2.append((2*al + 2*cl/3)*tauBpmGeV*1e7)
    for i in range(len(B1)):
        B.append((B1[i]+B2[i])/2)
    Bmean,Berr = unmake_gvar_vec(B)
    Bupp,Blow = make_upp_low(B)
    plt.figure(figsize=figsize)
    plt.plot(qsq, Bmean, color='b')
    plt.fill_between(qsq,Blow,Bupp, color='b',alpha=alpha)
    cols = ['r','g','purple','m']
    for i,exp in enumerate(exps):
        x,xerr = exp.make_x()
        if exp.label == "CDF '11":
            y,yerr = exp.make_y(exp.binBmu)
        else:
            y,yerr = exp.make_y(exp.binBemu)
        plt.errorbar(x,y,xerr=xerr, yerr=yerr, color=cols[i], fmt=exp.sym,ms=ms,label=(exp.label),capsize=capsize)
        
    plt.fill_between([8.68,10.11],[-5,-5],[45,45], color='k',alpha=alpha)
    plt.fill_between([12.86,14.18],[-5,-5],[45,45], color='k',alpha=alpha)
    plt.text((10.11+8.68)/2,0.1,r'$J/\Psi$',va='center',ha='center',fontsize=fontsizelab)
    plt.text((12.86+14.18)/2,0.1,r'$\Psi(2S)$',va='center',ha='center',fontsize=fontsizelab)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel('$10^7 d\mathcal{B}/dq^2~[\mathrm{GeV}^{-2}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    #plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper right')
    plt.axes().set_ylim([-0.01,0.51])
    plt.tight_layout()
    plt.savefig('Plots/dBdq2_emu.pdf')
    plt.close()
    
    return()
###########################################################################################

def dBdq2_the_p(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    m_lep = m_mu # use mu as most data muon and make no difference
    p = make_p_physical_point_BK(pfit,Fits)
    B = []
    qsq = []
    for q2 in np.linspace(4*m_lep**2,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        al,cl = make_al_cl(p,q2,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        B.append((2*al + 2*cl/3)*tauBpmGeV*1e7)
    Bmean,Berr = unmake_gvar_vec(B)
    Bupp,Blow = make_upp_low(B)
    plt.figure(figsize=figsize)
    plt.plot(qsq, Bmean, color='b')
    plt.fill_between(qsq,Blow,Bupp, color='b',alpha=alpha)
    cols = ['r','g','purple','m']
    thes = [Bobeth07,Bobeth11,Bobeth16,Fermi16A]
    for i,the in enumerate(thes):
        x,xerr = the.make_x()
        if the.label in ["Bobeth et al. '07","Bobeth '11"]:
            y,yerr = the.add_up_down_errs(the.fix_B_bins(the.binBmum))
        elif the.label in ["FNAL/MILC '15"]:
            y,yerr = the.add_errs(the.fix_B_bins(the.binBmup))
        elif the.label in ["Bobeth et al. '12"]:
            y,yerr = the.add_up_down_errs(the.fix_B_bins(the.binBemum))
        plt.errorbar(x,y,xerr=xerr, yerr=yerr, color=cols[i],fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
        
    plt.fill_between([8.68,10.11],[-5,-5],[45,45], color='k',alpha=alpha)
    plt.fill_between([12.86,14.18],[-5,-5],[45,45], color='k',alpha=alpha)
    plt.text((10.11+8.68)/2,0.05,r'$J/\Psi$',va='center',ha='center',fontsize=fontsizelab)
    plt.text((12.86+14.18)/2,0.05,r'$\Psi(2S)$',va='center',ha='center',fontsize=fontsizelab)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel('$10^7 d\mathcal{B}^{(\pm)}/dq^2~[\mathrm{GeV}^{-2}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    #plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper right')
    plt.axes().set_ylim([-0.02,0.5])
    plt.tight_layout()
    plt.savefig('Plots/dBdq2_the_p.pdf')
    plt.close()
    
    return()

def dBdq2_the_0(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    m_lep = m_mu # use mu as most data muon and make no difference
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    B = []
    qsq = []
    for q2 in np.linspace(4*m_lep**2,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        al,cl = make_al_cl(p,q2,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        B.append((2*al + 2*cl/3)*tauB0GeV*1e7)
    Bmean,Berr = unmake_gvar_vec(B)
    Bupp,Blow = make_upp_low(B)
    plt.figure(figsize=figsize)
    plt.plot(qsq, Bmean, color='b')
    plt.fill_between(qsq,Blow,Bupp, color='b',alpha=alpha)
    cols = ['r','g','purple','m','c']
    thes = [Bobeth07,Bobeth11,Khod,Fermi16A,Bobeth16]
    for i,the in enumerate(thes):
        x,xerr = the.make_x()
        if the.label in ["Bobeth et al. '07","Bobeth '11","Khod. et al. '12"]:
            y,yerr = the.add_up_down_errs(the.fix_B_bins(the.binBmu0))
        elif the.label in ["FNAL/MILC '15"]:
            y,yerr = the.add_errs(the.fix_B_bins(the.binBmu0))
        elif the.label in ["Bobeth et al. '12"]:
            y,yerr = the.add_up_down_errs(the.fix_B_bins(the.binBemu0))
        plt.errorbar(x,y,xerr=xerr, yerr=yerr, color=cols[i],fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    plt.fill_between([8.68,10.11],[-5,-5],[45,45], color='k',alpha=alpha)
    plt.fill_between([12.86,14.18],[-5,-5],[45,45], color='k',alpha=alpha)
    plt.text((10.11+8.68)/2,0.1,r'$J/\Psi$',va='center',ha='center',fontsize=fontsizelab)
    plt.text((12.86+14.18)/2,0.1,r'$\Psi(2S)$',va='center',ha='center',fontsize=fontsizelab)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel('$10^7 d\mathcal{B}^{(0)}/dq^2~[\mathrm{GeV}^{-2}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    #plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper right')
    plt.axes().set_ylim([-0.01,0.49])
    plt.tight_layout()
    plt.savefig('Plots/dBdq2_the_0.pdf')
    plt.close()
    
    return()

def dBdq2_the(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    m_lep = m_mu # use mu as most data muon and make no difference
    B1 = []
    B2 = []
    B = []
    qsq = []
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    for q2 in np.linspace(4*m_lep**2,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        al,cl = make_al_cl(p,q2,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        B1.append((2*al + 2*cl/3)*tauB0GeV*1e7)
        
    p = make_p_physical_point_BK(pfit,Fits)
    for q2 in np.linspace(4*m_lep**2,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        al,cl = make_al_cl(p,q2,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        B2.append((2*al + 2*cl/3)*tauBpmGeV*1e7)
    for i in range(len(B1)):
        B.append((B1[i]+B2[i])/2)
    Bmean,Berr = unmake_gvar_vec(B)
    Bupp,Blow = make_upp_low(B)
    plt.figure(figsize=figsize)
    plt.plot(qsq, Bmean, color='b')
    plt.fill_between(qsq,Blow,Bupp, color='b',alpha=alpha)
    cols = ['r','g','purple','m']
    thes = [HPQCD14A,Altmann12]
    for i,the in enumerate(thes):
        x,xerr = the.make_x()
        y,yerr = the.add_errs(the.fix_B_bins(the.binBemu))
        plt.errorbar(x,y,xerr=xerr, yerr=yerr, color=cols[i],fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    plt.fill_between([8.68,10.11],[-5,-5],[45,45], color='k',alpha=alpha)
    plt.fill_between([12.86,14.18],[-5,-5],[45,45], color='k',alpha=alpha)
    plt.text((10.11+8.68)/2,0.1,r'$J/\Psi$',va='center',ha='center',fontsize=fontsizelab)
    plt.text((12.86+14.18)/2,0.1,r'$\Psi(2S)$',va='center',ha='center',fontsize=fontsizelab)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel('$10^7 d\mathcal{B}/dq^2~[\mathrm{GeV}^{-2}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    #plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper right')
    plt.axes().set_ylim([-0.01,0.49])
    plt.tight_layout()
    plt.savefig('Plots/dBdq2_the.pdf')
    plt.close()
    
    return()

#############################################################################################################

def dBdq2_the_tau(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    m_lep = m_tau # use mu as most data muon and make no difference
    B1 = []
    B2 = []
    B = []
    qsq = []
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    for q2 in np.linspace(4*m_lep**2,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        al,cl = make_al_cl(p,q2,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        B1.append((2*al + 2*cl/3)*tauB0GeV*1e7)
        
    p = make_p_physical_point_BK(pfit,Fits)
    for q2 in np.linspace(4*m_lep**2,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        al,cl = make_al_cl(p,q2,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        B2.append((2*al + 2*cl/3)*tauBpmGeV*1e7)
    for i in range(len(B1)):
        B.append((B1[i]+B2[i])/2)
    Bmean,Berr = unmake_gvar_vec(B)
    Bupp,Blow = make_upp_low(B)
    plt.figure(figsize=figsize)
    #plt.plot(qsq, Bmean, color='purple')
    #plt.fill_between(qsq,Blow,Bupp, color='purple',alpha=alpha)
    B1mean,B1err = unmake_gvar_vec(B1)
    B1upp,B1low = make_upp_low(B1)
    plt.plot(qsq, B1mean, color='b')
    plt.fill_between(qsq,B1low,B1upp, color='b',alpha=alpha)
    B2mean,B2err = unmake_gvar_vec(B2)
    B2upp,B2low = make_upp_low(B2)
    plt.plot(qsq, B2mean, color='r')
    plt.fill_between(qsq,B2low,B2upp, color='r',alpha=alpha)

    the = HPQCD14B
    x,xerr = the.make_x()
    y,yerr = the.add_errs(the.fix_B_bins(the.binBtau))
    plt.errorbar(x,y,xerr=xerr, yerr=yerr, color='purple',fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)

    the = Fermi16C
    x,xerr = the.make_x()
    y,yerr = the.add_errs(the.fix_B_bins(the.binBtaup))
    plt.errorbar(x,y,xerr=xerr, yerr=yerr, color='r',fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)

    the = Fermi16C
    x,xerr = the.make_x()
    y,yerr = the.add_errs(the.fix_B_bins(the.binBtau0))
    plt.errorbar(x,y,xerr=xerr, yerr=yerr, color='b',fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    
    #plt.fill_between([8.68,10.11],[-5,-5],[45,45], color='k',alpha=alpha)
    plt.fill_between([12.86,14.18],[-5,-5],[45,45], color='k',alpha=alpha)
    #plt.text((10.11+8.68)/2,0.1,r'$J/\Psi$',va='center',ha='center',fontsize=fontsizelab)
    plt.text((12.86+14.18)/2,0.05,r'$\Psi(2S)$',va='center',ha='center',fontsize=fontsizelab)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$10^7 d\mathcal{B}_{\tau}/dq^2~[\mathrm{GeV}^{-2}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    #plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=1,loc='lower center')
    plt.axes().set_ylim([0,0.25])
    plt.axes().set_xlim([12.5,23])
    plt.tight_layout()
    plt.savefig('Plots/dBdq2_the_tau.pdf')
    plt.close()
    
    return()

#############################################################################################################

def B_exp_plots(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    exps = [BELLE09,BaBar09,BaBar12,BaBar16,CDF11,LHCb12A,LHCb12B,LHCb14A,LHCb14A2,LHCb14B,LHCb14C]
    m_lep = m_mu
    qsq_min = 4*m_lep**2
    qsq_max = qsqmaxphysBK.mean
    ############### p ########
    numbs = [7,2,9,6,1,8,3,4,5,10]
    labs = []
    i = 0
    plt.figure(figsize=figsize)
    for exp in exps:
        if exp.Bep != []:
            labs.append(exp.label)
            x,xerr = exp.make_1_y(exp.Bep)
            plt.errorbar(x,numbs[i], xerr=xerr,ms=ms,fmt=exp.sym,color='r' ,capsize=capsize,lw=lw)
            i += 1
        if exp.Bmup != []:
            labs.append(exp.label)
            x,xerr = exp.make_1_y(exp.Bmup)
            plt.errorbar(x,numbs[i], xerr=xerr,ms=ms,fmt=exp.sym,color='b' ,capsize=capsize,lw=lw)
            i += 1
        if exp.Bemup != []:
            labs.append(exp.label)
            x,xerr = exp.make_1_y(exp.Bemup)
            plt.errorbar(x,numbs[i], xerr=xerr,ms=ms,fmt=exp.sym,color='purple' ,capsize=capsize,lw=lw)
            i += 1

    plt.plot([1,8],[5.5,5.5],color='k')
    plt.plot([1,8],[7.5,7.5],color='k')
    p = make_p_physical_point_BK(pfit,Fits)
    res = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)*tauBpmGeV*1e7
    res2 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250,gaps=True)*tauBpmGeV*1e7
    tau_res = integrate_Gamma(p,4*m_tau**2,qsq_max,m_tau,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=150)*tauBpmGeV*1e7
    tau_x,tau_xerr = BaBar16.make_1_y(BaBar16.Btaup)
    print('Btau',tau_res,'BaBar16','{0}+{1}-{2}'.format(tau_x[0]*1e7,tau_xerr[1][0]*1e7,tau_xerr[0][0]*1e7))
    plt.errorbar(res.mean,10, xerr=res.sdev,ms=ms,fmt='*',color='k',capsize=capsize,lw=lw)
    plt.fill_between([res.mean-res.sdev,res.mean+res.sdev],[0,0],[11,11], color='k',alpha=alpha/2)
    plt.fill_between([res2.mean-res2.sdev,res2.mean+res2.sdev],[0,0],[11,11], edgecolor='k',fc='none',alpha=alpha,hatch='X')    
    labs.append("HPQCD '21")
    plt.text(2.3,6.5,r'$B^+\to{}K^+e^+e^-$',fontsize=fontsizelab, va='center')
    plt.text(2.3,3,r'$B^+\to{}K^+\mu^+\mu^-$',fontsize=fontsizelab, va='center')
    plt.text(2.3,9,r'$B^+\to{}K^+\ell^+\ell^-$',fontsize=fontsizelab, va='center')
    plt.xlabel(r'$10^{7}\mathcal{B}^{(+)}$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=False,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('none')
    plt.xlim([2.2,6.8])
    plt.ylim([0.5,10.5])
    plt.gca().set_yticks(numbs)
    plt.gca().set_yticklabels(labs)
    plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.5))
    #plt.text(1.01,8.5,r'$\Gamma$',fontsize=fontsizelab*2,va='top',ha='right')
    #plt.plot([1,1.1],[7.5,7.5],color='k')
    #plt.plot([1,1],[7.5,8.5],color='k')
    plt.tight_layout()
    plt.savefig('Plots/Bp_exp.pdf')
    plt.close()

    ########################## 0 ##########################################################
    numbs = [7,2,9,6,1,8,3,4,5,10]
    labs = []
    i = 0
    plt.figure(figsize=figsize)
    for exp in exps:
        if exp.Be0 != []:
            labs.append(exp.label)
            x,xerr = exp.make_1_y(exp.Be0)
            plt.errorbar(x,numbs[i], xerr=xerr,ms=ms,fmt=exp.sym,color='r' ,capsize=capsize,lw=lw)
            i += 1
        if exp.Bmu0 != []:
            labs.append(exp.label)
            x,xerr = exp.make_1_y(exp.Bmu0)
            plt.errorbar(x,numbs[i], xerr=xerr,ms=ms,fmt=exp.sym,color='b' ,capsize=capsize,lw=lw)
            i += 1
        if exp.Bemu0 != []:
            labs.append(exp.label)
            x,xerr = exp.make_1_y(exp.Bemu0)
            plt.errorbar(x,numbs[i], xerr=xerr,ms=ms,fmt=exp.sym,color='purple' ,capsize=capsize,lw=lw)
            i += 1

    plt.plot([-2,11],[7.5,7.5],color='k')
    plt.plot([-2,11],[5.5,5.5],color='k')
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    res = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)*tauB0GeV*1e7
    res2 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250,gaps=True)*tauB0GeV*1e7
    plt.errorbar(res.mean,10, xerr=res.sdev,ms=ms,fmt='*',color='k',capsize=capsize,lw=lw)
    plt.fill_between([res.mean-res.sdev,res.mean+res.sdev],[0,0],[11,11], color='k',alpha=alpha/2)
    plt.fill_between([res2.mean-res2.sdev,res2.mean+res2.sdev],[0,0],[11,11], edgecolor='k',fc='none',alpha=alpha,hatch='X') 
    labs.append("HPQCD '21")
    plt.text(6.15,6.5,r'$B^0\to{}K^0e^+e^-$',fontsize=fontsizelab, va='center')
    plt.text(6.15,3.0,r'$B^0\to{}K^0\mu^+\mu^-$',fontsize=fontsizelab, va='center')
    plt.text(6.15,9,r'$B^0\to{}K^0\ell^+\ell^-$',fontsize=fontsizelab, va='center')
    plt.xlabel(r'$10^{7}\mathcal{B}^{(0)}$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=False,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('none')
    plt.xlim([-0.5,9.3])
    plt.ylim([0.5,10.5])
    plt.gca().set_yticks(numbs)
    plt.gca().set_yticklabels(labs)
    plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.5))
    #plt.text(1.01,8.5,r'$\Gamma$',fontsize=fontsizelab*2,va='top',ha='right')
    #plt.plot([1,1.1],[7.5,7.5],color='k')
    #plt.plot([1,1],[7.5,8.5],color='k')
    plt.tight_layout()
    plt.savefig('Plots/B0_exp.pdf')
    plt.close()

    ########################## both  ##########################################################
    numbs = [5,2,7,4,1,6,8,3,9]
    labs = []
    i = 0
    plt.figure(figsize=figsize)
    for exp in exps:
        if exp.Be != []:
            labs.append(exp.label)
            x,xerr = exp.make_1_y(exp.Be)
            plt.errorbar(x,numbs[i], xerr=xerr,ms=ms,fmt=exp.sym,color='r' ,capsize=capsize,lw=lw)
            i += 1
        if exp.Bmu != []:
            labs.append(exp.label)
            x,xerr = exp.make_1_y(exp.Bmu)
            plt.errorbar(x,numbs[i], xerr=xerr,ms=ms,fmt=exp.sym,color='b' ,capsize=capsize,lw=lw)
            i += 1
        if exp.Bemu != []:
            labs.append(exp.label)
            x,xerr = exp.make_1_y(exp.Bemu)
            plt.errorbar(x,numbs[i], xerr=xerr,ms=ms,fmt=exp.sym,color='purple' ,capsize=capsize,lw=lw)
            i += 1

    plt.plot([-2,11],[5.5,5.5],color='k')
    plt.plot([-2,11],[3.5,3.5],color='k')
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    res1 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)*tauB0GeV*1e7
    resB1 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250,gaps=True)*tauB0GeV*1e7
    p = make_p_physical_point_BK(pfit,Fits)
    res2 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)*tauBpmGeV*1e7
    resB2 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250,gaps=True)*tauBpmGeV*1e7
    res = (res1+res2)/2
    resB = (resB1+resB2)/2
    plt.errorbar(res.mean,9, xerr=res.sdev,ms=ms,fmt='*',color='k',capsize=capsize,lw=lw)
    plt.fill_between([res.mean-res.sdev,res.mean+res.sdev],[0,0],[11,11], color='k',alpha=alpha/2)
    plt.fill_between([resB.mean-resB.sdev,resB.mean+resB.sdev],[0,0],[11,11], edgecolor='k',fc='none',alpha=alpha,hatch='X')
    labs.append("HPQCD '21")
    plt.text(1.55,4.5,r'$B\to{}Ke^+e^-$',fontsize=fontsizelab, va='center')
    plt.text(1.55,2,r'$B\to{}K\mu^+\mu^-$',fontsize=fontsizelab, va='center')
    plt.text(1.55,7.5,r'$B\to{}K\ell^+\ell^-$',fontsize=fontsizelab, va='center')
    plt.xlabel(r'$10^{7}\mathcal{B}$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=False,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('none')
    plt.xlim([1.5,6.6])
    plt.ylim([0.5,9.5])
    plt.gca().set_yticks(numbs)
    plt.gca().set_yticklabels(labs)
    plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.5))
    #plt.text(1.01,8.5,r'$\Gamma$',fontsize=fontsizelab*2,va='top',ha='right')
    #plt.plot([1,1.1],[7.5,7.5],color='k')
    #plt.plot([1,1],[7.5,8.5],color='k')
    plt.tight_layout()
    plt.savefig('Plots/B_exp.pdf')
    plt.close()
    return()

############################################################################################################

def B_the_plots(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    m_lep = m_mu
    qsq_min = 4*m_lep**2
    qsq_max = qsqmaxphysBK.mean
    ############### e,mu ########
    numbs = [9,7,4,1,6,3,8,5,2,10]
    labs = []
    i = 0
    plt.figure(figsize=figsize)
    the = HPQCD14A
    #labs.append(the.label)
    #x,xerr = the.add_errs(the.Be)
    #plt.errorbar(x,1, xerr=xerr,ms=ms,fmt=the.sym,color='purple' ,capsize=capsize,lw=lw)
    labs.append(the.label)
    x,xerr = the.add_errs(the.Bmu)
    plt.errorbar(x,9, xerr=xerr,ms=ms,fmt=the.sym,color='purple' ,capsize=capsize,lw=lw)
    
    the = Fermi16A
    labs.append(the.label)
    x,xerr = the.add_errs(the.Bmup)
    plt.errorbar(x,7, xerr=xerr,ms=ms,fmt=the.sym,color='r' ,capsize=capsize,lw=lw)
    labs.append(the.label)
    x,xerr = the.add_errs(the.Bmu0)
    plt.errorbar(x,4, xerr=xerr,ms=ms,fmt=the.sym,color='b' ,capsize=capsize,lw=lw)

    the = Bobeth11
    labs.append(the.label)
    x,xerr = the.add_up_down_errs(the.Bemum)# only from 14.18 upwards
    plt.errorbar(x,1, xerr=xerr,ms=ms,fmt=the.sym,color='r' ,capsize=capsize,lw=lw)

    the = Wang12
    labs.append(the.label)
    x,xerr = the.add_up_down_errs(the.Bemum)
    plt.errorbar(x,6, xerr=xerr,ms=ms,fmt=the.sym,color='r' ,capsize=capsize,lw=lw)
    labs.append(the.label)
    x,xerr = the.add_up_down_errs(the.Bemu0)
    plt.errorbar(x,3, xerr=xerr,ms=ms,fmt=the.sym,color='b' ,capsize=capsize,lw=lw)
    
    
    plt.plot([0,8],[2.5,2.5],color='k')
    plt.plot([0,8],[5.5,5.5],color='k')
    plt.plot([0,8],[8.5,8.5],color='k')
    p = make_p_physical_point_BK(pfit,Fits)
    res = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)*tauBpmGeV*1e7
    #res2 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250,gaps=True)*tauBpmGeV*1e7
    res0 = res
    res1418 = integrate_Gamma(p,14.18,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)*tauBpmGeV*1e7
    plt.errorbar(res.mean,8, xerr=res.sdev,ms=ms,fmt='*',color='r',capsize=capsize,lw=lw)
    #plt.fill_between([res.mean-res.sdev,res.mean+res.sdev],[2.5,2.5],[11,11], color='r',alpha=alpha/2)
    labs.append("HPQCD '21")
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    res = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)*tauB0GeV*1e7
    #res2 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250,gaps=True)*tauBpmGeV*1e7
    res0 = (res0+res)/2
    plt.errorbar(res.mean,5, xerr=res.sdev,ms=ms,fmt='*',color='b',capsize=capsize,lw=lw)
    #plt.fill_between([res.mean-res.sdev,res.mean+res.sdev],[2.5,2.5],[11,11], color='b',alpha=alpha/2)
    labs.append("HPQCD '21")
    plt.errorbar(res1418.mean,2, xerr=res1418.sdev,ms=ms,fmt='*',color='r',capsize=capsize,lw=lw)
    #plt.fill_between([res1418.mean-res1418.sdev,res1418.mean+res1418.sdev],[0,0],[2.5,2.5], color='r',alpha=alpha/2)    
    labs.append("HPQCD '21")
    plt.errorbar(res0.mean,10, xerr=res0.sdev,ms=ms,fmt='*',color='purple',capsize=capsize,lw=lw)
    #plt.fill_between([res0.mean-res0.sdev,res0.mean+res0.sdev],[2.5,2.5],[11,11], color='b',alpha=alpha/2)
    labs.append("HPQCD '21")
    plt.text(1.9,4,r'$B^0\to{}K^0\ell^+\ell^-$',fontsize=fontsizelab, va='center')
    plt.text(1.9,7,r'$B^{\pm}\to{}K^{\pm}\ell^+\ell^-$',fontsize=fontsizelab, va='center')
    plt.text(1.9,9.5,r'$B\to{}K\mu^+\mu^-$',fontsize=fontsizelab, va='center')
    plt.text(1.9,1.5,r'$B^-\to{}K^-\ell^+\ell^-_{(14.18\mathrm{GeV}^2,q^2_{\mathrm{max}})}$',fontsize=fontsizelab, va='center')
    plt.xlabel(r'$10^{7}\mathcal{B}$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=False,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('none')
    plt.xlim([0.6,7.5])
    plt.ylim([0.5,10.5])
    plt.gca().set_yticks(numbs)
    plt.gca().set_yticklabels(labs)
    plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.5))
    plt.tight_layout()
    plt.savefig('Plots/Bemu_the.pdf')
    plt.close()

    ########################## tau ##########################################################
    m_lep = m_tau
    qsq_min = 4*m_lep**2
    qsq_max = qsqmaxphysBK.mean
    numbs = [3,9,6,1,8,5,10,2,7,4]
    labs = []
    i = 0
    plt.figure(figsize=figsize)
    
    the = HPQCD14A #14.18
    labs.append(the.label)
    x,xerr = the.add_errs(the.Btau)
    plt.errorbar(x,3, xerr=xerr,ms=ms,fmt=the.sym,color='purple' ,capsize=capsize,lw=lw)
    #labs.append(the.label)
    #x,xerr = the.add_errs(the.Bmu)
    #plt.errorbar(x,7, xerr=xerr,ms=ms,fmt=the.sym,color='purple' ,capsize=capsize,lw=lw)
    
    the = Fermi16A
    labs.append(the.label)
    x,xerr = the.add_errs(the.Btaup)
    plt.errorbar(x,9, xerr=xerr,ms=ms,fmt=the.sym,color='r' ,capsize=capsize,lw=lw)
    labs.append(the.label)
    x,xerr = the.add_errs(the.Btau0)
    plt.errorbar(x,6, xerr=xerr,ms=ms,fmt=the.sym,color='b' ,capsize=capsize,lw=lw)

    the = Bobeth11
    labs.append(the.label)
    x,xerr = the.add_up_down_errs(the.Btaum)# only from 14.18 upwards
    plt.errorbar(x,1, xerr=xerr,ms=ms,fmt=the.sym,color='r' ,capsize=capsize,lw=lw)

    the = Wang12
    labs.append(the.label)
    x,xerr = the.add_up_down_errs(the.Btaum)
    plt.errorbar(x,8, xerr=xerr,ms=ms,fmt=the.sym,color='r' ,capsize=capsize,lw=lw)
    labs.append(the.label)
    x,xerr = the.add_up_down_errs(the.Btau0)
    plt.errorbar(x,5, xerr=xerr,ms=ms,fmt=the.sym,color='b' ,capsize=capsize,lw=lw)
    
    plt.plot([0,8],[2.5,2.5],color='k')
    plt.plot([0,8],[4.5,4.5],color='k')
    plt.plot([0,8],[7.5,7.5],color='k')
    p = make_p_physical_point_BK(pfit,Fits)
    res = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)*tauBpmGeV*1e7
    #res2 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250,gaps=True)*tauBpmGeV*1e7
    res1418 = integrate_Gamma(p,14.18,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)*tauBpmGeV*1e7
    plt.errorbar(res.mean,10, xerr=res.sdev,ms=ms,fmt='*',color='r',capsize=capsize,lw=lw)
    #plt.fill_between([res.mean-res.sdev,res.mean+res.sdev],[2.5,2.5],[11,11], color='r',alpha=alpha/2)
    plt.errorbar(res1418.mean,2, xerr=res1418.sdev,ms=ms,fmt='*',color='r',capsize=capsize,lw=lw)
    #plt.fill_between([res1418.mean-res1418.sdev,res1418.mean+res1418.sdev],[0,0],[2.5,2.5], color='r',alpha=alpha/2)  
    labs.append("HPQCD '21")
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    res = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)*tauB0GeV*1e7
    res1418b = integrate_Gamma(p,14.18,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)*tauB0GeV*1e7
    #res2 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250,gaps=True)*tauBpmGeV*1e7
    plt.errorbar(res.mean,7, xerr=res.sdev,ms=ms,fmt='*',color='b',capsize=capsize,lw=lw)
    #plt.fill_between([res.mean-res.sdev,res.mean+res.sdev],[2.5,2.5],[11,11], color='b',alpha=alpha/2)
    res1418 = (res1418+res1418b)/2
    plt.errorbar(res1418.mean,4, xerr=res1418.sdev,ms=ms,fmt='*',color='purple',capsize=capsize,lw=lw)
    #plt.fill_between([res1418.mean-res1418.sdev,res1418.mean+res1418.sdev],[0,0],[2.5,2.5], color='b',alpha=alpha/2) 
    labs.append("HPQCD '21")  
    labs.append("HPQCD '21")
    labs.append("HPQCD '21")
    plt.text(0.72,6,r'$B^0\to{}K^0\tau^+\tau^-$',fontsize=fontsizelab*0.9, va='center')
    plt.text(0.72,9,r'$B^{\pm}\to{}K^{\pm}\tau^+\tau^-$',fontsize=fontsizelab*0.9, va='center')
    plt.text(0.72,3.8,r'$B\to{}K\tau^+\tau^-_{(14.18\mathrm{GeV}^2,q^2_{\mathrm{max}})}$',fontsize=fontsizelab*0.9, va='center')
    plt.text(0.72,1.8,r'$B^-\to{}K^-\tau^+\tau^-_{(14.18\mathrm{GeV}^2,q^2_{\mathrm{max}})}$',fontsize=fontsizelab*0.9, va='center')
    plt.xlabel(r'$10^{7}\mathcal{B}$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=False,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('none')
    plt.xlim([0.7,1.9])
    plt.ylim([0.5,10.5])
    plt.gca().set_yticks(numbs)
    plt.gca().set_yticklabels(labs)
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig('Plots/Btau_the.pdf')
    plt.close()
    return()

############################################################################################################

def invert_y(y,yerr):
    newy = []
    newyerr = []
    for i in range(len(y)):
        temp = gv.gvar(y[i],yerr[i])+1
        newy.append((1/temp).mean)
        newyerr.append((1/temp).sdev)
    return(newy,newyerr)

def convert_Remu_bob(y,yerr):
    newy = []
    newyerrupp = []
    newyerrlow = []
    for i in range(len(y)):
        newy.append((y[i]-1)*1000)
        newyerrlow.append(yerr[0][i]*1000)
        newyerrupp.append(yerr[1][i]*1000)
    newyerr = [newyerrlow,newyerrupp]
    return(newy,newyerr)

############################################################################################################

def nu_in_qsq(pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,const2):
    qsq = []
    y0 = []
    yp = []
    p = make_p_physical_point_BK(pfit,Fits)
    for q2 in np.linspace(0,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        p3 = ((q2-p['MKphys']**2-p['MBphys']**2)**2/(4*p['MBphys']**2)-p['MKphys']**2)**(3/2)
        if math.isnan(p3.mean):
            p3 = 0
        fp = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
        yp.append(1e7*(tauBpmGeV * VtbVts**2 * GF**2 * alphaEW**2 * Xt**2 * p3 * fp**2) /(32 * (np.pi)**5 * sinthw2**2)) #only need one fit
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    for q2 in np.linspace(0,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        p3 = ((q2-p['MKphys']**2-p['MBphys']**2)**2/(4*p['MBphys']**2)-p['MKphys']**2)**(3/2)
        if math.isnan(p3.mean):
            p3 = 0
        fp = make_fp_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0,const2=const2)
        y0.append(1e7*(tauB0GeV * VtbVts**2 * GF**2 * alphaEW**2 * Xt**2 * p3 * fp**2) /(32 * (np.pi)**5 * sinthw2**2))
    y0mean,y0err = unmake_gvar_vec(y0)
    y0upp,y0low = make_upp_low(y0)
    ypmean,yperr = unmake_gvar_vec(yp)
    ypupp,yplow = make_upp_low(yp)
    plt.figure(figsize=figsize)
    plt.plot(qsq, y0mean, color='b',linestyle='-',label=r'$B^0\to K^0\nu\bar{\nu}_{~\mathrm{SD}}$')
    plt.fill_between(qsq,y0low,y0upp, color='b',alpha=alpha)
    plt.plot(qsq, ypmean, color='r',linestyle='-',label=r'$B^+\to K^+\nu\bar{\nu}_{~\mathrm{SD}}$')
    plt.fill_between(qsq,yplow,ypupp, color='r',alpha=alpha)

    handles, labels = plt.gca().get_legend_handles_labels()
    #handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,loc='lower left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$10^7\times d\mathcal{B}/dq^2~[\mathrm{GeV}^{-2}]$',fontsize=fontsizelab)
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
    plt.savefig('Plots/nuinqsq.pdf')
    plt.close()
    return()

###########################################################################################################
def Rbybin_the(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    #m_lep = m_tau # use mu as most data muon and make no difference
    ##########tau ###############
    plt.figure(figsize=figsize)
    bins = [14.18,16,18,20,22]
    x = []
    xerr = []
    res = []
    reserr = []
    for b in range(len(bins)-1):
        qsq_min = bins[b]
        qsq_max = bins[b+1]
        x.append( (qsq_min+qsq_max)/2)
        xerr.append((qsq_min-qsq_max)/2)
        p = make_p_physical_point_BK(pfit,Fits)
        res1top = integrate_Gamma(p,qsq_min,qsq_max,m_tau,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        res1bot = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        res2top = integrate_Gamma(p,qsq_min,qsq_max,m_tau,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        res2bot = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        res.append(((res1top/res1bot +res2top/res2bot)/2).mean)
        reserr.append(((res1top/res1bot +res2top/res2bot)/2).sdev)
        
    plt.errorbar(x,res,xerr=xerr, yerr=reserr,ms=ms,fmt='*',mfc='none',color='k',label="HPQCD '21",capsize=capsize,lw=lw)
        
    the = HPQCD14B
    x,xerr = the.make_x()
    y,yerr = the.add_errs(the.binRtaumu)
    plt.errorbar(x,y,xerr=xerr, yerr=yerr, color='purple',mfc='none',fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    #y,yerr = the.add_errs(the.binRtaue)
    #plt.errorbar(x,y,xerr=xerr, yerr=yerr, color='b',mfc='none',fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    #y,yerr = the.add_errs(the.binRtauemu)
    #plt.errorbar(x,y,xerr=xerr, yerr=yerr, color='purple',mfc='none',fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)

    the = Fermi16C
    x,xerr = the.make_x()
    y,yerr = the.add_errs(the.binRmutaup)
    y,yerr = invert_y(y,yerr)
    #print(y)
    plt.errorbar(x,y,xerr=xerr, yerr=yerr, color='r',mfc='none',fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    #y,yerr = the.add_errs(the.binRmutau0)
    #y,yerr = invert_y(y,yerr)
    #print(y)
    #plt.errorbar(x,y,xerr=xerr, yerr=yerr, color='b',mfc='none',fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    
    #plt.fill_between([8.68,10.11],[-5,-5],[45,45], color='k',alpha=alpha)
    plt.fill_between([12.86,14.18],[-5,-5],[45,45], color='k',alpha=alpha)
    #plt.text((10.11+8.68)/2,0.1,r'$J/\Psi$',va='center',ha='center',fontsize=fontsizelab)
    plt.text((12.86+14.18)/2,1.25,r'$\Psi(2S)$',va='center',ha='center',fontsize=fontsizelab)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$R^{\tau}_{\mu}$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=1,loc='lower right')
    plt.axes().set_ylim([0.6,2.0])
    plt.axes().set_xlim([12.5,23])
    plt.tight_layout()
    plt.savefig('Plots/Rbybin_the_tau.pdf')
    plt.close()

    ##################### mu e ###########################################
    plt.figure(figsize=figsize)
    bins = [0.1,2,4,6,8,10,12,14,16,18,20,22]
    x = []
    xerr = []
    res = []
    reserr = []
    for b in range(len(bins)-1):
        qsq_min = bins[b]
        qsq_max = bins[b+1]
        x.append( (qsq_min+qsq_max)/2)
        xerr.append((qsq_min-qsq_max)/2)
        p = make_p_physical_point_BK(pfit,Fits)
        res1top = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        res1bot = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        res2top = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        res2bot = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        temp = (res1top/res1bot +res2top/res2bot)/2
        temp = (temp-1)*1000
        res.append(temp.mean)
        reserr.append(temp.sdev)
        
    plt.errorbar(x,res,xerr=xerr, yerr=reserr,ms=ms,fmt='*',mfc='none',color='k',label="HPQCD '21",capsize=capsize,lw=lw)

    the = Bobeth07
    x,xerr = the.make_x()
    y,yerr = the.add_up_down_errs(the.binRmuem)
    y,yerr = convert_Remu_bob(y,yerr)
    plt.errorbar(x[2],y[2],xerr=xerr[2], yerr=[[yerr[0][2]],[yerr[1][2]]], color='r',mfc='none',fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    #plt.errorbar(x[-1],y[-1],xerr=xerr[-1], yerr=[[yerr[0][-1]],[yerr[1][-1]]], color='r',mfc='none',fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    #show only 1,7 bin 
    #y,yerr = the.add_up_down_errs(the.binRmue0)
    #y,yerr = convert_Remu_bob(y,yerr)
    #plt.errorbar(x,y,xerr=xerr, yerr=yerr, color='b',mfc='none',fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    
    the = HPQCD14A
    x,xerr = the.make_x()
    y,yerr = the.add_errs(the.binRmue)
    plt.errorbar(x,y,xerr=xerr, yerr=yerr, color='purple',mfc='none',fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)

    the = Fermi16B
    x,xerr = the.make_x()
    y,yerr = the.add_errs(the.binRmuep)
    plt.errorbar(x,y,xerr=xerr, yerr=yerr, color='r',mfc='none',fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    y,yerr = the.add_errs(the.binRmue0)
    plt.errorbar(x[0],y[0],xerr=xerr[0], yerr=yerr[0], color='b',mfc='none',fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    # show only the bin where the charges differ
    plt.fill_between([8.68,10.11],[-10,-10],[45,45], color='k',alpha=alpha)
    plt.fill_between([12.86,14.18],[-10,-10],[45,45], color='k',alpha=alpha)
    plt.text((10.11+8.68)/2,8,r'$J/\Psi$',va='center',ha='center',fontsize=fontsizelab)
    plt.text((12.86+14.18)/2,8,r'$\Psi(2S)$',va='center',ha='center',fontsize=fontsizelab)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$10^3(R^{\mu}_{e}-1)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(1))

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper left')
    plt.axes().set_ylim([-5,11.2])
    plt.axes().set_xlim([-0.1,23])
    plt.tight_layout()
    plt.savefig('Plots/Rbybin_the_emu.pdf')
    plt.close()
    
    return()

############################################################################################################


def Rmue_plots(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    #2 chargeless and one not (this one over q^2 range 1-6).
    exps = [BELLE09,BaBar09,BaBar12,BaBar16,CDF11,LHCb12A,LHCb12B,LHCb14A,LHCb14A2,LHCb14B,LHCb14C]
    numbs = [4,3,1,2,5]
    labs = []
    i = 0
    plt.figure(figsize=figsize)
    for exp in exps:
        if exp.Rmue != []:
            labs.append(exp.label)
            x,xerr = exp.make_1_y(exp.Rmue)
            plt.errorbar(x,numbs[i], xerr=xerr,ms=ms,fmt=exp.sym,color='r' ,capsize=capsize,lw=lw)
            i += 1
        if exp.Rmuep != []:
            labs.append(exp.label)
            x,xerr = exp.make_1_y(exp.Rmuep)
            plt.errorbar(x,numbs[i], xerr=xerr,ms=ms,fmt=exp.sym,color='b' ,capsize=capsize,lw=lw)
            i += 1

    plt.plot([1,1],[-1,10],color='k',linestyle='--')
    plt.plot([0,2],[2.5,2.5],color='k')
    p = make_p_physical_point_BK(pfit,Fits)
    qsq_max = qsqmaxphysBK.mean
    res1top = integrate_Gamma(p,4*m_mu**2,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)
    res1bot = integrate_Gamma(p,4*m_e**2,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    res2top = integrate_Gamma(p,4*m_mu**2,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)
    res2bot = integrate_Gamma(p,4*m_e**2,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)
    res = (res1top/res1bot +res2top/res2bot)/2
    plt.errorbar(res.mean,5, xerr=res.sdev,ms=ms,fmt='*',color='k',capsize=capsize,lw=lw)

    p = make_p_physical_point_BK(pfit,Fits)
    qsq_max = 6
    qsq_min = 1
    res1top = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)
    res1bot = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    res2top = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)
    res2bot = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)
    resp = (res1top/res1bot +res2top/res2bot)/2
    plt.errorbar(resp.mean,2, xerr=resp.sdev,ms=ms,fmt='*',color='k',capsize=capsize,lw=lw)
    
    #plt.fill_between([res.mean-res.sdev,res.mean+res.sdev],[0,0],[11,11], color='k',alpha=alpha/2)
    labs.append("HPQCD '21")
    labs.append("HPQCD '21")
    plt.text(1.02,4.5,r'$R^{\mu}_e(q^2_{\mathrm{min}},q^2_{\mathrm{max}})$',fontsize=fontsizelab, va='center')
    plt.text(1.02,1.5,r'$R^{\mu (+)}_e (1\mathrm{GeV}^2,6\mathrm{GeV}^2)$',fontsize=fontsizelab, va='center')
    #plt.text(2.3,9,r'$B^+\to{}K^+\ell^+\ell^-$',fontsize=fontsizelab, va='center')
    plt.xlabel(r'$R^{\mu}_{e}$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=False,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('none')
    plt.xlim([0.6,1.44])
    plt.ylim([0.5,5.5])
    plt.gca().set_yticks(numbs)
    plt.gca().set_yticklabels(labs)
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    #plt.text(1.01,8.5,r'$\Gamma$',fontsize=fontsizelab*2,va='top',ha='right')
    #plt.plot([1,1.1],[7.5,7.5],color='k')
    #plt.plot([1,1],[7.5,8.5],color='k')
    plt.tight_layout()
    plt.savefig('Plots/Rmueexp.pdf')
    plt.close()
    return()

#############################################################################################################

def R_the_plot(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    ############### e,mu ########
    numbs = [1,3,2,4]
    labs = []
    i = 0
    plt.figure(figsize=figsize)
    the = HPQCD14A
    labs.append(the.label)
    x,xerr = the.add_errs(the.Rmue)
    R = gv.gvar(x[0],xerr[0])
    R = (R-1)*100
    x = R.mean
    xerr = R.sdev
    plt.errorbar(x,1, xerr=xerr,ms=ms,fmt=the.sym,color='b' ,capsize=capsize,lw=lw)
    labs.append(the.label)
    x,xerr = the.add_errs(the.Rtaumu)
    R = gv.gvar(x[0],xerr[0])
    R = (R-1)
    x = R.mean
    xerr = R.sdev
    plt.errorbar(x,3, xerr=xerr,ms=ms,fmt=the.sym,color='r' ,capsize=capsize,lw=lw)
    plt.plot([0,0],[-1,10],color='k',linestyle='--')
    plt.plot([-1,1],[2.5,2.5],color='k')
    p = make_p_physical_point_BK(pfit,Fits)
    qsq_max = qsqmaxphysBK.mean
    resmue1top = integrate_Gamma(p,4*m_mu**2,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)
    resmue1bot = integrate_Gamma(p,4*m_mu**2,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)
    restaumu1top = integrate_Gamma(p,14.18,qsq_max,m_tau,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)
    restaumu1bot = integrate_Gamma(p,14.18,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    resmue2top = integrate_Gamma(p,4*m_mu**2,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)
    resmue2bot = integrate_Gamma(p,4*m_mu**2,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)
    restaumu2top = integrate_Gamma(p,14.18,qsq_max,m_tau,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)
    restaumu2bot = integrate_Gamma(p,14.18,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=250)
    resmue = (resmue1top/resmue1bot +resmue2top/resmue2bot)/2
    resmue = (resmue-1)*100
    restaumu = (restaumu1top/restaumu1bot +restaumu2top/restaumu2bot)/2
    restaumu =restaumu-1
    plt.errorbar(resmue.mean,2, xerr=resmue.sdev,ms=ms,fmt='*',color='b',capsize=capsize,lw=lw)
    labs.append("HPQCD '21")
    plt.errorbar(restaumu.mean,4, xerr=restaumu.sdev,ms=ms,fmt='*',color='r',capsize=capsize,lw=lw)
    labs.append("HPQCD '21")
    plt.text(-0.205,3.5,r'$(R^{\tau}_{\mu}-1)_{(14.18\mathrm{GeV}^2,q^2_{\mathrm{max}})}$',fontsize=fontsizelab, va='center')
    plt.text(-0.205,1.5,r'$10^2\times(R^{\mu}_{e}-1)_{(4m_{\mu}^2,q^2_{\mathrm{max}})}$',fontsize=fontsizelab, va='center')
    plt.axes().tick_params(labelright=False,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('none')
    plt.xlim([-0.21,0.21])
    plt.ylim([0.5,4.5])
    plt.gca().set_yticks(numbs)
    plt.gca().set_yticklabels(labs)
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.05))
    plt.tight_layout()
    plt.savefig('Plots/R_the.pdf')
    plt.close()
    return()

#############################################################################################################

def F_h_plots(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    #exps = [LHCb12B,LHCb14B]
    m_lep = m_mu # use mu as most data muon and make no difference
    p = make_p_physical_point_BK(pfit,Fits)
    F = []
    qsq = []
    #for q2 in np.linspace(4*m_lep**2,qsqmaxphysBK.mean,nopts): #q2 now in GeV
    #    qsq.append(q2)
    #    al,cl = make_al_cl(p,q2,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
    #    F.append((2*al + 2*cl/3)*tauBpmGeV*1e7)
    qsq_min = 4*m_lep**2
    qsq.append((qsq_min+1)/2)
    F.append(integrate_FH(p,qsq_min,1,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2))
    for qsq_min in range(1,23):
        qsq.append(qsq_min+0.5)
        F.append(integrate_FH(p,qsq_min,qsq_min+1,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2))
    Fmean,Ferr = unmake_gvar_vec(F)
    Fupp,Flow = make_upp_low(F)
    plt.figure(figsize=figsize)
    plt.errorbar(qsq, Fmean,yerr=Ferr, color='k',fmt='*',ms=ms,label="HPQCD '21")
    #plt.fill_between(qsq,Flow,Fupp, color='purple',alpha=alpha)
    exp = LHCb12B
    x,xerr = exp.make_x()
    y,yerr = exp.make_y(exp.FHmup)
    plt.errorbar(x,y, yerr=yerr, color='r', fmt=exp.sym,ms=ms,label=(exp.label),capsize=capsize)
    exp = LHCb14B
    x,xerr = exp.make_x()
    y,yerr = make_y_LHCb14B(exp.FHmup)
    plt.errorbar(x,y, yerr=yerr, color='b', fmt='none',ms=ms,label=(exp.label),capsize=capsize)
    plt.plot([-2,25],[0,0],color='k',linestyle='--')
    plt.fill_between([8.68,10.11],[-1,-1],[1,1], color='k',alpha=alpha)
    plt.fill_between([12.86,14.18],[-1,-1],[1,1], color='k',alpha=alpha)
    plt.text((10.11+8.68)/2,0.4,r'$J/\Psi$',va='center',ha='center',fontsize=fontsizelab)
    plt.text((12.86+14.18)/2,0.4,r'$\Psi(2S)$',va='center',ha='center',fontsize=fontsizelab)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel('$F^{\mu (+)}_H$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))

    #handles, labels = plt.gca().get_legend_handles_labels()
    #handles = [h[0] for h in handles]
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper left')
    plt.axes().set_ylim([-0.02,0.46])
    plt.axes().set_xlim([0,qsqmaxphysBK.mean])
    
    plt.tight_layout()
    plt.savefig('Plots/F_H.pdf')
    plt.close()
    
    return()

def make_y_LHCb14B(Fs):
    # we make a Gaussian assumption to add the systematic errors to the statistical one
    means = []
    errs = []
    for F in Fs:
        mean = (F[0]+F[1])/2
        sdev = (F[1]-F[0])/2
        sys = F[2]
        err = np.sqrt(sdev**2+sys**2)
        means.append(mean)
        errs.append(err)
    return(means,errs)

#############################################################################################################



#############################################################################################################

def F_H_the_plots(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    ####### e ###################################
    m_lep = m_e
    numbs = [1,3,4,2]
    labs = []
    i = 0
    plt.figure(figsize=figsize)
    the = Fermi16B
    labs.append(the.label)
    x,xerr = the.add_errs(the.FHe0) #10^8
    plt.errorbar(x,1, xerr=xerr,ms=ms,fmt=the.sym,color='b' ,capsize=capsize,lw=lw)
    labs.append(the.label)
    x,xerr = the.add_errs(the.FHep)#10^8
    plt.errorbar(x,3, xerr=xerr,ms=ms,fmt=the.sym,color='r' ,capsize=capsize,lw=lw)
    
    #plt.plot([0,0],[-1,10],color='k',linestyle='--')
    plt.plot([30,100],[2.5,2.5],color='k')
    #plt.plot([0,30],[4.5,4.5],color='k')
    p = make_p_physical_point_BK(pfit,Fits)
    qsq_max = qsqmaxphysBK.mean
    qsq_min = 4*m_lep**2
    res = 1e8*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=200)
    plt.errorbar(res.mean,4, xerr=res.sdev,ms=ms,fmt='*',color='r',capsize=capsize,lw=lw)
    labs.append("HPQCD '21")
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    res = 1e8*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=200)
    plt.errorbar(res.mean,2, xerr=res.sdev,ms=ms,fmt='*',color='b',capsize=capsize,lw=lw)
    labs.append("HPQCD '21")
    #res = 1000*(res1418A+res1418B)/2
    #plt.errorbar(res.mean,2, xerr=res.sdev,ms=ms,fmt='*',color='purple',capsize=capsize,lw=lw)
    #labs.append("HPQCD '21")
    plt.text(50,1.5,r'$B^0\to{}K^0e^+e^-$',fontsize=fontsizelab*0.9, va='center')
    plt.text(50,3.5,r'$B^+\to{}K^+e^+e^-$',fontsize=fontsizelab*0.9, va='center')
    plt.axes().tick_params(labelright=False,which='both',width=2,labelsize=fontsizelab)
    plt.xlabel('$10^8F_H^{e}$',fontsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('none')
    plt.xlim([38,81])
    plt.ylim([0.5,4.5])
    plt.gca().set_yticks(numbs)
    plt.gca().set_yticklabels(labs)
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.tight_layout()
    plt.savefig('Plots/FH_the_e.pdf')
    plt.close()
    ####### mu ###################################
    m_lep = m_mu
    numbs = [1,3,5,6,4,2]
    labs = []
    i = 0
    plt.figure(figsize=figsize)
    the=Bobeth11
    labs.append(the.label)
    x,xerr = the.add_up_down_errs(the.FHmu) # 10^3 from 14.18 up
    plt.errorbar(x,1, xerr=xerr,ms=ms,fmt=the.sym,color='purple' ,capsize=capsize,lw=lw)

    
    the = Fermi16B
    labs.append(the.label)
    x,xerr = the.add_errs(the.FHmu0) #10^3
    plt.errorbar(x,3, xerr=xerr,ms=ms,fmt=the.sym,color='b' ,capsize=capsize,lw=lw)
    labs.append(the.label)
    x,xerr = the.add_errs(the.FHmup)
    plt.errorbar(x,5, xerr=xerr,ms=ms,fmt=the.sym,color='r' ,capsize=capsize,lw=lw)
    
    #plt.plot([0,0],[-1,10],color='k',linestyle='--')
    plt.plot([0,30],[2.5,2.5],color='k')
    plt.plot([0,30],[4.5,4.5],color='k')
    p = make_p_physical_point_BK(pfit,Fits)
    qsq_max = qsqmaxphysBK.mean
    qsq_min = 4*m_lep**2
    res = 1000*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=200)
    res1418A = integrate_FH(p,14.18,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=200)
    plt.errorbar(res.mean,6, xerr=res.sdev,ms=ms,fmt='*',color='r',capsize=capsize,lw=lw)
    labs.append("HPQCD '21")
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    res = 1000*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=200)
    res1418B = integrate_FH(p,14.18,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=200)
    plt.errorbar(res.mean,4, xerr=res.sdev,ms=ms,fmt='*',color='b',capsize=capsize,lw=lw)
    labs.append("HPQCD '21")
    res = 1000*(res1418A+res1418B)/2
    plt.errorbar(res.mean,2, xerr=res.sdev,ms=ms,fmt='*',color='purple',capsize=capsize,lw=lw)
    labs.append("HPQCD '21")
    plt.text(5,3.5,r'$B^0\to{}K^0\mu^+\mu^-$',fontsize=fontsizelab*0.9, va='center')
    plt.text(5,5.5,r'$B^+\to{}K^+\mu^+\mu^-$',fontsize=fontsizelab*0.9, va='center')
    plt.text(12,1.5,r'$B\to{}K\mu^+\mu^-_{(14.18\mathrm{GeV}^2,q^2_{\mathrm{max}})}$',fontsize=fontsizelab*0.9, va='center')
    plt.axes().tick_params(labelright=False,which='both',width=2,labelsize=fontsizelab)
    plt.xlabel('$10^3F_H^{\mu}$',fontsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('none')
    plt.xlim([4,25])
    plt.ylim([0.5,6.5])
    plt.gca().set_yticks(numbs)
    plt.gca().set_yticklabels(labs)
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.tight_layout()
    plt.savefig('Plots/FH_the_mu.pdf')
    plt.close()
    ####### tau ###################################
    m_lep = m_tau
    numbs = [1,2,4,6,7,5,3]
    labs = []
    i = 0
    plt.figure(figsize=figsize)
    the=Bobeth11
    labs.append(the.label)
    x,xerr = the.add_up_down_errs(the.FHtau) # from 14.18 up
    plt.errorbar(x,1, xerr=xerr,ms=ms,fmt=the.sym,color='purple' ,capsize=capsize,lw=lw)

    the = HPQCD14A
    labs.append(the.label)
    x,xerr = the.add_errs(the.FHtau) # 14.18 up 
    plt.errorbar(x,2, xerr=xerr,ms=ms,fmt=the.sym,color='purple' ,capsize=capsize,lw=lw)
    
    the = Fermi16B
    labs.append(the.label)
    x,xerr = the.add_errs(the.FHtau0) #10^3
    plt.errorbar(x,4, xerr=xerr,ms=ms,fmt=the.sym,color='b' ,capsize=capsize,lw=lw)
    labs.append(the.label)
    x,xerr = the.add_errs(the.FHtaup)
    plt.errorbar(x,6, xerr=xerr,ms=ms,fmt=the.sym,color='r' ,capsize=capsize,lw=lw)
    
    #plt.plot([0,0],[-1,10],color='k',linestyle='--')
    plt.plot([0,2],[5.5,5.5],color='k')
    plt.plot([0,2],[3.5,3.5],color='k')
    p = make_p_physical_point_BK(pfit,Fits)
    qsq_max = qsqmaxphysBK.mean
    qsq_min = 4*m_lep**2
    res = integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=200)
    res1418A = integrate_FH(p,14.18,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=200)
    plt.errorbar(res.mean,7, xerr=res.sdev,ms=ms,fmt='*',color='r',capsize=capsize,lw=lw)
    labs.append("HPQCD '21")
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    res = integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=200)
    res1418B = integrate_FH(p,14.18,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2,iters=200)
    plt.errorbar(res.mean,5, xerr=res.sdev,ms=ms,fmt='*',color='b',capsize=capsize,lw=lw)
    labs.append("HPQCD '21")
    res = (res1418A+res1418B)/2
    plt.errorbar(res.mean,3, xerr=res.sdev,ms=ms,fmt='*',color='purple',capsize=capsize,lw=lw)
    labs.append("HPQCD '21")
    plt.text(0.831,4.5,r'$B^0\to{}K^0\tau^+\tau^-$',fontsize=fontsizelab*0.9, va='center')
    plt.text(0.831,6.5,r'$B^+\to{}K^+\tau^+\tau^-$',fontsize=fontsizelab*0.9, va='center')
    plt.text(0.831,2,r'$B\to{}K\tau^+\tau^-_{(14.18\mathrm{GeV}^2,q^2_{\mathrm{max}})}$',fontsize=fontsizelab*0.9, va='center')
    plt.axes().tick_params(labelright=False,which='both',width=2,labelsize=fontsizelab)
    plt.xlabel(r'$F_H^{\tau}$',fontsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('none')
    plt.xlim([0.83,0.925])
    plt.ylim([0.5,7.5])
    plt.gca().set_yticks(numbs)
    plt.gca().set_yticklabels(labs)
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.02))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.tight_layout()
    plt.savefig('Plots/FH_the_tau.pdf')
    plt.close()
    return()
#############################################################################################################

def FHbybin_the_plots(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    #exps = [LHCb12B,LHCb14B]
    ############### e ##############
    m_lep = m_e 
    F = []
    qsq = []
    qsqerr = []
    qsq_bins = [0.1,2,4,6,8,10,12,14,16,18,20,22]
    for q in range(len(qsq_bins)-1):
        qsq_min = qsq_bins[q]
        qsq_max = qsq_bins[q+1]
        qsq.append((qsq_min+qsq_max)/2)
        qsqerr.append((qsq_max-qsq_min)/2)
        p = make_p_physical_point_BK(pfit,Fits)
        Fp = (integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2))
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        F0 = (integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2))
        F.append(1e6*(Fp+F0)/2)
    Fmean,Ferr = unmake_gvar_vec(F)
    Fupp,Flow = make_upp_low(F)
    plt.figure(figsize=figsize)
    plt.errorbar(qsq, Fmean,xerr=qsqerr,yerr=Ferr, color='k',fmt='*',mfc='none',ms=ms,label="HPQCD '21",capsize=capsize)

    the = HPQCD14A # *10^6
    x,xerr = the.make_x()
    y,yerr = the.add_errs(the.binFHe)
    plt.errorbar(x,y, xerr=xerr,yerr=yerr, color='purple',mfc='none', fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)

    the = Fermi16B # *10^8
    x,xerr = the.make_x()
    y,yerr = the.add_errs(the.binFHep)
    y = np.array(y)*0.01
    yerr = np.array(yerr)*0.01
    plt.errorbar(x,y, xerr=xerr,yerr=yerr, color='r',mfc='none', fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    y,yerr = the.add_errs(the.binFHe0)
    y = np.array(y)*0.01
    yerr = np.array(yerr)*0.01
    plt.errorbar(x[0],y[0], xerr=xerr[0],yerr=yerr[0], color='b',mfc='none', fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)

    plt.fill_between([8.68,10.11],[-1,-1],[3,3], color='k',alpha=alpha)
    plt.fill_between([12.86,14.18],[-1,-1],[3,3], color='k',alpha=alpha)
    plt.text((10.11+8.68)/2,2,r'$J/\Psi$',va='center',ha='center',fontsize=fontsizelab)
    plt.text((12.86+14.18)/2,2,r'$\Psi(2S)$',va='center',ha='center',fontsize=fontsizelab)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel('$10^6F^e_H$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper right')
    plt.axes().set_ylim([-0.01,2.7])
    plt.axes().set_xlim([-0.2,qsqmaxphysBK.mean+0.2])
    plt.tight_layout()
    plt.savefig('Plots/FHbybin_the_e.pdf')
    plt.close()

    ############### mu ##############
    m_lep = m_mu 
    F = []
    qsq = []
    qsqerr = []
    qsq_bins = [0.1,2,4,6,8,10,12,14,16,18,20,22]
    for q in range(len(qsq_bins)-1):
        qsq_min = qsq_bins[q]
        qsq_max = qsq_bins[q+1]
        qsq.append((qsq_min+qsq_max)/2)
        qsqerr.append((qsq_max-qsq_min)/2)
        p = make_p_physical_point_BK(pfit,Fits)
        Fp = (integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2))
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        F0 = (integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2))
        F.append(1e2*(Fp+F0)/2)
    Fmean,Ferr = unmake_gvar_vec(F)
    Fupp,Flow = make_upp_low(F)
    plt.figure(figsize=figsize)
    plt.errorbar(qsq, Fmean,xerr=qsqerr,yerr=Ferr, color='k',fmt='*',mfc='none',ms=ms,label="HPQCD '21",capsize=capsize)

    the = Bobeth07
    x,xerr = the.make_x()#*10^1
    y,yerr = the.add_up_down_errs(the.binFHmum)
    y = np.array(y)*100
    yerrd = np.array(yerr[0])*100
    yerru = np.array(yerr[1])*100
    yerr = [yerrd,yerru]
    plt.errorbar(x[2],y[2], xerr=xerr[2],yerr=[[yerr[0][2]],[yerr[1][2]]], color='r',mfc='none', fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    #y,yerr = the.add_up_down_errs(the.binFHmu0)
    #y = np.array(y)*100
    #yerrd = np.array(yerr[0])*100
    #yerru = np.array(yerr[1])*100
    #yerr = [yerrd,yerru]
    #plt.errorbar(x,y, xerr=xerr,yerr=yerr, color='b',mfc='none', fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    
    the = HPQCD14A # *10^2
    x,xerr = the.make_x()
    y,yerr = the.add_errs(the.binFHmu)
    plt.errorbar(x,y, xerr=xerr,yerr=yerr, color='purple',mfc='none', fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)

    the = Fermi16B # *10^3
    x,xerr = the.make_x()
    y,yerr = the.add_errs(the.binFHmup)
    y = np.array(y)*0.1
    yerr = np.array(yerr)*0.1
    plt.errorbar(x,y, xerr=xerr,yerr=yerr, color='r',mfc='none', fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    y,yerr = the.add_errs(the.binFHmu0)
    y = np.array(y)*0.1
    yerr = np.array(yerr)*0.1
    plt.errorbar(x[0],y[0], xerr=xerr[0],yerr=yerr[0], color='b',mfc='none', fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)

    plt.fill_between([8.68,10.11],[-1,-1],[20,20], color='k',alpha=alpha)
    plt.fill_between([12.86,14.18],[-1,-1],[20,20], color='k',alpha=alpha)
    plt.text((10.11+8.68)/2,8,r'$J/\Psi$',va='center',ha='center',fontsize=fontsizelab)
    plt.text((12.86+14.18)/2,8,r'$\Psi(2S)$',va='center',ha='center',fontsize=fontsizelab)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel('$10^2F^{\mu}_H$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(1))
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper right')
    plt.axes().set_ylim([-0.01,10.5])
    plt.axes().set_xlim([-0.2,qsqmaxphysBK.mean+0.2])
    plt.tight_layout()
    plt.savefig('Plots/FHbybin_the_mu.pdf')
    plt.close()

    ############### emu ##############
    #m_lep = m_mu 
    F = []
    qsq = []
    qsqerr = []
    m_mu_paper = 0.106 #GeV used in 1212.2321    
    qsq_bins = [4*m_mu_paper**2,2,4.3,8.68,10,12,14.18,16,18,20,22]
    for q in range(len(qsq_bins)-1):
        qsq_min = qsq_bins[q]
        qsq_max = qsq_bins[q+1]
        qsq.append((qsq_min+qsq_max)/2)
        qsqerr.append((qsq_max-qsq_min)/2)
        #p = make_p_physical_point_BK(pfit,Fits)
        #Fpe = (integrate_FH(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2))
        #Fpmu = (integrate_FH(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2))
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        F0e = (integrate_FH(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2))
        F0mu = (integrate_FH(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2))
        #F.append(100*(F0e+F0mu)/2)
        F.append(100*F0mu)
    Fmean,Ferr = unmake_gvar_vec(F)
    Fupp,Flow = make_upp_low(F)
    plt.figure(figsize=figsize)
    plt.errorbar(qsq, Fmean,xerr=qsqerr,yerr=Ferr, color='k',fmt='*',mfc='none',ms=ms,label="HPQCD '21",capsize=capsize)

    the = Bobeth16
    x,xerr = the.make_x()#*10^1
    #y,yerr = the.add_up_down_errs(the.binFHemum)
    #y = np.array(y)*100
    #yerrd = np.array(yerr[0])*100
    #yerru = np.array(yerr[1])*100
    #yerr = [yerrd,yerru]
    #plt.errorbar(x,y, xerr=xerr,yerr=yerr, color='r',mfc='none', fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    y,yerr = the.add_up_down_errs(the.binFHemu0)
    y = np.array(y)*100
    yerrd = np.array(yerr[0])*100
    yerru = np.array(yerr[1])*100
    yerr = [yerrd,yerru]
    plt.errorbar(x,y, xerr=xerr,yerr=yerr, color='b',mfc='none', fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)

    plt.fill_between([8.68,10.11],[-1,-1],[20,20], color='k',alpha=alpha)
    plt.fill_between([12.86,14.18],[-1,-1],[20,20], color='k',alpha=alpha)
    plt.text((10.11+8.68)/2,8,r'$J/\Psi$',va='center',ha='center',fontsize=fontsizelab)
    plt.text((12.86+14.18)/2,8,r'$\Psi(2S)$',va='center',ha='center',fontsize=fontsizelab)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel('$10^2F^{\ell(0)}_H$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(1))
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper right')
    plt.axes().set_ylim([-0.2,11.1])
    plt.axes().set_xlim([-0.2,qsqmaxphysBK.mean+0.2])
    plt.tight_layout()
    plt.savefig('Plots/FHbybin_the_emu.pdf')
    plt.close()

    ####################### tau ################################################
    m_lep = m_tau 
    F = []
    qsq = []
    qsqerr = []
    qsq_bins = [14.18,16,18,20,22]
    for q in range(len(qsq_bins)-1):
        qsq_min = qsq_bins[q]
        qsq_max = qsq_bins[q+1]
        qsq.append((qsq_min+qsq_max)/2)
        qsqerr.append((qsq_max-qsq_min)/2)
        p = make_p_physical_point_BK(pfit,Fits)
        Fp = (integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2))
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        F0 = (integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2))
        F.append((Fp+F0)/2)
    Fmean,Ferr = unmake_gvar_vec(F)
    Fupp,Flow = make_upp_low(F)
    plt.figure(figsize=figsize)
    plt.errorbar(qsq, Fmean,xerr=qsqerr,yerr=Ferr, color='k',fmt='*',mfc='none',ms=ms,label="HPQCD '21",capsize=capsize)
    
    the = HPQCD14B 
    x,xerr = the.make_x()
    y,yerr = the.add_errs(the.binFHtau)
    plt.errorbar(x,y, xerr=xerr,yerr=yerr, color='purple',mfc='none', fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)

    the = Fermi16C # *10^3
    x,xerr = the.make_x()
    y,yerr = the.add_errs(the.binFHtaup)
    plt.errorbar(x,y, xerr=xerr,yerr=yerr, color='r',mfc='none', fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)
    #y,yerr = the.add_errs(the.binFHtau0)
    #plt.errorbar(x,y, xerr=xerr,yerr=yerr, color='b',mfc='none', fmt=the.sym,ms=ms,label=(the.label),capsize=capsize)

    #plt.fill_between([8.68,10.11],[-1,-1],[20,20], color='k',alpha=alpha)
    plt.fill_between([12.86,14.18],[-1,-1],[20,20], color='k',alpha=alpha)
    #plt.text((10.11+8.68)/2,8,r'$J/\Psi$',va='center',ha='center',fontsize=fontsizelab)
    plt.text(13,0.85,r'$\Psi(2S)$',va='center',fontsize=fontsizelab)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$F^{\tau}_H$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(2))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.02))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper right')
    plt.axes().set_ylim([0.81,0.94])
    plt.axes().set_xlim([12.9,qsqmaxphysBK.mean+0.2])
    plt.tight_layout()
    plt.savefig('Plots/FHbybin_the_tau.pdf')
    plt.close()
    
    return()

#############################################################################################################

def table_of_as(Fits,pfit,Nijk,Npow,Nm,fpf0same,addrho,Del):
    list0 = []
    listp = []
    listT = []
    Fit = Fits[0]
    mass = Fit['masses'][0]
    fit = Fit['conf']
    p = make_p_physical_point_BK(pfit,Fits)
    logs = make_logs(p,mass,Fit)
    atab = open('Tables/tablesofas.txt','w')
    for n in range(Npow):
        if n == 0:
            atab.write('      {0}&'.format(make_an_BK(n,Nijk,Nm,addrho,p,'0',Fit,mass,0,fpf0same)))
        else:
            atab.write('{0}&'.format(make_an_BK(n,Nijk,Nm,addrho,p,'0',Fit,mass,0,fpf0same)))
        list0.append(make_an_BK(n,Nijk,Nm,addrho,p,'0',Fit,mass,0,fpf0same))
        listp.append(make_an_BK(n,Nijk,Nm,addrho,p,'p',Fit,mass,0,fpf0same))
        listT.append(make_an_BK(n,Nijk,Nm,addrho,p,'T',Fit,mass,0,fpf0same))
    MBsstar = p['MBsstarphys']
    MBs0 = p['MBphys']+Del
    for n in range(Npow):           
        atab.write('{0}&'.format(make_an_BK(n,Nijk,Nm,addrho,p,'p',Fit,mass,0,fpf0same)))
    for n in range(Npow):           
        atab.write('{0}&'.format(make_an_BK(n,Nijk,Nm,addrho,p,'T',Fit,mass,0,fpf0same)))
    atab.write('{0}&{1}&{2}\\\\ [1ex]\n'.format(MBs0,MBsstar,logs))
    atab.write('      \hline \n')
    list0.extend(listp)
    list0.extend(listT)
    list0.append(MBs0)
    list0.append(MBsstar)
    list0.append(logs)
    covar = gv.evalcorr(list0)
    for i in range(3*Npow+3):
            atab.write('\n      ')
            for k in range(i):
                atab.write('&')
            for j in range(i,3*Npow+3):
                #print(covar[i][j])
                atab.write('{0:.5f}'.format(covar[i][j]))
                if j != 3*Npow+2:
                    atab.write('&')
                else:
                    atab.write('\\\\ [0.5ex]')
    atab.close()
    return()
#############################DK tensor##########################################################

def DKfT_table_of_as(Fits,pfit,Nijk,Npow,Nm,fpf0same,addrho):
    listT = []
    Fit = Fits[0]
    mass = Fit['masses'][0]
    fit = Fit['conf']
    p = make_p_Mh_BK(pfit,Fits,pfit['MDphys'])
    logs = make_logs(p,mass,Fit)
    atab = open('Tables/DKfTtablesofas.txt','w')
    for n in range(Npow):
        if n == 0:
            atab.write('      {0}&'.format(make_an_BK(n,Nijk,Nm,addrho,p,'T',Fit,mass,0,fpf0same)))
        else:
            atab.write('{0}&'.format(make_an_BK(n,Nijk,Nm,addrho,p,'T',Fit,mass,0,fpf0same)))
        listT.append(make_an_BK(n,Nijk,Nm,addrho,p,'T',Fit,mass,0,fpf0same))
    MDsstar = p['MDsstarphys']
    atab.write('{0}&{1}\\\\ [1ex]\n'.format(MDsstar,logs))
    atab.write('      \hline \n')
    listT.append(MDsstar)
    listT.append(logs)
    covar = gv.evalcorr(listT)
    for i in range(Npow+2):
            atab.write('\n      ')
            for k in range(i):
                atab.write('&')
            for j in range(i,Npow+2):
                #print(covar[i][j])
                atab.write('{0:.5f}'.format(covar[i][j]))
                if j != Npow+1:
                    atab.write('&')
                else:
                    atab.write('\\\\ [0.5ex]')
    atab.close()
    return()

################################################################################################################
def results_tables(fs_data,Fit):
    table = open('Tables/{0}table.txt'.format(Fit['conf']),'w')
    for mass in Fit['masses']:
        table.write('      \hline \n')
        table.write('      &{0}'.format(mass))
        for tw,twist in enumerate(Fit['twists']):
            if tw == 0:
                table.write('&{0}&{1}&{2}&{3}&{4}&-'.format(Fit['M_parent_m{0}'.format(mass)],fs_data['qsq_m{0}_tw{1}'.format(mass,twist)],Fit['E_daughter_tw{0}_theory'.format(twist)],Fit['S_m{0}_tw{1}'.format(mass,twist)],Fit['V_m{0}_tw{1}'.format(mass,twist)]))
            elif 'T_m{0}_tw{1}'.format(mass,twist) in Fit:
                table.write('      &&&{0}&{1}&{2}&{3}&{4}'.format(fs_data['qsq_m{0}_tw{1}'.format(mass,twist)],Fit['E_daughter_tw{0}_theory'.format(twist)],Fit['S_m{0}_tw{1}'.format(mass,twist)],Fit['V_m{0}_tw{1}'.format(mass,twist)],Fit['T_m{0}_tw{1}'.format(mass,twist)]))
            else:
                table.write('      &&&{0}&{1}&{2}&{3}&-'.format(fs_data['qsq_m{0}_tw{1}'.format(mass,twist)],Fit['E_daughter_tw{0}_theory'.format(twist)],Fit['S_m{0}_tw{1}'.format(mass,twist)],Fit['V_m{0}_tw{1}'.format(mass,twist)]))
            if fs_data['fp_m{0}_tw{1}'.format(mass,twist)] != None:
                if fs_data['fT_m{0}_tw{1}'.format(mass,twist)] != None:
                    table.write('&{0}&{1}&{2}\\\\ [1ex]\n'.format(fs_data['f0_m{0}_tw{1}'.format(mass,twist)],fs_data['fp_m{0}_tw{1}'.format(mass,twist)],fs_data['fT_m{0}_tw{1}'.format(mass,twist)]))
                else:
                    table.write('&{0}&{1}&-\\\\ [1ex]\n'.format(fs_data['f0_m{0}_tw{1}'.format(mass,twist)],fs_data['fp_m{0}_tw{1}'.format(mass,twist)]))
            else:
                table.write('&{0}&-&-\\\\ [1ex]\n'.format(fs_data['f0_m{0}_tw{1}'.format(mass,twist)]))

        
    table.close()
    return()

###########################################################################################################################

def error_plot(pfit,prior,Fits,Nijk,Npow,Nm,f,t_0,addrho,fpf0same,const2):
    qsqs = np.linspace(0,qsqmaxphysBK.mean,nopts)
    f0,fp,fT = output_error_BK(pfit,prior,Fits,Nijk,Npow,Nm,f,qsqs,t_0,addrho,fpf0same,const2)
    plt.figure(18,figsize=figsize)
    ax1 = plt.subplot(311)
    ax1b = ax1.twinx()
    ax1.plot(qsqs,f0[1],color='r', ls='--',lw=2,label='Inputs')
    ax1.plot(qsqs,f0[2],color='purple',ls=':',lw=2,label='+ q mistunings')
    ax1.plot(qsqs,f0[3],color='b',ls='-',lw=2,label='+ Statistics')
    ax1.plot(qsqs,f0[4],color='g',ls='-.',lw=2,label='+ HQET')
    ax1.plot(qsqs,f0[5],color='k',ls='-',lw=4,label='+ Discretisation')
    ax1.set_ylabel('$(f_0(q^2)~\% \mathrm{err})^2 $ ',fontsize=fontsizelab)
    ax1.tick_params(width=2,labelsize=fontsizelab)
    ax1.tick_params(which='major',length=major)
    ax1.tick_params(which='minor',length=minor)
    #plt.gca().yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('none')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.yaxis.set_major_locator(MultipleLocator(5))
    #ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.set_xlim([0,qsqmaxphysBK.mean])
    ####################################### right hand y axis ###
    ax1b.plot(qsqs,f0[1],color='r', ls='--',lw=2,label='Inputs')
    ax1b.plot(qsqs,f0[2],color='purple',ls=':',lw=2,label='+ q mistunings')
    ax1b.plot(qsqs,f0[3],color='b',ls='-',lw=2,label='+ Statistics')
    ax1b.plot(qsqs,f0[4],color='g',ls='-.',lw=2,label='+ HQET')
    ax1b.plot(qsqs,f0[5],color='k',ls='-',lw=4,label='+ Discretisation')
    ax1b.set_ylabel('$f_0(q^2)~\% \mathrm{err}$ ',fontsize=fontsizelab)
    ax1b.tick_params(width=2,labelsize=fontsizelab)
    ax1b.tick_params(which='major',length=major)
    ax1b.tick_params(which='minor',length=minor)
    low,upp = ax1.get_ylim()
    ax1b.set_ylim([low,upp])
    points = []
    rootpoints = []
    i = 1.0
    while i != 'stop':
        if i**2 < upp:
            points.append(i**2)
            if int(10*i)%2 == 0:
                rootpoints.append('{0}'.format(i))
            else:
                rootpoints.append('')
            i+=0.5
        else:
            i ='stop'
    ax1b.set_yticks(points)
    ax1b.set_yticklabels(rootpoints)
    
    plt.legend(loc='upper right',ncol=2,fontsize=fontsizeleg,frameon=False)
    ############ midddle ############################################
        
    ax2 = plt.subplot(312,sharex=ax1)
    ax2b = ax2.twinx()
    ax2.plot(qsqs,fp[1],color='r', ls='--',lw=2)
    ax2.plot(qsqs,fp[2],color='purple',ls=':',lw=2)
    ax2.plot(qsqs,fp[3],color='b',ls='-',lw=2)
    ax2.plot(qsqs,fp[4],color='g',ls='-.',lw=2)
    ax2.plot(qsqs,fp[5],color='k',ls='-',lw=4)
    ax2.set_ylabel('$(f_+(q^2)~\%\mathrm{err})^2$',fontsize=fontsizelab)
    ax2.tick_params(width=2,labelsize=fontsizelab)
    ax2.tick_params(which='major',length=major)
    ax2.tick_params(which='minor',length=minor)
    ax2.xaxis.set_ticks_position('none')
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.yaxis.set_major_locator(MultipleLocator(50))
    #ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.set_xlim([0,qsqmaxphysBK.mean])
    #plt.axes().set_ylim([-0.8,2.5])
    #plt.axes().set_xlim([lower-0.22,upper+0.22])
    ######## right hand axis ##
    ax2b.plot(qsqs,fp[1],color='r', ls='--',lw=2)
    ax2b.plot(qsqs,fp[2],color='purple',ls=':',lw=2)
    ax2b.plot(qsqs,fp[3],color='b',ls='-',lw=2)
    ax2b.plot(qsqs,fp[4],color='g',ls='-.',lw=2)
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
    i = 1.0
    while i != 'stop':
        if i**2 < upp:
            points.append(i**2)
            if int(10*i)%2 == 0:
                rootpoints.append('{0}'.format(i))
            else:
                rootpoints.append('')
            i+=4.0
        else:
            i ='stop'
    ax2b.set_yticks(points)
    ax2b.set_yticklabels(rootpoints)
    #############bottom ############################################
    
    ax3 = plt.subplot(313,sharex=ax1)
    ax3b = ax3.twinx()
    ax3.plot(qsqs,fT[1],color='r', ls='--',lw=2)
    ax3.plot(qsqs,fT[2],color='purple',ls=':',lw=2)
    ax3.plot(qsqs,fT[3],color='b',ls='-',lw=2)
    ax3.plot(qsqs,fT[4],color='g',ls='-.',lw=2)
    ax3.plot(qsqs,fT[5],color='k',ls='-',lw=4)
    ax3.set_xlabel(r'$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    ax3.set_ylabel('$(f_T(q^2)~\%\mathrm{err})^2$',fontsize=fontsizelab)
    ax3.tick_params(width=2,labelsize=fontsizelab)
    ax3.tick_params(which='major',length=major)
    ax3.tick_params(which='minor',length=minor)
    ax3.xaxis.set_major_locator(MultipleLocator(5))
    ax3.xaxis.set_minor_locator(MultipleLocator(1))
    ax3.yaxis.set_major_locator(MultipleLocator(100))
    #ax3.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax3.set_xlim([0,qsqmaxphysBK.mean])
    #plt.axes().set_ylim([-0.8,2.5])
    #plt.axes().set_xlim([lower-0.22,upper+0.22])
    ######## right hand axis ##
    ax3b.plot(qsqs,fT[1],color='r', ls='--',lw=2)
    ax3b.plot(qsqs,fT[2],color='purple',ls=':',lw=2)
    ax3b.plot(qsqs,fT[3],color='b',ls='-',lw=2)
    ax3b.plot(qsqs,fT[4],color='g',ls='-.',lw=2)
    ax3b.plot(qsqs,fT[5],color='k',ls='-',lw=4)
    ax3b.set_xlabel(r'$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    ax3b.set_ylabel('$f_T(q^2)~\%\mathrm{err}$',fontsize=fontsizelab)
    ax3b.tick_params(width=2,labelsize=fontsizelab)
    ax3b.tick_params(which='major',length=major)
    ax3b.tick_params(which='minor',length=minor)
    low,upp = ax3.get_ylim()
    ax3b.set_ylim([low,upp])
    points = []
    rootpoints = []
    i = 1.0
    while i != 'stop':
        if i**2 < upp:
            points.append(i**2)
            if int(10*i)%2 == 0:
                rootpoints.append('{0}'.format(i))
            else:
                rootpoints.append('')
            i+=4.0
        else:
            i ='stop'
    ax3b.set_yticks(points)
    ax3b.set_yticklabels(rootpoints)
    plt.tight_layout()
    plt.savefig('Plots/f0fpfTerr.pdf')
    plt.close()
    return()


################################################################################################

def DK_fT_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata):
    qsq = []
    z = []
    y = []
    p = make_p_Mh_BK(pfit,Fits,pfit['MDphys'])
    for q2 in np.linspace(0,qsqmaxphysDK.mean,nopts): #q2 now in GeV
        qsq.append(q2)
        zed = make_z(q2,t_0,p['MDphys'],p['MKphys']) #all GeV dimensions
        if zed ==0:
            z.append(zed)
        else:
            z.append(zed.mean)
        y.append(make_fT_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(figsize=figsize)
    plt.plot(qsq, ymean, color='g')
    plt.fill_between(qsq,ylow,yupp, color='g',alpha=alpha)
    plt.errorbar(0,0.687, yerr=0.054,fmt='k*',ms=ms,mfc='none',label = r'arXiv:1803.04807',lw=lw)
    plt.errorbar(0.269,0.741, yerr=0.053,fmt='k*',ms=ms,mfc='none',lw=lw)
    plt.errorbar(0.538,0.799, yerr=0.052,fmt='k*',ms=ms,mfc='none',lw=lw)
    plt.errorbar(0.808,0.862, yerr=0.051,fmt='k*',ms=ms,mfc='none',lw=lw)
    plt.errorbar(1.077,0.930, yerr=0.051,fmt='k*',ms=ms,mfc='none',lw=lw)
    plt.errorbar(1.346,1.003, yerr=0.051,fmt='k*',ms=ms,mfc='none',lw=lw)
    plt.errorbar(1.615,1.083, yerr=0.053,fmt='k*',ms=ms,mfc='none',lw=lw)
    plt.errorbar(1.885,1.170, yerr=0.056,fmt='k*',ms=ms,mfc='none',lw=lw)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,loc='upper left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$f_T(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.tight_layout()
    plt.savefig('Plots/DKfTpoleinqsq.pdf')
    plt.close()
    
    plt.figure(figsize=figsize)
    plt.plot(z,ymean, color='g')
    plt.fill_between(z,ylow,yupp, color='g',alpha=alpha)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False)
    plt.xlabel('$z$',fontsize=fontsizelab)
    plt.ylabel(r'$f_T(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.tight_layout()
    plt.savefig('Plots/DKfTpoleinz.pdf')
    plt.close()
    return()

############################################################################################################
def sort_for_bar(data,spacing):
    y = []
    x = []
    i = 0
    while i+spacing <= 1:
        y.append(len(list(x for x in data if i <= x < i + spacing)))
        x.append(i+spacing/2)
        i += spacing
    return(x,y)
####################################################################################################################

def mass_corr_plots(Fit,fs,thpts,F,fsF):
    if Fit['conf'] in ['VCp','Cp','VC']:
        return()
    corrs = collections.OrderedDict()
    corrs['f0'] = []
    corrs['fp'] = []
    corrs['fT'] = []
    corrs['S'] = []
    corrs['V'] = []
    corrs['T'] = []
    corrs['Ss'] = []
    corrs['Vs'] = []
    corrs['f0s'] = []
    corrs['fps'] = []
    for i,mass in enumerate(Fit['masses']):
        for j in range(i+1,len(Fit['masses'])):
            mass2 = Fit['masses'][j]
            for k,twist in enumerate(Fit['twists']):
                for l in range(k,len(Fit['twists'])):
                    twist2 = Fit['twists'][l]
                    f0 = fs['f0_m{0}_tw{1}'.format(mass,twist)]
                    fp = fs['fp_m{0}_tw{1}'.format(mass,twist)]
                    fT = fs['fT_m{0}_tw{1}'.format(mass,twist)]

                    f02 = fs['f0_m{0}_tw{1}'.format(mass2,twist2)]
                    fp2 = fs['fp_m{0}_tw{1}'.format(mass2,twist2)]
                    fT2 = fs['fT_m{0}_tw{1}'.format(mass2,twist2)]
                    corrs['f0'].append(gv.evalcorr([f0,f02])[0][1])
                    if fp != None:
                        corrs['fp'].append(gv.evalcorr([fp,fp2])[0][1])
                    if fT != None:
                        corrs['fT'].append(gv.evalcorr([fT,fT2])[0][1])
                    for thpt in thpts[Fit['conf']]:
                        if twist != '0' or thpt != 'T':
                            V = Fit['{0}Vnn_m{1}_tw{2}'.format(thpt,mass,twist)]
                            V2 = Fit['{0}Vnn_m{1}_tw{2}'.format(thpt,mass2,twist2)]
                            corrs[thpt].append(gv.evalcorr([V,V2])[0][1])
    if Fit['conf'] == 'Fs':
        for mass in Fit['masses']:
            for twist in Fit['twists']:
                corrs['Ss'].append(gv.evalcorr([Fit['SVnn_m{0}_tw{1}'.format(mass,twist)],F['SVnn_m{0}_tw{1}'.format(mass,twist)]])[0][1])
                corrs['Vs'].append(gv.evalcorr([Fit['VVnn_m{0}_tw{1}'.format(mass,twist)],F['VVnn_m{0}_tw{1}'.format(mass,twist)]])[0][1])
                corrs['f0s'].append(gv.evalcorr([fs['f0_m{0}_tw{1}'.format(mass,twist)],fsF['f0_m{0}_tw{1}'.format(mass,twist)]])[0][1])
                if twist != '0':
                    corrs['fps'].append(gv.evalcorr([fs['fp_m{0}_tw{1}'.format(mass,twist)],fsF['fp_m{0}_tw{1}'.format(mass,twist)]])[0][1])
    for tag in ['f0','fp','fT','S','V','T']:
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
        plt.savefig('MassCorrs/sep_mass_{0}corrs{1}.pdf'.format(Fit['conf'],tag))
        plt.close()
    if Fit['conf'] == 'Fs':
        for tag in ['Ss','Vs','f0s','fps']:
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
            plt.savefig('MassCorrs/sep_mass_{0}sandlcorrs{1}.pdf'.format(Fit['conf'],tag))
            plt.close()
    return()

############################################################################################################



def new_f0_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata):
    i = 0
    plotfits = []
    for Fit in Fits:
        if Fit['conf'] not in ['UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
        if Fit['conf' ] == 'F':
            F = Fit
        if Fit['conf' ] == 'SF':
            SF = Fit
        j = 0
        for mass in Fit['masses']:
            qsq = []
            z = []
            y = []
            y2 = []
            qsq2 = []
            z2 = []
            for twist in Fit['twists']:
                q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                if Fit['conf'] == 'Fs':
                    z.append(make_z(q2,t_0,F['M_parent_m{0}'.format(mass)],F['M_daughter'],Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))#all lat units
                elif Fit['conf'] == 'SFs':
                    z.append(make_z(q2,t_0,SF['M_parent_m{0}'.format(mass)],SF['M_daughter'],Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))
                else:
                    z.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))
                y.append(fs_data[Fit['conf']]['f0_m{0}_tw{1}'.format(mass,twist)])
            q2max = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,Fit['twists'][0])]
            q2min = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,Fit['twists'][-1])]
            for q2 in np.linspace(q2min.mean,q2max.mean,nopts):
                y2.append(make_f0_BK(Nijk,Npow,Nm,addrho,pfit,Fit,q2,t_0,mass,fpf0same,float(mass)).mean)
                qsq2.append((q2/Fit['a']**2).mean)
                if Fit['conf'] == 'Fs':
                    z2.append(make_z(q2,t_0,F['M_parent_m{0}'.format(mass)],F['M_daughter'],Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']).mean)#all lat units
                elif Fit['conf'] == 'SFs':
                    z2.append(make_z(q2,t_0,SF['M_parent_m{0}'.format(mass)],SF['M_daughter'],Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']).mean)
                else:
                    z2.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']).mean)
            qsq,qsqerr = unmake_gvar_vec(qsq)
            z,zerr = unmake_gvar_vec(z)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(19,figsize=figsize)
            plt.plot(qsq2, y2, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            plt.figure(20,figsize=figsize)
            plt.plot(z2, y2, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(z, y, xerr=zerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            j += 1
        i += 1
    qsq = []
    z = []
    y = []
    #y2 =[]
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
        #y2.append(make_f0_BK(Nijk,Npow,Nm,addrho,pfit,Fits[7],q2*Fits[7]['a']**2,t_0,Fits[7]['masses'][3],fpf0same,0.8).mean)
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(19,figsize=figsize)
    plt.plot(qsq, ymean, color='b')
    #plt.plot(qsq, y2, color='c')
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
    plt.savefig('Plots/newf0poleinqsq.pdf')
    plt.close()
    
    plt.figure(20,figsize=figsize)
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
    plt.savefig('Plots/newf0poleinz.pdf')
    plt.close()
    return()
###############################################################################################################

def make_emu_bins(m_l):
    #bins = [[4*m_l**2,2],[2,4.3],[4.3,8.68],[10.09,12.86],[14.18,16.00],[16.00,qsqmaxphysBK.mean],[1,6],[0.1,2]]
    #bins = [[0.1,0.98],[1.1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[11,11.8]]#
    bins = [[11.8,12.5],[15,16],[16,17],[17,18],[18,19],[19,20],[20,21],[21,22],[1.1,6],[15,22]]
    return(bins)
###############################################################################################################
def Bemu_results_tables(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    m_lep = m_e # just does the bins 
    bins = make_emu_bins(m_lep)
    #bins2 =[[0.1,0.98],[1.1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[11,11.8],[11.8,12.5],[15,16],[16,17],[17,18],[18,19],[19,20],[20,21],[21,22],[1.1,6],[15,22]]
    line1 = '    \\begin{tabular}{ c |'
    line2 = '     $q^2$ bin &'
    for element in bins:
        line1 = '{0} c'.format(line1)
        qsq_min = element[0]
        qsq_max = element[1]
        if qsq_min == 4*m_lep**2:
            qsq_min = '4m_{\ell}^2'
        if qsq_max == qsqmaxphysBK.mean:
            qsq_max = 'q^2_{\\text{max}}'
        line2 = '{0} $({1},{2})$ '.format(line2,qsq_min,qsq_max)
        if element != bins[-1]:
            line2 = '{0}&'.format(line2)
    line1 = '{0} {1}'.format(line1,'}')
    table = open('Tables/Bemu.txt','w')
    table.write('{0}\n'.format(line1))
    table.write('      \hline\n')
    table.write('{0}\\\\ [0.5ex]\n'.format(line2))
    table.write('      \hline\n')
    table.write('      \hline\n')
    ########################## e p########################################
    m_lep = m_e
    bins = make_emu_bins(m_lep)
    p = make_p_physical_point_BK(pfit,Fits)
    line3 = '     $10^7\\mathcal{B}(B^+\\to K^+e^+e^-)$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        B = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauBpmGeV*1e7
        line3 = '{0} {1} '.format(line3,B)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ####################### e 0 ##########################################
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    line3 = '     $10^7\\mathcal{B}(B^0\\to K^0e^+e^-)$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        B = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauB0GeV*1e7
        line3 = '{0} {1} '.format(line3,B)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ######################## e ###########################################
    
    line3 = '     $10^7\\mathcal{B}(B\\to Ke^+e^-)$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        p = make_p_physical_point_BK(pfit,Fits)
        B1 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauBpmGeV*1e7
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        B2 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauB0GeV*1e7
        line3 = '{0} {1} '.format(line3,(B1+B2)/2)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    table.write('      \hline\n')
    ########################## mu ###############################################
    ########################## mu p########################################
    m_lep = m_mu
    bins = make_emu_bins(m_lep)
    p = make_p_physical_point_BK(pfit,Fits)
    line3 = '     $10^7\\mathcal{B}(B^+\\to K^+\\mu^+\\mu^-)$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        B = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauBpmGeV*1e7
        line3 = '{0} {1} '.format(line3,B)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ####################### e 0 ##########################################
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    line3 = '     $10^7\\mathcal{B}(B^0\\to K^0\\mu^+\\mu^-)$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        B = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauB0GeV*1e7
        line3 = '{0} {1} '.format(line3,B)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ######################## e ###########################################
    
    line3 = '     $10^7\\mathcal{B}(B\\to K\\mu^+\\mu^-)$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        p = make_p_physical_point_BK(pfit,Fits)
        B1 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauBpmGeV*1e7
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        B2 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauB0GeV*1e7
        line3 = '{0} {1} '.format(line3,(B1+B2)/2)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    table.write('      \hline\n')
    ########################## e mu ###############################################
     ########################## emu p########################################
    m_lep = m_mu
    bins = make_emu_bins(m_lep)
    p = make_p_physical_point_BK(pfit,Fits)
    line3 = '     $10^7\\mathcal{B}(B^+\\to K^+\\ell^+\\ell^-)$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        m_lep = m_e
        B1 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauBpmGeV*1e7
        m_lep = m_mu
        B2 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauBpmGeV*1e7
        line3 = '{0} {1} '.format(line3,(B1+B2)/2)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ####################### e 0 ##########################################
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    line3 = '     $10^7\\mathcal{B}(B^0\\to K^0\\ell^+\\ell^-)$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        m_lep = m_e
        B1 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauB0GeV*1e7
        m_lep = m_mu
        B2 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauB0GeV*1e7
        line3 = '{0} {1} '.format(line3,(B1+B2)/2)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ######################## e ###########################################
    
    line3 = '     $10^7\\mathcal{B}(B\\to K\\ell^+\\ell^-)$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        p = make_p_physical_point_BK(pfit,Fits)
        m_lep = m_e
        B1p = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauBpmGeV*1e7
        m_lep = m_mu
        B2p = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauBpmGeV*1e7
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        m_lep = m_e 
        B10 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauB0GeV*1e7
        m_lep = m_mu 
        B20 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauB0GeV*1e7
        line3 = '{0} {1} '.format(line3,(B1p+B2p+B10+B20)/4)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    table.write('      \hline\n')


    
    table.write('    \end{tabular}')        
    table.close()
    return()

#################################################################################################################



def Remu_results_tables(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    m_lep = m_e # just does the bins 
    bins = make_emu_bins(m_lep)
    #bins2 =[[0.1,0.98],[1.1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[11,11.8],[11.8,12.5],[15,16],[16,17],[17,18],[18,19],[19,20],[20,21],[21,22],[1.1,6],[15,22]]
    line1 = '    \\begin{tabular}{ c |'
    line2 = '     $q^2$ bin &'
    for element in bins:
        line1 = '{0} c'.format(line1)
        qsq_min = element[0]
        qsq_max = element[1]
        if qsq_min == 4*m_lep**2:
            qsq_min = '4m_{\mu}^2'
        if qsq_max == qsqmaxphysBK.mean:
            qsq_max = 'q^2_{\\text{max}}'
        line2 = '{0} $({1},{2})$ '.format(line2,qsq_min,qsq_max)
        if element != bins[-1]:
            line2 = '{0}&'.format(line2)
    line1 = '{0} {1}'.format(line1,'}')
    table = open('Tables/Remu.txt','w')
    table.write('{0}\n'.format(line1))
    table.write('      \hline\n')
    table.write('{0}\\\\ [0.5ex]\n'.format(line2))
    table.write('      \hline\n')
    table.write('      \hline\n')
    ########################## R mu/e p########################################
    m_lep = m_mu
    bins = make_emu_bins(m_lep)
    p = make_p_physical_point_BK(pfit,Fits)
    line3 = '     $10^3R^{\\mu(+)}_e$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        Bmu = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Be = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        R = 1e3*(Bmu/Be -1)
        line3 = '{0} {1} '.format(line3,R)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ####################### Rmue 0 ##########################################
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    line3 = '     $10^3R^{\\mu(0)}_e$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        Bmu = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Be = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        R = 1e3*(Bmu/Be -1)
        line3 = '{0} {1} '.format(line3,R)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ######################## Rmue ###########################################
    line3 = '     $10^3R^{\\mu}_e$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        p = make_p_physical_point_BK(pfit,Fits)
        B1mu = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        B1e = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        B2mu = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        B2e = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        R = 1e3*((B1mu+B2mu)/(B1e+B2e) -1)
        line3 = '{0} {1} '.format(line3,R)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    table.write('      \hline\n')
    #######################################################
    table.write('    \end{tabular}')        
    table.close()
    return()
##################################################################################################################

def Bnu_results_tables(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    bins = [[0.001,2],[2,4],[4,6],[6,8],[8,10],[10,12],[12,14],[14,16],[16,18],[18,20],[20,qsqmaxphysBK.mean]]
    line1 = '    \\begin{tabular}{ c |'
    line2 = '     $q^2$ bin &'
    for element in bins:
        line1 = '{0} c'.format(line1)
        qsq_min = element[0]
        qsq_max = element[1]
        if qsq_max == qsqmaxphysBK.mean:
            qsq_max = 'q^2_{\\text{max}}'
        line2 = '{0} $({1},{2})$ '.format(line2,qsq_min,qsq_max)
        if element != bins[-1]:
            line2 = '{0}&'.format(line2)
    line1 = '{0} {1}'.format(line1,'}')
    table = open('Tables/Bnu.txt','w')
    table.write('{0}\n'.format(line1))
    table.write('      \hline\n')
    table.write('{0}\\\\ [0.5ex]\n'.format(line2))
    table.write('      \hline\n')
    table.write('      \hline\n')
    ########################## B p ########################################
    p = make_p_physical_point_BK(pfit,Fits)
    line3 = '     $10^6\\mathcal{B}(B^+\\to K^+\\nu\\bar{\\nu}_{~\\text{SD}})$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        result = 1e6*integrate_fp_B(p,qsq_min,qsq_max,pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2)
        Bp = (tauBpmGeV * VtbVts**2 * GF**2 * alphaEW**2 * Xt**2 * result) /(32 * (np.pi)**5 * sinthw2**2)
        line3 = '{0} {1} '.format(line3,Bp)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ####################### Rmue 0 ##########################################
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    line3 = '     $10^6\\mathcal{B}(B^0\\to K^0\\nu\\bar{\\nu}_{~\\text{SD}})$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        result = 1e6*integrate_fp_B(p,qsq_min,qsq_max,pfit,Fits,Nijk,Npow,Nm,addrho,t_0,fpf0same,const2)
        B0 = (tauB0GeV * VtbVts**2 * GF**2 * alphaEW**2 * Xt**2 * result) /(32 * (np.pi)**5 * sinthw2**2)
        line3 = '{0} {1} '.format(line3,B0)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    table.write('      \hline\n')
    #######################################################
    table.write('    \end{tabular}')        
    table.close()
    return()

##################################################################################################################

def FHemu_results_tables(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    m_lep = m_e # just does the bins 
    bins = make_emu_bins(m_lep)
    #bins2 =[[0.1,0.98],[1.1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[11,11.8],[11.8,12.5],[15,16],[16,17],[17,18],[18,19],[19,20],[20,21],[21,22],[1.1,6],[15,22]]
    line1 = '    \\begin{tabular}{ c |'
    line2 = '     $q^2$ bin &'
    for element in bins:
        line1 = '{0} c'.format(line1)
        qsq_min = element[0]
        qsq_max = element[1]
        if qsq_min == 4*m_lep**2:
            qsq_min = '4m_{\ell}^2'
        if qsq_max == qsqmaxphysBK.mean:
            qsq_max = 'q^2_{\\text{max}}'
        line2 = '{0} $({1},{2})$ '.format(line2,qsq_min,qsq_max)
        if element != bins[-1]:
            line2 = '{0}&'.format(line2)
    line1 = '{0} {1}'.format(line1,'}')
    table = open('Tables/FHemu.txt','w')
    table.write('{0}\n'.format(line1))
    table.write('      \hline\n')
    table.write('{0}\\\\ [0.5ex]\n'.format(line2))
    table.write('      \hline\n')
    table.write('      \hline\n')
    ########################## e p########################################
    m_lep = m_e
    bins = make_emu_bins(m_lep)
    p = make_p_physical_point_BK(pfit,Fits)
    line3 = '     $10^8F_H^{e(+)}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        F = 1e8*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,F)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ####################### e 0 ##########################################
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    line3 = '     $10^8F_H^{e(0)}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        F = 1e8*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,F)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ######################## e ###########################################
    line3 = '     $10^8F_H^{e}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        p = make_p_physical_point_BK(pfit,Fits)
        F1 = 1e8*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        F2 = 1e8*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,(F1+F2)/2)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    table.write('      \hline\n')
    ########################## mu ###############################################
    m_lep = m_mu
    bins = make_emu_bins(m_lep)
    p = make_p_physical_point_BK(pfit,Fits)
    line3 = '     $10^3F_H^{\\mu(+)}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        F = 1e3*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,F)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ####################### mu 0 ##########################################
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    line3 = '     $10^3F_H^{\\mu(0)}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        F = 1e3*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,F)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ######################## mu ###########################################
    line3 = '     $10^3F_H^{\\mu}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        p = make_p_physical_point_BK(pfit,Fits)
        F1 = 1e3*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        F2 = 1e3*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,(F1+F2)/2)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    table.write('      \hline\n')
    ########################## e mu ###############################################
     ########################## emu p########################################
    m_lep = m_mu
    bins = make_emu_bins(m_lep)
    p = make_p_physical_point_BK(pfit,Fits)
    line3 = '     $10^3F_H^{\\ell(+)}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        m_lep = m_e
        F1 = 1e3*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        m_lep = m_mu
        F2 = 1e3*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,(F1+F2)/2)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ####################### e 0 ##########################################
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    line3 = '     $10^3F_H^{\\ell(+)}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        m_lep = m_e
        F1 = 1e3*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        m_lep = m_mu
        F2 = 1e3*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,(F1+F2)/2)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ######################## e ###########################################
    line3 = '     $10^3F_H^{\\ell(+)}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        p = make_p_physical_point_BK(pfit,Fits)
        m_lep = m_e
        F1p = 1e3*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        m_lep = m_mu
        F2p = 1e3*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        m_lep = m_e 
        F10 = 1e3*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        m_lep = m_mu 
        F20 = 1e3*integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,(F1p+F2p+F10+F20)/4)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    table.write('      \hline\n')


    
    table.write('    \end{tabular}')        
    table.close()
    return()
###############################################################################################################################

def Btau_results_tables(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    m_lep = m_tau 
    bins = [[4*m_lep**2,16],[16,18],[18,qsqmaxphysBK.mean],[14.18,16],[16,qsqmaxphysBK.mean],[15,17],[17,19],[19,22]]
    line1 = '    \\begin{tabular}{ c |'
    line2 = '     $q^2$ bin &'
    for element in bins:
        line1 = '{0} c'.format(line1)
        qsq_min = element[0]
        qsq_max = element[1]
        if qsq_min == 4*m_lep**2:
            qsq_min = '4m_{\\tau}^2'
        if qsq_max == qsqmaxphysBK.mean:
            qsq_max = 'q^2_{\\text{max}}'
        line2 = '{0} $({1},{2})$ '.format(line2,qsq_min,qsq_max)
        if element != bins[-1]:
            line2 = '{0}&'.format(line2)
    line1 = '{0} {1}'.format(line1,'}')
    table = open('Tables/Btau.txt','w')
    table.write('{0}\n'.format(line1))
    table.write('      \hline\n')
    table.write('{0}\\\\ [0.5ex]\n'.format(line2))
    table.write('      \hline\n')
    table.write('      \hline\n')
    ########################## tau p########################################
    p = make_p_physical_point_BK(pfit,Fits)
    line3 = '     $10^7\\mathcal{B}(B^+\\to K^+\\tau^+\\tau^-)$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        B = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauBpmGeV*1e7
        line3 = '{0} {1} '.format(line3,B)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ####################### tau 0 ##########################################
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    line3 = '     $10^7\\mathcal{B}(B^0\\to K^0\\tau^+\\tau^-)$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        B = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauB0GeV*1e7
        line3 = '{0} {1} '.format(line3,B)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ######################## tau ###########################################
    
    line3 = '     $10^7\\mathcal{B}(B\\to K\\tau^+\\tau^-)$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        p = make_p_physical_point_BK(pfit,Fits)
        B1 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauBpmGeV*1e7
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        B2 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)*tauB0GeV*1e7
        line3 = '{0} {1} '.format(line3,(B1+B2)/2)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    table.write('      \hline\n')

    
    table.write('    \end{tabular}')        
    table.close()
    return()
##########################################################################################


def Rtau_results_tables(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    m_lep = m_tau # just does the bins 
    bins = [[4*m_lep**2,16],[16,18],[18,qsqmaxphysBK.mean],[14.18,16],[16,qsqmaxphysBK.mean],[15,17],[17,19],[19,22]]
    line1 = '    \\begin{tabular}{ c |'
    line2 = '     $q^2$ bin &'
    for element in bins:
        line1 = '{0} c'.format(line1)
        qsq_min = element[0]
        qsq_max = element[1]
        if qsq_min == 4*m_lep**2:
            qsq_min = '4m_{\\tau}^2'
        if qsq_max == qsqmaxphysBK.mean:
            qsq_max = 'q^2_{\\text{max}}'
        line2 = '{0} $({1},{2})$ '.format(line2,qsq_min,qsq_max)
        if element != bins[-1]:
            line2 = '{0}&'.format(line2)
    line1 = '{0} {1}'.format(line1,'}')
    table = open('Tables/Rtau.txt','w')
    table.write('{0}\n'.format(line1))
    table.write('      \hline\n')
    table.write('{0}\\\\ [0.5ex]\n'.format(line2))
    table.write('      \hline\n')
    table.write('      \hline\n')
    ########################## e p########################################
    p = make_p_physical_point_BK(pfit,Fits)
    line3 = '     $R^{\\tau(+)}_e$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        Btau = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Be = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,Btau/Be)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ####################### e 0 ##########################################
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    line3 = '     $R^{\\tau(0)}_e$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        Btau = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Be = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,Btau/Be)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ######################## e ###########################################
    line3 = '     $R^{\\tau}_e$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        p = make_p_physical_point_BK(pfit,Fits)
        Btau1 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Be1 = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        Btau2 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Be2 = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,(Btau1/Be1 + Btau2/Be2)/2)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    table.write('      \hline\n')
    ########################## mu ###############################################
    ########################## mu p########################################
    p = make_p_physical_point_BK(pfit,Fits)
    line3 = '     $R^{\\tau(+)}_{\\mu}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        Btau = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Bmu = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,Btau/Bmu)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ####################### mu 0 ##########################################
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    line3 = '     $R^{\\tau(0)}_{\\mu}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        Btau = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Bmu = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,Btau/Bmu)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ######################## mu ###########################################
    line3 = '     $R^{\\tau}_{\\mu}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        p = make_p_physical_point_BK(pfit,Fits)
        Btau1 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Bmu1 = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        Btau2 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Bmu2 = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,(Btau1/Bmu1+Btau2/Bmu2)/2)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    table.write('      \hline\n')
    ########################## e mu ###############################################
     ########################## emu p########################################
    p = make_p_physical_point_BK(pfit,Fits)
    line3 = '     $R^{\\tau(+)}_{\\ell}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        Btau = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Be = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Bmu = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,(Btau/Be + Btau/Bmu)/2)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ####################### emu 0 ##########################################
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    line3 = '     $R^{\\tau(0)}_{\\ell}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        Btau = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Be = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Bmu = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,(Btau/Be + Btau/Bmu)/2)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ######################## emu ###########################################
    
    line3 = '     $R^{\\tau}_{\\ell}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        p = make_p_physical_point_BK(pfit,Fits)
        Btau1 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Be1 = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Bmu1 = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        Btau2 = integrate_Gamma(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Be2 = integrate_Gamma(p,qsq_min,qsq_max,m_e,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        Bmu2 = integrate_Gamma(p,qsq_min,qsq_max,m_mu,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,(Btau1/Be1 + Btau1/Bmu1 + Btau2/Be2 + Btau2/Bmu2)/4)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    table.write('      \hline\n')


    
    table.write('    \end{tabular}')        
    table.close()
    return()

#################################################################################################################

def FHtau_results_tables(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    m_lep = m_tau 
    bins = [[4*m_lep**2,16],[16,18],[18,qsqmaxphysBK.mean],[14.18,16],[16,qsqmaxphysBK.mean],[15,17],[17,19],[19,22]]
    line1 = '    \\begin{tabular}{ c |'
    line2 = '     $q^2$ bin &'
    for element in bins:
        line1 = '{0} c'.format(line1)
        qsq_min = element[0]
        qsq_max = element[1]
        if qsq_min == 4*m_lep**2:
            qsq_min = '4m_{\\tau}^2'
        if qsq_max == qsqmaxphysBK.mean:
            qsq_max = 'q^2_{\\text{max}}'
        line2 = '{0} $({1},{2})$ '.format(line2,qsq_min,qsq_max)
        if element != bins[-1]:
            line2 = '{0}&'.format(line2)
    line1 = '{0} {1}'.format(line1,'}')
    table = open('Tables/FHtau.txt','w')
    table.write('{0}\n'.format(line1))
    table.write('      \hline\n')
    table.write('{0}\\\\ [0.5ex]\n'.format(line2))
    table.write('      \hline\n')
    table.write('      \hline\n')
    ########################## tau p########################################
    p = make_p_physical_point_BK(pfit,Fits)
    line3 = '     $F_H^{\\tau(+)}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        F = integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,F)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ####################### tau 0 ##########################################
    p = make_p_physical_point_BK(pfit,Fits,B0=True)
    line3 = '     $F_H^{\\tau(0)}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        F = integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,F)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    ######################## tau ###########################################
    line3 = '     $F_H^{\\tau}$ &'
    for element in bins:
        qsq_min = element[0]
        qsq_max = element[1]
        p = make_p_physical_point_BK(pfit,Fits)
        F1 = integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        p = make_p_physical_point_BK(pfit,Fits,B0=True)
        F2 = integrate_FH(p,qsq_min,qsq_max,m_lep,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2)
        line3 = '{0} {1} '.format(line3,(F1+F2)/2)
        if element != bins[-1]:
            line3 = '{0}&'.format(line3)    
    table.write('{0}\\\\ [1.0ex]\n'.format(line3))
    table.write('      \hline\n')

    
    table.write('    \end{tabular}')        
    table.close()
    return()
##########################################################################################






























































































































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

def old_error_plot(pfit,prior,Fits,Nijk,Npow,f,t_0,Del,addrho,fpf0same):
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

##################################################

def old_dBdq2_plots(pfit,t_0,Fits,fpf0same,Nijk,Npow,Nm,addrho,const2):
    #Jpsim = gv.gvar('3.096900(6)')
    #Jpsiw = gv.gvar('0.0000929(28)')
    #Psi2sm = gv.gvar('3.68610(6)')
    #Psi2sw = gv.gvar('0.000294(8)')
    p = make_p_physical_point_BK(pfit,Fits)
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

    plt.fill_between([8.68,10.11],[-5,-5],[45,45], color='k',alpha=alpha)
    plt.fill_between([12.86,14.18],[-5,-5],[45,45], color='k',alpha=alpha)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel('$10^9 d\mathcal{B}_{\mu}/dq^2~[\mathrm{GeV}^{-2}]$',fontsize=fontsizelab)
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
    plt.axes().set_ylim([-2,42])
    plt.tight_layout()
    plt.savefig('Plots/dBdq2.pdf')
    plt.close()
    
    return()
################################
def fT_no_pole_in_qsq_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata):
    i = 0
    plotfits = []
    for Fit in Fits:
        if Fit['conf'] not in ['VCp','Cp','Fs','SFs','UFs']:
            plotfits.append(Fit)
    for Fit in plotfits:
        fit = Fit['conf']
        j = 0
        for mass in Fit['masses']:
            qsq = []
            z = []
            y = []
            y2 = []
            qsq2 = []
            z2 = []
            MHsstar = make_MHsstar(Fit['M_parent_m{0}'.format(mass)],pfit,Fit['a'])
            for twist in Fit['twists']:
                if fs_data[Fit['conf']]['fT_m{0}_tw{1}'.format(mass,twist)] != None:
                    q2 = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,twist)] # lat units
                    qsq.append(q2/Fit['a']**2) #want qsq for the x value in GeV
                    z.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']))#all lat units
                    pole = 1-(q2/MHsstar**2)
                    y.append(pole*fs_data[Fit['conf']]['fT_m{0}_tw{1}'.format(mass,twist)])
            q2max = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,Fit['twists'][1])]
            q2min = fs_data[Fit['conf']]['qsq_m{0}_tw{1}'.format(mass,Fit['twists'][-1])]
            MHsstar = make_MHsstar(pfit['MH_{0}_m{1}'.format(fit,mass)],pfit,pfit['a_{0}'.format(fit)])
            for q2 in np.linspace(q2min.mean,q2max.mean,nopts):
                qsq2.append((q2/Fit['a']**2).mean)
                z2.append(make_z(q2,t_0,Fit['M_parent_m{0}'.format(mass)],Fit['M_daughter']).mean)
                pole = 1-(q2/MHsstar**2)
                y2.append((pole*make_fT_BK(Nijk,Npow,Nm,addrho,pfit,Fit,q2,t_0,mass,fpf0same,float(mass))).mean)
            qsq,qsqerr = unmake_gvar_vec(qsq)
            z,zerr = unmake_gvar_vec(z)
            y,yerr = unmake_gvar_vec(y)
            
            plt.figure(12,figsize=figsize)
            plt.plot(qsq2, y2, color=cols[j], mfc='none',linestyle=lines[i])
            plt.errorbar(qsq, y, xerr=qsqerr, yerr=yerr, color=cols[j], fmt=symbs[i],ms=ms, mfc='none',label=('{0} m{1}'.format(Fit['label'],mass)))
            
            plt.figure(13,figsize=figsize)
            plt.plot(z2, y2, color=cols[j], mfc='none',linestyle=lines[i])
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
        #        make_fp_BsEtas(Nijk,Npow,addrho,p,Fit,alat,qsq,z,mass,fpf0same,amh)
        y.append(pole*make_fT_BK(Nijk,Npow,Nm,addrho,p,Fits[0],q2,t_0,Fits[0]['masses'][0],fpf0same,0)) #only need one fit
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(12,figsize=figsize)
    plt.plot(qsq, ymean, color='g')
    plt.fill_between(qsq,ylow,yupp, color='g',alpha=alpha)
    #if datafTmaxBK != None and adddata:
    #    plt.errorbar(qsqmaxphysBK.mean, datafTmaxBK.mean, xerr=qsqmaxphysBK.sdev, yerr=datafTmaxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False, ncol=3)
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.ylabel(r'$\left(1-\frac{q^2}{M^2_{H_{s}^*}} \right)f_T(q^2)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig('Plots/fTnopoleinqsq.pdf')
    plt.close()
    
    plt.figure(13,figsize=figsize)
    plt.plot(z,ymean, color='g')
    plt.fill_between(z,ylow,yupp, color='g',alpha=alpha)
    #if datafTmaxBK != None and adddata:
    #    plt.errorbar( make_z(qsqmaxphysBK,t_0,MBphys,MKphys).mean, datafTmaxBK.mean, xerr=make_z(qsqmaxphysBK,t_0,MBphys,MKphys).sdev, yerr=datafTmaxBK.sdev, color='purple', fmt='D',ms=ms, mfc='none',label = r'$arXiv 1510.07446$')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False, ncol=3)
    plt.xlabel('$z$',fontsize=fontsizelab)
    plt.ylabel(r'$\left(1-\frac{q^2}{M^2_{H_{s}^*}} \right)f_T(z)$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig('Plots/fTnopoleinz.pdf')
    plt.close()
    return()

################################################################################################
def p_in_z(fs_data,pfit,Fits,t_0,Nijk,Npow,Nm,addrho,fpf0same,adddata,const2):
    z = []
    y = []
    p = make_p_physical_point_BK(pfit,Fits)
    for q2 in np.linspace(0,qsqmaxphysBK.mean,nopts): #q2 now in GeV
        zed = make_z(q2,t_0,p['MBphys'],p['MKphys']) #all GeV dimensions
        if zed ==0:
            z.append(zed)
        else:
            z.append(zed.mean)
        y.append((q2-p['MKphys']**2-p['MDphys']**2)**2/(4*p['MDphys']**2)-p['MKphys']**2) 
    ymean,yerr = unmake_gvar_vec(y)
    yupp,ylow = make_upp_low(y)
    plt.figure(figsize=figsize)
    plt.plot(z,ymean, color='r')
    plt.fill_between(z,ylow,yupp, color='r',alpha=alpha)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=fontsizeleg,frameon=False,ncol=3)
    plt.xlabel('$z$',fontsize=fontsizelab)
    plt.ylabel(r'$|\vec{p}_K|^2[\mathrm{GeV}]^2$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    #plt.axes().yaxis.set_major_locator(MultipleLocator(1.0))
    #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.2))
    #plt.ylim([0.2,3.4])
    plt.tight_layout()
    plt.savefig('Plots/pinz.pdf')
    plt.close()
    return()

################################

