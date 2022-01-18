from functionsBK import *
import numpy as np
import gvar as gv
import scipy
import matplotlib.pyplot as plt
import collections
#import cython
#################################### Matplotlib stuff ######################
plt.rc("font",**{"size":20})
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
factor = 1.0 #multiplies everything to make smaller for big plots (0.5) usually 1
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


####################################################################
#Here we seek to calculate the corrections in appendix B1 and B2 of 1510.02349. Makes use of C++ header files from 0810.4077

#first, corrections to C_9^eff. Currently, the real part of this is of size 4-5.
#The largest correction is alpha_s/4pi (C1*F_1,c(9) + C2*F_2,c(9) + C8F_8^(9)]
#C8 = gv.gvar('-0.152(15)')#from 1510.02349, can't find this in paper we are using
C8 = C8eff - (C3 - 1/6 * C4 + 20 * C5 - 10/3 * C6 )
alpha_s = gv.gvar('0.2253(28)') #Used qcdevol
lambda_us = gv.gvar('0.01980(62)') #should be (PDG) VusVub/VtsVtb
N_c = 3 # Assuming this is N_colours
M_B = (MBphysp + MBphys0)/2#get this from p need mean for Ei
M_K = (MKphysp + MKphys0)/2
w0 = 1/gv.gvar('3(1)') # from paper (look up again)
C_F = 4/3 # see page 44
a_1 = gv.gvar('0.0453(30)')    # at 4 GeV**2 hep-lat/0606012 updates?
a_2 = gv.gvar('0.175(50)')     # at 4 GeV**2 0606012 updates?
print('4m_e^2 = ',4*m_e**2)
###################################### A couple of general functions ##########################################
def mymean(x):
    #returns the mean of a gvar is passed on, or a float if passed a float. Neeeded when function takes both gvars and floats and have if statements about them i.e. if x.mean <0
    if isinstance(x,gv._gvarcore.GVar):
        return(x.mean)
    else:
        return(x)

def zeros(x):  # takes gvars and sets them to floats if their mean is 0. This hopefully avoids issues with 0(0) 
    if mymean(x[0]) == 0:
        x[0] = 0
    if mymean(x[1]) == 0:
        x[1] = 0
    return(x)

def c_arg(topR,topI,botR,botI): #retunrs the real and imaginary parts of a complex argument for example x/x-1
    denom = botR**2 + botI**2
    R = (topR*botR + topI*botI)/denom
    I = (topI*botR - topR*botI)/denom
    return([R,I])

def check_converged(a):
    #takes a list of values and looks for relative change to asses if the integral has converged. Needs to check in cases with floats, gvars, gavrs with 0 mean and gvars with 0 sdev. Returns True if the
    if isinstance(a[-1],gv._gvarcore.GVar):
        if a[-1].mean == 0: # some parts are 0 becase the integral is purely real or imaginary. We bypass this case.
            return(True)
        elif a[-1].sdev == 0:
            check = abs((a[-1]-a[-2])/a[-1])
            if check < 0.001: #absolute value changes by less than 1%
                return(True)
            else:
                return(False)
        else:
            check = abs((a[-1].mean-a[-2].mean)/a[-1].sdev)
            if check < 0.05: #changes by % of a sigma - allows for slower/quicker convergence 5% seems fine
                return(True)
            else:
                return(False)
    else:
        if a[-1] == 0.0:
            return(True)
        else:    
            check = abs((a[-1]-a[-2])/a[-1])
            if check < 0.001: #absolute value changes by less than 1%
                return(True)
            else:
                return(False)

######################################################################################################################
    
def make_correction1(qsq): # dominant O(alpha_s) correction to C9eff in B11. Currently missing F1c9 and F2c9 (in C++)
    F19R,F19I,F29R,F29I = make_F1c9_F2c9(qsq) 
    F89 = make_F89(qsq)
    corrR = (alpha_s/(4*np.pi)) * (C1*F19R + C2*F29R + C8*F89 )
    corrI = (alpha_s/(4*np.pi)) * (C1*F19I + C2*F29I)
    print('C9eff correction O(alpha_s) =  {0} + {1}i'.format(corrR,corrI))
    return(corrR,corrI)

def make_correction2(qsq): #O(lambda_u^s) correction to C9eff in B11
    hcR,hcI = make_h(qsq,m_c)
    h0R,h0I = make_h(qsq,0)
    corrR = lambda_us * (hcR-h0R)*(4/3 *C1 + C2)
    corrI = lambda_us * (hcI-h0I)*(4/3 *C1 + C2)
    print('C9eff correction O(lambd_us) = {0} + {1}i'.format(corrR,corrI))
    return(corrR,corrI)

def make_correction3(qsq): #this is the O(alpha_s) correction to C7eff which is as defined in FNAL)
    F17R,F17I = make_F1c7(qsq)
    F87R,F87I = make_F87(qsq)
    corrR = (alpha_s/(4*np.pi)) * ((C1-6*C2)*F17R + C8*F87R)
    corrI = (alpha_s/(4*np.pi)) * ((C1-6*C2)*F17I + C8*F87I)
    print('C7eff correction O(alpha_s) =  {0} + {1}i'.format(corrR,corrI))
    return(corrR,corrI)

def read_fileF171929():
    f = collections.OrderedDict()
    f['qsqs'] = []
    f['F17Rs'] = []
    f['F17Is'] = []
    f['F19Rs'] = []
    f['F19Is'] = []
    f['F29Rs'] = []
    f['F29Is'] = []
    data = open('Fits/F17F19F29.txt','r')
    #data = open('Fits/F19F29_c02removed.txt','r')
    #data = open('Fits/F19F29_b02removed.txt','r')
    lines = data.readlines()
    for line in lines:
        numbs = line.split()
        #print(numbs)
        f['qsqs'].append(float(numbs[0])) 
        f['F17Rs'].append(float(numbs[1]))
        f['F17Is'].append(float(numbs[2]))
        f['F19Rs'].append(float(numbs[3]))
        f['F19Is'].append(float(numbs[4]))
        f['F29Rs'].append(float(numbs[5]))
        f['F29Is'].append(float(numbs[6]))
    data.close()
    return(f)

#def read_fileF17():
#    f = collections.OrderedDict()
#    f['qsqs'] = []
#    f['F17Rs'] = []
#    f['F17Is'] = []
#    data = open('Fits/F17_new.txt','r')
#    lines = data.readlines()
#    for line in lines:
#        numbs = line.split()
#        f['qsqs'].append(float(numbs[0])) 
#        f['F17Rs'].append(float(numbs[1]))
#        f['F17Is'].append(float(numbs[2]))
#    data.close()
#    return(f)

F19_dat = read_fileF171929()

def make_F1c9_F2c9(qsq):
    #we have evalauted these where possible for some (not evnely spaced) choice of q^2
    # Saved in file Fits/F19F29_final.txt in form qsq F19R F19I F29R F29I
    # We want to read these and lineraly interpolate
    low = 0
    high = 25
    for q in F19_dat['qsqs']:
        if q <= qsq and q > low:
            low = q
        if q >= qsq and q < high:
            high = q
    i = F19_dat['qsqs'].index(low)
    j = F19_dat['qsqs'].index(high)
    if qsq == high:
        grad = 1
    else:
        grad = (qsq - low)/(high-low)
    F19R = F19_dat['F19Rs'][i] + grad*(F19_dat['F19Rs'][j]-F19_dat['F19Rs'][i])
    F19I = F19_dat['F19Is'][i] + grad*(F19_dat['F19Is'][j]-F19_dat['F19Is'][i])
    F29R = F19_dat['F29Rs'][i] + grad*(F19_dat['F29Rs'][j]-F19_dat['F29Rs'][i])
    F29I = F19_dat['F29Is'][i] + grad*(F19_dat['F29Is'][j]-F19_dat['F29Is'][i])
    return(F19R,F19I,F29R,F29I)

def make_F1c7(qsq):
    low = 0
    high = 25
    for q in F19_dat['qsqs']:
        if q <= qsq and q > low:
            low = q
        if q >= qsq and q < high:
            high = q
    i = F19_dat['qsqs'].index(low)
    j = F19_dat['qsqs'].index(high)
    if qsq == high:
        grad = 1
    else:
        grad = (qsq - low)/(high-low)
    F17R = F19_dat['F17Rs'][i] + grad*(F19_dat['F17Rs'][j]-F19_dat['F17Rs'][i])
    F17I = F19_dat['F17Is'][i] + grad*(F19_dat['F17Is'][j]-F19_dat['F17Is'][i])
    return(F17R,F17I)

def make_F87(qsq):
    s = (qsq/m_b**2)
    B0 = make_B0s(s)
    C0 = make_C0s(s)
    R = -32/9 * gv.log(mu_scale/m_b) - 8/9 * s/(1-s) * gv.log(s) - 4/9 * (11-16*s+8*s**2)/(1-s)**2 + 4/9 * 1/(1-s)**3 * ((9*s-5*s**2+2*s**3)*B0 - (4+2*s)*C0)
    I = -8/9 * np.pi
    return(R,I)

def make_F89(qsq):
    s = (qsq/m_b**2)
    B0 = make_B0s(s)
    C0 = make_C0s(s)
    F =  16/9 * 1/(1-s) * gv.log(s) + 8/9 * (5-2*s)/(1-s)**2 - 8/9 * (4-s)/(1-s)**3 * ( (1+s)*B0 - 2*C0)
    return(F)

def make_B0s(s):
    B = -2 * gv.sqrt(4/s -1) * gv.arctan(1/gv.sqrt(4/s - 1))
    #print('B = ',B)
    return(B)

def make_C0s(s):
    s = mymean(s) # the uncertainty here is tiny so we do this to speed up the integral
    eps = 1e-2 #1e-2 works but very slow for small q^2. Could add more terms to expansion and use larger eps
    def fcn1(x):
        y = 1/(x*(1-s)+1) * gv.log(x**2/(1-x*(1-x)*s))
        return(y)
    
    def fcn2(x):
        if x ==0:
            y = 0  
        else:
            expansion = 0
            for i in range(1,10): #include 10 terms
                expansion += (-1)**i * (x*(1-s))**i # expand 1/(1+x(1-s))about small x
            y = - (1+expansion) * gv.log((1-x*(1-x)*s)) + 2*expansion * gv.log(x)
        return(y)
    C1 = do_integral(fcn1,eps,1)
    C2 = do_integral(fcn2,0,eps)
    C = C1 + C2 + 2*(eps*gv.log(eps)-eps)
    #print('C = ',C)
    return(C)



def do_integral(fcn,low,upp): # generic integrator for a real function (one value). Takes the function of the integrand and uses the trapeziodal rule. Starts with 16 iters and doubles until stability condition is met. 
    integrands = gv.BufferDict() # Use this to save integrands at certain values, to avoid calculating again
    iters = int(16)
    Test = False
    results = []
    while Test == False:
        points = np.linspace(low,upp,iters+1)
        del_qsq =  (upp-low) /iters
        if low in integrands:
            l = integrands[low]
        else:
            l = fcn(low)
            integrands[low] = l
        if upp in integrands:
            h = integrands[upp]
        else:
            h = fcn(upp)
            integrands[upp] = h
        funcs = l + h
        for i in range(1,iters):
            if points[i] in integrands:
                f = integrands[points[i]]
            else:
                f = fcn(points[i])
                integrands[points[i]] = f
            funcs += 2*f
        result1 = del_qsq*funcs/2
        results.append(result1)
        iters *= 2
        print('iters',int(iters/2),results[-2:])
        if len(results)>=2:
            Test = check_converged(results)
                
    return(result1)

def do_complex_integral(fcn,low,upp): #Same as above but takes functions with real and imaginary parts. Integrates both separately.
    integrandsR = gv.BufferDict()
    integrandsI = gv.BufferDict()
    iters = int(16)
    Test = False
    resultsR = []
    resultsI = []
    while Test == False:
        points = np.linspace(low,upp,iters+1) 
        del_qsq =  (upp-low) /iters
        if low in integrandsR:
            lR = integrandsR[low]
            lI = integrandsI[low]
        else:
            lR,lI = fcn(low)
            integrandsR[low] = lR
            integrandsI[low] = lI
        if upp in integrandsR:
            hR = integrandsR[upp]
            hI = integrandsI[upp]
        else:
            hR,hI = fcn(upp)
            integrandsR[upp] = hR
            integrandsI[upp] = hI
        funcsR = lR+hR
        funcsI = lI+hI
        for i in range(1,iters):
            if points[i] in integrandsR:
                fR = integrandsR[points[i]]
                fI = integrandsI[points[i]]
            else:
                fR,fI = fcn(points[i])
                integrandsR[points[i]] = fR
                integrandsI[points[i]] = fI
            funcsR += 2*fR
            funcsI += 2*fI
        resultR = del_qsq*funcsR/2
        resultI = del_qsq*funcsI/2
        resultsR.append(resultR)
        resultsI.append(resultI)
        iters *= 2
        print('iters',int(iters/2),resultsR[-2:],resultsI[-2:])
        #print(resultsR)
        #print(resultsI)
        if len(resultsR)>=2:
            checkR = check_converged(resultsR)
            checkI = check_converged(resultsI)
            if checkR == True and checkI == True:
                Test = True
    #print('Final iters',int(iters/2))
    return(resultR,resultI)


######################## Non factorisable bits  Delta C9eff in eq B26 #######################################################

def make_DelC9eff(qsq): #makes whole Del C9 eff
    Del_taupRp,Del_taupIp,Del_taupR0,Del_taupI0 = make_Del_taup(qsq) 
    DelRp = (2 * m_b * Del_taupRp )/(M_B) # need to divide by f+
    DelIp = (2 * m_b * Del_taupIp )/(M_B) # need to divide by f+, do this in functionsBK
    DelR0 = (2 * m_b * Del_taupR0 )/(M_B) # need to divide by f+
    DelI0 = (2 * m_b * Del_taupI0 )/(M_B) # need to divide by f+, do this in functionsBK
    print('Del C_9^eff(B^+) * f_+ = {0} + {1}i'.format(DelRp,DelIp))
    print('Del C_9^eff(B^0) * f_+ = {0} + {1}i'.format(DelR0,DelI0))
    return(DelRp,DelIp,DelR0,DelI0)

def make_Del_taup(qsq):#Makes Del tau, split into 4 parts plus1 plus2, minus1,minus2 these are the 4 parts B27 can be split into. I.e. plus 1 is the Tp+^(0) bit , plus2 is the alpha_sCF TP+^(nf) bit and same for minus. 
    N = (np.pi**2*fBp*fKp)/(N_c*M_B) # Factor out front
    plus1 = 0 # Tp+ ^0 =0#
    minus1R,minus1I = make_minus1_integral(qsq)
    minus2R,minus2I = make_minus2_integral(qsq)
    print('plus')
    plus2R,plus2I = make_plus2_integral(qsq)
    eq = 2/3 #put in charge
    DelRp = N * (plus1 + eq*minus1R + plus2R + eq*minus2R)
    DelIp = N * (plus1 + eq*minus1I + plus2I + eq*minus2I)
    eq = -1/3
    DelR0 = N * (plus1 + eq*minus1R + plus2R + eq*minus2R)
    DelI0 = N * (plus1 + eq*minus1I + plus2I + eq*minus2I)
    return(DelRp,DelIp,DelR0,DelI0)

def make_Del_bits(qsq,bit):
    fac = 2 * m_b/M_B
    N = (np.pi**2*fBp*fKp)/(N_c*M_B) # Factor out front just use + for this 
    plus1 = 0 # Tp+ ^0 =0#
    if bit == 'TP-0':
        minus1R,minus1I = make_minus1_integral(qsq)
        return(fac*N*minus1R,fac*N*minus1I)
    if bit == 'TP-nf':
        minus2R,minus2I = make_minus2_integral(qsq)
        return(fac*N*minus2R,fac*N*minus2I)
    if bit == 'TP+nf':
        plus2R,plus2I = make_plus2_integral(qsq)
        return(fac*N*plus2R,fac*N*plus2I)

def make_minus1_integral(qsq):
    #T_P-^(0) is not a function of u, so we just need to integrate phi_K, which we can do manually, and then perform the omega integral. We leave out e_q which is put in later
    # Different value for + ('p') or 0 charge. Here we take 'p' as default and we can pass the correct one later
    fac = ( 4 * M_B/m_b ) *  (C3 + 4/3*C4 + 16*C5 + 64/3*C6) 
    #u_int_0 = 1
    #u_int_1 = 0 #integral of phi_K over u # C_1^3/2(x) =  3x
    #u_int_2 = 0 #  C_2^3/2(x) = 15/2x^2 -3/2
    # so the 0th order u integral is 1 and the first and second order ones are 0; we only need to integrate over w:
    Ei = scipy.special.expi(qsq/(M_B.mean*w0.mean))# This function won't take Gvars 
    integralR = M_B * gv.exp(-qsq/(M_B*w0))/w0 * (-Ei) # There is an extra factor of M_B in TP-^0 vs that in eq B43 (check)
    integralI = M_B * gv.exp(-qsq/(M_B*w0))/w0 * np.pi
    resultR = fac * integralR
    resultI = fac * integralI
    return(resultR,resultI)

def make_minus2_integral(qsq): #Similar to minus1 but this time TP-^nf has 3 distinct parts. Do p and 0 above
    overall_fac = alpha_s/(4*np.pi) * C_F
    # last term in TP-nf only has w dependence, which is the same as Tp- with int_u =1
    Ei = scipy.special.expi(qsq/(M_B.mean*w0.mean))
    integralR = M_B * gv.exp(-qsq/(M_B*w0))/w0 * (-Ei) # There is an extra factor of M_B in TP-^nf vs that in eq B43 (check)
    integralI = M_B * gv.exp(-qsq/(M_B*w0))/w0 * np.pi
    fac_last =  6 * M_B/m_b * 8/27 * (-15/2 * C4 + 12*C5 -32 *C6)
    lastR =  fac_last * integralR
    lastI =  fac_last * integralI
    #first term is next easiest. Here we need to just integrate over u. The w integral is the same.
    fac_first = -8*C8eff
    def first_func(u):
        overall = 6*u*(1-u) * 1/(1 - u + qsq*u/M_B**2) # real
        R0 = overall #0th order, can add others as we wish
        R1 = overall * a_1 * 3*(2*u-1)
        R2 = overall * a_2 * (15/2 *(2*u-1)**2 -3/2)
        R = R0 + R1 + R2
        return(R)
        
    u_int = do_integral(first_func,0,1) 
    firstR = fac_first * u_int * integralR # w integral same as usual
    firstI = fac_first * u_int * integralI
    #Middle bit. We again need to do the u integral numerically. w integral is the same. 
    fac_mid = -6 * M_B/m_b
    
    def mid_func(u): # for 0th component of u int. Can add others later
        harg = (1-u)*M_B.mean**2 + u*qsq  # make_h doesn't like gvars for qsq
        h1R,h1I = make_h(harg,m_c)
        h2R,h2I = make_h(harg,m_b)
        h3R,h3I = make_h(harg,0)
        fac1 = -C1/6 + C2 + C4 + 10*C6
        fac2 = C3 + 5/6 * C4 + 16*C5 + 22/3 *C6
        fac3 = C3 + 17/6 * C4 + 16 * C5 + 82/3 *C6
        overallR = 6*u*(1-u)*(fac1*h1R + fac2*h2R + fac3*h3R)
        overallI = 6*u*(1-u)*(fac1*h1I + fac2*h2I + fac3*h3I)
        R0 = overallR
        I0 = overallI
        R1 = overallR * a_1 * 3*(2*u-1)
        I1 = overallI * a_1 * 3*(2*u-1)
        R2 = overallR * a_2 * (15/2 * (2*u-1)**2 -3/2)
        I2 = overallI * a_2 * (15/2 * (2*u-1)**2 -3/2)
        R = R0 + R1 + R2
        I = I0 + I1 + I2
        return(R,I)
    
    u_intR,u_intI = do_complex_integral(mid_func,0,1)
    midR = fac_mid * ( u_intR * integralR - u_intI * integralI)
    midI = fac_mid * ( u_intI * integralR + u_intR * integralI)
    ################################################
    resultR = overall_fac * (lastR + firstR + midR)
    resultI = overall_fac * (lastI + firstI + midI)
    return(resultR,resultI)

def make_plus2_integral(qsq): # This is the trickiest: TP+^nf but has no w dependence so we just need to do the u integral
    fac = alpha_s/(4*np.pi) * C_F * M_B/m_b
    e_u = 2/3
    e_d = -1/3
    def func(u):
        fac1 = e_u * (-C1/6 + C2 + 6*C6)
        fac2 = e_d * (C3 - C4/6 + 16*C5 + 10*C6/3)
        fac3 = e_d * (C3 - C4/6 + 16*C5 - 8*C6/3)
        t1R,t1I = make_t_parr(u,m_c,qsq)
        t2R,t2I = make_t_parr(u,m_b,qsq)
        t3R,t3I = make_t_parr(u,0,qsq)
        overallR = 6*u* (fac1*t1R + fac2*t2R + fac3*t3R)#We manually cancal the 1-u in B40 with the ubar in the denom of t_parr
        overallI = 6*u* (fac1*t1I + fac2*t2I + fac3*t3I)#We manually cancal the 1-u in B40 with the ubar in the denom of t_parr
        itg0R = overallR #0th order
        itg0I = overallI
        itg1R = overallR * a_1 * 3*(2*u-1)
        itg1I = overallI * a_1 * 3*(2*u-1)
        itg2R = overallR * a_2 * (15/2 * (2*u-1)**2 -3/2)
        itg2I = overallI * a_2 * (15/2 * (2*u-1)**2 -3/2)
        itgR = itg0R + itg1R + itg2R
        itgI = itg0I + itg1I + itg2I
        return(itgR,itgI)
    #this integral can be tricky near u = 1, so we help the integrator by adding 0-0.9 + 0.9-0.99 + ...
    lims  = [0,0.9,0.99,0.999,0.9999,1]
    int_uR = 0
    int_uI = 0
    for i in range(len(lims)-1):
        low = lims[i]
        high = lims[i+1]
        print('Doing int',low,high)
        R,I = do_complex_integral(func,low,high)
        int_uR += R
        int_uI += I
    int_wR = 1/w0  # Trivial integral B42
    int_wI = 0
    resultR = fac*(int_uR*int_wR - int_uI*int_wI )
    resultI = fac*(int_uR*int_wI + int_uI*int_wR )
    return(resultR,resultI)

def make_t_parr(u,m,qsq): #t_parrallell function
    ubar = 1-u
    if m == 0: #these cancel in I1 
        I1R,I1I = 1.0,0
        B01R,B01I,B02R,B02I = 0,0,0,0 #theses cancel in tparr if m =0
    elif mymean(u) == 1: # in this case we use the theory for (B0-B0/ubar)
        I1R,I1I = make_I1(m,u,qsq)
        beta = 4*m**2/qsq -1
        if mymean(beta) >=0:
            arg = gv.sqrt(beta)
            tanR = gv.arctan(1/arg)
            tanI = 0
            theoryR = 4*m**2*(M_B**2-qsq)/qsq**2 * (tanR/arg - 1/(beta + 1)) # my versions
            theoryI = 0
        else:
            arg = gv.sqrt(-beta) #purely imaginary
            x = [0,-1/arg] # x= 1/arg so just -1/arg x = [xR,xI]
            tanR,tanI = make_arctan(x)
            theoryR = 4*m**2*(M_B**2-qsq)/qsq**2 * (tanI/arg - 1/(beta+1) ) # my versions
            theoryI = 4*m**2*(M_B**2-qsq)/qsq**2 * (-tanR/arg ) # my versions
    else:
        I1R,I1I = make_I1(m,u,qsq)
        argB = ubar*M_B**2+u*qsq
        B01R,B01I = make_B0(argB,m)
        B02R,B02I = make_B0(qsq,m)
    E = (M_B**2 + M_K**2 -qsq)/(2*M_B)
    if m ==0:
        tR = 2*M_B/E
        tI = 0
    elif mymean(u) == 1:
        tR = 2*M_B*I1R/(E) + (ubar*M_B**2+u*qsq)/E**2 *theoryR# we remove a factor of ubar in the denom as this is multiplied in B40 anyway 
        tI = 2*M_B*I1I/(E) + (ubar*M_B**2+u*qsq)/E**2 *theoryI
    else:
        tR = 2*M_B*I1R/(E) + (ubar*M_B**2+u*qsq)/(ubar*E**2) * (B01R-B02R)# we remove a factor of ubar in the denom as this is multiplied in B40 anyway 
        tI = 2*M_B*I1I/(E) + (ubar*M_B**2+u*qsq)/(ubar*E**2) * (B01I-B02I)# we remove a factor of ubar in the denom as this is multiplied in B40 anyway
    return(tR,tI)

def make_B0(qsq,m):
    beta = 4*m**2/qsq -1
    if mymean(beta) >=0:
        arg = gv.sqrt(beta)
        BR = -2*arg*gv.arctan(1/arg)
        BI = 0
    else:
        arg = gv.sqrt(-beta) #purely imaginary
        x = [0,-1/arg] # x= 1/arg so just -1/arg x = [xR,xI]
        tanR,tanI = make_arctan(x)
        BR = 2*arg*tanI
        BI = -2*arg*tanR
    return(BR,BI)

def make_arctan(x):
    #makes arctan using the fact that arctan(x) = -i/2 log(1+ix/1-ix) taking complex x = [xR,xI]
    topR = 1 - x[1]
    topI = x[0]
    botR = 1 + x[1]
    botI = -x[0]
    logarg = c_arg(topR,topI,botR,botI) # Makes into complex number [a,b]
    LR,LI = make_log(logarg)
    tanR = 0.5*LI
    tanI = -0.5*LR
    return(tanR,tanI)

def make_I1(m,u,qsq):
    ubar = 1-u
    xarg = 1/4 - m**2/(ubar*M_B**2+u*qsq)
    if mymean(xarg) >=0:
        xpR = 1/2 + (xarg)**(1/2)
        xpI = 0
        xmR = 1/2 - (xarg)**(1/2)
        xmI = 0
    else:   # in this case log(x-1/x) is 0 because |x-1/x| =1. We leave this in as the code has no issue
        xpR = 1/2
        xpI = (-xarg)**(1/2)
        xmR = 1/2
        xmI = -(-xarg)**(1/2)
    yarg = 1/4 - m**2/qsq
    if mymean(yarg) >=0:
        ypR = 1/2 + (yarg)**(1/2)
        ypI = 0
        ymR = 1/2 - (yarg)**(1/2)
        ymI = 0
    else: # in this case log(y-1/y) is 0 because |x-1/x| =1. We leave this in as the code has no issue
        ypR = 1/2
        ypI = (-yarg)**(1/2)
        ymR = 1/2
        ymI = -(-yarg)**(1/2)
    xp = [xpR,xpI] # need to worry about real and imaginary parts 
    xm = [xmR,xmI]
    yp = [ypR,ypI]
    ym = [ymR,ymI]
    L1xpR,L1xpI = make_L1(xp)
    L1xmR,L1xmI = make_L1(xm)
    L1ypR,L1ypI = make_L1(yp)
    L1ymR,L1ymI = make_L1(ym)
    if ubar !=0:
        IR = 1 + 2*m**2/(ubar*(M_B**2-qsq)) * ( L1xpR + L1xmR - L1ypR - L1ymR )
        II = 2*m**2/(ubar*(M_B**2-qsq)) * ( L1xpI + L1xmI - L1ypI - L1ymI )
    ##### Need to find this in limit ubar ->0
    elif ubar == 0:
        predfac = m**2*(M_B**2-qsq)/qsq**2
        beta = 1-4*m**2/qsq
        dLypR,dLypI = make_derivative_L1(yp)
        dLymR,dLymI = make_derivative_L1(ym)
        if mymean(beta) >= 0:
            arg = gv.sqrt(beta)
            theoryR = predfac/arg * (dLypR-dLymR)
            theoryI = predfac/arg * (dLypI-dLymI)
        else:
            arg = gv.sqrt(-beta)
            theoryR = predfac/arg * (dLypI-dLymI)
            theoryI = -predfac/arg * (dLypR-dLymR)
        IR = 1 + 2*m**2/(M_B**2-qsq) * theoryR
        II = 2*m**2/(M_B**2-qsq) * theoryI
    return(IR,II)

def make_derivative_L1(x):#derivative of L1 for finding limit as ubar ->0
    x = zeros(x)
    topR = x[0] - 1
    topI = x[1]
    botR = x[0]
    botI = x[1]
    logarg1 = c_arg(topR,topI,botR,botI) #x-1/x
    logarg1 = zeros(logarg1)
    denom = (x[0]-1)**2 +x[1]**2
    fac = [(x[0]-1)/denom,-x[1]/denom] #1/x-1
    if mymean(x[0]) == 0 and mymean(x[1]) == 0:
        print("Error, shouldn't be passed 0 here, means m=0")
    elif x[0] == 0.5:
        logR,logI =make_log(logarg1,zero=True)
    else:
        logR,logI =make_log(logarg1)
    dLR = fac[0]*logR - fac[1]*logI
    dLI = fac[0]*logI + fac[1]*logR    
    return(dLR,dLI)
    

def make_L1(x): # have to worry about real and imaginay parts, for x =[xreal,ximaginary]
    x = zeros(x)
    topR = x[0] - 1
    topI = x[1]
    botR = x[0]
    botI = x[1]
    logarg1 = c_arg(topR,topI,botR,botI) #x-1/x
    logarg1 = zeros(logarg1)
    logarg2 = [1-x[0],-x[1]] #1-x
    logarg2 = zeros(logarg2)
    topR = x[0]
    topI = x[1]
    botR = x[0] - 1
    botI = x[1]
    dilogarg = c_arg(topR,topI,botR,botI) #x/x-1
    dilogarg = zeros(dilogarg)
    if mymean(x[0]) == 0 and mymean(x[1]) == 0:
        log1R,log1I = 0,0 # double check this is correct
        log2R,log2I = 0,0 #as these will always be multiplied
        dilogR,dilogI = make_dilog(dilogarg)
    elif x[0] == 0.5:
        log1R,log1I =make_log(logarg1,zero=True)
        log2R,log2I =make_log(logarg2)
        dilogR,dilogI = make_dilog(dilogarg,zero=True)
    else:
        log1R,log1I =make_log(logarg1)
        log2R,log2I =make_log(logarg2)
        dilogR,dilogI = make_dilog(dilogarg)
    LR = (log1R*log2R-log1I*log2I) - np.pi**2/6 + dilogR
    LI = (log1I*log2R+log1R*log2I) + dilogI
    return(LR,LI)

def make_dilog(x,zero=False):
    #Pass this zero because if x[0] =0.5 |x| =1, but it may come out as 0.999999999 and so end up in the wrong part of this.
    x = zeros(x)
    #if |x|<1 we can simply proceed as usual. If not, we use an identity to relate to dilog(1/x).
    if mymean(x[0]) == 1 and mymean(x[1]) ==0: # fixed value at x = 1
        LR = np.pi**2/6
        LI = 0
    elif mymean(x[0]**2 + x[1]**2) < 1 and zero == False: # if zero = true the |x| = 1
        LR,LI = make_dilog_small_x(x)
    else:#LI(z) = -pi**2/6 -ln**2(-z)/2 - Li(1/z) so if x[0] =1 ln(x) need to be passed zero= True 
        xinv = [x[0]/(x[0]**2+x[1]**2),-x[1]/(x[0]**2+x[1]**2)]
        logR,logI = make_log([-x[0],-x[1]],zero=zero)
        log2R = logR**2 - logI**2
        log2I = 2*logR*logI
        LiR,LiI = make_dilog_small_x(xinv,zero=zero)
        LR = -np.pi**2/6 - log2R/2 - LiR
        LI = - log2I/2 - LiI
    return(LR,LI)

def make_dilog_small_x(x,zero=False): # x is complex in general and is passed in form x = [xR,xI] works for |x|<1
    x = zeros(x)
    if mymean(x[0]**2+x[1]**2) > 1 and zero == False:
        print('Error x in dilog greater than and shouldnt be 1 x =',x,'|x| =', x[0]**2+x[1]**2)        
    LRs = []
    LIs = []
    LR = 0
    LI = 0
    Test = False
    i = 1
    while Test == False:
        xiR,xiI = make_power(x,i,zero=zero)# i is power
        LR += xiR/i**2
        LI += xiI/i**2
        LRs.append(LR)
        LIs.append(LI)
        i+=1
        if len(LRs) >= 2:
            checkR = check_converged(LRs)
            checkI = check_converged(LIs)
            if checkR == True and checkI == True:
                Test = True
    return(LR,LI)

def make_log(x,zero=False): # x is complex in general and is passed in form x = [xR,xI]. Zero == True allows us to set A=1 exactly when x+-[0] = 0.5 and log (x-1/x)
    #log(z) = log(Ae^{itheta}) = log(A)+itheta
    x = zeros(x)
    A = gv.sqrt(x[0]**2+x[1]**2)
    if zero == True:
        A = 1
    if x[0] == 0:
        if mymean(x[1])>0:
            theta = np.pi/2
        else:
            theta = -np.pi/2
    elif x[1] == 0:
        if mymean(x[0])>0:
            theta = 0
        else:
            theta = np.pi
    else:
        theta = gv.arctan(x[1]/x[0])
        if mymean(x[0])<0:
            if mymean(x[1]) >=0:
                theta += np.pi 
            else:
                theta -= np.pi   # this should mean that theta is always in range -pi to pi.
    if mymean(A) == 1: # removes case of zero log with vanishing uncertainty
        LR = 0
    else:
        LR = gv.log(A)
    LI = theta
    return(LR,LI)

def make_power(x,k,zero=False): #returns real and impaginary parts of x^k
    #write x =Ae^{itheta}
    A = gv.sqrt(x[0]**2+x[1]**2)
    if zero ==True:
        A = 1
    theta = gv.arctan(x[1]/x[0])
    if mymean(x[0])<0:
        if mymean(x[1]) >=0:
            theta += np.pi
        else:
            theta -= np.pi    # this should mean that theta is always in range -pi to pi. 
    #now x^i A^k e{iktheta}
    A_new = A**k
    theta_new = k*theta # don't need to worry about the theta in this direction, can
    xiR = A_new * gv.cos(theta_new)
    xiI = A_new * gv.sin(theta_new)
    return(xiR,xiI)
##########################################

def save_results():
    saved_result = collections.OrderedDict()
    saved_result['qsq'] = []
    saved_result['DelC9Rp'] = []
    saved_result['DelC9Ip'] = []
    saved_result['DelC9R0'] = []
    saved_result['DelC9I0'] = []
    saved_result['C9OalphasR'] = []
    saved_result['C9OalphasI'] = []
    saved_result['C7OalphasR'] = []
    saved_result['C7OalphasI'] = []
    saved_result['C9OlambR'] = []
    saved_result['C9OlambI'] = []

    for qsq in np.linspace(1e-6,23.5,1000): # evalutes over range of q^2 values. 
        print('################### qsq =',qsq)
        DRp,DIp,DR0,DI0 = make_DelC9eff(qsq)
        alR,alI = make_correction1(qsq) 
        lamR,lamI = make_correction2(qsq) 
        C7alR,C7alI = make_correction3(qsq) 
        saved_result['qsq'].append(qsq)
        saved_result['DelC9Rp'].append(DRp)
        saved_result['DelC9Ip'].append(DIp)
        saved_result['DelC9R0'].append(DR0)
        saved_result['DelC9I0'].append(DI0)
        saved_result['C9OalphasR'].append(alR)
        saved_result['C9OalphasI'].append(alI)
        saved_result['C7OalphasR'].append(C7alR)
        saved_result['C7OalphasI'].append(C7alI)
        saved_result['C9OlambR'].append(lamR)
        saved_result['C9OlambI'].append(lamI)

    gv.dump(saved_result,'Fits/C9_corrections.pickle')
    return()


###########################################

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

def do_plots():
    x = []
    Tm0R = []
    Tm0I = []
    TmnfR = []
    TmnfI = []
    TpnfR = []
    TpnfI = []
    alphaR = []
    alphaI = []
    lambdaR = []
    lambdaI = []
    C7alphaR = []
    C7alphaI = []
    TotC9Rp = []
    TotC9Ip = []
    TotC9R0 = []
    TotC9I0 = []    
    for qsq in np.linspace(4*m_e**2,qsqmaxphysBK.mean,1000):
        print('################### qsq =',qsq)
        Rp = 0
        Ip = 0
        R0 = 0
        I0 = 0
        x.append(qsq)
        R,I = make_Del_bits(qsq,'TP-0')
        Rp += 2/3 *R
        Ip += 2/3 *I
        R0 += -1/3 *R
        I0 += -1/3 *I 
        Tm0R.append(R)
        Tm0I.append(I)
        R,I = make_Del_bits(qsq,'TP-nf')
        Rp += 2/3 *R
        Ip += 2/3 *I
        R0 += -1/3 *R
        I0 += -1/3 *I 
        TmnfR.append(R)
        TmnfI.append(I)
        R,I = make_Del_bits(qsq,'TP+nf')
        Rp += R
        Ip += I
        R0 += R
        I0 += I 
        TpnfR.append(R)
        TpnfI.append(I)
        R,I = make_correction1(qsq)
        alphaR.append(-R) # minus because subtracts
        alphaI.append(-I)
        R,I = make_correction3(qsq)
        C7alphaR.append(-R) # minus because subtracts
        C7alphaI.append(-I)
        R,I = make_correction2(qsq)
        lambdaR.append(R)
        lambdaI.append(I)
        TotC9Rp.append(Rp)
        TotC9Ip.append(Ip)
        TotC9R0.append(R0)
        TotC9I0.append(I0)
    Tm0Rm,Tm0Rs = unmake_gvar_vec(Tm0R)
    Tm0Ru,Tm0Rl = make_upp_low(Tm0R)
    TmnfRm,TmnfRs = unmake_gvar_vec(TmnfR)
    TmnfRu,TmnfRl = make_upp_low(TmnfR)
    TpnfRm,TpnfRs = unmake_gvar_vec(TpnfR)
    TpnfRu,TpnfRl = make_upp_low(TpnfR)    
    plt.figure(figsize=figsize)
    plt.plot(x, Tm0Rm, color='r',label=r'$(f_+/e_q)\mathrm{Re}[\Delta C_9^{\mathrm{eff}}(T_{K,-}^{(0)})]$')
    plt.fill_between(x,Tm0Rl,Tm0Ru, color='r',alpha=alpha)
    plt.plot(x, TmnfRm, color='b',label=r'$(f_+/e_q) \mathrm{Re}[\Delta C_9^{\mathrm{eff}}(T_{K,-}^{(nf)})]$')
    plt.fill_between(x,TmnfRl,TmnfRu, color='b',alpha=alpha)
    plt.plot(x, TpnfRm, color='g',label=r'$f_+\mathrm{Re}[\Delta C_9^{\mathrm{eff}}(T_{K,+}^{(nf)})]$')
    plt.fill_between(x,TpnfRl,TpnfRu, color='g',alpha=alpha)
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=2,loc='lower right')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(2))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.plot([-10,30],[0,0],linestyle='--',color='k')
    plt.axes().set_ylim([-0.25,0.25])
    plt.axes().set_xlim([0,corrcut])
    plt.tight_layout()
    plt.savefig('Plots/ReDelC9.pdf')
    plt.close()

    Tm0Im,Tm0Is = unmake_gvar_vec(Tm0I)
    Tm0Iu,Tm0Il = make_upp_low(Tm0I)
    TmnfIm,TmnfIs = unmake_gvar_vec(TmnfI)
    TmnfIu,TmnfIl = make_upp_low(TmnfI)
    TpnfIm,TpnfIs = unmake_gvar_vec(TpnfI)
    TpnfIu,TpnfIl = make_upp_low(TpnfI)   
    plt.figure(figsize=figsize)
    plt.plot(x, Tm0Im, color='r',label=r'$(f_+/e_q)\mathrm{Im}[\Delta C_9^{\mathrm{eff}}(T_{K,-}^{(0)})]$')
    plt.fill_between(x,Tm0Il,Tm0Iu, color='r',alpha=alpha)
    plt.plot(x, TmnfIm, color='b',label=r'$(f_+/e_q)\mathrm{Im}[\Delta C_9^{\mathrm{eff}}(T_{K,-}^{(nf)})]$')
    plt.fill_between(x,TmnfIl,TmnfIu, color='b',alpha=alpha)
    plt.plot(x, TpnfIm, color='g',label=r'$f_+\mathrm{Im}[\Delta C_9^{\mathrm{eff}}(T_{K,+}^{(nf)})]$')
    plt.fill_between(x,TpnfIl,TpnfIu, color='g',alpha=alpha)
    #handles, labels = plt.gca().get_legend_handles_labels()
    #handles = [h[0] for h in handles]
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=1,loc='lower right')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    #plt.ylabel(r'$f_T(q$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(2))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.plot([-10,30],[0,0],linestyle='--',color='k')
    plt.axes().set_ylim([-0.8,0.1])
    plt.axes().set_xlim([0,corrcut])
    plt.tight_layout()
    plt.savefig('Plots/ImDelC9.pdf')
    plt.close()

    alRm,alRs = unmake_gvar_vec(alphaR)
    alRu,alRl = make_upp_low(alphaR)
    alIm,alIs = unmake_gvar_vec(alphaI)
    alIu,alIl = make_upp_low(alphaI)
    
    C7alRm,C7alRs = unmake_gvar_vec(C7alphaR)
    C7alRu,C7alRl = make_upp_low(C7alphaR)
    C7alIm,C7alIs = unmake_gvar_vec(C7alphaI)
    C7alIu,C7alIl = make_upp_low(C7alphaI)
    
    laRm,laRs = unmake_gvar_vec(lambdaR)
    laRu,laRl = make_upp_low(lambdaR)
    laIm,laIs = unmake_gvar_vec(lambdaI)
    laIu,laIl = make_upp_low(lambdaI)
    plt.figure(figsize=figsize)
    plt.plot(x, alRm, color='k',label=r'$\mathrm{Re}[C_9^{\mathrm{eff}}(\mathcal{O}(\alpha_s))]$')
    plt.fill_between(x,alRu,alRl, color='k',alpha=alpha)
    plt.plot(x, alIm, color='r',label=r'$\mathrm{Im}[C_9^{\mathrm{eff}}(\mathcal{O}(\alpha_s))]$')
    plt.fill_between(x,alIu,alIl, color='r',alpha=alpha)

    plt.plot(x, C7alRm, color='c',label=r'$\mathrm{Re}[C_7^{\mathrm{eff}}(\mathcal{O}(\alpha_s))]$')
    plt.fill_between(x,C7alRu,C7alRl, color='c',alpha=alpha)
    plt.plot(x, C7alIm, color='purple',label=r'$\mathrm{Im}[C_7^{\mathrm{eff}}(\mathcal{O}(\alpha_s))]$')
    plt.fill_between(x,C7alIu,C7alIl, color='purple',alpha=alpha)
    
    plt.plot(x, laRm, color='b',label=r'$\mathrm{Re}[C_9^{\mathrm{eff}}(\mathcal{O}(\lambda_u^{(s)}))]$')
    plt.fill_between(x,laRu,laRl, color='b',alpha=alpha)
    plt.plot(x, laIm, color='g',label=r'$\mathrm{Im}[C_9^{\mathrm{eff}}(\mathcal{O}(\lambda_u^{(s)}))]$')
    plt.fill_between(x,laIu,laIl, color='g',alpha=alpha)
   
    #handles, labels = plt.gca().get_legend_handles_labels()
    #handles = [h[0] for h in handles]
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=2,loc='lower left')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    #plt.ylabel(r'$f_T(q$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.plot([-10,30],[0,0],linestyle='--',color='k')
    plt.axes().set_ylim([-0.6,0.6])
    plt.axes().set_xlim([-1,24])
    plt.tight_layout()
    plt.savefig('Plots/C9effOalOla.pdf')
    plt.close()

    Rpm,Rps = unmake_gvar_vec(TotC9Rp)
    R0m,R0s = unmake_gvar_vec(TotC9R0)
    Ipm,Ips = unmake_gvar_vec(TotC9Ip)
    I0m,I0s = unmake_gvar_vec(TotC9I0)
    
    Rpu,Rpl = make_upp_low(TotC9Rp)
    R0u,R0l = make_upp_low(TotC9R0)
    Ipu,Ipl = make_upp_low(TotC9Ip)
    I0u,I0l = make_upp_low(TotC9I0)
    plt.figure(figsize=figsize)
    plt.plot(x, Rpm, color='r',label=r'$f_+\mathrm{Re}[\Delta C_9^{\mathrm{eff}}(B^+)]$')
    plt.fill_between(x,Rpl,Rpu, color='r',alpha=alpha)
    plt.plot(x, R0m, color='k',label=r'$f_+\mathrm{Re}[\Delta C_9^{\mathrm{eff}}(B^0)]$')
    plt.fill_between(x,R0l,R0u, color='k',alpha=alpha)
    
    plt.plot(x, Ipm, color='b',label=r'$f_+\mathrm{Im}[\Delta C_9^{\mathrm{eff}}(B^+)]$')
    plt.fill_between(x,Ipl,Ipu, color='b',alpha=alpha)
    plt.plot(x, I0m, color='g',label=r'$f_+\mathrm{Im}[\Delta C_9^{\mathrm{eff}}(B^0)]$')
    plt.fill_between(x,I0l,I0u, color='g',alpha=alpha)
     
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=2,loc='lower right')
    plt.xlabel('$q^2[\mathrm{GeV}^2]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.plot([-10,30],[0,0],linestyle='--',color='k')
    plt.axes().set_ylim([-0.5,0.5])
    plt.axes().set_xlim([0,corrcut])
    plt.tight_layout()
    plt.savefig('Plots/TotalDelC9eff.pdf')
    plt.close()
    return()






def plot_F1_F2():
    F1R = []
    F2R = []
    F1I = []
    F2I = []
    s = []
    for qsq in np.linspace(1e-4,23,200):
        F19R,F19I,F29R,F29I = make_F1c9_F2c9(qsq)
        s.append(qsq/m_b.mean**2)
        F1R.append(F19R)
        F1I.append(F19I)
        F2R.append(F29R)
        F2I.append(F29I)

    
    plt.figure(figsize=figsize)
    plt.plot(s, F1R, color='r')     
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper right')
    plt.xlabel('$s$',fontsize=fontsizelab)
    plt.ylabel('Re$[F_1^{(9)}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    #plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    #plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    #plt.axes().set_xlim([0.4,1])
    plt.tight_layout()
    plt.savefig('Plots/ReF19.pdf')
    plt.close()

    plt.figure(figsize=figsize)
    plt.plot(s, F1I, color='r')     
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper right')
    plt.xlabel('$s$',fontsize=fontsizelab)
    plt.ylabel('Im$[F_1^{(9)}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    #plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    #plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    #plt.axes().set_xlim([0.4,1])
    plt.tight_layout()
    plt.savefig('Plots/ImF19.pdf')
    plt.close()

    plt.figure(figsize=figsize)
    plt.plot(s, F2R, color='r')     
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper right')
    plt.xlabel('$s$',fontsize=fontsizelab)
    plt.ylabel('Re$[F_2^{(9)}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    #plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    #plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
   # plt.axes().set_xlim([0.4,1])
    plt.tight_layout()
    plt.savefig('Plots/ReF29.pdf')
    plt.close()

    plt.figure(figsize=figsize)
    plt.plot(s, F2I, color='r')     
    plt.legend(fontsize=fontsizeleg,frameon=False,ncol=1,loc='upper right')
    plt.xlabel('$s$',fontsize=fontsizelab)
    plt.ylabel('Im$[F_2^{(9)}]$',fontsize=fontsizelab)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=fontsizelab)
    plt.axes().tick_params(which='major',length=major)
    plt.axes().tick_params(which='minor',length=minor)
    plt.axes().yaxis.set_ticks_position('both')
    #plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    #plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    #plt.axes().yaxis.set_major_locator(MultipleLocator(0.2))
    #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    #plt.axes().set_xlim([0.4,1])
    plt.tight_layout()
    plt.savefig('Plots/ImF29.pdf')
    plt.close()

    return()


#save_results()
#do_plots()
#plot_F1_F2()
