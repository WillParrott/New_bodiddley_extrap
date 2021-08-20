from functionsBK import *
import numpy as np
import gvar as gv
import scipy
import matplotlib.pyplot as plt
#import cython

#Here we seek to calculate the corrections in appendix B1 and B2 of 1510.02349. Makes use of C++ header files from 0810.4077

#first, corrections to C_9^eff. Currently, the real part of this is of size 4-5.
#The largest correction is alpha_s/4pi (C1*F_1,c(9) + C2*F_2,c(9) + C8F_8^(9)]
C8 = gv.gvar('-0.152(15)')#from 1510.02349, can't find this in paper we are using
C8eff = C8 + C3 - 1/6 * C4 + 20 * C5 - 10/3 * C6 
alpha_s = gv.gvar('0.2253(28)') #Used qcdevol
lambda_us = 0.02 #what is this number?
fK = gv.gvar('0.1556(4)')# 1509.02220 for fK+ find better value for this?
N_c = 3 # Assuming this is N_colours
M_B = MBphysp#get this from p need mean for Ei
M_K = MKphysp
w0 = 1/gv.gvar('3(1)') # from Christine's notes. Not sure where this is from originally.
C_F = 4/3 # see page 44
a_1 = 1    # get a_1 and a_2 values
a_2 = 1

###################################### A couple of general functions ##########################################
def mymean(x):
    #returns the mean of a gvar is passed on, or a float if passed a float. Neeeded when function takes both gvars and floats and have if statements about them i.e. if x.mean <0
    if isinstance(x,gv._gvarcore.GVar):
        return(x.mean)
    else:
        return(x)
    
def check_converged(a):
    #takes a list of values and looks for relative change to asses if the integral has converged. Needs to check in cases with floats, gvars, gavrs with 0 mean and gvars with 0 sdev. Returns True if the 
    if isinstance(a[-1],gv._gvarcore.GVar):
        if a[-1].mean == 0: # some parts are 0 becase the integral is purely real or imaginary. We bypass this case.
            return(True)
        elif a[-1].sdev == 0:
            check = (a[-1]-a[-2])/a[-1]
            if check < 0.001:
                return(True)
            else:
                return(False)
        else:
            check = (a[-1]-a[-2]).mean/a[-1].sdev
            if check < 0.02:
                return(True)
            else:
                return(False)
    else:
        check = (a[-1]-a[-2])/a[-1]
        if check < 0.001:
            return(True)
        else:
            return(False)

######################################################################################################################
    
def make_correction1(qsq): # dominant O(alpha_s) correction to C9eff in B11. Currently missing F1c9 and F2c9 (in C++)
    F1c9 = 1#make_F1c9(qsq)
    F2c9 = 1#make_F2c9(qsq)
    F89 = make_F89(qsq)
    corr = (alpha_s/(4*np.pi)) * (C1*F1c9 + C2*F2c9 + C8*F89 )
    print('correction O(alpha_s) = ',corr)
    return(corr)

def make_correction2(qsq): #O(lambda_u^s) correction to C9eff in B11
    hcR,hcI = make_h(qsq,m_c)
    h0R,h0I = make_h(qsq,0)
    corrR = lambda_us * (hcR-h0R)*(4/3 *C1 + C2)
    corrI = lambda_us * (hcI-h0I)*(4/3 *C1 + C2)
    print('correction O(lambd_us) = {0} + {1}i'.format(corrR,corrI))
    return(corrR,corrI)

#F1c9 and F2c9 are difficult

def make_F89(qsq):
    s = qsq/m_b**2
    B0 = make_B0s(s)
    C0 = make_C0s(s)
    F =  16/9 * 1/(1-s) * gv.log(s) + 8/9 * (5-2*s)/(1-s)**2 - 8/9 * (4-s)/(1-s)**3 * ( (1+s)*B0 - 2*C0) 
    return(F)

def make_B0s(s):
    B = -2 * gv.sqrt(4/s -1) * gv.arctan(1/gv.sqrt(4/s - 1))
    return(B)

def make_C0s(s):
    eps = 1e-2
    def fcn1(x):
        y = 1/(x*(1-s)+1) * gv.log(x**2/(1-x*(1-x)*s))
        return(y)
    
    def fcn2(x):
        if x ==0:
            y = 0  
        else:
            y = (1-x*(1-s)+(x*(1-s))**2 -(x*(1-s))**3) * gv.log(1/(1-x*(1-x)*s)) + (- x*(1-s)  + (x*(1-s))**2 -(x*(1-s))**3) * gv.log(x**2)
        return(y)
    C1 = do_integral(fcn1,eps,1)
    C2 = do_integral(fcn2,0,eps)
    C = C1 + C2 + 2*(eps*gv.log(eps)-eps)
    #print('eps =',eps,'C0 =',C)
    return(C)


def do_integral(fcn,low,upp): # generic integrator for a real function (one value). Takes the function of the integrand and uses the trapeziodal rule. Starts with 16 iters and doubles until stability condition is met. 
    integrands = gv.BufferDict()
    iters = int(16)
    Test = False
    results = []
    while Test == False:
        points = np.linspace(low,upp,iters+1) 
        del_qsq =  (upp-low) /iters
        funcs = fcn(low) + fcn(upp)
        for i in range(1,iters):
            funcs += 2*fcn(points[i])
        result1 = del_qsq*funcs/2
        results.append(result1)
        iters *= 2
        #print('iters',int(iters/2))
        #print(results)
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
        lR,lI = fcn(low)
        hR,hI = fcn(upp)
        funcsR = lR+hR
        funcsI = lI+hI
        for i in range(1,iters):
            fR,fI = fcn(points[i])
            funcsR += 2*fR
            funcsI += 2*fI
        resultR = del_qsq*funcsR/2
        resultI = del_qsq*funcsI/2
        resultsR.append(resultR)
        resultsI.append(resultI)
        iters *= 2
        #print('iters',int(iters/2))
        #print(resultsR)
        #print(resultsI)
        if len(resultsR)>=2:
            checkR = check_converged(resultsR)
            checkI = check_converged(resultsI)
            if checkR == True and checkI == True:
                Test = True
    
    return(resultR,resultI)


######################## Non factorisable bits  Delta C9eff in eq B26 #######################################################

def make_DelC9eff(qsq): #makes whole Del C9 eff
    fp = 0.3 #change this or remove it from calc so it appears in Y_eff bit. This is just approx fp(0) for now.
    Del_taupR,Del_taupI = make_Del_taup(qsq) 
    DelR = (2 * m_b * Del_taupR )/(M_B*fp)
    DelI = (2 * m_b * Del_taupI )/(M_B*fp)
    #print(qsq,DelR,DelI)
    return(DelR,DelI)

def make_Del_taup(qsq):#Makes Del tau, split into 4 parts plus1 plus2, minus1,minus2 these are the 4 parts B27 can be split into. I.e. plus 1 is the Tp+^(0) bit , plus2 is the alpha_sCF TP+^(nf) bit and same for minus. 
    N = (np.pi**2*fB*fK)/(N_c*M_B) # Factor out front
    plus1 = 0 # Tp+ ^0 =0
    print('doing minus1')
    minus1R,minus1I = make_minus1_integral(qsq)
    print('doing minus2')
    minus2R,minus2I = make_minus2_integral(qsq)
    print('doing plus2')
    plus2R,plus2I = make_plus2_integral(qsq)
    DelR = N * (plus1 + minus1R + plus2R + minus2R)
    DelI = N * (plus1 + minus1I + plus2I + minus2I)
    return(DelR,DelI)

def make_minus1_integral(qsq,charge ='p'):
    #T_P-^(0) is not a function of u, so we just need to integrate phi_K, which we can do manually, and then perform the omega integral
    # Different value for + ('p') or 0 charge. Here we take 'p' as default and we can pass the correct one later
    if charge == 'p':
        e_q = 2/3
    elif charge == '0':
        e_q = -1/3
    fac = ( e_q * 4 * M_B/m_b ) *  (C3 + 4/3*C4 + 16*C5 + 64/3*C6) 
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

def make_minus2_integral(qsq,charge='p'): #Similar to minus1 but this time TP-^nf has 3 distinct parts
    overall_fac = alpha_s/(4*np.pi) * C_F
    if charge == 'p':
        e_q = 2/3
    elif charge == '0':
        e_q = -1/3
    # last term in TP-nf only has w dependence, which is the same as Tp- with int_u =1
    Ei = scipy.special.expi(qsq/(M_B.mean*w0.mean))
    integralR = M_B * gv.exp(-qsq/(M_B*w0))/w0 * (-Ei) # There is an extra factor of M_B in TP-^nf vs that in eq B43 (check)
    integralI = M_B * gv.exp(-qsq/(M_B*w0))/w0 * np.pi
    fac_last = e_q * 6 * M_B/m_b * 8/27 * (-15/2 * C4 + 12*C5 -32 *C6)
    lastR = fac_last * integralR
    lastI = fac_last * integralI
    #first term is next easiest. Here we need to just integrate over u. The w integral is the same.
    fac_first = -e_q*8*C8eff
    
    def first_func(u):
        overall = 6*u*(1-u) * 1/(1 - u + qsq*u/M_B**2) # real
        R0 = overall #0th order, can add others as we wish
        R = R0 
        return(R)
        
    u_int = do_integral(first_func,0,1) 
    firstR = fac_first * u_int * integralR # w integral same as usual
    firstI = fac_first * u_int * integralI
    #Middle bit. We again need to do the u integral numerically. w integral is the same. 
    fac_mid = -e_q * 6 * M_B/m_b
    
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
        R = R0
        I = I0
        return(R,I)
    
    u_intR,u_intI = do_complex_integral(mid_func,0,1)
    midR = fac_mid * ( u_intR * integralR - u_intI * integralI)
    midI = fac_mid * ( u_intI * integralR + u_intR * integralI)
    ################################################
    #print('R',qsq,firstR,midR,lastR)
    #print('I',qsq,firstI,midI,lastI)
    resultR = overall_fac * (lastR + firstR + midR)
    resultI = overall_fac * (lastI+ firstI + midI)
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
        if mymean(u) == 1: # in this case we must manually cancal the 1-u in B40 with the ubar in the denom of t_parr
            overallR = 6*u* (fac1*t1R + fac2*t2R + fac3*t3R)
            overallI = 6*u* (fac1*t1I + fac2*t2I + fac3*t3I)
        else:
            overallR = 6*u*(1-u)* (fac1*t1R + fac2*t2R + fac3*t3R)
            overallI = 6*u*(1-u)* (fac1*t1I + fac2*t2I + fac3*t3I)
        itg0R = overallR #0th order
        itg0I = overallI
        itgR = itg0R
        itgI = itg0I
        return(itgR,itgI)
    x = []
    yR = []
    yI = []
    for u in np.linspace(0,1,100):
        x.append(u)
        r,i = func(u)
        yR.append(mymean(r))
        YI.append(mymean(i))
    plt.figure()
    plt.plot(x,yR,color='r')
    plt.plot(x,yI,color='b')
    plt.savefig('Plots/plus2int.pdf')
    plt.close()
    int_uR,int_uI = do_complex_integral(func,0,1)
    int_wR = 1/w_0  # Trivial integral B42
    int_wI = 0
    resultR = fac*(int_uR*int_wR - int_uI*int_wI )
    resultI = fac*(int_uR*int_wI + int_uI*int_wR )
    return(resultR,resultI)

def make_t_parr(u,m,qsq): #t_parrallell function
    ubar = 1-u
    if m == 0: #these cancel in I1 
        I1R,I1I = 1.0,0
        B01R,B01I,B02R,B02I = 0,0,0,0 #theses cancel in tparr if m =0
    elif mymean(u) == 1:
        I1R,I1I = 1.0,0
        B01R,B01I,B02R,B02I = 0,0,0,0 #theses cancel in tparr if u=1 
    else:
        I1R,I1I = make_I1(m,u,qsq)
        B01R,B01I = make_B0(ubar*M_B**2+u*qsq,m)
        B02R,B02I = make_B0(qsq,m)
    E = (M_B**2 + M_K**2 -qsq)/(2*M_B)
    if mymean(u) == 1:
        tR = 2*M_B*I1R/(E) # this creates a problem 1/(1-u) which cancels in the u integral but needs to be addressed manually. We remove u bar here, and in this case we only integrate over 6u * thing. This affects all 
        tI = 0
    else:
        tR = 2*M_B*I1R/(ubar*E) + (ubar*M_B**2+u*qsq)/(ubar**2*E**2) * (B01R-B02R)
        tI = 2*M_B*I1I/(ubar*E) + (ubar*M_B**2+u*qsq)/(ubar**2*E**2) * (B01I-B02I)
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
    #makes arctan using the fact that arctan(x) = -2i log(1+ix/1-ix) taking complex x = [xR,xI]
    # 1+ix/1-ix = 1-x[1]+ix[0]/1+x[1]-ix[0] = (1-x[1]+ix[0])(1+x[1]+ix[0])/denom where denom is (1+x[1])**2 + x[0]**2
    # = -x[0]**2 + 1 - x[1]**2 + 2ix[0]
    denom = (1+x[1])**2 + x[0]**2 
    logarg = [(1-x[0]**2-x[1]**2)/denom,2*x[0]/denom]
    LR,LI = make_log(logarg)
    tanR = 2*LI
    tanI = -2*LR
    return(tanR,tanI)

def make_I1(m,u,qsq):
    ubar = 1-u
    xarg = 1/4 - m**2/(ubar*M_B**2-qsq)
    if mymean(xarg) >=0:
        xpR = 1/2 + (xarg)**(1/2)
        xpI = 0
        xmR = 1/2 - (xarg)**(1/2)
        xmI = 0
    else:
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
    else:
        ypR = 1/2
        ypI = (-yarg)**(1/2)
        ymR = 1/2
        ymI = -(-yarg)**(1/2)
    xp = [xpR,xpI] # need to worry about real and imaginary parts 
    xm = [xmR,xmI]
    yp = [ypR,ypI]
    ym = [ymR,ymI]
    #print(m,u,qsq,xp,xm,yp,ym)
    L1xpR,L1xpI = make_L1(xp)
    L1xmR,L1xmI = make_L1(xm)
    L1ypR,L1ypI = make_L1(yp)
    L1ymR,L1ymI = make_L1(ym)
    #print(u,ubar)
    IR = 1 + 2*m**2/(ubar*(M_B**2-qsq)) * ( L1xpR + L1xmR - L1ypR - L1ymR )
    II = 1 + 2*m**2/(ubar*(M_B**2-qsq)) * ( L1xpI + L1xmI - L1ypI - L1ymI )
    return(IR,II)

def make_L1(x): # have to worry about real and imaginay parts, for x =[xreal,ximaginary]
    if mymean(x[0]) == 0:
        x[0] = 0
    if mymean(x[1]) == 0:
        x[1] = 0
    logarg1 = [x[0]*(x[0]-1)/(x[0]**2+x[1]**2),-x[1]*(x[0]-1)/(x[0]**2+x[1]**2)] #(x-1)/x  = xR-1/(xR+ixI) * (xR-ixI)/(xR-ixI) = (xR(xR-1) -ixI(xR-1))/xR**2+xI**2
    logarg2 = [1-x[0],-x[1]] #x-1
    denom = (x[0]-1)**2 + x[1]**2
    if mymean(x[0]) == 1 and mymean(x[1]) == 0:
        dilogarg = [0,0]
    else:
        dilogarg = [(x[0]*(x[0]-1)+x[1]**2)/denom,-x[1]/denom] #x/x-1 = (xR+ixI)*(xR-1-ixI)/((xR-1)**2+xi**2) =(xR*(xR-1)+xI**2+ixI(-1))/((xR-1)**2+xI**2)
    if mymean(x[0]) == 0 and mymean(x[1]) == 0:
        log1R,log1I = 0,0
        log2R,log2I = 0,0 #as these will always be multiplied 
    else:
        log1R,log1I =make_log(logarg1)
        log2R,log2I =make_log(logarg2)
    dilogR,dilogI = make_dilog(dilogarg)
    LR = (log1R*log2R-log1I*log2I) - np.pi**2/6 + dilogR
    LI = (log1I*log2R+log1R*log2I) - np.pi**2/6 + dilogI
    return(LR,LI)

def make_dilog(x):
    if mymean(x[0]) == 0:
        x[0] = 0
    if mymean(x[1]) == 0:
        x[1] = 0
    #if |x|<1 we can simply proceed as usual. If not, we use an identity to relate to dilog(1/x).
    if mymean(x[0]**2 + x[1]**2) < 1:
        LR,LI = make_dilog_small_x(x)
    else:#LI(z) = -pi**2/6 -ln**2(-z)/2 - Li(1/z)
        xinv = [x[0]/(x[0]**2+x[1]**2),-x[1]/(x[0]**2+x[1]**2)]
        logR,logI = make_log([-x[0],-x[1]])
        log2R = logR**2 - logI**2
        log2I = 2*logR*logI
        LiR,LiI = make_dilog_small_x(xinv)
        LR = -np.pi**2/6 - log2R/2 - LiR
        LI = - log2I/2 - LiI 
    return(LR,LI)

def make_dilog_small_x(x): # x is complex in general and is passed in form x = [xR,xI]
    if mymean(x[0]**2+x[1]**2) > 1:
        print('Error x in dilog greater than 1 x =',x)
    LRs = []
    LIs = []
    LR = 0
    LI = 0
    Test = False
    i = 1
    while Test == False:
        xiR,xiI = make_power(x,i)# i is power
        LR += xiR/i**2
        LI += xiI/i**2
        LRs.append(LR)
        LIs.append(LI)
        i+=1
        #print('dilog ints',int(i-1))
        #print(LRs)
        #print(LIs)
        if len(LRs) >= 2:
            checkR = check_converged(LRs)
            checkI = check_converged(LIs)
            if checkR == True and checkI == True:
                Test = True
    #print('Final L',L)
    return(LR,LI)

def make_log(x): # x is complex in general and is passed in form x = [xR,xI]
    #log(z) = log(Ae^{itheta}) = log(A)+itheta
    if mymean(x[0]) == 0:
        x[0] = 0
    if mymean(x[1]) == 0:
        x[1] = 0
    A = gv.sqrt(x[0]**2+x[1]**2)
    theta = gv.arctan(x[1]/x[0])
    if mymean(x[0])<0:
        if mymean(x[1]) >0:
            theta += np.pi 
        else:
            theta -= np.pi   # this should mean that theta is always in range -pi to pi. 
    LR = gv.log(A)
    LI = theta
    return(LR,LI)

def make_power(x,k): #returns real and impaginary parts of x^k
    #write x =Ae^{itheta}
    A = gv.sqrt(x[0]**2+x[1]**2)
    theta = gv.arctan(x[1]/x[0])
    if mymean(x[0])<0:
        if mymean(x[1]) >0:
            theta += np.pi
        else:
            theta -= np.pi    # this should mean that theta is always in range -pi to pi. 
    #now x^i A^k e{iktheta}
    A_new = A**k
    theta_new = k*theta # don't need to worry about the theta in this direction, can
    xiR = A_new * gv.cos(theta_new)
    xiI = A_new * gv.sin(theta_new)
    return(xiR,xiI)
    
for qsq in range(1,2):
    make_DelC9eff(qsq)
    #make_correction1(qsq)
