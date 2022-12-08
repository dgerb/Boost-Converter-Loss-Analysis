
#%% Start
import sys
import os
from IPython.display import display
from sympy import *
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sb

CONV = 'boost' # 'boost', 'flyback', 'inverter'
IGBT = False # switch is an IGBT, otherwise is MOSFET
AC = False
DOPARAMETRIC = True

init_printing()
Vpk, Ipk, Vo, Po, T, f, L, theta = \
    symbols('V_pk, I_pk, V_o, P_o, T, f, L, theta', positive=True, real=True)
A,B,D,t = symbols('A, B, D, t', positive=True, real=True)
params = {'Vi': 170, 'Vo': 250, 'Po':250, 'f':65e3, 'L':500e-6} # AC prototype
# params = {'Vi': 24, 'Vo': 23.96*2, 'Po':0.165*23.96*2, 'f':65e3, 'L':500e-6} # DC prototype

if CONV == 'boost': # assign parasitics in the boost converter
    # Inductor: 750343810
    # Diode Bridge: GBU804
    # Switch: STP9NK60Z
    # Boost Diode: LQA08TC600
    # Capacitor: B43544A6477M000

    # Conduction Loss
    RL = .1 # inductor ESR
    VB = .73*2 # bridge diode drop
    RB = .044*2 # bridge ESR
    RQ = .95 # switch on resistance (MOSFET), or linearized resistance (IGBT)
    # VQ = 0 # switch drop (IGBT)
    VD = 2.4 # boost diode drop
    RD = .0625 # boost diode linearized resistance
    RC = 0.180/2 # output capacitor ESR
    # Switch Switching Loss
    QCISS = 1030e-12 # switch input capacitance at max VDS (Cgs + Cgd)
    QCOSS = 109e-12 # switch output capacitance at max VDS (Cds + Cgd)
    QRG = 10+4.7 # switch gate resistance
    QVGS = 10 # switch VGS max value
    QVTH = 3.75 # switch gate threshold voltage
    QVGP = 6 # switch gate plateau voltage
    QQGD = 21e-9 # switch gate-drain charge at VDS value on datasheet
    QVDS = 480 # switch VDS value on datasheet cooresponding to QQGD
    # Diode switching loss
    DTRR = 11e-9 # diode reverse recovery time (total, ta + tb) from datsheet
    DIRR = 1 # diode reverse recovery peak reverse current from datasheet
    DIF = 8 # diode reverse recovery forward current from datasheet
    DDIDT = 200e6 # diode reverse recovery di/dt slope from datasheet
    DCJ = 11e-12 # diode junction capacitance


# create symbols for parasitics and parameters for symoblic math
qtir, qton, qtoff, qcoss, dkq0, ds, dcj, btrr, birrm, bqrr, bcj = \
    symbols('T_{IR}, T_{ON}, T_{OFF}, C_oss, K_{Q}, S, C_j, t_rr, I_rrm, Q_rr, C_j', \
    positive=True, real=True)
ln, ll, la, lbha, lbhb, lbhc, lbhd, lbhe, lbhx, lpa, lpb, lpc = \
    symbols('N, l_c, A_c, a_b, b_b, c_b, d_b, e_b, x_b, a_p, b_p, c_p', \
    positive=True, real=True)

# assign symbolic inputs and parameters relevant to AC or DC input
if AC:
    Ipk = Po*2/Vpk
    vIn = Vpk*sin(theta)
    iRef = Ipk*sin(theta)
    distTag = 'AC'
    iIn = params['Po']*2/params['Vi']
    params['VpkAC'] = params['Vi']
    params['VinDC'] = 0
    params['IpkAC'] = iIn
    params['IinDC'] = 0
else: # DC
    Ipk = Po/Vpk
    vIn = Vpk
    iRef = Ipk
    distTag = 'DC'
    iIn = params['Po']/params['Vi']
    params['VpkAC'] = 0
    params['VinDC'] = params['Vi']
    # params['Ii'] = params['Po']/params['Vi']
    params['IpkAC'] = 0
    params['IpkDC'] = iIn
# assign parameter substitutions to turn symbolic expressions to real numbers
subs = {Vpk:params['Vi'], Ipk:iIn, Vo:params['Vo'], \
        Po:params['Po'], T:1/params['f'], f:params['f'], L:params['L'], \
        qcoss:QCOSS, qton:0, qtoff:0, \
        dcj:DCJ}
# assign file names, CSV data matrix, and latex header
latexFileName = os.path.join('Latex', CONV + distTag + '.tex')
simParamFileName = os.path.join('SimParams', CONV + 'SimParams.txt')
simDataFileName = os.path.join('SimParams', 'sim.csv')
dataCSV = [['Param', 'Model Simple', 'Model with Ripple', 'Ideal Sim'], \
           ['I_L,rms (A)', 0, 0, 0], \
           ['I_B,rms (A)', 0, 0, 0], \
           ['I_B,avg (A)', 0, 0, 0], \
           ['I_Q,rms (A)', 0, 0, 0], \
           ['I_D,rms (A)', 0, 0, 0], \
           ['I_D,avg (A)', 0, 0, 0], \
           ['I_C,rms (A)', 0, 0, 0]]
ROW = {'ilr':1, 'ibr':2, 'iba':3, 'iqr':4, 'idr':5, 'ida':6, 'icr':7}
COL = {'simple':1, 'ripple':2, 'sim':3}
latexLines = ['\\documentclass[12pt]{report}','\\usepackage{amsmath}',\
              '\\usepackage{graphicx}','\\begin{document}', CONV, '']

# assign duty cycle symbolically
if CONV == 'boost':
    dQ = 1 - vIn/Vo
    dD = vIn/Vo
elif CONV == 'flyback':
    dQ = 1 - vIn/Vo
    dD = vIn/Vo
elif CONV == 'inverter':
    dQ = 1 - vIn/Vo
    dD = vIn/Vo
display('dQ', dQ)
display('dD', dD)

# beautifies square root
def fixRoot(frac):
    frac = factor(expand(simplify(frac ** 2)))
    numer, denom = frac.as_numer_denom()
    return Mul(simplify(sqrt(numer)), \
               Pow(simplify(sqrt(denom)),Integer(-1),evaluate=False), \
               evaluate=False)

# helper functions for latex to begin an equation
def beginEQ(title=None):
    if not title is None:
        text('\n' + title)
        print('')
        print(title)
    latexLines.append('\\begin{align}')

# helper functions for latex to end an equation
def endEQ():
    latexLines.append('\\end{align}')
    
# helper functions for latex to add text
def text(text):
    latexLines.append(text)

# helper functions for latex to add an expression
def expr(expr, var=None, scale=-1):
    eqStr = ''
    scaleB = ''
    scaleE = ''
    varStr = ''
    latexStr = ''
    if not expr is None: # manually input equation or expression
        latexStr = latex(expr)
    if not var is None:
        varStr = var
        if not expr is None:
            eqStr = ' = '
    if scale > 0:
        scaleB = '\\text{\\scalebox{' + str(scale) + '}{$'
        scaleE = '$}}'
        eqStr = ' = '
    latexLines.append(scaleB + varStr + eqStr + latexStr + scaleE + '\\\\')

# helper functions to read from CSV
def getCSVData(filePath):
    datalist = np.genfromtxt(filePath, delimiter=',')
    return datalist

    
# helper functions to write to CSV
def writeCSVFile(filePath, data):
    csvfile, writer = openCSVFile(filePath)
    writer.writerows(data)
    csvfile.close()

# helper functions to open CSV
def openCSVFile(filePath):
    csvfile = openFileWrite(filePath)
    csvfile.truncate()
    writer = csv.writer(csvfile)
    return csvfile, writer

# helper functions to open file to write to CSV
def openFileWrite(filePath):
    if sys.version_info[0] == 2:  # Not named on 2.6
        if not os.path.exists(os.path.dirname(filePath)):
            try:
                os.makedirs(os.path.dirname(filePath))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        access = 'wb'
        kwargs = {}
    else:
        if not os.path.exists(os.path.dirname(filePath)):
            os.makedirs(os.path.dirname(filePath), exist_ok=True)
        access = 'wt'
        kwargs = {'newline':''}
    return open(filePath, access, **kwargs)

#%% Basic Equations ------------------------------------------------------------------
beginEQ('Basic Equations')
expr(iRef,'I_{ref}(\\theta)')
expr(dQ,'\\delta_{Q}(\\theta)')
expr(dD,'\\delta_{D}(\\theta)')
endEQ()

beginEQ('Bilateral Triangle $\\Delta^B$ \n\nA = triangle peak-to-peak \n\nDT = time of triangle peak')
tri1rms = sqrt((1/T)*(Integral((-A/2 + A/(D*T)*t)**2, (t, 0, D*T)) + \
                       Integral((-A/2 + A/((1-D)*T)*t)**2, (t, 0, (1-D)*T))))
expr(tri1rms,'\\Delta^B_{rms}(A,B,D,T)')
tri1rms = fixRoot(expand(tri1rms.doit()))
expr(tri1rms,'\\Delta^B_{rms}(A,B,D,T)')
display(tri1rms)

tri1avg = (1/T)*(Integral((-A/2 + A/(D*T)*t), (t, 0, D*T)) + \
                       Integral((-A/2 + A/((1-D)*T)*t), (t, 0, (1-D)*T)))
expr(tri1avg,'\\Delta^B_{avg}(A,B,D,T)')
tri1avg = simplify(fixRoot(expand(tri1avg.doit())))
expr(tri1avg,'\\Delta^B_{avg}(A,B,D,T)')
endEQ()
display(tri1avg)

beginEQ('Elevated Right Triangle $\\Delta^R$ \n\nB = Tri Y Midpoint \n\nA = triangle height \n\nDT = time of triangle peak')
tri2rms = sqrt((1/T)*Integral((B - A/2 + A/(D*T)*t)**2, (t, 0, D*T)))
expr(tri2rms,'\\Delta^R_{rms}(A,B,D,T)')
tri2rms = fixRoot(tri2rms.doit())
expr(tri2rms,'\\Delta^R_{rms}(A,B,D,T)')
display(tri2rms)

tri2avg = (1/T)*Integral((B - A/2 + A/(D*T)*t), (t, 0, D*T))
expr(tri2avg,'\\Delta^R_{avg}(A,B,D,T)')
tri2avg = simplify(fixRoot(expand(tri2avg.doit())))
expr(tri2avg,'\\Delta^R_{avg}(A,B,D,T)')
endEQ()
display(tri2avg)

#%% INDUCTOR ------------------------------------------------------------------
#%% Inductor rms simple
beginEQ('Inductor rms simple')
if CONV == 'boost':
    iRefrms = sqrt((1/pi)*Integral(iRef**2, (theta, 0, pi)))
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
expr(iRefrms,'I_{ref,rms}')
iRefrms = fixRoot(iRefrms.doit())
iLrms = iRefrms
expr(iLrms,'I_{ref,rms}')
expr(None,'I_{L,rms} = I_{ref,rms}')
display('iLrms', iLrms)

endEQ()
dataCSV[ROW['ilr']][COL['simple']] = iLrms.evalf(subs=subs)
display(iLrms.evalf(subs=subs))

#%% Inductor rms with ripple
beginEQ('Inductor rms with ripple')
if CONV == 'boost':
    expr(None,'\\Delta i_{L,pp}(\\theta) = v_{In}(\\theta)\\frac{\\delta_{Q}(\\theta)}{Lf}')
    iLRpp = vIn*(dQ/f)/L
    expr(iLRpp,'\\Delta i_{L,pp}(\\theta)')
    expr(None,'\\Delta i_{L,rms,t}(\\theta) = \\Delta^B_{rms}(A=i_{LR,pp})')
    iLRrmsT = tri1rms.subs(A,iLRpp) # = iLRpp/(2*sqrt(3))
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
display('iLRpp', iLRpp)
iLRrmsT = fixRoot(iLRrmsT)
expr(iLRrmsT,'\\Delta i_{L,rms,t}(\\theta)')
display('iLRrmsT', iLRrmsT)

iLRrms = sqrt((1/pi)*Integral(iLRrmsT**2, (theta, 0, pi)))
expr(iLRrms,'\\Delta I_{L,rms}')
iLRrms = fixRoot(iLRrms.doit())
expr(iLRrms,'\\Delta I_{L,rms}')
display('iLRrms', iLRrms)

expr(None,'I_{L,rms} = \\sqrt{I_{ref,rms}^2 + \\Delta I_{L,rms}^2}')
iLrmsWR = sqrt(iRefrms**2 + iLRrms**2)
iLrmsWR = fixRoot(iLrmsWR)
expr(iLrmsWR,'I_{L,rms}')
display('iLrmsWR', iLrmsWR)

endEQ()
dataCSV[ROW['ilr']][COL['ripple']] = iLrmsWR.evalf(subs=subs)
display(iLrmsWR.evalf(subs=subs))

#%% Inductor current ripple ratio

pprSubs = subs
pprSubs[theta]=pi/2
iLcrr = iLRpp.evalf(subs=pprSubs)/iRef.evalf(subs=pprSubs)
resultsFileName = os.path.join('SimResults', 'ResultsBoost') + distTag \
    + str(round(iLcrr*1000)) + '.csv'
display('Inductor current ripple ratio: iLcrr', iLcrr)

#%% Inductor core loss

# beginEQ('Inductor core loss with ripple')
# if CONV == 'boost':
#     expr(None,'H_{max}(\\theta) = \\frac{N}{l_{c}}\\Big(i_{Ref}(\\theta)+\\frac{\\Delta i_{L,pp}}{2}\\Big)')
#     HACMax = ln/ll*(iRef + iLRpp/2)
#     expr(fixRoot(HACMax), 'H_{max}(\\theta)')
#     expr(None,'B_{max}(\\theta) = \\bigg(\\frac{a_{b} + b_{b}H_{max} + c_{b}H_{max}^{2}}{1 + d_{b}H_{max} + e_{b}H_{max}^{2}}\\bigg)^{x_{b}}')
#     HACMin = ln/ll*(iRef - iLRpp/2)
#     expr(None,'H_{min}(\\theta) = \\frac{N}{l_{c}}\\Big(i_{Ref}(\\theta)-\\frac{\\Delta i_{L,pp}}{2}\\Big)')
#     expr(fixRoot(HACMin), 'H_{min}(\\theta)')
#     expr(None,'B_{min}(\\theta) = \\bigg(\\frac{a_{b} + b_{b}H_{min} + c_{b}H_{min}^{2}}{1 + d_{b}H_{min} + e_{b}H_{min}^{2}}\\bigg)^{x_{b}}')
#     expr(None,'B_{pk}(\\theta) = \\frac{\\Delta B}{2} = \\frac{B_{max} - B_{min}}{2}')
#     expr(None,'P_{d,t}(\\theta) = a_{p} B_{pk}^{b_{p}}(\\theta) f^{c_{p}}')
#     expr(None,'P_{L,mg,t}(\\theta) = P_{d,t}(\\theta) l_{c} A_{c}')
#     endEQ()
#     BACMax = ((lbha + lbhb*HACMax + lbhc*HACMax**2)/(1 + lbhd*HACMax + lbhe*HACMax**2))**lbhx
#     BACMin = ((lbha + lbhb*HACMin + lbhc*HACMin**2)/(1 + lbhd*HACMin + lbhe*HACMin**2))**lbhx
#     Bpk = (BACMax - BACMin)/2
#     PdensT = lpa*(Bpk**lpb)*(f*1e-3)**lpc
#     PLcoreT = PdensT*LL*LA/1000
# elif CONV == 'flyback':
#     0
# elif CONV == 'inverter':
#     0

# beginEQ()
# expr(None,'P_{L,mg} = \\frac{1}{\\pi}\\int^{\\pi}_{0} P_{L,mg,t}(\\theta)')
# expr(None,'P_{L,mg} = \\frac{1}{\\pi}\\int^{\\pi}_{0} a_{p} B_{pk}^{b_{p}}(\\theta) f^{c_{p}} l_{c} A_{c}')
# expr(None,'P_{L,mg} = \\frac{1}{1000}\\sum^{1000}_{n=0} P_{L,mg,t}(\\frac{\\pi}{1000}n)')
# expr(None,'P_{L,mg} = \\frac{1}{1000}\\sum^{1000}_{n=0} a_{p} B_{pk}^{b_{p}}(\\frac{\\pi}{1000}n) f^{c_{p}} l_{c} A_{c}')
# endEQ()
# # Integral Method
# # PLcore = (1/pi)*Integral(PLcoreT, (theta, 0, pi))
# # PLcore = fixRoot(PLcore.doit())
# # display(PLcore.evalf(subs=subs))
# # Sum Approximation
# display("Sum approximation")
# PLcoreT2 = PLcoreT.evalf(subs=subs)
# PLcore2 = 0
# for n in range(0, 1000):
#     PLcore2 = PLcore2 + PLcoreT2.evalf(subs={theta:pi*n/1000})
# PLcore2 = PLcore2/1000
# display(PLcore2)
# PLcore = PLcore2

#%% BRIDGE --------------------------------------------------------------------
#%% Bridge rms simple
beginEQ('Bridge rms simple')
if CONV == 'boost':
    expr(None,'I_{B,rms} = I_{L,rms}')
    iBrms = iLrms
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
expr(iBrms,'I_{B,rms}')

endEQ()
dataCSV[ROW['ibr']][COL['simple']] = iBrms.evalf(subs=subs)
display('iBrms', iBrms.evalf(subs=subs))

#%% Bridge rms with ripple
beginEQ('Bridge rms with ripple')
if CONV == 'boost':
    expr(None,'I_{B,rms} = I_{L,rms}')
    iBrmsWR = iLrmsWR
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
expr(iBrmsWR,'I_{B,rms}')

endEQ()
dataCSV[ROW['ibr']][COL['ripple']] = iBrmsWR.evalf(subs=subs)
display('iBrmsWR', iBrmsWR.evalf(subs=subs))

#%% Bridge avg
beginEQ('Bridge avg')
if CONV == 'boost':
    iBavg = (1/pi)*Integral(iRef + 0, (theta, 0, pi))
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
expr(iBavg,'I_{B,avg}')
iBavg = fixRoot(iBavg.doit())
expr(iBavg,'I_{B,avg}')
display('iBavg', iBavg)

endEQ()
dataCSV[ROW['iba']][COL['simple']] = iBavg.evalf(subs=subs)
dataCSV[ROW['iba']][COL['ripple']] = iBavg.evalf(subs=subs)
display(iBavg.evalf(subs=subs))

#%% SWITCH --------------------------------------------------------------------
#%% Switch rms simple
beginEQ('Switch rms simple')
if CONV == 'boost':
    expr(None,'i_{Q,rms,t}(\\theta) = \\Delta^R_{rms} \\big(' + \
         'B=i_{ref}, A=0, D=\\delta_{Q} \\big)')
    iQrmsT = fixRoot(tri2rms.subs([(B,iRef),(A,0),(D,dQ)]))
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
expr(iQrmsT,'i_{Q,rms,t}(\\theta)')
display('iQrmsT', iQrmsT)

iQrms = sqrt((1/pi)*Integral(iQrmsT**2, (theta, 0, pi)))
expr(iQrms,'I_{Q,rms}')
iQrms = fixRoot(iQrms.doit())
expr(iQrms,'I_{Q,rms}')
display('iQrms', iQrms)

endEQ()
dataCSV[ROW['iqr']][COL['simple']] = iQrms.evalf(subs=subs)
display(iQrms.evalf(subs=subs))

#%% Switch rms with ripple
beginEQ('Switch rms with ripple')
if CONV == 'boost':
    expr(None,'i_{Q,rms,t}(\\theta) = \\Delta^R_{rms} \\big(' + \
         'B=i_{ref},' +
         'A=i_{LR,pp},D=\\delta_{Q} \\big)')
    iQrmsT = tri2rms.subs([(B,iRef),(A,iLRpp),(D,dQ)])
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
iQrmsT = fixRoot(iQrmsT)
expr(iQrmsT,'i_{Q,rms,t}(\\theta)', 0.75)
display('iQrmsT', iQrmsT)

iQrmsWR = sqrt((1/pi)*Integral(iQrmsT**2, (theta, 0, pi)))
expr(iQrmsWR,'I_{Q,rms}')
iQrmsWR = fixRoot(iQrmsWR.doit())
expr(iQrmsWR,'I_{Q,rms}', 0.75)
display('iQrmsWR', iQrmsWR)

endEQ()
dataCSV[ROW['iqr']][COL['ripple']] = iQrmsWR.evalf(subs=subs)
display(iQrmsWR.evalf(subs=subs))

#%% Switch average
beginEQ('Switch avg')
if CONV == 'boost':
    expr(None,'i_{Q,avg,t}(\\theta) = \\Delta^R_{avg} \\big(' + \
         'B=i_{ref}, D=\\delta_{Q} \\big)')
    iQavgT = fixRoot(tri2avg.subs([(B,iRef),(D,dQ)]))
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
expr(iQavgT,'I_{Q,avg,t}')
iQavg = (1/pi)*Integral(iRef*dQ, (theta, 0, pi))
expr(iQavg,'I_{Q,avg}')
iQavg = fixRoot(iQavg.doit())
expr(iQavg,'I_{Q,avg}')
display('iQavg', iQavg)

endEQ()
display(iQavg.evalf(subs=subs))

#%% Switch switching timings
beginEQ('Switch switching loss timings')
expr(None,'t_{IR} = t_{2} - t_{1} = R_{g}C_{iss,test}\\text{ln}\\bigg(\\frac{V_{gs,max}-V_{th}}{V_{gs,max}-V_{gp}}\\bigg)')
expr(None,'t_{VF} = t_{3} = \\frac{R_{g}Q_{gd,test}V_{ds,max}}{V_{ds,test}(V_{gs,max} - V_{gp})}')
expr(None,'t_{VR} = t_{5} = \\frac{R_{g}Q_{gd,test}V_{ds,max}}{V_{ds,test}V_{gp}}')
expr(None,'t_{IF} = t_{6} = R_{g}C_{iss,test}\\text{ln}\\bigg(\\frac{V_{gp}}{V_{th}}\\bigg)')
QTIR = QRG*QCISS*np.log((QVGS - QVTH)/(QVGS - QVGP))
subs[qtir] = QTIR
QTIF = QRG*QCISS*np.log(QVGP/QVTH)
if CONV == 'boost':
    qtvf = QRG*(QQGD/QVDS)*(Vo/(QVGS - QVGP))
    qtvr = QRG*(QQGD/QVDS)*(Vo/QVGP)
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
QTVF = qtvf.evalf(subs=subs)
QTVR = qtvr.evalf(subs=subs)
display("QTIR: " + str(QTIR))
display("QTIF: " + str(QTIF))
display("QTVF: " + str(QTVF))
display("QTVR: " + str(QTVR))
expr(None,'t_{ON} = t_{IR} + t_{VF}')
expr(None,'t_{OFF} = t_{IF} + t_{VR}')
subs[qton] = QTIR + QTVF
subs[qtoff] = QTIF + QTVR
endEQ()

#%% Switch switching loss simple
beginEQ('Switch switching loss simple')
if CONV == 'boost':
    expr(None,'P_{Q,sw,t}(\\theta) = \\frac{V_{o}f}{2}I_{ref}(\\theta)(t_{ON} + t_{OFF})')
    PQswT = fixRoot((Vo*f*iRef/2)*(qton + qtoff))
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
expr(PQswT,'P_{Q,sw,t}(\\theta)')

PQsw = (1/pi)*Integral(PQswT, (theta, 0, pi))
expr(PQsw,'P_{Q,sw}')
PQsw = fixRoot(PQsw.doit())
expr(PQsw,'P_{Q,sw}')
display('PQsw', PQsw)
endEQ()
display(PQsw.evalf(subs=subs))

#%% Switch switching loss with ripple
beginEQ('Switch switching loss with ripple')
if CONV == 'boost':
    expr(None,'P_{Q,sw,t}(\\theta) = \\frac{V_{o}f}{2}\\bigg(\\big(I_{ref}(\\theta)-\\Delta i_{L,pp}(\\theta)\\big)t_{ON}' + \
    ' + \\big(I_{ref}(\\theta)+\\Delta i_{L,pp}(\\theta)\\big)t_{OFF}\\bigg)')
    PQswT = (Vo*f/2)*((iRef-iLRpp/2)*qton + (iRef+iLRpp/2)*qtoff)
    # can be assumed the simple form if qton approx equals qtoff
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
expr(PQswT,'P_{Q,sw,t}(\\theta)', 0.75)

PQswWR = (1/pi)*Integral(PQswT, (theta, 0, pi))
display('PQswWR', PQswWR)
expr(PQswWR,'P_{Q,sw}')
PQswWR = fixRoot(PQswWR.doit())
if AC:
    endEQ()
    beginEQ('')
    expr(PQswWR,'P_{Q,sw}', 0.35)
else:
    expr(PQswWR,'P_{Q,sw}', 0.6)
display('PQswWR', PQswWR)
endEQ()
display(PQswWR.evalf(subs=subs))

#%% Switch switching loss output capacitance
beginEQ('Switch switching loss output capacitance')
if CONV == 'boost':
    PQswT = fixRoot(qcoss*Vo**2*f/2)
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
expr(PQswT,'P_{Q,sw,c}(\\theta)')

PQswc = (1/pi)*Integral(PQswT, (theta, 0, pi))
expr(PQswc,'P_{Q,sw,c}')
PQswc = fixRoot(PQswc.doit())
expr(PQswc,'P_{Q,sw,c}')
display('PQswc', PQswc)

endEQ()
display(PQswc.evalf(subs=subs))

#%% BOOST DIODE ---------------------------------------------------------------
#%% Boost Diode simple
beginEQ('Boost Diode simple')
if CONV == 'boost':
    expr(None,'i_{D,rms,t}(\\theta) = \\Delta^R_{rms} \\big(' + \
         'B=i_{ref}, A=0, D=\\delta_{D} \\big)')
    iDrmsT = fixRoot(tri2rms.subs([(B,iRef),(A,0),(D,dD)]))
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
expr(iDrmsT,'i_{D,rms,t}(\\theta)')
display('iDrmsT', iDrmsT)

iDrms = sqrt((1/pi)*Integral(iDrmsT**2, (theta, 0, pi)))
expr(iDrms,'I_{D,rms}')
iDrms = fixRoot(iDrms.doit())
expr(iDrms,'I_{D,rms}')
display('iDrms', iDrms)

dataCSV[ROW['idr']][COL['simple']] = iDrms.evalf(subs=subs)
endEQ()
display(iDrms.evalf(subs=subs))

#%% Boost Diode with ripple
beginEQ('Boost Diode with ripple')
if CONV == 'boost':
    expr(None,'i_{D,rms,t}(\\theta) = \\Delta^R_{rms} \\big(' + \
         'B=i_{ref},' +
         'A=i_{LR,pp},D=\\delta_{D} \\big)')
    iDrmsT = tri2rms.subs([(B,iRef),(A,iLRpp),(D,dD)])
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
iDrmsT = fixRoot(iDrmsT)
expr(iDrmsT,'i_{D,rms,t}(\\theta)', 0.75)
display('iDrmsT', iDrmsT)

iDrmsWR = sqrt((1/pi)*Integral(iDrmsT**2, (theta, 0, pi)))
expr(iDrmsWR,'I_{D,rms}')
iDrmsWR = fixRoot(iDrmsWR.doit())
expr(iDrmsWR,'I_{D,rms}')
display('iDrmsWR', iDrmsWR)

endEQ()
dataCSV[ROW['idr']][COL['ripple']] = iDrmsWR.evalf(subs=subs)
display(iDrmsWR.evalf(subs=subs))

#%% Boost Diode average
beginEQ('Boost Diode avg')
if CONV == 'boost':
    expr(None,'i_{D,avg,t}(\\theta) = \\Delta^R_{avg} \\big(' + \
         'B=i_{ref}, D=\\delta_{D} \\big)')
    iDavgT = fixRoot(tri2avg.subs([(B,iRef),(D,dD)]))
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
expr(iDavgT,'I_{D,avg,t}')
iDavg = (1/pi)*Integral(iRef*dD, (theta, 0, pi))
expr(iDavg,'I_{D,avg}')
iDavg = fixRoot(iDavg.doit())
expr(iDavg,'I_{D,avg}')
display('iDavg', iDavg)

endEQ()
dataCSV[ROW['ida']][COL['simple']] = iDavg.evalf(subs=subs)
dataCSV[ROW['ida']][COL['ripple']] = iDavg.evalf(subs=subs)
display(iDavg.evalf(subs=subs))

#%% Boost Diode swiching loss reverse recovery timings
beginEQ('Boost Diode switching loss timings')
expr(None,'K_{Q} = \\frac{I_{rr,0}T_{rr,0}}{2\\sqrt{I_{F,0}}}')
expr(None,'S = \\frac{T_{rr,0}\\frac{dI_{D,0}}{dt}}{I_{rr,0}} - 1')
subs[dkq0] = DIRR*DTRR/(2*sqrt(DIF))
subs[ds] = DTRR*DDIDT/DIRR - 1 # diode reverse recovery softness (also called IR)
display('KQ0 = ', dkq0.evalf(subs=subs))
display('S = ', ds.evalf(subs=subs))
expr(None,'I_{rr} = \\sqrt{\\frac{2\\frac{dI_{D}}{dt}K_{Q}\\sqrt{I_{F}}}{1+S}}')
expr(None,'T_{a} = \\frac{I_{rr}}{\\frac{dI_{D}}{dt}}')
expr(None,'T_{b} = ST_{a}')
expr(None,'E_{Q} = V_{DS}\\big(\\frac{I_{rr}}{2}T_{a}+\\frac{I_{rr}}{4}T_{b}\\big)')
expr(None,'E_{D} = \\frac{V_{R}I_{rr}}{4}T_{b}')
endEQ()

#%% Boost Diode switching loss reverse recovery simple
beginEQ('Boost Diode switching loss reverse recovery simple')
if CONV == 'boost':
    dif = iRef
    expr(dif, 'I_{F}')
    display('Iref = ', iRef.evalf(subs=subs))
    display('iLRpp = ', iLRpp.evalf(subs=subs))
    display('IF = ', dif.evalf(subs=subs))
    ddidt = dif/qtir
    expr(ddidt, '\\frac{dI_{D}}{dt}')
    display('didt = ', ddidt.evalf(subs=subs))
    dirr = sqrt((2*ddidt*dkq0*sqrt(dif))/(1+ds))
    expr(dirr, 'I_{rr}')
    display('Irr = ', dirr.evalf(subs=subs))
    dta = dirr/ddidt
    expr(dta, 'T_{a}')
    display('ta = ', dta.evalf(subs=subs))
    dtb = ds*dta
    expr(dtb, 'T_{b}')
    display('tb = ', dtb.evalf(subs=subs))
    deq = Vo*dirr*(dta/2 + dtb/4)
    ded = Vo*dirr*dtb/4
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
deq = simplify(deq)
ded = simplify(ded)
expr(deq,'E_{Qrr}')
expr(ded,'E_{Drr}')
PDswT = (deq+ded)*f
PDswT = simplify(PDswT)
expr(PDswT,'P_{D,sw,rr}(\\theta)')
display('PDswT = ', PDswT.evalf(subs=subs))
display('PDswT ', PDswT)

PDswrr = (1/pi)*Integral(PDswT, (theta, 0, pi))
expr(PDswrr,'P_{D,sw,rr}')
display('PDswrr', PDswrr)
if not AC:
    PDswrr = fixRoot(PDswrr.doit())
else:
    A = 1
    B = 0
    PDswrr = sqrt(2*Po/Vpk)*(dkq0*Vo*f/pi)*Integral( \
        sqrt(A + B) - (theta - pi/2)**2*(A + 2*B)/(4*sqrt(A + B)), (theta, 0, pi))
    expr(PDswrr,'P_{D,sw,rr}')
    display('PDswrr', PDswrr)
    PDswrr = fixRoot(PDswrr.doit())
expr(PDswrr,'P_{D,sw,rr}')
display('PDswrr', PDswrr)

endEQ()
display(PDswrr.evalf(subs=subs))

#%% Boost Diode switching loss reverse recovery with ripple
beginEQ('Boost Diode switching loss reverse recovery with ripple')
if CONV == 'boost':
    dif = iRef - iLRpp/2
    expr(dif, 'I_{F}')
    display('Iref = ', iRef.evalf(subs=subs))
    display('iLRpp = ', iLRpp.evalf(subs=subs))
    display('IF = ', dif.evalf(subs=subs))
    ddidt = dif/qtir
    expr(ddidt, '\\frac{dI_{D}}{dt}')
    display('didt = ', ddidt.evalf(subs=subs))
    dirr = sqrt((2*ddidt*dkq0*sqrt(dif))/(1+ds))
    expr(dirr, 'I_{rr}')
    display('Irr = ', dirr.evalf(subs=subs))
    dta = dirr/ddidt
    expr(dta, 'T_{a}')
    display('ta = ', dta.evalf(subs=subs))
    dtb = ds*dta
    expr(dtb, 'T_{b}')
    display('tb = ', dtb.evalf(subs=subs))
    deq = Vo*dirr*(dta/2 + dtb/4)
    ded = Vo*dirr*dtb/4
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
deq = simplify(deq.doit())
ded = simplify(ded.doit())
expr(deq,'E_{Qrr}')
expr(ded,'E_{Drr}')
endEQ()
beginEQ('')
PDswT = (deq+ded)*f
PDswT = simplify(PDswT.doit())
expr(PDswT,'P_{D,sw,rr}(\\theta)')
display('PDswT = ', PDswT.evalf(subs=subs))

PDswrrWR = (1/pi)*Integral(PDswT, (theta, 0, pi))
expr(PDswrrWR,'P_{D,sw,rr}')
display('PDswrrWR', PDswrrWR)
if not AC:
    PDswrrWR = fixRoot(PDswrrWR.doit())
else:
    A = 4*L*Po*Vo*f - Vo*Vpk**2
    B = Vpk**3
    PDswrrWR = sqrt(2*Vo*f/(L*Vpk))*(dkq0/(2*pi))*Integral( \
        sqrt(A + B) - (theta - pi/2)**2*(A + 2*B)/(4*sqrt(A + B)), (theta, 0, pi))
    expr(PDswrrWR,'P_{D,sw,rr}')
    display('PDswrrWR', PDswrrWR)
    PDswrrWR = fixRoot(PDswrrWR.doit())
expr(PDswrrWR,'P_{D,sw,rr}')
display('PDswrrWR', PDswrrWR)

endEQ()
display(PDswrrWR.evalf(subs=subs))

#%% Boost Diode switching loss junction capacitance
beginEQ('Boost Diode switching loss junction capacitance')
expr(None, 'P_{D,sw,c}(\\theta) = \\frac{C_{j}V_{D,bl}^{2}(\\theta)f}{2}')
if CONV == 'boost':
    PDswT = fixRoot(dcj*Vo**2*f/2)
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
expr(PDswT,'P_{D,sw,c}(\\theta)')

PDswc = (1/pi)*Integral(PDswT, (theta, 0, pi))
expr(PDswc,'P_{D,sw,c}')
PDswc = fixRoot(PDswc.doit())
expr(PDswc,'P_{D,sw,c}')
display('PDswc', PDswc)

endEQ()
display(PDswc.evalf(subs=subs))

#%% CAPACITOR -----------------------------------------------------------------
#%% Cap rms simple
beginEQ('Cap rms simple')
if CONV == 'boost':
    expr(None,'I_{C,rms} = \\sqrt{I_{D,rms}^2 - (\\frac{P_o}{V_o})^2}')
    iCrms = sqrt(iDrms**2 - (Po/Vo)**2)
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
iCrms = fixRoot(iCrms)
expr(iCrms,'I_{C,rms}')
display('iCrms', iCrms)

endEQ()
dataCSV[ROW['icr']][COL['simple']] = iCrms.evalf(subs=subs)
display(iCrms.evalf(subs=subs))

#%% Cap rms with ripple
beginEQ('Cap rms with ripple')
if CONV == 'boost':
    expr(None,'I_{C,rms} = \\sqrt{I_{D,rms}^2 - (\\frac{P_o}{V_o})^2}')
    iCrmsWR = sqrt(iDrmsWR**2 - (Po/Vo)**2)
elif CONV == 'flyback':
    0
elif CONV == 'inverter':
    0
iCrmsWR = fixRoot(iCrmsWR)
expr(iCrmsWR,'I_{C,rms}')
display('iCrmsWR', iCrmsWR)

endEQ()
dataCSV[ROW['icr']][COL['ripple']] = iCrmsWR.evalf(subs=subs)
display(iCrmsWR.evalf(subs=subs))

#%% END -----------------------------------------------------------------------
#%% End
# create latex equations
latexLines.append('\\end{document}')
with openFileWrite(latexFileName) as latexFile:
    latexFile.writelines(line + '\n' for line in latexLines)
display('', 'equations saved to ' + latexFileName)
# create sim params file
with openFileWrite(simParamFileName) as paramFile:
    for key in params.keys():
        paramFile.write(key + ' = ' + str(params[key]) + ';\n')
display('sim params saved to ' + simParamFileName)
# get sim data
with open(simDataFileName, 'r') as simDataFile:
    reader = csv.reader(simDataFile, delimiter=',')
    csvData = list(reader)
simHeader = csvData[0]
simData = np.array(csvData[1:])
simData = simData.astype(float)
display('data read from ' + simDataFileName)
# output results comparison
dataCSV[ROW['ilr']][COL['sim']] = np.sqrt(np.mean(simData[:,simHeader.index('I(L)')]**2))
dataCSV[ROW['ibr']][COL['sim']] = np.sqrt(np.mean(simData[:,simHeader.index('I(B1)')]**2))
dataCSV[ROW['iba']][COL['sim']] = np.mean(simData[:,simHeader.index('I(B1)')])
dataCSV[ROW['iqr']][COL['sim']] = np.sqrt(np.mean(simData[:,simHeader.index('I(Q)')]**2))
dataCSV[ROW['idr']][COL['sim']] = np.sqrt(np.mean(simData[:,simHeader.index('I(D)')]**2))
dataCSV[ROW['ida']][COL['sim']] = np.mean(simData[:,simHeader.index('I(D)')])
dataCSV[ROW['icr']][COL['sim']] = np.sqrt(np.mean(simData[:,simHeader.index('I(C)')]**2))
writeCSVFile(resultsFileName, dataCSV)
display('data stored in ' + resultsFileName)
display(dataCSV)

#%% Parametric Evaluation, storing loss data

WR = True   
NPARALOSS = 5 + 0 + 2 # conduction, magnetic, switching
if DOPARAMETRIC:
    VOLOSSANALYSIS = 400 # the output voltage for the loss analysis
    # arrVo = [48]
    # arrPo = arrVo[0]*np.linspace(0, 2, num=11)
    arrVo = [200, 300, 400]
    arrPo = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
else:
    VOLOSSANALYSIS = params['Vo'] # the output voltage for the loss analysis
    arrPo = [params['Po']]
    arrVo = [params['Vo']]

# arrPo = [100*n for n in range(1, 6)]
# arrVo = [100*n for n in range(2, 5)]
effMat = np.zeros((len(arrVo), len(arrPo)))
lossMat = np.zeros((len(arrPo), NPARALOSS))

for nVo in range(0, len(arrVo)):
    for nPo in range(0, len(arrPo)):
        subs2 = subs.copy()
        subs2[Vo] = arrVo[nVo]
        subs2[Po] = arrPo[nPo]
# Inductor total loss
        PLco = RL*(iLrms.evalf(subs=subs2))**2
        PLcoWR = RL*(iLrmsWR.evalf(subs=subs2))**2
        # PLmgWR = PLcore.evalf(subs=subs2)
# Bridge total loss
        if AC:
            PBco = VB*iBavg.evalf(subs=subs2) + RB*(iBrms.evalf(subs=subs2))**2
            PBcoWR = VB*iBavg.evalf(subs=subs2) + RB*(iBrmsWR.evalf(subs=subs2))**2
        else:
            PBco = 0
            PBcoWR = 0
# Switch total loss
        if IGBT:
            PQco = arrVo[nVo]*iQavg.evalf(subs=subs2) + RQ*(iQrms.evalf(subs=subs2))**2
            PQcoWR = arrVo[nVo]*iQavg.evalf(subs=subs2) + RQ*(iQrmsWR.evalf(subs=subs2))**2
        else:
            PQco = RQ*(iQrms.evalf(subs=subs2))**2
            PQcoWR = RQ*(iQrmsWR.evalf(subs=subs2))**2
        PQsw = PQsw.evalf(subs=subs2) + PQswc.evalf(subs=subs2)
        PQswWR = PQswWR.evalf(subs=subs2) + PQswc.evalf(subs=subs2)      
# Boost diode total loss
        PDco = VD*iDavg.evalf(subs=subs2) + RB*(iDrms.evalf(subs=subs2))**2
        PDcoWR = VD*iDavg.evalf(subs=subs2) + RB*(iDrmsWR.evalf(subs=subs2))**2
        PDsw = PDswrr.evalf(subs=subs2) + PDswc.evalf(subs=subs2)
        tempPDswrrWR = PDswrrWR.evalf(subs=subs2)
        if not tempPDswrrWR.is_real:
            tempPDswrrWR = 0
        PDswWR = tempPDswrrWR + PDswc.evalf(subs=subs2)
# Capacitor total loss
        PCco = RC*(iCrms.evalf(subs=subs2))**2
        PCcoWR = RC*(iCrmsWR.evalf(subs=subs2))**2

# Efficiency and Loss Array
        if WR:
            effMat[nVo, nPo] = 1-(PLcoWR + PQcoWR + PQswWR + \
                  PDcoWR + PDswWR + PCcoWR + PBcoWR)/float(arrPo[nPo])
            if arrVo[nVo] == VOLOSSANALYSIS:
                lossMat[nPo,:] = np.array([PLcoWR, PQcoWR, PQswWR, \
                    PDcoWR, PDswWR, PCcoWR, PBcoWR])/float(arrPo[nPo])
        else:
            effMat[nVo, nPo] = 1-(PLco + PQco + PQsw + \
                  PDco + PDsw + PCco + PBco)/float(arrPo[nPo])
            if arrVo[nVo] == VOLOSSANALYSIS:
                lossMat[nPo,:] = np.array([PLco, PQco, PQsw, \
                    PDco, PDsw, PCco, PBco])/float(arrPo[nPo])        
if AC:
    effFileName = os.path.join('LossResults', 'ACEff.csv')
    lossFileName = os.path.join('LossResults', 'ACLoss.csv')
else:
    effFileName = os.path.join('LossResults', 'DCEff.csv')
    lossFileName = os.path.join('LossResults', 'DCLoss.csv')
display('data stored in ' + effFileName)
writeCSVFile(effFileName, effMat.tolist())
display('data stored in ' + lossFileName)
writeCSVFile(lossFileName, lossMat.tolist())
if not DOPARAMETRIC:
    print("Loss Breakdown")
    print("Lcond, Qcond, Qsw, Dcond, Dsw, Ccond, Bcond")
    print(lossMat/np.sum(lossMat))
    print("Total Efficiency")
    print(effMat[0,0])

#%% Plot loss bars, get statistics
fontSize = 12
def formatPlot(fig, title, projectFolderDir, outputFileName, xlabel, ylabel):
    plt.title(title, fontsize=fontSize*1.5)
    plt.ylabel(ylabel, fontsize=fontSize)
    plt.yticks(fontsize=fontSize)
    plt.xticks(fontsize=fontSize)
    plt.xlabel(xlabel, fontsize=fontSize)
    plt.tight_layout()
    sb.set_style("darkgrid")
    plt.savefig(os.path.join(projectFolderDir, outputFileName+'.pdf'))
    plt.close(fig)

# Graphing stored parametric loss data
effAC = np.genfromtxt(os.path.join('LossResults', 'ACEff.csv'), delimiter=',')
effDC = np.genfromtxt(os.path.join('LossResults', 'DCEff.csv'), delimiter=',')
lossAC = np.genfromtxt(os.path.join('LossResults', 'ACLoss.csv'), delimiter=',')
lossDC = np.genfromtxt(os.path.join('LossResults', 'DCLoss.csv'), delimiter=',')

# Get max and min of (AC losses)/(DC losses)
lossRatioMat = (1-effAC)/(1-effDC)
display('Min Loss Ratio AC:DC = ' + str(np.min(lossRatioMat)))
display('Max Loss Ratio AC:DC = ' + str(np.max(lossRatioMat)))

# Plot Efficiency
fig = plt.figure()
xTicks = len(arrVo)
for nVo in range(0, len(arrVo)):
    plt.plot(arrPo, 100*effAC[nVo, :], color= (1, float(nVo)/(xTicks-1), 0), \
             label='AC, Vo='+str(arrVo[nVo]))
for nVo in range(0, len(arrVo)):
    plt.plot(arrPo, 100*effDC[nVo, :], color= (0, float(nVo)/(xTicks-1), 1), \
             label='DC, Vo='+str(arrVo[nVo]))
plt.legend(loc='lower right')
plt.ylim((85,100))
plt.locator_params(axis='y', nbins=6)
plt.grid(linestyle=':',linewidth=.5)
formatPlot(fig, 'Efficiency Curves', 'LossResults', 'EffCurves', \
    'Output Power (W)', 'Efficiency (%)')
display('Efficiency Plot stored to... ' + 'LossResults/EffCurves')

# Plot Loss Analysis
numRuns = len(arrPo)
numGrids = 2
barWidth = 1.0/(numGrids + 1)
tickOffset = (barWidth*numGrids)/2.0
index = np.arange(numRuns)
labelSet = ['L Cond', 'Q Cond', 'Q Sw', \
            'D Cond', 'D Sw', 'C Cond', 'B Cond']
fig = plt.figure()
plt.title('System Loss', fontsize=20)
dataset = []
for n in range(0, NPARALOSS):
    dataset.extend([[100*lossAC[:, n], 100*lossDC[:, n]]])
labelsUsed = np.zeros((NPARALOSS))
hatches = ['\\\\', '//', 'X', 'o', '*', '.']
hatches = hatches[0:numGrids]
gridLabels = ['AC', 'DC']
handles = []
labels = []
for n in range(0, numGrids):
    handles.extend([mpatches.Patch(facecolor='white', edgecolor='black', \
        hatch=hatches[n])])
    labels.extend([gridLabels[n]])
categoryHandles = ['']*len(dataset)
categoryLabels = ['']*len(dataset)
for n in range(0, numGrids):
    bottom = 0
    for m in range(0, len(dataset)):
        allZero = True
        for value in dataset[m][n]:
            if value > 0.001:
                allZero = False
        if allZero:
            continue
        color = cm.Set2((len(dataset) - m - 1.0)/(len(dataset) - 1.0))
        plt.bar(index + barWidth*n, \
            dataset[m][n], barWidth, bottom=bottom, color=color, \
            edgecolor='black', hatch=hatches[n])
        if labelsUsed[m] == 0:
            categoryHandles[m] = mpatches.Patch(facecolor=color, \
                edgecolor='black', hatch='')
            categoryLabels[m] = labelSet[m]
            labelsUsed[m] = 1
        bottom = bottom + dataset[m][n]
categoryHandles = list(filter(lambda a: a != '', categoryHandles))
categoryLabels = list(filter(lambda a: a != '', categoryLabels))
handles.extend(list(reversed(categoryHandles)))
labels.extend(list(reversed(categoryLabels)))
plt.xticks(index + tickOffset, [str(n) for n in arrPo], fontsize=fontSize)
plt.xlabel('Output Power', fontsize=fontSize)
plt.legend(handles=handles, labels=labels, loc='lower left', fontsize=fontSize*.8)
# lgd = plt.legend(handles=handles, labels=labels, loc='center left', \
#     bbox_to_anchor=(1, 0.5), fontsize=fontSize)
# fig.artists.append(lgd)
plt.grid(True, linestyle='dotted')
plt.gca().set_axisbelow(True)
plt.gca().xaxis.grid(False)
formatPlot(fig, 'Loss Analysis', 'LossResults', 'LossAnalysis', \
           'Output Power (W)', 'Loss (%)')
display('Loss Analysis Plot stored to... ' + 'LossResults/LossAnalysis')


