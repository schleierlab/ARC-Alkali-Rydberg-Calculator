# import arc 
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd

from arc.advanced.dressing import DressedInteractions
from arc import Cesium
from scipy.constants import pi, hbar
#from ARC_Alkali_Rydberg_Calculator.arc.advanced.dressing import DressedInteractions

'''
1) Setup Python 3.7 (maybe 3.8?) virtual environment
2) Install numpy, scipy, matplotlib, sympy, lmfit
## 3) pip3.7 install ARC-Alkali-Rydberg-Calculator==3.0.19
4) Download the ARC source code from Schlier Smith Lab GitHub
5) cd into it
6) python setup.py install
7) Delete the folder if you like and enjoy :)
'''
'''
calc = DressedInteractions(Cesium(),43,1,1.5)
calc.defineBasis(0,0,2,2,20e9,1e-4)

calc.diagonalize(np.linspace(1,10,100),250,progressOutput = True)

calc.defineDressing(16e6,1,[(6,0,0.5,0.5,3.5)],
                    [1],groundStateOffset=4.021776399375e9,
                    measuredRabiFreq=2e6)

calc.calculateDressedPotential()

calc.plotDressedPotential(includePertPrediction=True,color='mediumseagreen')
calc.showPlot()
'''

###Basis parameters###
theta=0*np.pi
phi=0; #Azimuthal Angle [0-2pi]
dn = 2; #Range of n to consider (n0-dn:n0+dn)
dl = 2; #Range of l values
deltaMax = 20e9 #Max pair-state energy difference [Hz], must be at least 20e9 to converge
Bz = 1e-4 #in T
nEig= 250 #Number of eigenstates to extract

####Dressing parameters###

atom = Cesium()
nRydberg = 74
lRydberg= 1
jRydberg= 1.5
laserDetuning = (atom.getZeemanEnergyShift(lRydberg,jRydberg,1.5,Bz)/(2*pi*hbar)+8*1e6) # Hz from zero field energy
# 16 MHz detuning is from the mj=3/2
# 02/02/26 changed to 8 MHz detuning

## check the mj, mI
groundState = [(6,0,0.5,0.5,-0.5),(6,0,0.5,-0.5,0.5)] # |4,0> with (n, l, j, mj, mI)
groundStateCoeffs = [1/np.sqrt(2),1/np.sqrt(2)]
groundStateOffset = 4.021776399375e9 #in Hz, for Cs F=4
laserPolarization = 1

## info about the measured Rabi frequency
measuredRabiFreq = 2e6 #Hz

###Calculation###
calc = DressedInteractions(atom,nRydberg,lRydberg,jRydberg)
r = np.linspace(calc.getLeRoyRadius(),10,100)

calc.defineBasis(theta,phi, dn,dl, deltaMax,Bz,progressOutput=False,debugOutput=False) 
# Diagonalize
calc.diagonalise(r,nEig)

calc.plotLevelDiagram(highlightColor='darkblue')
calc.ax.set_ylim([-.25,.05])
calc.fig.set_size_inches(10,5.0)
calc.ax.grid()
calc.ax.collections[0].set_rasterized(True)
calc.showPlot()

calc.defineDressing(laserDetuning,laserPolarization,groundState,
                    groundStateCoeffs,groundStateOffset,
                    measuredRabiFreq=measuredRabiFreq, debugOutput=True)

calc.calculateDressedPotential()

#Plot
calc.plotDressedPotential(includePertPrediction=True,color='mediumseagreen')
calc.showPlot()