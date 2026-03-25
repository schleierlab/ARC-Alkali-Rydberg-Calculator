# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e as C_e, h as C_h, hbar
from arc.alkali_atom_functions import *
from arc.calculations_atom_pairstate import PairStateInteractions

class DressedInteractions(PairStateInteractions):

    # =============================== Methods ===============================

    # currently only supports interaction between identical Alkali atoms
    def __init__(self, atom, n, l, j,
                 s=0.5, interactionsUpTo=1):
        """Initialize the dressed state calculation. Both "atoms" are defined
        to be in the same state as we are dressing to only one state identical
        to both atoms.

        Args:
            atom (type): atom (:obj:`AlkaliAtom`): = {
                :obj:`arc.alkali_atom_data.Lithium6`,
                :obj:`arc.alkali_atom_data.Lithium7`,
                :obj:`arc.alkali_atom_data.Sodium`,
                :obj:`arc.alkali_atom_data.Potassium39`,
                :obj:`arc.alkali_atom_data.Potassium40`,
                :obj:`arc.alkali_atom_data.Potassium41`,
                :obj:`arc.alkali_atom_data.Rubidium85`,
                :obj:`arc.alkali_atom_data.Rubidium87`,
                :obj:`arc.alkali_atom_data.Caesium` }
            n (int): principal quantum number for the coupled state.
            l (int): orbital angular momentum for the coupled state.
            j (float): total angular momentum for the coupled state.
            s (float): optional, spin state of the atoms. Default value
                of 0.5 is correct for :obj:`AlkaliAtom`.

                Example:
                    Dressing a Cesium :math:`43 P_{3/2} m_j=3/2` state,
                    from :math:`F=4, m_F=4` ground state with circular
                    polarization :math:`\sigma_+`.

                    from arc import *
                    from arc.advanced.dressing import DressedInteractions

                    calc = DressedInteractions(Cesium(),43,1,1.5)
                    calc.defineBasis(0,0,2,2,20e9,1e-4)
                    calc.diagonalize(np.linspace(1,10,100),250)

                    calc.defineDressing(20e6,1,[(6,0,0.5,0.5,3.5)],
                                        [1],groundStateOffset=4.021776399375e9,
                                        measuredRabiFreq=2e6)
                    calc.calculateDressedPotential()
                    calc.plotDressedPotential(includePertPrediction=True,color='mediumseagreen')
                    calc.showPlot()

        """
        m1 = 0.5
        m2 = -0.5
        # pair state (n,l,j,0.5) (n,l,j,-0.5) defines the zero pair
        # state energy

        super().__init__(atom, n, l, j, n, l, j, m1, m2,
                         s=s, interactionsUpTo=interactionsUpTo,
                         s2=None, atom2=None)

    def defineBasis(self, theta, phi, dn, dl, deltaMax, Bz=0,
                    progressOutput=False, debugOutput=False):
        """Overloads the PairStateInteractions method. Makes sure forceNolimitMj
        is True while diagonalizing to give access to all pair states
        independent of the reference state chosen.
        """
        super().defineBasis(theta, phi, dn, dl, deltaMax, Bz=Bz,
                            progressOutput=progressOutput,
                            forceNolimitMj=True, debugOutput=debugOutput)

    def diagonalise(self, rangeR, noOfEigenvectors, eigenstateDetuning=0.,
                    sortEigenvectors=False, progressOutput=False,
                    debugOutput=False):
        """Finds eigenstates in the atom pair basis. Overloads the
            PairStateInteractions method.

        """

        super().diagonalise(rangeR, noOfEigenvectors,
                            drivingFromState=[0, 0, 0, 0, 0],
                            eigenstateDetuning=eigenstateDetuning,
                            sortEigenvectors=sortEigenvectors,
                            progressOutput=progressOutput,
                            debugOutput=debugOutput)

        self.eigVals = np.array(self.y)
        self.eigVecs = np.array(self.eigVecs)
        self.originalOverlap = self.eigVecs[:, :,
                                            self.originalPairStateIndex]**2

    def defineDressing(self, laserDetuning, laserPolarization, groundState,
                       groundStateCoeffs, groundStateOffset,
                       measuredRabiFreq=None,
                       laserWaist=100e-6,
                       laserPower=1, debugOutput=False):
        """Define dressing parameters.

        Args:
            laserDetuning (float): Laser detuning in Hz from the zero-field
                state defined by (self.n,self.l,self.j).
            laserPolarization (int): specifies transition that the driving field
                couples to, +1, 0 or -1 corresponding to driving :
                math:`\sigma^+`, :math:`\pi` and :math:`\sigma^-` transitions
                respectively.
            groundState (list): List of states in the ground state
                superposition.
            groundStateCoeffs (list): Amplitude of each state in the ground
                state superposition.
            groundStateOffset (float): Frequency offset in Hz of the ground
                state. In e.g. Cs, this is the shift of a hyperfine F=3 or F=4
                ground state from the 6S1/2 ground state COM.
            measuredRabiFreq (float): optional, None by default. Empirically
                measured Rabi frequency from the ground state to state
                (self.n,self.l,self.j).
            laserWaist (float): Optional, by default 100e-6. If not using
                measuredRabiFreq, use this laserWaist in m to calculate Rabi
                frequencies.
            laserPower (float): Optional, by default 1. If not using
                measuredRabiFreq, use this laser power in W to calculate
                Rabi frequencies.
            debugOutput (float): Optional, by default False. If True,
                print debug output.

        """

        self.laserDetuning = laserDetuning
        self.laserPolarization = laserPolarization

        self.groundState = groundState
        self.groundStateCoeffs = groundStateCoeffs

        self.measuredRabiFreq = measuredRabiFreq
        self.laserWaist = laserWaist
        self.laserPower = laserPower

        # make sure ground state coefficient squares sum to 1
        assert np.isclose(sum([gsc**2 for gsc in groundStateCoeffs]), 1)

        # Joule
        self.groundStateEnergy = self.atom1.getEnergy(groundState[0][0],
                                                      groundState[0][1],
                                                      groundState[0][2])*C_e +\
                                                      groundStateOffset*C_h

        # reference the laser frequency to the first atom excited state energy
        # Joules/hbar
        self.laserFreq = (self.atom1.getEnergy(self.n, self.l, self.j)*C_e -
                          self.groundStateEnergy + laserDetuning*C_h)/hbar

        # initialize the list of ground state mI numbers
        self.mIlist = []
        for indexgs, gs in enumerate(self.groundState):
            if groundStateCoeffs[indexgs] > 0:
                self.mIlist.append(gs[4])
        self.mIlist = list(set(self.mIlist))
        self.mIMultiplicity = len(self.mIlist)**2

        # create a list of extended basis states
        self.extendedBasisStates = []
        for state in self.basisStates:
            for mI1 in self.mIlist:
                for mI2 in self.mIlist:
                    self.extendedBasisStates.append(np.append(state,
                                                              [mI1, mI2]))
        self.extendedBasisStates = np.array(self.extendedBasisStates)

        # create a dictionary of (n,l,j,mj,mI) => rabi frequency
        # Rabi frequencies are in rad/s
        self.rabiFreqDict = {}

        # In case you are using experimentally determined Rabi frequency,
        # calculate the reference DME and the proportionality constant
        # to use later in calculating Rabi frequency couplings to different
        # excited states.
        if measuredRabiFreq:
            # need to manually make sure the first ground state
            # couples to an excited state to not divide by 0 later
            proportionalityConstant = groundStateCoeffs[0]**2

            # verify mj < J for first ground state
            if np.abs(groundState[0][3]+laserPolarization) - 0.1 > self.j:
                self.dmeGSReference = 0
            else:

                self.dmeGSReference = \
                    self.atom1.getDipoleMatrixElement(groundState[0][0],
                                                      groundState[0][1],
                                                      groundState[0][2],
                                                      groundState[0][3],
                                                      self.n, self.l, self.j,
                                                      groundState[0][3] +
                                                      laserPolarization,
                                                      laserPolarization)

            if debugOutput:
                print('Using an experimentally measured Rabi Frequency {measuredRabiFreq/1e6} MHz')
                print('DME from {groundState[0]} to the reference excited state ({self.n},{self.l},{self.j}):{self.dmeGSReference}')

            if len(groundState) > 1:
                for gs in groundState[1:]:
                    # verify mj < J for second ground state
                    if np.abs(gs[3]+laserPolarization) - 0.1 > self.j:
                        dmeGS = 0
                    else:
                        dmeGS = self.atom1.getDipoleMatrixElement(gs[0], gs[1],
                                gs[2], gs[3], self.n, self.l, self.j, gs[3] +
                                laserPolarization, laserPolarization)
                    if debugOutput:
                        print('DME from {gs} to the reference excited state ({self.n},{self.l},{self.j}):{self.dmeGSReference}')

                # uses the first ground state as the reference
                dmeRatio = dmeGS/self.dmeGSReference
                proportionalityConstant += groundStateCoeffs[1]**2*dmeRatio**2

            self.proportionalityConstant = np.sqrt(proportionalityConstant)
            if debugOutput:
                print(f'Sqrt of a sum of ground state coefficients multiplied with the DME ratio w.r.t. the first ground state: {self.proportionalityConstant}')

        for state in self.extendedBasisStates:
            state1 = (int(state[0]),int(state[1]),state[2],state[3],state[8])
            state2 = (int(state[4]), int(state[5]), state[6], state[7], state[9])

            self._addStateToRabiFreqDict(state1, debugOutput)
            self._addStateToRabiFreqDict(state2, debugOutput)

    def calculateDressedPotential(self, debugOutput=False):
        """Calculate the dressed energy of the ground state versus position.

        Args:
            debugOutput (bool): Optional, False by default. If True,
                print debug output.

        """
        # calculates Rabi frequency of coupling from |up up> to Rydberg pair eigenstate
        # this gives an array where each row corresponds to all eigenstates at a given distance
        self.pairRabiFreqs = np.zeros((len(self.r),self.eigVecs.shape[1]*self.mIMultiplicity), dtype=complex)

        for basisStateIndex, basisState in enumerate(self.extendedBasisStates):
            [n1, l1, j1, mj1, n2, l2, j2, mj2, mi1, mi2] = basisState
            n1 = int(n1)
            l1 = int(l1)
            n2 = int(n2)
            l2 = int(l2)
            # Joules
            energyState1 = (self.atom1.getEnergy(n1, l1, j1)*C_e +
                            self.atom1.getZeemanEnergyShift(l1, j1, mj1, self.Bz))
            energyState2 = (self.atom1.getEnergy(n2, l2, j2)*C_e +
                            self.atom1.getZeemanEnergyShift(l2, j2, mj2, self.Bz))

            # calculate overlap between the basisState and all eigenstates
            # for each distance, this overlap will have nEig*self.mIMultiplicity
            # numbers, due to a larger basis when considering mI states
            overlap = np.zeros((len(self.r),
                               self.eigVecs.shape[1]*self.mIMultiplicity),
                               dtype=complex)
            mi1Idx = self.mIlist.index(mi1)
            mi2Idx = self.mIlist.index(mi2)
            overlap[:, mi1Idx*len(self.mIlist)+mi2Idx::self.mIMultiplicity] = \
                            self.eigVecs[:, :,int(basisStateIndex/self.mIMultiplicity)]

            RabiFreq1 = self.rabiFreqDict[(n1,l1,j1,mj1,mi1)]
            RabiFreq2 = self.rabiFreqDict[(n2, l2, j2, mj2, mi2)]
            pairRF = overlap*RabiFreq1*RabiFreq2/2.*(1/(self.laserFreq+(self.groundStateEnergy-energyState1)/hbar)+ \
                                             1/(self.laserFreq + (self.groundStateEnergy - energyState2)/hbar))
            self.pairRabiFreqs += pairRF

            if debugOutput:
                print(basisState)
                print(f'Rabi frequency for coupling to state 1: {RabiFreq1/(2*np.pi*10**6):.3f} MHz')
                print(f'Rabi frequency for coupling to state 2: {RabiFreq2/(2*np.pi*10**6):.3f} MHz')
                print(f'Sum of the pair Rabi frequency over pair eigenstates: {np.sum(np.abs(pairRF),axis=1):.3f}')

        # need to make an extended eigVals list for different mI values,
        # but the values of eigenvalues are not different as they don't depend
        # on the nuclear spin quantum number
        self.eigValsExtended = np.repeat(self.eigVals,
                                         self.mIMultiplicity, axis=1)
        twoPhotonDetunings = (2*self.laserDetuning -
                              self.eigValsExtended*1e9)*2*np.pi  # rad/
        # finally calculate the dressed potential as a sum of
        # \Omega_pair^2/\Delta_2 photon
        V = np.abs(self.pairRabiFreqs)**2/twoPhotonDetunings
        self.dressedPotential = 0.25*np.sum(V, axis=1)  # rad/s

    def loadData(self, filename, directory):
        """Load data generated by DressedInteractions.exportData.

        Args:
            filename (string):  Filename to export the data to.
            directory (string): Directorz to export the data to.

        """

        self.eigVals = np.load(directory + r'eigVals_' + filename + '.npy')
        self.eigVecs = np.load(directory + r'eigVecs_' + filename + '.npy')
        self.basisStates = np.load(directory + r'basisStates_' + filename + '.npy')
        self.r = np.load(directory + r'distanceList_' + filename + '.npy')

        self.y = self.eigVals

        self.noOfEigenvectors = self.eigVecs.shape[1]
        self.dimension = self.eigVecs.shape[2]
        self.highlight = self.eigVecs[:, :, self.originalPairStateIndex]**2


    def exportData(self, filename, directory):
        """Export data to npy files along with a summary text file.

        Args:
            filename (string):  Filename to export the data to.
            directory (string): Directorz to export the data to.

        """

        np.save(directory + 'eigVals_' + filename + '.npy', self.y)
        np.save(directory + 'eigVecs_' + filename + '.npy', self.eigVecs)
        np.save(directory + 'basisStates_' + filename + '.npy', self.basisStates)
        np.save(directory + 'distanceList_' + filename + '.npy', self.r)

        file = open(directory + r'eigVecDataFile_' + filename + '.txt', 'w')
        file.write("n = " + str(self.n) + '\n')
        file.write("l = " + str(self.l) + '\n')
        file.write("j = " + str(self.j) + '\n')
        file.write("m1 = " + str(self.m1) + '\n')
        file.write("nn = " + str(self.nn) + '\n')
        file.write("ll = " + str(self.ll) + '\n')
        file.write("jj = " + str(self.jj) + '\n')
        file.write("m2 = " + str(self.m2) + '\n')
        file.write("theta = " + str(self.theta) + '\n')
        file.write("phi = " + str(self.phi) + '\n')
        file.write("dn = " + str(self.dn) + '\n')
        file.write("dl = " + str(self.dl) + '\n')
        file.write("deltaMax = " + str(self.deltaMax))
        file.close()

    def _addStateToRabiFreqDict(self, state, debugOutput=False):
        """Calculates Rabi frequency from the ground state to the excited state
        populates a dictionary of the Rabi frequency values.

        Args:
            state ([int,int,float,float,float]): State we couple to from
            self.groundState. Last member of state is the nuclear spin quantum
            number mI.
        """
        # checks if a state is already in the Rabi frequency dictionary,
        # and if not calculates Rabi frequency
        # and adds it to dictionary
        if not (state in self.rabiFreqDict):
            for idx, gs in enumerate(self.groundState):
                if (self.groundStateCoeffs[idx] > 0) \
                        and (state[3] == gs[3]+self.laserPolarization) \
                        and (state[4] == gs[4]):
                    # calculate Rabi frequency coupling from the ground state
                    # gs to state
                    if self.measuredRabiFreq:
                        try:
                            rf = (self.measuredRabiFreq*2*np.pi)/self.proportionalityConstant*\
                                    self.groundStateCoeffs[idx]*(self.atom1.getDipoleMatrixElement(gs[0], gs[1],\
                                    gs[2], gs[3], state[0], state[1], state[2], state[3],\
                                    self.laserPolarization))/self.dmeGSReference
                        except ValueError as e:#catch 6j symbol errors
                            if debugOutput:
                                print(f'For states {gs=} and {state=}')
                                print(e)
                            rf = 0
                    else:
                        # if not using a measured Rabi frequency, get the coupling from DMEs
                        rf = self.groundStateCoeffs[idx]*self.atom1.getRabiFrequency(gs[0], gs[1],\
                            gs[2], gs[3], state[0], state[1], state[2], self.laserPolarization, self.laserPower,\
                            self.laserWaist)
                    break
                else:
                    rf = 0
            self.rabiFreqDict[state] = rf

    def plotLevelDiagram(self,  highlightColor='red',
                         highlightScale='linear'):
        """Plots pair state level diagram. In case the dressed potential has
           been calculated, highlits the states based on the pair Rabi frequency coupling from the ground state
        to the pair state eigenvalue.
        Args:
            highlightColor (string): specifies the colour used
                for state highlighting.
            highlightScale (string): optional, specifies scaling of
                state highlighting. Default is 'linear'. Use 'log-2' or
                'log-3' for logarithmic scale going down to 1e-2 and 1e-3
                respectively. Logarithmic scale is useful for spotting
                weakly admixed states.
        """

        self.y = self.eigVals
        self.highlight = self.originalOverlap

        super().plotLevelDiagram(highlightColor=highlightColor,
                                 highlightScale=highlightScale)

    def plotLevelDiagramDressed(self, highlightColor='red',
                                highlightScale='linear'):
        """Similar to plotLevelDiagram, plots the level diagram but
            here the shading of the plot is based on dressing pair Rabi
            frequencies.
        Args:
            highlightColor (string): specifies the colour used
                for state highlighting.
            highlightScale (string): optional, specifies scaling of
                state highlighting. Default is 'linear'. Use 'log-2' or
                'log-3' for logarithmic scale going down to 1e-2 and 1e-3
                respectively. Logarithmic scale is useful for spotting
                weakly admixed states.
        """
        # check if dressing calculation has been done
        if not hasattr(self, 'eigValsExtended'):
            raise Exception('calculateDressedPotential has not been run.')

        self.y = self.eigValsExtended
        self.highlight = np.abs(self.pairRabiFreqs/(2*np.pi*1e3))  # in kHz
        super().plotLevelDiagram(highlightColor=highlightColor,
                                 highlightScale=highlightScale)
        self.cb.set_label(r"$\Omega_\psi / 2 \pi$ (kHz)")

    def showPlot(self):
        '''
        Shows level diagram printed by
        :obj:`PairStateInteractions.plotLevelDiagram`

        Note that this effectively disables the interactive plot available in
        calculations_atom_pairstate.
        '''
        plt.show()

    def plotDressedPotential(self, includePertPrediction=False, color=None):
        """Plot the dressed potential versus distance.

        Call :obj:`showPlot` if you want to display a plot afterwards.

        Args:
            includePertPrediction (боол): optional, False by default.
            Calculate and plot the perturbative prediction for the dressed
            potential. color (string): Color of the dressing potential line.
        """

        self.fig_d, self.ax_d = plt.subplots()

        if includePertPrediction:
            if self.measuredRabiFreq:
                rf = self.measuredRabiFreq
            else:
                rf = 0
                for idx, gs in enumerate(self.groundState):
                    rf += self.groundStateCoeffs[idx]*self.atom1.getRabiFrequency(gs[0],gs[1],\
                        gs[2], gs[3], self.n, self.l, self.j,
                        self.laserPolarization,self.laserPower,
                        self.laserWaist)

            # figure out what is the highest mj of the states we couple to
            # to use in calculating the Zeeman shift

            # this is asumming the laser detuning is either higher than the highest
            # coupled zeeman sublevel or lower than the lowest coupled zeeman sublevel
            mjCoupledMax = 0
            for idx, gs in enumerate(self.groundState):
                mjCoupled = gs[3]+self.laserPolarization
                if abs(mjCoupled) <= self.j and abs(mjCoupled) > abs(mjCoupledMax):
                        mjCoupledMax = mjCoupled

            # detuning from the nearest coupled state
            laserDetuningNearestState = self.laserDetuning - self.atom1.getZeemanEnergyShift(self.l,self.j,mjCoupledMax,self.Bz)/(2*np.pi*hbar)

            # simple perturbation theory prediction - \Omega^4/(8 \Delta^3)
            pertPrediction = rf**4/(8*laserDetuningNearestState**3)/1e3
            self.ax_d.axhline(pertPrediction, 0, 10, color='gray',
                              linewidth=2.5, zorder=1)

        self.ax_d.scatter(self.r, self.dressedPotential/(2*np.pi*1e3),
                          color=color)

        self.ax_d.grid()
        self.ax_d.set_xlabel('Interatomic distance ($\mu$m)', fontsize=20)
        self.ax_d.set_ylabel('Pair energy (kHz)', fontsize=20)
        self.ax_d.set_xlim(left=0, right=None)

    def savePlotDressedPotential(self, filename):
        """Save the dressed potential plot.

        Args:
            filename (string): Filename to save the plot to.


        """
        if (self.fig_d != 0):
            self.fig_d.savefig(filename, bbox_inches='tight', dpi=200)
        else:
            print("Error while saving a plot: dressing potential is plotted yet")
        return 0
