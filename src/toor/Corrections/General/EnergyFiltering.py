import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class EnergyFiltering:
    def __init__(self):
        # import excel datas to dataframe
        path_file = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._radioisotopeData = pd.read_excel(os.path.join(path_file, "configurations",
                                                            "RadioIsotopes_decayscheme.xlsx"))
        self._filteredListMode = None
        self._currentRadioisotopeFather = None
        self._currentRadioisotopeSon = None
        self._currentRadioisotopeState = None
        self._currentRadioIsotopeHalfLife = None
        self._currentNumberOfBranches = None
        self._currentBranchID = None
        self._currentRatioBranch = None
        self._currentNumberOfGammas = None
        self._currentPeakID = None
        self._currentPeakName = None
        self._currentPeakEnergy = None
        self._currentLowCutEnergy = None
        self._currentHighCutEnergy = None
        self._currentProcess = None
        self._totalNumberOfGammas = None
        self._allDecayGammas = None

    @property
    def totalNumberOfGammas(self):
        return self._totalNumberOfGammas

    @property
    def allDecayGammas(self):
        return self._allDecayGammas

    def detectEnergyWindows(self, radioisotopeParent="Ac-225"):
        """
        Detect the energy windows for each gamma peak of the radioisotope
        If the radioisotope son is not stable goes to the rows of the son and join the energy windows of that gammas
        :param radioisotopeParent:
        :return:

        """
        # get the radioisotope parent

        radioisotopeParent = self._radioisotopeData[self._radioisotopeData['Radioisotope_Father'] == radioisotopeParent]
        # Isolate the gammas
        mask = (radioisotopeParent['DecayProcess'] == "Gamma") | (radioisotopeParent['DecayProcess'] == "X-ray")
        mask = mask & (radioisotopeParent.SetForReconstruction == True)

        allDecayGammas = radioisotopeParent[mask]
        while radioisotopeParent['State'].values[0] != "Stable":
            radioisotopeSon = radioisotopeParent['Decays_to'].values
            # check if there is more than one different son
            uniqueSon = np.unique(radioisotopeSon)
            for radioisotopeSon in uniqueSon:
                # get the radioisotope son

                radioisotopeSon = self._radioisotopeData[self._radioisotopeData['Radioisotope_Father'] == radioisotopeSon]
                #Isoalte the gammas
                mask_son = (radioisotopeSon['DecayProcess'] == "Gamma") | (radioisotopeSon['DecayProcess'] == "X-ray")
                mask_son = mask_son & (radioisotopeSon.SetForReconstruction == True)
                decayGammasSon = radioisotopeSon[mask_son]
                #Join the gammas
                allDecayGammas = pd.concat([allDecayGammas, decayGammasSon])
                radioisotopeParent = radioisotopeSon
        self._allDecayGammas = allDecayGammas
        self._totalNumberOfGammas = len(allDecayGammas)
        print("Number of Gammas: ", len(allDecayGammas))
        pd.set_option('display.max_columns', None)
        print(allDecayGammas)

    def setCurrentGammaInformation(self, id=0):
        """
        The id is the index of the gamma in the dataframe
        :param id:
        :return:
        """
        self._currentRadioisotopeFather = self._allDecayGammas['Radioisotope_Father'].values[id]
        self._currentRadioisotopeSon = self._allDecayGammas['Decays_to'].values[id]
        self._currentRadioisotopeState = self._allDecayGammas['State'].values[id]
        self._currentRadioIsotopeHalfLife = self._allDecayGammas['Half-life_s'].values[id]
        self._currentNumberOfBranches = self._allDecayGammas['Number_of_branchs'].values[id]
        self._currentBranchID = self._allDecayGammas['Branch_id'].values[id]
        self._currentRatioBranch = self._allDecayGammas['Ratio_branch'].values[id]
        self._currentNumberOfGammas = self._allDecayGammas['Number_of_gammas'].values[id]
        self._currentPeakID = self._allDecayGammas['Peak_id'].values[id]
        self._currentPeakName = self._allDecayGammas['Peak_name'].values[id]
        self._currentPeakEnergy = self._allDecayGammas['Energy_peak'].values[id]
        self._currentLowCutEnergy = self._allDecayGammas['Low_cut'].values[id]
        self._currentHighCutEnergy = self._allDecayGammas['High_cut'].values[id]
        self._currentProcess = self._allDecayGammas['DecayProcess'].values[id]

    def overrideEnergyCut(self, lowCutEnergy=0, highCutEnergy=400, systemEnergyResolution=None):
        if systemEnergyResolution is not None:
            self._currentLowCutEnergy = self._currentPeakEnergy - systemEnergyResolution/100
            self._currentHighCutEnergy = self._currentPeakEnergy + systemEnergyResolution/100
        else:
            self._currentLowCutEnergy = lowCutEnergy
            self._currentHighCutEnergy = highCutEnergy

    def setEnergyCutToListMode(self, listMode, save_plot=True, study_folder=None):
        """
        Set the energy cut to the list mode
        :param listMode:
        :param save_plot:
        :return:
        """
        if save_plot:
            fig_energy_cut = plt.figure()
            plt.hist(listMode['energy'], bins=400, range=(0, 400))
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts")
            plt.axvline(x=self._currentLowCutEnergy, color='r', linestyle='--')
            plt.axvline(x=self._currentHighCutEnergy, color='r', linestyle='--')

        listMode = listMode[(listMode['energy'] >= self._currentLowCutEnergy) &
                            (listMode['energy'] < self._currentHighCutEnergy)]
        listMode = listMode.reset_index(drop=True)

        if save_plot:
            plt.hist(listMode['energy'], bins=400, range=(0, 400))
            fig_energy_cut.savefig(os.path.join(study_folder, "energy_cut_{}.png".format(self._currentPeakName)))
        return listMode

    def printCurrentInfoCut(self):
        print("---------Gamma Information---------")
        print("Current Radioisotope: ", self._currentRadioisotopeFather)
        print("Current Radioisotope Son: ", self._currentRadioisotopeSon)
        print("Current Radioisotope State: ", self._currentRadioisotopeState)
        print("Current Radioisotope Half Life: ", self._currentRadioIsotopeHalfLife)
        print("Current Number of Branches: ", self._currentNumberOfBranches)
        print("Current Branch ID: ", self._currentBranchID)
        print("Current Ratio Branch: ", self._currentRatioBranch)
        print("Current Number of Gammas: ", self._currentNumberOfGammas)
        print("Current Peak ID: ", self._currentPeakID)
        print("Current Peak Name: ", self._currentPeakName)
        print("Current Peak Energy: ", self._currentPeakEnergy)
        print("Current Low Cut Energy: ", self._currentLowCutEnergy)
        print("Current High Cut Energy: ", self._currentHighCutEnergy)
        print("Current Process: ", self._currentProcess)
        print("-----------------------------------")

    @property
    def currentRadioisotopeFather(self):
        return self._currentRadioisotopeFather

    @property
    def currentRadioisotopeSon(self):
        return self._currentRadioisotopeSon

    @property
    def currentRadioisotopeState(self):
        return self._currentRadioisotopeState

    @property
    def currentRadioIsotopeHalfLife(self):
        return self._currentRadioIsotopeHalfLife

    @property
    def currentNumberOfBranches(self):
        return self._currentNumberOfBranches

    @property
    def currentBranchID(self):
        return self._currentBranchID

    @property
    def currentRatioBranch(self):
        return self._currentRatioBranch

    @property
    def currentNumberOfGammas(self):
        return self._currentNumberOfGammas

    @property
    def currentPeakID(self):
        return self._currentPeakID

    @property
    def currentPeakName(self):
        return self._currentPeakName

    @property
    def currentPeakEnergy(self):
        return self._currentPeakEnergy

    @property
    def currentLowCutEnergy(self):
        return self._currentLowCutEnergy

    @property
    def currentHighCutEnergy(self):
        return self._currentHighCutEnergy

    @property
    def currentProcess(self):
        return self._currentProcess


if __name__ == "__main__":
    energyFiltering = EnergyFiltering()
    energyFiltering.detectEnergyWindows()
    for i in range(0, len(energyFiltering._allDecayGammas)):
        energyFiltering.setCurrentGammaInformation(i)
        energyFiltering.printCurrentInfoCut()