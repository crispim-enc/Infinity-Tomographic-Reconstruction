import numpy as np
import matplotlib.pyplot as plt
from Device import Device


class RegularPolygonalGeometry(Device):
    """
    This class is used to create a cylindrical geometry.
    parameters:
    detector_module: DetectorModule object
    radius: radius of the cylinder
    height: height of the cylinder
    fill_circunference: if True, the circunference of the cylinder will be filled with modules
    """

    def __init__(self, detector_module=None, radius=40, fill_circle=False):
        super().__init__()
        if detector_module is None:
            Warning(
                "Detector module cannot be None. Remember to set DetectorModule object. ")

        self._detectorModuleObject = detector_module
        self._radius = radius
        self._numberOfModulesZ = 1
        self._numberOfModulesPerSide = 1
        self._numberOfModulesPhi = 2
        self._startPhi = 0
        self._endPhi = 360
        self._anglePhi = (self._endPhi - self._startPhi) / self._numberOfModulesPhi
        self._fillCircle = fill_circle
        self._axialGapBetweenModules = 3
        self._radialGapBetweenModules = 0
        self._zTranslation = 0
        self._structureType = "static"
        self._numberOfModules = self._numberOfModulesZ * self._numberOfModulesPhi * self._numberOfModulesPerSide

        # self._detectorModule = [self._detectorModuleObject(i) for i in range(self._numberOfModules)]
        self._detectorModule = None
        if self._detectorModuleObject is not None:
            if fill_circle:
                # self._radius = 12.8 + 2*12.8* (self._numberOfModulesPhi/4-1)*np.cos(np.deg2rad((self._numberOfModulesPhi/4 - 1)*(90-self._anglePhi)))+10

                self._radius = 12.8*self._numberOfModulesPerSide +10 # half module width plus half crystal width
                if self._numberOfModulesPhi % 4 == 0:
                    for inter_module in range(int(self._numberOfModulesPhi / 4 - 1)):
                        self._radius += 2 * 12.8*self._numberOfModulesPerSide * np.cos(np.deg2rad((90 - (inter_module + 1) * self._anglePhi)))
                elif self._numberOfModulesPhi % 4 == 2:
                    self._radius = 10
                    for inter_module in range(int(np.ceil(self._numberOfModulesPhi / 4)- 1)):
                        self._radius += 2 * 12.8*self._numberOfModulesPerSide* np.cos(np.deg2rad((90 - (inter_module + 1) * self._anglePhi)))

            print("radius: ", self._radius)
        #     for inter_module in range(int(np.floor(self._numberOfModulesPhi / 4)- 1)):
        #         self._radius += 2 * 12.8*self._numberOfModulesPerSide* np.cos(np.deg2rad((90 - (inter_module + 1) * self._anglePhi)))
        # # self._detectorModule = [[self._detectorModuleObject(i+2*j) for i in range(self._numberOfModulesPhi)] for j in range(self._numberOfModulesZ)]

    @property
    def detectorModule(self):
        return self._detectorModule

    def setDetectorModule(self, detector_module):
        self._detectorModule = detector_module

    @property
    def radius(self):
        return self._radius

    def setRadius(self, radius):
        self._radius = radius

    @property
    def numberOfModulesZ(self):
        return self._numberOfModulesZ

    @property
    def numberOfModulesPhi(self):
        return self._numberOfModulesPhi

    @property
    def numberOfModulesPerSide(self):
        return self._numberOfModulesPerSide

    @property
    def numberOfModules(self):
        return self._numberOfModules

    def setNumberOfModulesZ(self, number_of_modules_z):
        self._numberOfModulesZ = number_of_modules_z
        self._numberOfModules = self._numberOfModulesZ * self._numberOfModulesPhi*self._numberOfModulesPerSide
        # self._detectorModule = [self._detectorModuleObject for i in range(self._numberOfModules)]

    def setNumberOfModulesPhi(self, number_of_modules_phi):
        self._numberOfModulesPhi = number_of_modules_phi
        self._numberOfModules = self._numberOfModulesZ * self._numberOfModulesPhi * self._numberOfModulesPerSide
        # self._detectorModule = [self._detectorModuleObject for i in range(self._numberOfModules)]

    def setNumberOfModulesPerSide(self, number_of_modules_per_side):
        self._numberOfModulesPerSide = number_of_modules_per_side
        self._numberOfModules = self._numberOfModulesZ * self._numberOfModulesPhi * self._numberOfModulesPerSide
        # self._detectorModule = [self._detectorModuleObject for i in range(self._numberOfModules)]

    def setAnglePhi(self, angle_phi):
        self._anglePhi = angle_phi

    def calculateInitialGeometry(self):
        for i in range(self._numberOfModulesPhi):
            for j in range(self._numberOfModulesZ):
                for k in range(self._numberOfModulesPerSide):
                    # self._detectorModule[
                    #     i + self._numberOfModulesPhi * j + self._numberOfModulesZ * self._numberOfModulesPhi * k].setXTranslation(self.radius*k-(self._numberOfModulesPerSide-1)*self.radius)
                    self._detectorModule[
                        i + self._numberOfModulesPhi * j + self._numberOfModulesZ * self._numberOfModulesPhi * k].setInitialGeometry()
                    # self._detectorModule[
                    #     i + self._numberOfModulesPhi * j + self._numberOfModulesZ * self._numberOfModulesPhi * k].setXTranslation(25.8*k* np.sin(np.deg2rad(self._anglePhi * i)))

                    self._detectorModule[i + self._numberOfModulesPhi * j+self._numberOfModulesZ*self._numberOfModulesPhi*k].setXTranslation(
                        self.radius * np.cos(np.deg2rad(self._anglePhi * i)))
                    self._detectorModule[i + self._numberOfModulesPhi * j+self._numberOfModulesZ*self._numberOfModulesPhi*k].setYTranslation(
                        self.radius * np.sin(np.deg2rad(self._anglePhi * i)))
                    self._detectorModule[i + self._numberOfModulesPhi * j+self._numberOfModulesZ*self._numberOfModulesPhi*k].setZTranslation(
                        j * 30)
                    self._detectorModule[i + self._numberOfModulesPhi * j+self._numberOfModulesZ*self._numberOfModulesPhi*k].setAlphaRotation(0)
                    self._detectorModule[i + self._numberOfModulesPhi * j+self._numberOfModulesZ*self._numberOfModulesPhi*k].setBetaRotation(0)
                    # self._detectorModule[j][i].setSigmaRotation(90 + 45 * i)
                    self._detectorModule[i + self._numberOfModulesPhi * j+self._numberOfModulesZ*self._numberOfModulesPhi*k].setSigmaRotation(90 + self._anglePhi * i)
                    self._detectorModule[i + self._numberOfModulesPhi * j+self._numberOfModulesZ*self._numberOfModulesPhi*k].setInitialGeometry()
                    # petModule.setYTranslation(30)

                #     self._detectorModule[i + self._numberOfModulesPhi * j].setYTranslation(
                #     self.radius * np.sin(np.deg2rad(self._anglePhi * i)))
                # self._detectorModule[i + self._numberOfModulesPhi * j].setZTranslation(j * 30)
                # self._detectorModule[i + self._numberOfModulesPhi * j].setAlphaRotation(0)
                # self._detectorModule[i + self._numberOfModulesPhi * j].setBetaRotation(0)a
                # # self._detectorModule[j][i].setSigmaRotation(90 + 45 * i)
                # self._detectorModule[i + self._numberOfModulesPhi * j].setSigmaRotation(90 + self._anglePhi * i)
                # self._detectorModule[i + self._numberOfModulesPhi * j].setInitialGeometry()
                # petModule.setYTranslation(30)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from src.DetectionLayout.Modules import PETModule
    from src.Designer import DeviceDesignerStandalone

    # newDevice = Device()
    # newDevice.setDeviceName("Test Device")
    # newDevice.setDeviceType("Test Device Type")
    # newDevice.setDeviceUUID("Test Device UUID")
    # newDevice.setDeviceStatus("Test Device Status")
    # newDevice.setDeviceDirectory("Test Device Directory")
    # print(newDevice.getDeviceName())
    # print(newDevice.getDeviceType())
    module_ = PETModule
    #
    newDevice = RegularPolygonalGeometry(detector_module=module_)
    newDevice.setDeviceName("Test Device")
    newDevice.calculateInitialGeometry()

    designer = DeviceDesignerStandalone(device=newDevice)
    designer.addDevice()
    designer.startRender()

    print(newDevice.getDeviceName())
