import numpy as np
import matplotlib.pyplot as plt
from Device import Device


class CylindricalGeometry(Device):
    """
    This class is used to create a cylindrical geometry.
    parameters:
    detector_module: DetectorModule object
    radius: radius of the cylinder
    height: height of the cylinder
    fill_circunference: if True, the circunference of the cylinder will be filled with modules
    """

    def __init__(self, detector_module=None, radius=60, fill_circle=True):
        super().__init__()
        if detector_module is None:
            raise ValueError("Detector module cannot be None. Please provide a DetectorModule object. Should be of type" )

        self._detectorModuleObject = detector_module
        self._radius = radius
        self._numberOfModulesZ = 4
        self._numberOfModulesPhi = 12
        self._startPhi = 0
        self._endPhi = 360
        self._anglePhi = (self._endPhi - self._startPhi) / self._numberOfModulesPhi
        self._fillCircle = fill_circle
        self._axialGapBetweenModules = 3
        self._axialDistanceBetweenModules = self._axialGapBetweenModules+25.8 # width of the module
        self._radialGapBetweenModules = 0
        self._zTranslation = 0
        self._structureType = "static"
        self.setGeometryType("cylindrical")
        self._numberOfModules = self._numberOfModulesZ * self._numberOfModulesPhi
        self._detectorModule = [self._detectorModuleObject(i) for i in range(self._numberOfModules)]
        if fill_circle:
            # self._radius = 12.8 + 2*12.8* (self._numberOfModulesPhi/4-1)*np.cos(np.deg2rad((self._numberOfModulesPhi/4 - 1)*(90-self._anglePhi)))+10

            self._radius = 12.8 + 10 # half module width plus half crystal width
            for inter_module in range(int(self._numberOfModulesPhi/4-1)):
                self._radius += 2*12.8*np.cos(np.deg2rad((90-(inter_module+1)*self._anglePhi)))
        print("Radius: {}".format(self._radius))
        # self._detectorModule = [[self._detectorModuleObject(i+2*j) for i in range(self._numberOfModulesPhi)] for j in range(self._numberOfModulesZ)]

    @property
    def detectorModule(self):
        return self._detectorModule

    def setDetectorModule(self, detector_module):
        self._detectorModule = detector_module

    @property
    def radius(self):
        return self._radius

    @property
    def numberOfModulesZ(self):
        return self._numberOfModulesZ

    def setNumberOfModulesZ(self, number_of_modules_z):
        self._numberOfModulesZ = number_of_modules_z
        self._numberOfModules = self._numberOfModulesZ * self._numberOfModulesPhi

    def calculateInitialGeometry(self):
        for i in range(self._numberOfModulesPhi):
            for j in range(self._numberOfModulesZ):
                self._detectorModule[i+self._numberOfModulesPhi*j].setXTranslation(self.radius * np.cos(np.deg2rad(self._anglePhi * i)))
                self._detectorModule[i+self._numberOfModulesPhi*j].setYTranslation(self.radius * np.sin(np.deg2rad(self._anglePhi * i)))
                self._detectorModule[i+self._numberOfModulesPhi*j].setZTranslation(j*self._axialDistanceBetweenModules)
                self._detectorModule[i+self._numberOfModulesPhi*j].setAlphaRotation(0)
                self._detectorModule[i+self._numberOfModulesPhi*j].setBetaRotation(0)
                # self._detectorModule[j][i].setSigmaRotation(90 + 45 * i)
                self._detectorModule[i+self._numberOfModulesPhi*j].setSigmaRotation(self._anglePhi * i)
                self._detectorModule[i+self._numberOfModulesPhi*j].setInitialGeometry()
                # petModule.setYTranslation(30)




            # petModule.setYTranslation(30)
            # petModule.setbe(45*i)

            # centers = [i.centroid for i in petModule.modelHighEnergyLightDetectors]
    def getRadius(self):
        return self.radius

    def getHeight(self):
        return self.height

    def setRadius(self, radius):
        self._radius = radius

    def setHeight(self, height):
        self.height = height

    # def __str__(self):
    #     return "Cylinder with radius {} and height {}".format(self.radius, self.height)

    # def __repr__(self):
    #     return self.__str__()
    #
    # def __eq__(self, other):
    #     if isinstance(other, CylindricalGeometry):
    #         return self.radius == other.radius and self.height == other.height
    #     else:
    #         return False
    #
    #
    # def __ne__(self, other):
    #     return not self.__eq__(other)
    #
    # def __hash__(self):
    #     return hash((self.radius, self.height))


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
    newDevice = CylindricalGeometry(detector_module=module_)
    newDevice.setDeviceName("Test Device")
    newDevice.calculateInitialGeometry()

    designer = DeviceDesignerStandalone(device=newDevice)
    designer.addDevice()
    designer.startRender()

    print(newDevice.getDeviceName())
