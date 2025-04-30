from .polygonal import RegularPolygonalGeometry


class PlanarGeometry(RegularPolygonalGeometry):
    def __init__(self, detector_module=None, distance_between_planes=10):
        super(PlanarGeometry, self).__init__(detector_module=detector_module)
        self._radius = distance_between_planes/2
        self._numberOfModulesPhi = 2
        self._geometryType = "planar"



if __name__ == "__main__":
    from toor.DetectionLayout.Modules import PETModule
    from toor.Designer import DeviceDesignerStandalone


    module_ = PETModule
    #
    newDevice = PlanarGeometry(detector_module=module_, distance_between_planes=40)
    newDevice.setDeviceName("Test Device")
    newDevice.setRadius(40)
    newDevice.setNumberOfModulesZ(4)
    newDevice.setNumberOfModulesPerSide(4)
    newDevice.setNumberOfModulesPhi(2)
    newDevice.calculateInitialGeometry()

    designer = DeviceDesignerStandalone(device=newDevice)
    designer.addDevice()
    designer.startRender()

    print(newDevice.getDeviceName())


