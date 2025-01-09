from src.Detectors.SiPM import GenericSiPM


class HamamatsuS14161Series(GenericSiPM):
    """
    Hamamatsu S14161 Series
    model: 3050HS-04 or 3050HS-08

    """
    def __init__(self, model="3050HS-08"):  # 3050HS-04 or 3050HS-08

        super(HamamatsuS14161Series, self).__init__()
        # self.applyModelProperties(model)
        self.setSeries("S14161 Series")
        self.setModel(model)
        self.setVendor("Hamamatsu")
        self.setPixelPitch(50)
        self.setPixelWidth(50)
        self.setPixelWidthTolerance(0.5)
        self.setPixelHeight(50)
        self.setPixelHeightTolerance(0.5)
        self.setPixelDepth(0.0)
        self.setPixelArea(2500)
        self.setPixelSpacingX(50)
        self.setPixelSpacingY(50)
        self.setBorderSizeX(0.2)
        self.setBorderSizeY(0.2)
        self.setResinThickness(0)
        self.setGeometricalFillFactor(0.74)
        self.setPhotonDetectionEfficiencyAtPeak(0.5)

        self.setNumberOfChannelsX(8)
        self.setNumberOfChannelsY(8)
        # self.setTotalNumberOfChannels(4)
        self.setEffectiveWidth(3)
        self.setEffectiveHeight(3)
        self.setEffectiveAreaPerChannel(9)
        self.setPackageType("Surface Mount")
        self.setWindowType("Silicone")
        self.setWindowRefractiveIndex(1.57)
        self.setBlockSPiMWidth(25.8)
        self.setBlockSPiMHeight(25.8)
        self.setBlockSPiMDepth(1.35)
        # self.setBlockSPiMArea(665.64)
        self.setExternalBorderSizeX(0.2)
        self.setExternalBorderSizeY(0.2)
        self.setChannelOriginalCentrePosition()

    def applyModelProperties(self, model):
        """
        Apply model properties
        :param model: model name
        :return:

        """
        if model == '3050HS-04':
            self.setSeries("S14161 Series")
            self.setModel(model)
            self.setVendor("Hamamatsu")
            self.setPixelPitch(50)
            self.setPixelWidth(50)
            self.setPixelWidthTolerance(0.5)
            self.setPixelHeight(50)
            self.setPixelHeightTolerance(0.5)
            self.setPixelDepth(0.0)
            self.setPixelArea(2500)
            self.setPixelSpacingX(50)
            self.setPixelSpacingY(50)
            self.setBorderSizeX(0)
            self.setBorderSizeY(0)
            self.setResinThickness(0)
            self.setGeometricalFillFactor(0.74)
            self.setPhotonDetectionEfficiencyAtPeak(0.5)

            self.setNumberOfChannelsX(4)
            self.setNumberOfChannelsY(4)
            # self.setTotalNumberOfChannels(1)
            self.setEffectiveWidth(3)
            self.setEffectiveHeight(3)
            self.setEffectiveAreaPerChannel(9)
            self.setPackageType("Surface Mount")
            self.setWindowType("Silicone")
            self.setWindowRefractiveIndex(1.57)
            # self.setWindowRefractiveIndexTolerance(0.01)

        elif model == '3050HS-08':
            self.setSeries("S14161 Series")
            self.setModel(model)
            self.setVendor("Hamamatsu")
            self.setPixelPitch(50)
            self.setPixelWidth(50)
            self.setPixelWidthTolerance(0.5)
            self.setPixelHeight(50)
            self.setPixelHeightTolerance(0.5)
            self.setPixelDepth(0.0)
            self.setPixelArea(2500)
            self.setPixelSpacingX(50)
            self.setPixelSpacingY(50)
            self.setBorderSizeX(0.2)
            self.setBorderSizeY(0.2)
            self.setResinThickness(0)
            self.setGeometricalFillFactor(0.74)
            self.setPhotonDetectionEfficiencyAtPeak(0.5)

            self.setNumberOfChannelsX(8)
            self.setNumberOfChannelsY(8)
            # self.setTotalNumberOfChannels(4)
            self.setEffectiveWidth(3)
            self.setEffectiveHeight(3)
            self.setEffectiveAreaPerChannel(9)
            self.setPackageType("Surface Mount")
            self.setWindowType("Silicone")
            self.setWindowRefractiveIndex(1.57)
            self.setBlockSPiMWidth(25.8)
            self.setBlockSPiMHeight(25.8)
            self.setBlockSPiMDepth(1.35)
            # self.setBlockSPiMArea(665.64)
            self.setExternalBorderSizeX(0.2)
            self.setExternalBorderSizeY(0.2)

            # self.setWindowRefractiveIndexTolerance(0.01)




if __name__ == "__main__":

    a = HamamatsuS14161Series(GenericSiPM)
    print(vars(a))



