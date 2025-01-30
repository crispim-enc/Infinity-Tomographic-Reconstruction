import numpy as np


class GenericSiPM:
    """
    Generic SiPM class
    parameters:
    """

    def __init__(self, idSiPM=0 ) -> object:
        # super().__init__(**kwargs)
        self.idSiPM = idSiPM
        self._series = None
        self._model = None
        self._vendor = None
        self._numberOfChannelsX = 1
        self._numberOfChannelsY = 1
        self._totalNumberOfChannels = self._numberOfChannelsX * self._numberOfChannelsY
        self._pixelPitch = 50
        self._numberPixelPerChannel = 3531
        self._packageType = "Surface Mount"
        self._windowType = "Silicone"
        self._windowRefractiveIndex = 1.57
        self._geometricalFillFactor = 0.74
        self._photonDetectionEfficiencyAtPeak = 0.5
        self._pixelWidth = 0.0
        self._pixelWidthTolerance = 0.0
        self._pixelHeight = 0.0
        self._pixelHeightTolerance = 0.0
        self._pixelDepth = 0.0
        self._resinThickness = 0.15
        self._pixelArea = self._pixelWidth * self._pixelHeight
        self._pixelSpacingX = 0.0
        self._pixelSpacingY = 0.0
        self._borderSizeX = 0.0
        self._borderSizeY = 0.0
        self._effectiveWidth = 3
        self._effectiveHeight = 3
        self._effectiveAreaPerChannel = self._effectiveWidth * self._effectiveHeight # mm^2
        self._blockSPiMWidth = 24
        self._blockSPiMHeight = 24
        self._blockSPiMDepth = 1.35
        self._blockSPiMArea = self._blockSPiMWidth * self._blockSPiMHeight
        self._externalBorderSizeX = 0.0
        self._externalBorderSizeY = 0.0
        self._centerSiPMModule = np.array([0, 0, 0])
        self._channelCentrePosition = None
        self._spacingToHighEnergyDetector = 0.0
        self._alphaRotation = 0.0
        self._betaRotation = 0.0
        self._sigmaRotation = 0.0
        self._xTranslation = 0.0
        self._yTranslation = 0.0
        self._zTranslation = 0.0

        self.setChannelOriginalCentrePosition()

    @property
    def centerSiPMModule(self):
        return self._centerSiPMModule

    def setCenterSiPMModule(self, center):
        self._centerSiPMModule = center
        # self.setChannelOriginalCentrePosition()

    @property
    def channelCentrePosition(self):
        return self._channelCentrePosition

    def setChannelCentrePosition(self, center):
        self._channelCentrePosition = center

    def setChannelOriginalCentrePosition(self):
        x_step = self._effectiveWidth + self.borderSizeX
        x_range = np.arange(0, self._numberOfChannelsX * x_step, x_step) - (
                self._numberOfChannelsX - 1) * x_step / 2
        z_step = self._effectiveHeight + self.borderSizeY
        z_range = np.arange(0, self._numberOfChannelsY * z_step, z_step) - (
                self._numberOfChannelsY - 1) * z_step / 2

        xx, zz = np.meshgrid(x_range, z_range)

        x_flat = xx.flatten() + self._centerSiPMModule[0]
        y_flat = -np.ones(self._numberOfChannelsX * self._numberOfChannelsX)*self.blockSPiMDepth/2 + self._centerSiPMModule[1]
        z_flat = zz.flatten() + self._centerSiPMModule[2]
        self._channelCentrePosition = np.array([x_flat, y_flat, z_flat]).T

    @property
    def series(self):
        return self._series

    def setSeries(self, series: object) -> object:
        self._series = series

    @property
    def model(self):
        return self._model

    def setModel(self, model):
        self._model = model

    @property
    def vendor(self):
        return self._vendor

    def setVendor(self, vendor):
        self._vendor = vendor

    # generate the functions to get and set the missing private properties
    # for the rest of the properties
    # ...
    @property
    def pixelPitch(self):
        return self._pixelPitch

    def setPixelPitch(self, pixelPitch):
        self._pixelPitch = pixelPitch

    @property
    def numberPixelPerChannel(self):
        return self._numberPixelPerChannel

    def setNumberPixelPerChannel(self, numberPixelPerChannel):
        self._numberPixelPerChannel = numberPixelPerChannel

    @property
    def packageType(self):
        return self._packageType

    def setPackageType(self, packageType):
        self._packageType = packageType

    @property
    def windowType(self):
        return self._windowType

    def setWindowType(self, windowType):
        self._windowType = windowType

    @property
    def windowRefractiveIndex(self):
        return self._windowRefractiveIndex

    def setWindowRefractiveIndex(self, windowRefractiveIndex):
        self._windowRefractiveIndex = windowRefractiveIndex

    @property
    def geometricalFillFactor(self):
        return self._geometricalFillFactor

    def setGeometricalFillFactor(self, geometricalFillFactor):
        self._geometricalFillFactor = geometricalFillFactor

    @property
    def photonDetectionEfficiencyAtPeak(self):
        return self._photonDetectionEfficiencyAtPeak

    def setPhotonDetectionEfficiencyAtPeak(self, photonDetectionEfficiencyAtPeak):
        self._photonDetectionEfficiencyAtPeak = photonDetectionEfficiencyAtPeak

    @property
    def pixelWidth(self):
        return self._pixelWidth

    def setPixelWidth(self, pixelWidth):
        self._pixelWidth = pixelWidth

    @property
    def pixelWidthTolerance(self):
        return self._pixelWidthTolerance

    def setPixelWidthTolerance(self, pixelWidthTolerance):
        self._pixelWidthTolerance = pixelWidthTolerance

    @property
    def pixelHeight(self):
        return self._pixelHeight

    def setPixelHeight(self, pixelHeight):
        self._pixelHeight = pixelHeight

    @property
    def pixelHeightTolerance(self):
        return self._pixelHeightTolerance

    def setPixelHeightTolerance(self, pixelHeightTolerance):
        self._pixelHeightTolerance = pixelHeightTolerance

    @property
    def pixelDepth(self):
        return self._pixelDepth

    def setPixelDepth(self, pixelDepth):
        self._pixelDepth = pixelDepth

    @property
    def resinThickness(self):
        return self._resinThickness

    def setResinThickness(self, resinThickness):
        self._resinThickness = resinThickness

    @property
    def pixelArea(self):
        return self._pixelArea

    def setPixelArea(self, pixelArea):
        self._pixelArea = pixelArea

    @property
    def pixelSpacingX(self):
        return self._pixelSpacingX

    def setPixelSpacingX(self, pixelSpacingX):
        self._pixelSpacingX = pixelSpacingX

    @property
    def pixelSpacingY(self):
        return self._pixelSpacingY

    def setPixelSpacingY(self, pixelSpacingY):
        self._pixelSpacingY = pixelSpacingY

    @property
    def borderSizeX(self):
        return self._borderSizeX

    def setBorderSizeX(self, borderSizeX):
        self._borderSizeX = borderSizeX

    @property
    def borderSizeY(self):
        return self._borderSizeY

    def setBorderSizeY(self, borderSizeY):
        self._borderSizeY = borderSizeY

    @property
    def effectiveWidth(self):
        return self._effectiveWidth

    def setEffectiveWidth(self, effectiveWidth):
        self._effectiveWidth = effectiveWidth

    @property
    def effectiveHeight(self):
        return self._effectiveHeight

    def setEffectiveHeight(self, effectiveHeight):
        self._effectiveHeight = effectiveHeight

    @property
    def effectiveAreaPerChannel(self):
        return self._effectiveAreaPerChannel

    def setEffectiveAreaPerChannel(self, effectiveAreaPerChannel):
        self._effectiveAreaPerChannel = effectiveAreaPerChannel

    @property
    def numberOfChannelsX(self):
        return self._numberOfChannelsX

    def setNumberOfChannelsX(self, numberOfChannelsX):
        self._numberOfChannelsX = numberOfChannelsX

    @property
    def numberOfChannelsY(self):
        return self._numberOfChannelsY

    def setNumberOfChannelsY(self, numberOfChannelsY):
        self._numberOfChannelsY = numberOfChannelsY
        self._totalNumberOfChannels = self._numberOfChannelsX * self._numberOfChannelsY

    @property
    def totalNumberOfChannels(self):
        return self._totalNumberOfChannels

    @property
    def blockSPiMWidth(self):
        return self._blockSPiMWidth

    def setBlockSPiMWidth(self, blockSPiMWidth):
        self._blockSPiMWidth = blockSPiMWidth

    @property
    def blockSPiMHeight(self):
        return self._blockSPiMHeight

    def setBlockSPiMHeight(self, blockSPiMHeight):
        self._blockSPiMHeight = blockSPiMHeight

    @property
    def blockSPiMDepth(self):
        return self._blockSPiMDepth

    def setBlockSPiMDepth(self, blockSPiMDepth):
        self._blockSPiMDepth = blockSPiMDepth

    @property
    def blockSPiMArea(self):
        return self._blockSPiMArea

    def setBlockSPiMArea(self, blockSPiMArea):
        self._blockSPiMArea = blockSPiMArea

    @property
    def externalBorderSizeX(self):
        return self._externalBorderSizeX

    @property
    def externalBorderSizeY(self):
        return self._externalBorderSizeY

    def setExternalBorderSizeX(self, externalBorderSizeX):
        self._externalBorderSizeX = externalBorderSizeX

    def setExternalBorderSizeY(self, externalBorderSizeY):
        self._externalBorderSizeY = externalBorderSizeY


    @property
    def alphaRotation(self):
        return self._alphaRotation

    def setAlphaRotation(self, value):
        self._alphaRotation = value

    @property
    def betaRotation(self):
        return self._betaRotation

    def setBetaRotation(self, value):
        self._betaRotation = value

    @property
    def sigmaRotation(self):
        return self._sigmaRotation

    def setSigmaRotation(self, value):
        self._sigmaRotation = value

    @property
    def xTranslation(self):
        return self._xTranslation

    def setXTranslation(self, value):
        self._xTranslation = value

    @property
    def yTranslation(self):
        return self._yTranslation

    def setYTranslation(self, value):
        self._yTranslation = value

    @property
    def zTranslation(self):
        return self._zTranslation

    def setZTranslation(self, value):
        self._zTranslation = value




