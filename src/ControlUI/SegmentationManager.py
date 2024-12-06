import PyQt5.QtCore
from PyQt5 import QtCore, QtWidgets
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util import numpy_support
import numpy as np


class SinogramVtk(QtCore.QObject):
    def __init__(self, parent=None):
        self.layout_vtk_sinogram = QtWidgets.QVBoxLayout()
        self.layout_vtk_sinogram.setContentsMargins(0, 0, 0, 0)
        self.vtkWidget_sinogram = vtkWidget_sinogram = QVTKRenderWindowInteractor(
            self.segmentation_frame)
        self.layout_vtk_sinogram.addWidget(self.vtkWidget_sinogram)
        self.interactor_sinogram = self.vtkWidget_sinogram.GetRenderWindow().GetInteractor()

        # rend.setBackground(backgroundColor.GetData())
        #
        # backgroundColor = colors.GetColor3d("SlateGray")
        # EnergyWindow._add_3d_histogram(self)
        # EnergyWindow._add_2d_histogram(self)

        # self.vtkWidget_sinogram.show()
        # self.interactor_sinogram.Initialize()
        self.vtkWidget_sinogram.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.vtkWidget_sinogram.GetRenderWindow().Render()
        self.segmentation_frame.setLayout(self.layout_vtk_sinogram)
        self.view_sinogram = vtk.vtkContextView()
        self.view_sinogram.SetRenderWindow(self.vtkWidget_sinogram.GetRenderWindow())
        self.sinogram_item_vtk = None
        SinogramVtk._add_2d_histogram(self)
        # self.view_sinogram.GetScene().RemoveItem(self.barchart)

    def _add_2d_histogram(self, listMode_original=None):
        listMode_original =np.random.randint(0,6000, (1000,7))

        listMode = listMode_original
        spectrumA = listMode[:, 0] + listMode[:, 1]
        spectrumB = listMode[:, 2] + listMode[:, 3]
        ratioA = (listMode[spectrumA != 0, 0] - listMode[spectrumA != 0, 1]) / spectrumA[spectrumA != 0]
        ratioB = (listMode[spectrumB != 0, 3] - listMode[spectrumB != 0, 2]) / spectrumB[spectrumB != 0]
        spectrumA = spectrumA[spectrumA != 0]
        spectrumB = spectrumB[spectrumB != 0]
        # Z_A, X, Y = np.histogram2d(ratioA, spectrumA, 750, range=[[np.nanmin(ratioA), np.nanmax(ratioB)], [np.nanmin(spectrumA), np.nanmax(spectrumA)]])
        number_of_bins = 750
        ratio_lim = [-0.95, 0.95]
        energy_lim = [200, 6000]
        range_ratio = np.abs(ratio_lim[1] - ratio_lim[0])
        range_energy = np.abs(energy_lim[1] - energy_lim[0])
        Z_A, X, Y = np.histogram2d(ratioA, spectrumA, 750, range=[ratio_lim, energy_lim])
        Z_B, X, Y = np.histogram2d(ratioB, spectrumB, 750, range=[ratio_lim, energy_lim])

        size = 400
        view_sinogram = vtk.vtkContextView()
        view_sinogram.SetRenderWindow(self.vtkWidget_sinogram.GetRenderWindow())
        renderer = view_sinogram.GetRenderer()        # renderer= vtk.vtkRenderer()
        self.vtkWidget_sinogram.GetRenderWindow().AddRenderer(renderer)
        # renwin.AddRenderer(renderer)
        renderer.SetBackground([0] * 3)


        # // Chart
        chart = vtk.vtkChartHistogram2D()
        # axis= chart.GetAxis(0).GetTitleProperties()
        # barChart = vtk.vtkChartXY()
        xAxis = chart.GetAxis(vtk.vtkAxis.BOTTOM)
        xAxis.SetTitle("Phi(º)")
        xAxis.GetTitleProperties().SetColor(1, 1, 1)
        # xAxis.GetTitleProperties().SetFontSize(16)
        # xAxis.GetTitleProperties().ItalicOn()
        xAxis.GetLabelProperties().SetColor(1, 1, 1)
        xAxis.GetPen().SetColor(255, 255, 255, 255)
        # xAxis.SetTicksVisible(False)
        xAxis.SetGridVisible(False)
        xAxis.SetRangeLabelsVisible(True)
        xAxis.SetRange(0, 1000)
        # xAxis.SetOffset(0)
        #
        # xAxis.SetUnscaledMinimum(200)

        # barChart = vtk.vtkChartXY()
        yAxis = chart.GetAxis(vtk.vtkAxis.LEFT)
        yAxis.SetTitle("S (mm)")
        yAxis.GetTitleProperties().SetColor(1, 1, 1)
        # xAxis.GetTitleProperties().SetFontSize(16)
        # xAxis.GetTitleProperties().ItalicOn()
        yAxis.GetLabelProperties().SetColor(1, 1, 1)
        yAxis.GetPen().SetColor(255, 255, 255, 255)
        # xAxis.SetTicksVisible(False)
        yAxis.SetRangeLabelsVisible(True)
        yAxis.SetGridVisible(False)
        view_sinogram.GetScene().AddItem(chart)

        yAxis_twin = chart.GetAxis(vtk.vtkAxis.TOP)
        yAxis_twin.GetTitleProperties().SetColor(1, 1, 1)
        yAxis_twin.GetLabelProperties().SetColor(1, 1, 1)
        yAxis_twin.SetAxisVisible(True)

        Z_A = np.rot90(Z_A)
        data = SinogramVtk.numpy_array_as_vtk_image_data(Z_A)
        data.SetSpacing(range_ratio / number_of_bins, range_energy / number_of_bins, 1.0)
        data.SetOrigin(-1, 0, 0.0)

        chart.SetInputData(data)
        transferFunction = vtk.vtkColorTransferFunction()
        transferFunction.AddHSVSegment(0, 0, 0, 0,
                                       np.max(Z_A), 0.497, 0.741, 1)

        transferFunction.Build()
        chart.SetTransferFunction(transferFunction)

    @staticmethod
    def numpy_array_as_vtk_image_data(source_numpy_array):
        """
        :param source_numpy_array: source array with 2-3 dimensions. If used, the third dimension represents the channel count.
        Note: Channels are flipped, i.e. source is assumed to be BGR instead of RGB (which works if you're using cv2.imread function to read three-channel images)
        Note: Assumes array value at [0,0] represents the upper-left pixel.
        :type source_numpy_array: np.ndarray
        :return: vtk-compatible image, if conversion is successful. Raises exception otherwise
        :rtype vtk.vtkImageData
        """

        if len(source_numpy_array.shape) > 2:
            channel_count = source_numpy_array.shape[2]
        else:
            channel_count = 1

        output_vtk_image = vtk.vtkImageData()
        output_vtk_image.SetDimensions(source_numpy_array.shape[1], source_numpy_array.shape[0], channel_count)

        vtk_type_by_numpy_type = {
            np.uint8: vtk.VTK_UNSIGNED_CHAR,
            np.uint16: vtk.VTK_UNSIGNED_SHORT,
            np.uint32: vtk.VTK_UNSIGNED_INT,
            np.uint64: vtk.VTK_UNSIGNED_LONG if vtk.VTK_SIZEOF_LONG == 64 else vtk.VTK_UNSIGNED_LONG_LONG,
            np.int8: vtk.VTK_CHAR,
            np.int16: vtk.VTK_SHORT,
            np.int32: vtk.VTK_INT,
            np.int64: vtk.VTK_LONG if vtk.VTK_SIZEOF_LONG == 64 else vtk.VTK_LONG_LONG,
            np.float32: vtk.VTK_FLOAT,
            np.float64: vtk.VTK_DOUBLE
        }
        vtk_datatype = vtk_type_by_numpy_type[source_numpy_array.dtype.type]

        source_numpy_array = np.flipud(source_numpy_array)

        # Note: don't flip (take out next two lines) if input is RGB.
        # Likewise, BGRA->RGBA would require a different reordering here.
        # if channel_count > 1:
        #     source_numpy_array = np.flip(source_numpy_array, 2)

        depth_array = numpy_support.numpy_to_vtk(source_numpy_array.ravel(), deep=True, array_type=vtk_datatype)
        depth_array.SetNumberOfComponents(channel_count)
        output_vtk_image.SetSpacing([1, 1, 1])
        output_vtk_image.SetOrigin([-1, -1, -1])
        output_vtk_image.GetPointData().SetScalars(depth_array)

        output_vtk_image.Modified()
        return output_vtk_image

