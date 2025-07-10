# *******************************************************
# * FILE: vtkrender.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import numpy as np
import vtk
import os
from scipy import ndimage
from toor.ImageReader import RawDataSetter
from .geometricdesigner import GeometryDesigner

filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\MOBY\\26-May-2023_23h58_8turn_0p005s_1p8bot_0p225top_range108_ListMode\whole_body\\ID_26 Jan 2022 - 00h 16m 02s_1p8bot_ IMAGE (78, 78, 129)mrp40ns50p.T"
filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\MOBY\\26-May-2023_23h58_8turn_0p005s_1p8bot_0p225top_range108_ListMode\whole_body\\ID_26 Jan 2022 - 00h 16m 02s_1p8bot_ IMAGE (129, 129, 216).T"
filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\MOBY\\26-May-2023_23h58_8turn_0p005s_1p8bot_0p225top_range108_ListMode\whole_body\\ID_26 Jan 2022 - 00h 16m 02s_1p8bot_ IMAGE (78, 78, 129).T"
filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\MOBY\\26-May-2023_23h58_8turn_0p005s_1p8bot_0p225top_range108_ListMode\whole_body\\ID_26 Jan 2022 - 00h 16m 02s_1p8bot_ IMAGE (104, 104, 129).T"
filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\MOBY\\26-May-2023_23h58_8turn_0p005s_1p8bot_0p225top_range108_ListMode\whole_body\\ID_26 Jan 2022 - 00h 16m 02s_1p8bot_ IMAGE (129, 129, 161).T"
filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\NEMA-NU-4-2008-IQ\\15-December-2022ListMode\\whole_body\\ID_26 Jan 2022 - 00h 16m 02s_1p80bot_ IMAGE (71, 71, 129).T"

size_file_m = [129,129,216]
size_file_m = [129,129,161]
size_file_m =[71,71,129]
# size_file_m = [104,104,129]

# size_file_m = [78,78,129]
# size_file_m = [53,53,65]
r = RawDataSetter(filepath)
# r = RawDataSetter(filepath)
r.read_files()
voxeldata = r.volume

# set to zero the pixels in which the neighbor is much lo

# voxeldata[:, :, -2:] = 0
# voxeldata[:, :, :2] = 0
v_lim = np.array([0.02, 0.8])
scale_x = float(0.5)
scale_y = float(0.5)
scale_z = float(0.5 / (0.4 / 0.44))

extent_x_y = [-scale_x * voxeldata.shape[0] / 2, scale_x * voxeldata.shape[0] / 2,
              -scale_y * voxeldata.shape[1] / 2, scale_y * voxeldata.shape[1] / 2]

# generate coordinates
height = r.size_file_m[0]
width = r.size_file_m[1]
x = np.arange(0, width)
y = np.arange(0, width)
xx, yy = np.meshgrid(x, y)

gd = GeometryDesigner(volume=voxeldata)
# gd._draw_image_reconstructed()


def create_viewport(renderer, position):
    renderer.SetViewport(position)
    return renderer


def create_reslice(input_connection, slab_thickness=None, slab_mode=None):
    reslice = vtk.vtkImageSlabReslice()
    reslice.SetInputConnection(input_connection)
    # reslice.SetOutputDimensionality(2)
    # if orientation == "sagittal":
    #     reslice.SetResliceAxesDirectionCosines(0, 0, 1, 0, 1, 0, 1, 0, 0)
    # elif orientation == "coronal":
    #     reslice.SetResliceAxesDirectionCosines(1, 0, 0, 0, 0, -1, 0, 1, 0)
    # elif orientation == "axial":
    #     reslice.SetResliceAxesDirectionCosines(1, 0, 0, 0, 1, 0, 0, 0, 1)
    if slab_thickness is not None:
        reslice.SetSlabThickness(slab_thickness)
        if slab_mode == "MIP":
            reslice.SetSlabModeToMax()
        elif slab_mode == "MinIP":
            reslice.SetSlabModeToMin()
        elif slab_mode == "Mean":
            reslice.SetSlabModeToMean()
    reslice.SetInterpolationModeToLinear()

    return reslice

def main():
    # Read the volume data
    direct = os.path.dirname(os.path.dirname(__file__))
    colormap = 'hot.cl'

    path = os.path.join(direct, "ControlUI", "colormap_files")
    z_pos = voxeldata.shape[2]
    data3D = voxeldata
    # data3D = self.volume
    #
    # data3D = ndimage.gaussian_filter(data3D, sigma=0.8)
    # data3D = ndimage.uniform_filter(data3D, size=3)
    # data3D = ndimage.convolve(data3D, np.ones((3, 3, 3)) / 27)
    # data3D = ndimage.gaussian_gradient_magnitude(data3D, sigma=0.6)
    # apply a wavelet filter to de image

    # data3D = pywt.idwtn(coeffs_thresh, 'haar')
    data3D = ndimage.median_filter(data3D, size=3)
    # Reconstruct the denoised image
    # Adjust the parameters according to your requirements
    patch_size = 3  # Size of patches used for denoising
    patch_distance = 3  # Maximum distance for patches to be considered similar
    h = 0.1  # Smoothing parameter (higher values give stronger smoothing)

    # Apply non-local means denoising
    # data3D = denoise_nl_means(data3D, patch_size=patch_size, patch_distance=patch_distance, h=h)

    # denoised_image = pywt.idwtn(coeffs_thresh, 'haar')
    xx = (np.tile(np.arange(0, data3D.shape[0]), (data3D.shape[0], 1)) - (
            data3D.shape[0] - 1) / 2) ** 2
    yy = (np.tile(np.arange(0, data3D.shape[1]), (data3D.shape[1], 1)) - (
            data3D.shape[1] - 1) / 2) ** 2
    xx = xx.T
    # circle_cut = xx + yy - (self.im_index_x.shape[1] * np.sin(np.radians(easypetdata.half_real_range))*0.5) ** 2
    circle_cut = xx + yy - (data3D.shape[1] * 0.4) ** 2
    # circle_cut = xx + yy - (self.im_index_x.shape[1] * np.sin(np.radians(120/2))*.50) ** 2

    circle_cut[circle_cut > 0] = 0
    circle_cut[circle_cut < 0] = 1
    circle_cut = np.tile(circle_cut[:, :, None], (1, 1, data3D.shape[2]))
    data3D = data3D * circle_cut
    # data3D[data3D < 0.02*data3D.max()] = 0
    data3D[:, :, :10] = 0
    data3D[:, :, -10:] = 0

    # data3D[:,:,-3:] = 0
    #  Volume inversion
    # data3D = np.flip(data3D, axis=2)

    size = data3D.shape
    w = size[0]
    h = size[1]
    d = size[2]
    stack = np.zeros((w, d, h))

    for j in range(0, z_pos):
        stack[:, j, :] = data3D[:, :, j]

    stack = np.require(stack, dtype=np.float32)  # stack = np.require(data3D,dtype=np.uint16)
    map_cut =1
    normalize_colormap = np.max(data3D) * map_cut
    start_color = 75
    end_color = 1200

    # --------IMPORT DATA-----------------

    dataImporter = vtk.vtkImageImport()
    # The preaviusly created array is converted to a string of chars and imported.
    data_string = stack.tobytes()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    # dataImporter.SetDataScalarTypeToUnsignedShort()
    dataImporter.SetDataScalarTypeToFloat()
    # dataImporter.SetDataScalarTypeToUnsignedChar()
    # dataImporter.SetDataSpacing(self.pixel_size_reconstruct_file[0], self.pixel_size_reconstruct_file[2],
    #                             self.pixel_size_reconstruct_file[1])

    # Because the data that is imported only contains an intensity value (it isnt RGB-coded or someting similar), the importer
    # must be told this is the case.
    dataImporter.SetNumberOfScalarComponents(1)
    # ---------------STORE DATA-------------------------------
    dataImporter.SetDataExtent(0, w - 1, 0, d - 1, 0, h - 1)
    dataImporter.SetWholeExtent(0, w - 1, 0, d - 1, 0, h - 1)
    # dataImporter.SetTransform(transL1)

    # -----------------------Scalar range-------------
    dataImporter.Update()
    reader = dataImporter
    # reader = vtk.vtkMetaImageReader()
    # reader.SetFileName("your_volume.mhd")  # Replace with your volume file path
    # reader.Update()

    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputConnection(reader.GetOutputPort())

    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

    composite_opacity = vtk.vtkPiecewiseFunction()
    composite_opacity.AddPoint(0, 0.0)
    composite_opacity.AddPoint(data3D.max(), 1.0)
    volume_property.SetScalarOpacity(composite_opacity)

    color = vtk.vtkColorTransferFunction()
    color.SetColorSpaceToRGB()
    file_name = os.path.join(path, colormap)
    color_map_file = np.loadtxt(file_name)
    # bins, res = np.histogram(stack.ravel(), len(reader), (stack.min(), stack.max()))
    # res2 = np.interp(res, [stack.min(), stack.max()], [0, 1])
    # opacitymap = np.vstack((res, res2)).T
    # opacitymap = opacitymap.astype('float32')

    increment_color = 0
    for row in range(0, len(color_map_file)):
        if row < start_color:
            color.AddRGBPoint(float(color_map_file[row][0]) * normalize_colormap,
                                              float(color_map_file[0][1]), float(color_map_file[0][2]), float(color_map_file[0][3]))

        elif row > end_color:
             color.AddRGBPoint(float(color_map_file[row][0]) * normalize_colormap,
                                              float(color_map_file[-1][1]), float(color_map_file[-1][2]), float(color_map_file[-1][3]))
        else:
            increment_color += len(color_map_file) / abs(start_color - end_color)
            if int(np.round(increment_color, 0)) < len(color_map_file):
                color.AddRGBPoint(float(color_map_file[row][0]) * normalize_colormap,
                                                  float(color_map_file[int(np.round(increment_color, 0))][1]),
                                                  float(color_map_file[int(np.round(increment_color, 0))][2]),
                                                  float(color_map_file[int(np.round(increment_color, 0))][3]))

    color.AddRGBPoint(0, float(color_map_file[0][1]), float(color_map_file[0][2]), float(color_map_file[0][3]))

    color.AddRGBPoint(data3D.max(), float(color_map_file[-1][1]), float(color_map_file[-1][2]), float(color_map_file[-1][3]))
    # color.AddRGBPoint(0, 0.0, 0.0, 0.0)
    # color.AddRGBPoint(64, 1.0, 0.5, 0.3)
    # color.AddRGBPoint(128, 1.0, 0.5, 0.3)
    # color.AddRGBPoint(192, 1.0, 1.0, 0.9)
    # color.AddRGBPoint(255, 1.0, 1.0, 0.9)
    volume_property.SetColor(color)
    volume_property.SetInterpolationType(2)
    volume_property.SetAmbient(1.1)
    # volumeProperty.SetDiffuse(0.1)
    volume_property.SetSpecular(0.4)

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)


    # Create the volume renderer
    volume_renderer = vtk.vtkRenderer()
    volume_renderer.AddVolume(volume)
    volume_renderer.SetBackground(0, 0, 0)
    # rotate the volume
    volume.RotateX(90)
    volume.RotateY(75)
    volume.RotateZ(60)
    # zoom camera
    volume_renderer.ResetCamera()
    volume_renderer.GetActiveCamera().Zoom(2)



    # Create the slice renderers
    slice_x_renderer = vtk.vtkRenderer()
    slice_y_renderer = vtk.vtkRenderer()
    slice_z_renderer = vtk.vtkRenderer()

    # Set viewports for the renderers
    create_viewport(slice_z_renderer, [0.0, 0.0, 0.5, 0.5])
    create_viewport(slice_x_renderer, [0.0, 0.5, 0.5, 1.0])
    create_viewport(slice_y_renderer, [0.5, 0.5, 1.0, 1.0])
    create_viewport(volume_renderer, [0.5, 0.0, 1.0, 0.5])

    mip_reslice = create_reslice(reader.GetOutputPort(), slab_thickness=100, slab_mode="Mean")
    color_mapper_x = vtk.vtkImageMapToColors()
    color_mapper_x.SetLookupTable(color)
    color_mapper_x.SetInputConnection(mip_reslice.GetOutputPort())

    color_mapper_x.Update()
    # Slice Mapper and Actor for X
    slice_x_mapper = vtk.vtkImageResliceMapper()
    slice_x_mapper.SetInputConnection(color_mapper_x.GetOutputPort())

    slice_x_mapper.SliceFacesCameraOn()
    # slice_x_mapper.SliceAtFocalPointOn()
    slice_x_mapper.BorderOff()
    # set plane of the slice
    plane_x = vtk.vtkPlane()
    plane_x.SetOrigin(0, 0, 0)
    plane_x.SetNormal(0, 0, 1)
    slice_x_mapper.SetSlicePlane(plane_x)



    slice_x_actor = vtk.vtkImageSlice()
    slice_x_actor.SetMapper(slice_x_mapper)
    # slice_x_actor.RotateZ(90)
    # # slice_x_actor.RotateY(90)
    # slice_x_actor.RotateZ(180)


    slice_x_renderer.AddViewProp(slice_x_actor)
    slice_x_renderer.SetBackground(0,0,0)

    # Slice Mapper and Actor for Y
    slice_y_mapper = vtk.vtkImageResliceMapper()
    slice_y_mapper.SetInputConnection(reader.GetOutputPort())
    slice_y_mapper.SliceFacesCameraOn()
    slice_y_mapper.SliceAtFocalPointOn()
    slice_y_mapper.BorderOff()

    slice_y_actor = vtk.vtkImageSlice()
    slice_y_actor.SetMapper(slice_y_mapper)

    slice_y_renderer.AddViewProp(slice_y_actor)
    slice_y_renderer.SetBackground(0,0,0)

    # Create the render window and interactor
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(volume_renderer)
    render_window.AddRenderer(slice_x_renderer)
    render_window.AddRenderer(slice_y_renderer)
    render_window.SetSize(800, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    # interactor per view port
    interactor_x = vtk.vtkRenderWindowInteractor()
    interactor_x.SetRenderWindow(render_window)
    interactor_x.SetInteractorStyle(vtk.vtkInteractorStyleImage())
    sliderRep = vtk.vtkSliderRepresentation2D()
    sliderRep.SetMinimumValue(0)
    sliderRep.SetMaximumValue(reader.GetOutput().GetDimensions()[2] - 1)
    sliderRep.SetValue(0)
    sliderRep.SetTitleText("Slice")
    sliderRep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
    sliderRep.GetPoint1Coordinate().SetValue(30, 30)
    sliderRep.GetPoint2Coordinate().SetCoordinateSystemToDisplay()
    sliderRep.GetPoint2Coordinate().SetValue(200, 30)

    sliderWidget = vtk.vtkSliderWidget()
    sliderWidget.SetInteractor(interactor_x)
    sliderWidget.SetRepresentation(sliderRep)
    sliderWidget.SetAnimationModeToAnimate()
    sliderWidget.EnabledOn()

    # Callback to update the slice position




    # Initialize the interactor
    interactor_x.Initialize()
    render_window.Render()
    interactor_x.Start()


if __name__ == "__main__":
    main()