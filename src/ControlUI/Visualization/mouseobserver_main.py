

class MouseObserverMainWindow:
    def __init__(self, parent=None):
        self.parent = parent
        self.actions = {"Slicing": 0}
        # self.actions = {}
        self.area_hover = {}
        self.renderer_entered = None

    def ButtonCallback(self, obj, event):
        print("entrei")
        if event == "RightButtonPressEvent":
            self.actions["Slicing"] = 1
        else:
            self.actions["Slicing"] = 0

    def MouseMoveCallback(self, obj, event):
        (lastX, lastY) = self.parent.populateMainWindowVTK.interactorMainWindow.GetLastEventPosition()
        (mouseX, mouseY) = self.parent.populateMainWindowVTK.interactorMainWindow.GetEventPosition()
        renderer_entered = self.parent.populateMainWindowVTK.interactorMainWindow.FindPokedRenderer(mouseX, mouseY)
        # print('Mouse moving')
        if self.parent.populateMainWindowVTK.listOfRenderers[0] == renderer_entered:
            # print('3D')

            self.parent.populateMainWindowVTK.interactorMainWindow.SetInteractorStyle(
                self.parent.populateMainWindowVTK.interactorStyle3D)
            self.parent.populateMainWindowVTK.interactorStyle3D.OnMouseMove()

        else:
            # print('Slicers')
            # self.interactorStyleImage = vtk.vtkInteractorStyleImage()
            # self.interactorStyleImage.SetInteractionModeToImageSlicing()
            # self.interactorMainWindow.SetInteractorStyle(self.interactorStyleImage)
            # index_renderer = self.parent.populateMainWindowVTK.listOfRenderers.index(
            #     renderer_entered) - 1  # testing -- put as dict
            # if self.actions["Slicing"] == 1:
            #     deltaY = mouseY - lastY
            #     # self.resliced_main_window[index_renderer].Update()
            #     sliceSpacing = \
            #     self.parent.populateMainWindowVTK.resliced_main_window[index_renderer].GetOutput().GetSpacing()[2]
            #     matrix = self.parent.populateMainWindowVTK.resliced_main_window[index_renderer].GetResliceAxes()
            #     center = matrix.MultiplyPoint((0, 0, sliceSpacing * deltaY, 1))
            #     matrix.SetElement(0, 3, center[0])
            #     matrix.SetElement(1, 3, center[1])
            #     matrix.SetElement(2, 3, center[2])
            #     self.parent.populateMainWindowVTK.vtkWidget.GetRenderWindow().Render()
            #
            # else:
            self.parent.populateMainWindowVTK.interactorMainWindow.SetInteractorStyle(
                self.parent.populateMainWindowVTK.interactorStyleImage)
            self.parent.populateMainWindowVTK.interactorStyleImage.OnMouseMove()

            self.renderer_entered = renderer_entered

    def wheelForwardCallback(self, obj, event):
        if event == "MouseWheelForwardEvent":
            increment = 1
        elif event == "MouseWheelBackwardEvent":
            increment = -1
        index_renderer = self.parent.populateMainWindowVTK.listOfRenderers.index(self.renderer_entered) - 1
        if self.parent.populateMainWindowVTK.resliced_main_window[index_renderer] is None:
            return
        else:
            sliceSpacing = \
            self.parent.populateMainWindowVTK.resliced_main_window[index_renderer].GetOutput().GetSpacing()[2]
            matrix = self.parent.populateMainWindowVTK.resliced_main_window[index_renderer].GetResliceAxes()
            center = matrix.MultiplyPoint((0, 0, sliceSpacing * increment, 1))
            matrix.SetElement(0, 3, center[0])
            matrix.SetElement(1, 3, center[1])
            matrix.SetElement(2, 3, center[2])
            self.parent.populateMainWindowVTK.vtkWidget.GetRenderWindow().Render()
