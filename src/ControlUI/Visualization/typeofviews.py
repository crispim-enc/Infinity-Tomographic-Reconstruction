import vtk


class ViewPortSelected(object):
    def __init__(self, parent=None):
        self.parent = parent
        self._type = "Scanning View"
        self._orientation = "horizontal"
        self._activeAreas = ActiveAreas()
        self._numberOfViews = 1
        self._numberOfActiveRenderers = 0
        self._xmins = []
        self._xmaxs = []
        self._ymins = []
        self._ymaxs = []
        self._borderColors = [[0.5, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]]

    @property
    def typeView(self):
        return self._type

    def setViewPort(self, listOfRenderers, vtkWidget):
        for i in range(self._activeAreas.quantity):
            # if self.info_type_of_views[self.type_view_selected["type"]]["renderer_area_3D"] is None:
            #     self.info_type_of_views[self.type_view_selected["type"]]["renderer_area_3D"] = vtk.vtkRenderer()
            # self.listOfRenderers[i] = vtk.vtkRenderer()

            listOfRenderers[i].SetViewport(self._xmins[i],
                                           self._ymins[i],
                                           self._xmaxs[i],
                                           self._ymaxs[i])

            ViewPortBorder(renderer=listOfRenderers[i], color=self._borderColors[i],
                           points_coordinates=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], last=False)

            listOfRenderers[i].ResetCamera()
            vtkWidget.GetRenderWindow().AddRenderer(listOfRenderers[i])

    def calculateRangeViewPort(self):
        if self._type == 'Scanning View':
            if self._orientation == 'vertical':
                if self._activeAreas.area3DView:
                    percentage_of_3D_occupancy = [0.5, 1]
                    xmins = [percentage_of_3D_occupancy[0]] * self._numberOfActiveRenderers
                    xmins[0] = 0
                    xmaxs = [1] * self._numberOfActiveRenderers
                    xmaxs[0] = percentage_of_3D_occupancy[0]
                    ymins = [0] * self._numberOfActiveRenderers
                    ymaxs = [1] * self._numberOfActiveRenderers
                    x_m = 0
                    y_m = 0
                    for k in range(1, self._numberOfActiveRenderers):
                        xmins[k] = percentage_of_3D_occupancy[0] + (x_m - percentage_of_3D_occupancy[0]) / \
                                   self._numberOfViews
                        xmaxs[k] = percentage_of_3D_occupancy[0] + (x_m - percentage_of_3D_occupancy[0] + 1) / \
                                   self._numberOfViews
                        ymins[k] = y_m / (self._activeAreas.quantity - 1)
                        ymaxs[k] = (y_m + 1) / (self._activeAreas.quantity - 1)

                        y_m += 1
                        if k == self._activeAreas.quantity:
                            y_m = 0
                            x_m += 1

                    self._xmins = xmins
                    self._xmaxs = xmaxs
                    self._ymins = ymins
                    self._ymaxs = ymaxs

            elif self._orientation == 'horizontal':
                begin_y = 2 / 5
                ymins = [begin_y, 0, 0, 0, begin_y, 3 / 5]
                ymaxs = [1, begin_y, begin_y, begin_y, 3 / 5, 1]
                xmins = [0, 0, 1 / 3, 2 / 3, 2 / 3, 2 / 3]
                xmaxs = [2 / 3, 1 / 3, 2 / 3, 1, 1, 1]

                self._xmins = xmins
                self._xmaxs = xmaxs
                self._ymins = ymins
                self._ymaxs = ymaxs

    # def _changeto_slicedviews(self, volume=None):
    #     print('sliced')
    #     PopulateMainWindowVTK.hide_qt_buttons(self, hide_camera_buttons=True, hide_bed_buttons=True)
    #     volume = np.random.randint(100, size=(40, 40, 36))
    #     number_of_views = volume.shape[2]
    #
    #     # height_per_view = self.main_frame.height() / number_in_y_direction
    #     # width_per_view = self.main_frame.width() / number_in_x_direction
    #     #
    #
    #     grid = np.array(list(set(reduce(list.__add__,
    #                                     ([i, number_of_views // i] for i in
    #                                      range(1, int(pow(number_of_views, 0.5) + 1))
    #                                      if number_of_views % i == 0)))))
    #     # a = np.array([16, 1, 2, 4, 8])
    #     rf = [None] * grid.shape[0]
    #     grid = np.sort(grid)
    #     grid_inverted = np.sort(grid)[::-1]
    #     r_h = [None] * grid.shape[0]
    #     for i in range(0, grid.shape[0]):
    #         r_h[i] = np.abs(self.main_frame.height() / (grid[i]) - self.main_frame.width() / (grid_inverted[i]))
    #
    #         # r = number_of_views / (grid * grid[i])
    #         #
    #         # ind = np.where(r == 1)
    #         # rf[i] = grid[i] / grid[ind]
    #
    #     # rf = np.abs(np.array(rf).T - 1)
    #     # rf = np.where(rf[0] == np.min(rf))
    #     # x_grid = grid[rf]
    #     # y_grid = number_of_views / x_grid
    #     print(r_h)
    #     print(grid[i])
    #     print(grid_inverted[i])
    #
    #     # print(y_grid)
    #     number_in_y_direction = 6
    #     number_in_x_direction = 6
    #     height_per_view = self.dynamic_view_frame.height() / number_in_y_direction
    #     width_per_view = self.dynamic_view_frame.width() / number_in_x_direction
    #
    #     ymins = [None] * number_of_views
    #     ymaxs = [None] * number_of_views
    #     xmins = [None] * number_of_views
    #     xmaxs = [None] * number_of_views
    #     # j += 1
    #     xj = 0
    #     yj = 0
    #
    #     for i in range(number_of_views):
    #
    #         ymins[i] = height_per_view * yj / self.dynamic_view_frame.height()
    #         ymaxs[i] = height_per_view * (yj + 1) / self.dynamic_view_frame.height()
    #         xmins[i] = width_per_view * xj / self.dynamic_view_frame.width()
    #         xmaxs[i] = width_per_view * (xj + 1) / self.dynamic_view_frame.width()
    #         xj += 1
    #         if (i + 1) % number_in_x_direction == 0:
    #             xj = 0
    #             yj += 1
    #
    #     self.listOfRenderers = [None] * number_of_views
    #     for i in range(number_of_views):
    #         self.listOfRenderers[i] = vtk.vtkRenderer()
    #
    #         self.listOfRenderers[i].SetViewport(xmins[i], ymins[number_of_views - 1 - i], xmaxs[i],
    #                                             ymaxs[number_of_views - 1 - i])
    #         # ViewPortBorder(renderer=self.listOfRenderers[i], color=[1,1,1],
    #         #                points_coordinates=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], last=False)
    #         self.listOfRenderers[i].ResetCamera()
    #         self.vtkWidget.GetRenderWindow().AddRenderer(self.listOfRenderers[i])
    #     self.vtkWidget.GetRenderWindow().Render()

    # def _define_viewport_areas(self):
    #     if type_view_selected == 'type_live_acquitision':
    #         self.resliced_main_window = [None] * 3
    #         if planes_view == 'Vertical':
    #             xmins = [0, .75, 0.75, .75]
    #             xmaxs = [0.75, 1, 1, 1]
    #             ymins = [0, 0, 1 / 3, 2 / 3]
    #             ymaxs = [1, 1 / 3, 2 / 3, 1]
    #
    #         elif planes_view == 'Horizontal':
    #             begin_y = 2 / 5
    #             ymins = [begin_y, 0, 0, 0, begin_y, 3 / 5]
    #             ymaxs = [1, begin_y, begin_y, begin_y, 3 / 5, 1]
    #             xmins = [0, 0, 1 / 3, 2 / 3, 2 / 3, 2 / 3]
    #             xmaxs = [2 / 3, 1 / 3, 2 / 3, 1, 1, 1]
    #
    #     elif type_view_selected == "type_2view_comparison":
    #         if planes_view == 'Vertical':
    #             xmins = [0, .75, 0.75, .75]
    #             xmaxs = [0.75, 1, 1, 1]
    #             ymins = [0, 0, 1 / 3, 2 / 3]
    #             ymaxs = [1, 1 / 3, 2 / 3, 1]
    #
    #         elif planes_view == 'Horizontal':
    #             begin_y = 2 / 5
    #             ymins = [begin_y, 0, 0, 0, begin_y, 3 / 5]
    #             ymaxs = [1, begin_y, begin_y, begin_y, 3 / 5, 1]
    #             xmins = [0, 0, 1 / 3, 2 / 3, 2 / 3, 2 / 3]
    #             xmaxs = [2 / 3, 1 / 3, 2 / 3, 1, 1, 1]


class ViewPortBorder:
    def __init__(self, renderer=None, color=None, last=True,
                 points_coordinates=None):
        if points_coordinates is None:
            points_coordinates = [[0, 0, 0], [0, 1, 0]]

        self._pointsCoordinates = points_coordinates
        if color is None:
            color = [0.95, 0.9, 0.95]

        self._color = color
        self._renderer = renderer
        self._last = last

    def updateViewPortBorder(self):
        number_of_points = len(self._pointsCoordinates)
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(number_of_points)
        for i in range(number_of_points):
            points.InsertPoint(i, self._pointsCoordinates[i])

        # points.InsertPoint(2, 0, 1, 0)
        # points.InsertPoint(3, 0, 0, 0)

        cells = vtk.vtkCellArray()
        cells.Initialize()

        lines = vtk.vtkPolyLine()

        if self._last:
            lines.GetPointIds().SetNumberOfIds(number_of_points + 1)
        else:
            lines.GetPointIds().SetNumberOfIds(number_of_points)

        for i in range(number_of_points):
            lines.GetPointIds().SetId(i, i)
            # cells.InsertNextCell(lines)
            # if last:
            #     lines.GetPointIds().SetId(number_of_points, 0)
            #     cells.InsertNextCell(lines)
        cells.InsertNextCell(lines)
        poly = vtk.vtkPolyData()
        poly.Initialize()
        poly.SetPoints(points)
        poly.SetLines(cells)

        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToNormalizedViewport()

        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(poly)
        mapper.SetTransformCoordinate(coordinate)
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self._color)
        actor.GetProperty().SetLineWidth(5.0)
        self._renderer.AddViewProp(actor)
        # self._return_actor(actor)


class ActiveAreas(object):
    def __init__(self):
        self._area3DView = True
        self._axialView = True
        self._coronalView = True
        self._sagittalView = True
        self._quantity = 4

    @property
    def area3DView(self):
        return self._area3DView

    def setArea3DView(self, boolean):
        if boolean != self._area3DView:
            self._area3DView = boolean
            if boolean:
                self._quantity += 1
            else:
                self._quantity -= 1
        return self._area3DView

    def setAxialView(self, boolean):
        if boolean != self._axialView:
            self._axialView = boolean
        return self._axialView

    def setCoronalView(self, boolean):
        if boolean != self._coronalView:
            self._coronalView = boolean
        return self._coronalView

    def setSagittalView(self, boolean):
        if boolean != self._sagittalView:
            self._sagittalView = boolean
        return self._sagittalView

    @property
    def quantity(self):
        return self._quantity

