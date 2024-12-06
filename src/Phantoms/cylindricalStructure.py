import  os
import numpy as np


class CylindricalStructure:
    """
    Class to define a cylindrical structure

    """
    def __init__(self):
        self._material = "Air"
        self._rMin = 0
        self._rMax = 16.75
        self._height = 80
        self._objectID = 1
        self._center = np.array([0, 0, 0], dtype=np.float32)
        # self._densitiesFile = "../../bin/GateMaterials.db"
        self._densitiesFile = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "bin", "GateMaterials.db")
        self._density = None

    @property
    def density(self):
        return self._density

    def setDensity(self, value=None):
        if value is None:
            try:
                with open(self._densitiesFile) as file:
                    # print(self._material)
                    data = file.read()
                    data = data.split("\n")
                    material = self._material
                    # if len(self._material) > 1:
                    #     material = self._material[0]
                    mask = [el.startswith(material) for el in data]


                self._density = float(np.array(data)[np.array(mask)][0].split("d=")[1].split(" ")[0])
            except IndexError:
                # print("Material not found. Density set to 1.15 g /cm3")
                self._density = 1.15
        else:
            self._density = value

        print("Density: {}".format(self._density))
        print("__________")

    @property
    def material(self):
        return self._material

    def setMaterial(self, value):
        if value != self._material:
            self._material = value
            self.setDensity()

    @property
    def rMin(self):
        return self._rMin

    def setRMin(self, value):
        if value != self._rMin:
            self._rMin = value

    @property
    def rMax(self):
        return self._rMax

    def setRMax(self, value):
        if value != self._rMax:
            self._rMax = value

    @property
    def height(self):
        return self._height

    def setHeight(self, value):
        if value != self._height:
            self._height = value

    @property
    def center(self):
        return self._center

    def setCenter(self, value):
        # if value != self._center:
        self._center = value
