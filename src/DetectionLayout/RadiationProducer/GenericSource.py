#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: GenericSource
# * AUTHOR: Pedro Encarnação
# * DATE: 28/01/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

"""
This functions create a generic source object for example x-ray emission.
"""


class GenericRadiativeSource:
   def __init__(self):
       self._sourceUUID = None
       self._sourceName = "Am-241"
       self._sourceHalfLife = 432.2 # years
       self._sourceActivity = 0.0
       self._sourceActivityUnit = "Bq"
       self._focalSpot = [0, 0, 0]
       self._focalSpotDiameter = 2

    
