#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: technician
# * AUTHOR: Pedro Encarnação
# * DATE: 24/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************


class Technician:
    def __init__(self):
        self._name = None
        self._role = None
        self._organization = None
        self._animalWelfareResponsiblePerson = None
        self._animalWelfareResponsiblePersonRole = None
        self._animalWelfareResponsibleOrganization = None

    def setName(self, name):
        """
        str: Name of the technician
        """
        self._name = name

    def setRole(self, role):
        """
        str: Role of the technician
        """
        self._role = role

    def setOrganization(self, organization):
        self._organization = organization

    def setAnimalWelfareResponsiblePerson(self, animalWelfareResponsiblePerson):
        self._animalWelfareResponsiblePerson = animalWelfareResponsiblePerson

    def setAnimalWelfareResponsiblePersonRole(self, animalWelfareResponsiblePersonRole):
        self._animalWelfareResponsiblePersonRole = animalWelfareResponsiblePersonRole

    def setAnimalWelfareResponsibleOrganization(self, animalWelfareResponsibleOrganization):
        self._animalWelfareResponsibleOrganization = animalWelfareResponsibleOrganization

