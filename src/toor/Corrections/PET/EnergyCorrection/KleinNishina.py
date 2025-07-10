# *******************************************************
# * FILE: KleinNishina.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

# import matplotlib.pyplot as plt
# import numpy as np
#
#
# class KleinNishina:
#     def __init__(self, incident_energy):
#         self.incident_energy = incident_energy
#         self.r0 = 2.8179e-15  # classical electron radius in meters
#
#     def cross_section(self, scattered_energy, theta):
#         e1 = self.incident_energy
#         e2 = scattered_energy
#         term1 = (self.r0**2) / (4 * np.pi)
#         term2 = (e2**2 / e1**2) * (1 + np.cos(theta)**2)
#         term3 = (2 * e1 * e2) / (e1**2) + np.sin(theta)**2
#         return term1 * (term2 - term3)
#
# incident_energy = 511*1.60218e-16 # incident photon energy = 6.4e-14 J
# scattered_energies = [3.2e-14, 4.0e-14, 4.8e-14] # scattered photon energies in Joules
#
# kn = KleinNishina(incident_energy)
# theta = np.linspace(0, 2*np.pi, 360)
# colors = ['red','green','blue']
#
# ax = plt.subplot(111, polar=True)
# for i in range(len(scattered_energies)):
#     cross_section = kn.cross_section(scattered_energies[i], theta)
#     ax.plot(theta, cross_section, label=f'Scattered energy = {scattered_energies[i]:.2e} J', color=colors[i])
#
# ax.legend()
# plt.title('Klein-Nishina Differential Cross Section')
# plt.show()

import math
class KleinNishina:
    def __init__(self, incident_energy_keV):
        self.incident_energy = incident_energy_keV * 1e3 # convert keV to J

    def scattering_angle(self, scattered_energy_keV):
        e1 = self.incident_energy
        e2 = scattered_energy_keV * 1e3 # convert keV to J
        return math.acos(1 - (e1/e2))

incident_energy = 6.4 # incident photon energy = 6.4 keV
scattered_energies = [3.2, 4.0, 4.8] # scattered photon energies in keV

kn = KleinNishina(incident_energy)

for i in range(len(scattered_energies)):
    angle = kn.scattering_angle(scattered_energies[i])
    print(f'Scattering angle for scattered energy of {scattered_energies[i]} keV  {angle}')

# kn = KleinNishina(incident_energy)
# theta = np.linspace(0, 2*np.pi, 360)
# colors = ['red','green','blue']
#
# ax = plt.subplot(111, polar=True)
# for i in range(len(scattered_energies)):
#     cross_section = kn.cross_section(scattered_energies[i], theta)
#     ax.plot(theta, cross_section, label=f'Scattered energy = {scattered_energies[i]:.2e} J', color=colors[i])