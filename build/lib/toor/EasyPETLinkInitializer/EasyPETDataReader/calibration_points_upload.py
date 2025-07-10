# *******************************************************
# * FILE: calibration_points_upload.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import glob
import numpy as np
import os
import logging


def calibration_points_init(numberOfCrystals):
    logging.debug('START')
    sub_path_calibration = "x_{}__y_{}".format(numberOfCrystals[0], numberOfCrystals[1])

    # calibration file
    mod_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    #print((mod_directory+sub_path_calibration+'/calibrationdata/*.easypetcal'))
    # newest = max(glob.iglob(os.path.join(mod_directory,'system_configurations',sub_path_calibration,'*.calbpeak')), key=os.path.getctime)

    newest = max(glob.iglob(os.path.join(os.path.dirname(os.path.dirname(mod_directory)),'system_configurations',sub_path_calibration,'*.calbpeak')), key=os.path.getctime)
    print('Name of calibration file: ' + newest)

    peakList = []
    file = open(newest, "r")
    for line in file.readlines():
        peakList.append([])
        for i in line.split():
            peakList[-1].append(float(i))

    peakArray = np.array(peakList).T
    numberOfCrystals = int(numberOfCrystals[0]*numberOfCrystals[1])
    peakMatrix = np.zeros((numberOfCrystals, 2))
    peakMatrix[:, 0] = peakArray[0][0:numberOfCrystals]
    peakMatrix[:, 1] = peakArray[0][numberOfCrystals: numberOfCrystals * 2]
    head, tail = os.path.split(newest)
    file_name = tail.split('.')
    file_name=file_name[0]
    file_name = file_name.split('_')
    file_name = file_name[0]
    file_name = file_name.split(' ')
    file_name = file_name[2:]
    space=' '
    file_name = space.join(file_name)
    file_name = 'Last Calibration: ' + file_name

    # Energy Factors
    #path = os.path.dirname(os.path.abspath(__file__))
    # newest = max(glob.iglob(mod_directory + '\\calibrationdata' + sub_path_calibration + '\\*.calbenergy'), key=os.path.getctime)
    newest = max(glob.iglob(os.path.join(os.path.dirname(os.path.dirname(mod_directory)), 'system_configurations', sub_path_calibration, '*.calbenergy')),
                 key=os.path.getctime)
    # newest = path + '/calibrationdata/MatrixA24_05_2017_18h40m_CF.out'
    # print('Name of calibration file: '+ newest)

    CF = []
    file = open(newest, "r")
    for line in file.readlines():
        CF.append([])
        for i in line.split():
            try:
                CF[-1].append(float(i))
            except ValueError:
                continue

    CFArray = np.array(CF).T
    logging.debug('END')
    return peakMatrix, file_name, CFArray

# teste calibration_points

# x = calibration_points_init([16,1])
# y = calibration_points_init([16,2])
# z = calibration_points_init([8,8])
# print(x)
