# *******************************************************
# * FILE: coincidencesRoot2Numpy.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import numpy as np
import uproot
import os


def read_file(filename):
    try:
        flat_tree = uproot.open(filename)['Singles']
    except uproot.KeyInFileError:
        print("No Singles trees available, trying SinglesScanner1")
        SinglesScanner1 = uproot.open(filename)['SinglesScanner1']
        SinglesScanner2 = uproot.open(filename)['SinglesScanner2']
    except:
        print("No trees available")

    for arrays in SinglesScanner1.iterate(
            ['runID', 'eventID', 'sourceID', 'sourcePosX', 'sourcePosY', 'sourcePosZ', 'time', 'energy'], library="np"):
        print(arrays)

    for arrays2 in SinglesScanner2.iterate(
            ['runID', 'eventID', 'sourceID', 'sourcePosX', 'sourcePosY', 'sourcePosZ', 'time', 'energy'], library="np"):
        print(arrays2)

    array2save = np.array(
        [arrays['runID'], arrays['eventID'], arrays['sourceID'], arrays['sourcePosX'], arrays['sourcePosY'],
         arrays['sourcePosZ'], arrays['time'], arrays['energy'], arrays2['energy']], np.float32)
    array2save = array2save.T

    np.save(os.path.splitext(filename)[0], array2save)

    # Plot it!

    # plt.hist(data)
    # plt.savefig("teste.png")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    read_file('easyPET_filtered.root')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
