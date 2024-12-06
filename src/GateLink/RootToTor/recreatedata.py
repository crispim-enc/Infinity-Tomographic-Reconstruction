import argparse
import glob
import os
import time
from addanglestorootfile import AddMotorInfoToRoot
from findcoincidencesRoot2Numpy import GenerateCoincidencesAndAnhnilationsPositions
from RootJoiner import RootJoiner
from validation_plots import CoincidenceValidationPlots

#  get arguments addanglestorootfile.py from command line

if __name__ == '__main__':
    arrays_keys = ['time', 'baseID', 'runID', 'eventID', 'sourcePosX', 'sourcePosY', 'sourcePosZ', 'energy',
                   'globalPosX', 'globalPosY',
                   'globalPosZ', 'level1ID', 'level2ID', 'level3ID', 'level4ID']

    parser = argparse.ArgumentParser(description='Add motor info to root file')
    parser.add_argument('-f', '--file', help='root file', required=False)
    parser.add_argument('-p', '--path', help='path root files', required=False)
    parser.add_argument('-mf', '--multiples_files', help='True to calculate for the directory', required=False)
    parser.add_argument('-j', '--join_root_files', help='True to calculate for the directory', required=False)
    parser.add_argument('-easypet', '--generate_easypet', help='True to calculate for the directory', required=False)
    parser.add_argument('-tor', '--generate_tor', help='True to calculate for the directory', required=False)
    parser.add_argument('-m', '--add_motor', help='root file', required=False)
    parser.add_argument('-c', '--generate_coincidences', help='True to generate numpy with coincidences', required=False)
    parser.add_argument('-a', '--array_keys', help="Default_keys: " + "\n".join(arrays_keys), required=False)
    parser.add_argument('-s', '--singles', help='If the data is recoverd from singles scanner 1 e 2 put False', required=False)
    # save types: root, numpy
    parser.add_argument('-r', '--record_type', help='root / numpy / root and numpy', required=False)
    # generate coincidences with event ID or coincidence window
    parser.add_argument('-ei', '--event_id', help='eventID (True) or coincidence window', required=False)
    parser.add_argument('-cw', '--coincidence_window', help='coincidence window', required=False)

    args = parser.parse_args()
    print(args.path)
    if args.add_motor is None:
        args.add_motor = "False"
    if args.generate_coincidences is None:
        args.generate_coincidences = "False"
    if args.array_keys is None:
        args.array_keys = arrays_keys
    if args.singles is None:
        args.singles = "False"
        doubleScannerFormat = True
    if args.record_type is None:
        args.record_type = "root"

    if args.event_id is None:
        args.event_id = "True"

    if args.coincidence_window is None:
        args.coincidence_window = 40
    print(args)
    #  create object to add motor info to root file
    root_file = args.file

    if args.multiples_files == "True":
        root_files = glob.glob(os.path.join(args.path, "*.root"))
        root_files.sort()
        print(root_files)
    else:
        root_files = [root_file]

    print(root_files)
    for root_file in root_files:

        if args.add_motor == "True":
            print("Adding motor info to root file: ", root_file)
            rootFile = AddMotorInfoToRoot(filename=root_file)
            rootFile.readRoot()
            if args.singles == "False":
                rootFile.setArraysToConvert(rootFile.singlesScanner1, args.array_keys)
                rootFile.setArraysToConvert(rootFile.singlesScanner2, args.array_keys)
            else:
                rootFile.setArraysToConvert(rootFile.singles, args.array_keys)
            rootFile.readMotorFiles()
            rootFile.createMotorArrays()
            rootFile.saveMotorArraysIntoRoot(args.array_keys)

        if args.generate_coincidences == "True":
            tic = time.time()
            coinc = GenerateCoincidencesAndAnhnilationsPositions(
                filename=root_file)
            try:
                partnumber = int(os.path.basename(root_file).split("_")[1].split(".")[0].split("part")[1])
            except IndexError:
                print("Error: part number not found in file name: ", root_file)

            coinc.setPartNumber(partnumber)
            coinc.setDoubleScannerFormat(doubleScannerFormat)
            coinc.readRoot()
            if args.singles == "False":
                coinc.setArraysToConvert(coinc.singlesScanner1, arrays_keys)
                coinc.setArraysToConvert(coinc.singlesScanner2, arrays_keys)
            else:
                coinc.setArraysToConvert(coinc.singles, arrays_keys)

            tec = time.time()
            # # #
            # With coincidence window
            if args.event_id == "False":
                coinc.findCoincidencesBigArrays(array_keys=arrays_keys, coincidenceWindow=args.coincidence_window)
            elif args.event_id == "True":
                coinc.findCoincidencesTrueEventID(array_keys=arrays_keys)


            toc = time.time()
            print("Time elapsed: ", toc - tic)
            print("Time finding coincidences: ", toc - tec)
            # coinc = GenerateCoincidencesAndAnhnilationsPositions(
            #     filename=root_file)
            # coinc.setPartNumber(100)
            # coinc.setDoubleScannerFormat(False)
            # coinc.readRoot()
            if args.record_type == "numpy" or args.record_type == "root and numpy":
                coinc.saveCoincidencesAsNumpyRecordsArray(arrays_keys)
                records_ = coinc.readCoincidencesNumpyRecordsArray()

            if args.record_type == "root" or args.record_type == "root and numpy":
                coinc.saveCoincidencesAsRootFile(arrays_keys)


    if args.join_root_files == "True":
        print("Joining Files")
        rootJoiner = RootJoiner(file_path=os.path.join(args.path, "motors_added"))
        rootJoiner.join()
    #
    records_ = rootJoiner.globalArray
    plots = CoincidenceValidationPlots(records_, arrays_keys, os.path.join(args.path, "motors_added"))
    plots.generateCharts()


