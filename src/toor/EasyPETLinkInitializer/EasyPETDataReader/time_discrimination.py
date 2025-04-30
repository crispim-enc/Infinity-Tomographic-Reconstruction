import numpy as np
# from lmodebin import binary_data


def time_discrimination(listMode, datatype=".easypet", simulationFile=False):
    if datatype ==".easypet":
        diff = np.diff(listMode[:, 4])
    elif datatype ==".easypetoriginal":
        diff = np.diff(listMode[:, 5])

    if simulationFile:
        diff = np.diff(listMode[:, 0])
        asign = np.sign(diff)
        time_indexes = np.where(asign == -1)
        return time_indexes

    listMode_deleted = np.delete(listMode, np.where(diff == 0), axis=0)
    if len(listMode_deleted) < 2:  # tem que ter pelo menos duas entradas
        return
    differ = np.delete(diff, np.where(diff == 0))
    # determine the signal of the matriz
    asign = np.sign(differ)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    signchange[0] = 1  # Does not detect all times the begining
    indexes_signchange = np.where(signchange == 1)

    listMode_deleted = listMode_deleted[indexes_signchange]

    # Broadcasting
    a = listMode[:][:, 6]
    b = listMode_deleted[:, 6]
    c = a[..., None] == b[None, ...]
    time_indexes = np.where(c == True)

    u, indices, counts = np.unique(time_indexes[:][1], return_index=True, return_counts=True)

    index_cut = np.int_(np.floor(counts / 2 + indices))

    #index_cut=index_cut[:-1]
    time_indexes = time_indexes[:][0]
    time_indexes = np.array(time_indexes)
    # print(time_indexes)
    # Give the begining of the turns and time in the respective matriz
    try:
        start_index_turn = listMode[time_indexes[index_cut], 0]

    except IndexError:
        index_cut[-1]=indices[-1]

    if not len(time_indexes) == 0:
        time_indexes = np.append(time_indexes, len(listMode) - 1)
        index_cut = np.append(index_cut,-1)

    #start_index_time = listMode[time_indexes[index_cut], 6]

    print("Time discriminition: {}".format(time_indexes[index_cut]))
    return time_indexes[index_cut]

def index_end_turn(reading_data, turn_end_time, last_turn_end_index_corrected_data):
    "compares the end turn time index from raw data and converted data"

    turn_end_index = np.where(reading_data[last_turn_end_index_corrected_data:,6]-turn_end_time[-1]<0.01)

    return turn_end_index[0][-1]+last_turn_end_index_corrected_data





# teste
# common_path= 'D:\github_easypet\easyPETtraining\Easypet_client with BD\Acquisitions' \
#             '\Easypet Scan 13 Feb 2019 - 18h 53m 45s\\'
# file_name =common_path+'Easypet Scan 13 Feb 2019 - 18h 53m 45s.easypet'
# file_name_original = common_path+'Easypet Scan 13 Feb 2019 - 18h 53m 45s Original data.easypetoriginal'
#
# [reading_data, Version_binary, header, dates, otherinfo] = binary_data().open(file_name)
# listmode = binary_data().open_original_data(file_name_original)
# import time
# turn_end_time = [listmode[150,6]/1000]
# #reading_data = np.ones((10000000,7))#print(reading_data[:,6])
#
# turn_end_index = index_end_turn(reading_data, turn_end_time=turn_end_time,last_turn_end_index_corrected_data=5)
# print(turn_end_index)