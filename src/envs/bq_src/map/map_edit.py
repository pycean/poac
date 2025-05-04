import h5py
import numpy as np

def read_h5py():
    map_data = h5py.File('map_'+str(0)+'.h5', 'r')['map_data'][:]
    print()

def write_h5py():
    # (13, 23, 2)
    # (17, 27, 2)
    imgData = np.zeros((67, 77, 2))
    f = h5py.File('map_5.h5', 'w')
    f['map_data'] = imgData
    f.close()


write_h5py()