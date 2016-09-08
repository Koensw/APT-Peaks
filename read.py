import numpy as np

def read_pos(file_name, count=-1):
    dt = np.dtype([('x','>f4'),
                   ('y', '>f4'),
                   ('z', '>f4'),
                   ('m', '>f4')])
    
    return np.fromfile(file_name, dt, count, "")
    