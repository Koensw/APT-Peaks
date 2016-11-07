"""
Functions to read APT data
"""

import numpy as np
import re

"""
Read a pos file

file_name: name of the file to read 
count: amount of records to read (or -1 for all)
"""
def read_pos(file_name, count=-1):
    dt = np.dtype([('x','>f4'),
                   ('y', '>f4'),
                   ('z', '>f4'),
                   ('m', '>f4')])
    
    return np.fromfile(file_name, dt, count, "")
    
"""
Read an epos file

file_name: name of the file to read 
count: amount of records to read (or -1 for all)
"""
def read_epos(file_name, count=-1):
    dt = np.dtype([('x','>f4'),
                   ('y', '>f4'),
                   ('z', '>f4'),
                   ('m', '>f4'),
                   ('ToF', '>f4'),
                   ('Vdc', '>f4'),
                   ('Vpulse', '>f4'),
                   ('Dx', '>f4'),
                   ('Dy', '>f4'),
                   ('Lpulse', '>i4'),
                   ('Mhit', '>i4')])
    
    return np.fromfile(file_name, dt, count, "")
    
"""
Read an rrng file and returns a tuple (start_range, end_range, atom_type, amount)

file_name: name of the file to read 
"""
def read_rrng(file_name):
    lst = []
    with open(file_name) as file:
        for line in file.readlines():
            if line[:5] != "Range": continue
            mtch = re.match("Range\d+=(\d+\.\d+) (\d+\.\d+) Vol:\d+\.\d+ (\w+):(\d)", line)
            grps = list(mtch.groups())
            grps[0] = float(grps[0])
            grps[1] = float(grps[1])
            grps[3] = int(grps[3])
            lst.append(tuple(grps))
    
    return sorted(lst)