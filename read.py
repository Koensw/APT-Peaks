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
    #np.dtype defines a numpy data type
    dt = np.dtype([('x','>f4'),
                   ('y', '>f4'),
                   ('z', '>f4'),
                   ('m', '>f4')])
    
    #numpy read the binary file typecasting to dt and returning an array of dt's
    return np.fromfile(file_name, dt, count, "")
    
"""
Read an epos file

file_name: name of the file to read 
count: amount of records to read (or -1 for all)
"""
def read_epos(file_name, count=-1):
    #np.dtype defines a numpy data type
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
    
    #numpy read the binary file typecasting to dt and returning an array of dt's
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
            #(\d+\.\d+) matches 2 integers separated by a ., the brackets group it to be one string that is returned
            #the numbers for volume are currently no returned!
            #match is a tuple os strings
            mtch = re.match("Range\d+=(\d+\.\d+) (\d+\.\d+) Vol:\d+\.\d+ (\w+):(\d)", line)
            #convert to list as tuple is immutable
            grps = list(mtch.groups())
            #grps[0] is begin of range
            grps[0] = float(grps[0])
            #grps[1] is end of range
            grps[1] = float(grps[1])
            #grps[2] is the ion name and already a string
            #grps[3] is multiplicity of the ions
            grps[3] = int(grps[3])
            lst.append(tuple(grps))
    #list is sorted by the begin of each range before returning
    return sorted(lst)