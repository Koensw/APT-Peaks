"""
Functions to read APT files
"""

import re
import numpy as np

def read_pos(file_name, count=-1):
    """
    Read a pos file and returns a list of dtype defined below

    Args:
        file_name: name of the file to read
        count: amount of records to read (or -1 for all)

    Returns:
        list of tuples with the dtype defined below
    """

    #np.dtype defines a numpy data type
    dtype = np.dtype([('x', '>f4'),     # x position
                      ('y', '>f4'),     # y position
                      ('z', '>f4'),     # z position
                      ('m', '>f4')])    # mass

    #numpy read the binary file typecasting to dt and returning an array of dt's
    return np.fromfile(file_name, dtype, count, "")

def read_epos(file_name, count=-1):
    """
    Read an epos file and returns a list of dtype defined below

    Args:
        file_name: name of the file to read
        count: amount of records to read (or -1 for all)

    Returns:
        list of tuples with the dtype defined below
    """

    #np.dtype defines a numpy data type
    dtype = np.dtype([('x', '>f4'),         # x position
                      ('y', '>f4'),         # y position
                      ('z', '>f4'),         # z position
                      ('m', '>f4'),         # mass
                      ('ToF', '>f4'),       # time of flight
                      ('Vdc', '>f4'),       # dc voltage
                      ('Vpulse', '>f4'),    # pulse voltage
                      ('Dx', '>f4'),        # impact x on detector
                      ('Dy', '>f4'),        # impact y on detector
                      ('Lpulse', '>i4'),    # length of pulse
                      ('Mhit', '>i4')])     # amount of multi hits

    #numpy read the binary file typecasting to dt and returning an array of dt's
    return np.fromfile(file_name, dtype, count, "")

def read_rrng(file_name):
    """
    Read an rrng file and returns a tuple (start_range, end_range, atom_type, amount)

    Args:
        file_name: name of the file to read

    Returns:
        Return a list of tuples (start-range, end-range, ion-name, ion-multiplicity)
    """

    lst = []
    with open(file_name) as file:
        for line in file.readlines():
            # ignore lines not starting with Range
            if line[:5] != "Range":
                continue

            #(\d+\.\d+) matches 2 integers separated by a .,
            # the brackets group it to be one string that is returned
            # the numbers for volume are currently not returned!
            #match is a tuple of strings
            mtch = re.match(r"Range\d+=(\d+\.\d+) (\d+\.\d+) Vol:\d+\.\d+ (\w+):(\d)", line)
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
