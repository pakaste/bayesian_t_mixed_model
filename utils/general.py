import os
import sys
import fnmatch

import datetime


def get_datetime_now():
    """
        Return the datetime now as a string, with the following format:
        YYYY_MM_DD
    """
    dt_now = datetime.datetime.now()
    year = str(dt_now.year)
    month = str(dt_now.month)
    day = str(dt_now.day)
    dt_now_string = year + '_' + month + '_' + day

    return dt_now_string


def get_filename(path, get_extension=False):
    '''
    Extracts filename from a path
        path          : path to a directory or file (str)
        get_extension : if True includes extension
    Returns : filename (str)
    Example : get_filename('./home/abc/pip_list.txt', True)
    '''

    base = os.path.basename(path)
    if get_extension:
        return base

    return os.path.splitext(base)[0]


def get_paths(
        dtory,
        typelist=[
            'jpg',
            'jpeg',
            'png',
            'tiff',
            'tif',
            'gif',
            'avi',
            'mat'],
    include_no_extensions=False,
    nesting=True,
    sort=True,
        verbose=True):
    '''
    Returns paths of files of certain type under a directory.
        dtory                 : directory path
        typelist              : python list of valid extensions
        include_no_extensions : if True, includes files without extensions
        nesting               : if True, recursively include all subdirectories
        sort                  : if True, return sorted list
    Returns : python list
    Example : get_paths('./home')
    '''

    matches = []

    if nesting:
        for root, dirnames, filenames in os.walk(dtory):
            for typ in typelist:
                for filename in fnmatch.filter(filenames, '*.' + typ):
                    matches.append(os.path.join(root, filename))
            if verbose:
                print(filenames)
            if include_no_extensions:  # include files without any extension
                for filename in filenames:
                    if '.' not in filename:
                        matches.append(os.path.join(root, filename))
    else:
        for typ in typelist:
            matches = matches + glob.glob(dtory + '\*' + typ)

    matches = list(set(matches))  # remove duplicates
    print("{} files found.".format(len(matches)))

    if sort:
        return sorted(matches)
    else:
        return matches
