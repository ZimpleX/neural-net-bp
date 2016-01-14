"""
utility functions often used by db operation
"""
import sqlite3
import numpy as np


def surround_by_brackets(name):
    """
    NOTE: name can be a string or a list of string
    ensure that every element in name is surrounded by bracket.
    return a list or a string: cast to other structures on return, if needed
    """
    if type(name) == type(''):
        return (name[0]=='[' and name[-1]==']') and name or '[{}]'.format(name)
    else:
        return [(n[0]=='[' and n[-1]==']') and n or '[{}]'.format(n) for n in name]

def desurround_by_brackets(name):
    """
    NOTE: name can be either a string or a list of string
    ensure that every element in name is not surrounded by bracker.
    return a list or a string
    """
    if type(name) == type(''):
        return (name[0]=='[' and name[-1]==']') and name[1:-1] or name
    else:
        return [(n[0]=='[' and n[-1]==']') and n[1:-1] or n for n in name]




def is_table_exist(db_fullpath, table):
    """
    check if table exists in db_fullpath
    table may or may not be surrounded by '[' and ']'
    """
    conn = sqlite3.connect(db_fullpath)
    c = conn.cursor()
    table = desurround_by_brackets(table)
    tbl_list = list(c.execute('SELECT name FROM sqlite_master WHERE type=\'table\''))
    conn.close()
    tbl_list = list(map(lambda x: x[0], tbl_list))
    return table in tbl_list


def count_entry(db_fullpath, table):
    """
    return total num of entries in table
    table may or may not be surrounded by '[' and ']'
    """
    table = surround_by_brackets(table)
    conn = sqlite3.connect(db_fullpath)
    c = conn.cursor()
    ret = c.execute('SELECT Count(*) FROM {}'.format(table)).fetchone()[0]
    conn.close()
    return ret


def get_attr_info(db_fullpath, table, enclosing=True):
    """
    return a dict for attr and the corresponding type
    with enclosing, the attr will be enclosed by '[' and ']'
    """
    table = surround_by_brackets(table)
    conn = sqlite3.connect(db_fullpath)
    c = conn.cursor()
    ret_dict = {}
    for t in list(c.execute('pragma table_info({})'.format(table))):
        f = enclosing and '[{}]' or '{}'
        ret_dict[f.format(t[1])] = t[2]
    conn.close()
    return ret_dict


def load_as_array(db_fullpath, table, attr_list):
    """
    load data from db as numpy array
    attr name in attr_list may or may not be surrounded by '[' and ']'
    """
    conn = sqlite3.connect(db_fullpath)
    c = conn.cursor()
    attr_list = surround_by_brackets(attr_list)
    table = surround_by_brackets(table)
    return \
        np.array(list(c.execute('SELECT {} FROM {}'.format(','.join(attr_list), table))))
