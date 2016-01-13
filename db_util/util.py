"""
utility functions often used by db operation
"""
import sqlite3

def is_table_exist(db_fullpath, table):
    """
    check if table exists in db_fullpath
    table may or may not be surrounded by '[' and ']'
    """
    conn = sqlite3.connect(db_fullpath)
    c = conn.cursor()
    if table[0] == '[' and table[-1] == ']':
        table = table[1][-1]
    tbl_list = list(c.execute('SELECT name FROM sqlite_master WHERE type=\'table\''))
    conn.close()
    tbl_list = list(map(lambda x: x[0], tbl_list))
    return table in tbl_list


def count_entry(db_fullpath, table):
    """
    return total num of entries in table
    table may or may not be surrounded by '[' and ']'
    """
    if table[0] != '[' and table[-1] != ']':
        table = '[{}]'.format(table)
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
    if table[0] != '[' and table[-1] != ']':
        table = '[{}]'.format(table)
    conn = sqlite3.connect(db_fullpath)
    c = conn.cursor()
    ret_dict = {}
    for t in list(c.execute('pragma table_info({})'.format(table))):
        f = enclosing and '[{}]' or '{}'
        ret_dict[f.format(t[1])] = t[2]
    conn.close()
    return ret_dict
