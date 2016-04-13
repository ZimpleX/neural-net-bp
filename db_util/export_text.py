"""
export db data values into a text file
"""

import os
import sqlite3
from db_util.conf import *
from db_util.basic import *
from db_util.util import *
from logf.printf import *
import logf.filef as ff
from time import strftime
from numpy import *
import logf.filef as filef

import pdb


def export_to_text(table, out_file, order_by, *col, db_name=DB_NAME, db_path=DB_DIR_PARENT):
    """
    export data entries from a sqlite3 db into a text file
        order_by        output entries order by the value in this column
        *col            the columns in table to be exported
    """
    db_fullpath = '{}/{}'.format(db_path, db_name)
    conn = sqlite3.connect(db_fullpath)
    c = conn.cursor()
    table = surround_by_brackets(table)
    order_by = surround_by_brackets(order_by)
    col = [surround_by_brackets(_) for _ in col]
    col_str = ','.join(col)
    retrieved = list(c.execute('SELECT DISTINCT {cols} FROM {table} ORDER BY {ob}'.format(cols=col_str, table=table, ob=order_by)))
    conn.close()
    from functools import reduce
    s = reduce(lambda _1, _2: '{}\n{}'\
            .format(_1, reduce(lambda __1, __2: '{}, {}'.format(__1,__2), _2)), \
            retrieved, '')
    s = s.strip()
    ff.print_to_file(out_file, s, type=None, log_dir='./')
    printf('retrieved data into {}', out_file)
