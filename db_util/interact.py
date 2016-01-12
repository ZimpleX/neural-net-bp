"""
some inteaction functions for analyzing data by proper operations
on the tables in the database
"""

import os
import sqlite3
from db_util.conf import *
from db_util.basic import *
from logf.printf import *
from time import strftime
from numpy import *

import pdb

def db_control_dim(meta_table, data_table, *var_attr, comm_key=TIME_ATTR,
    db_path=DB_DIR_PARENT, db_name=DB_NAME, db_temp=ANALYSIS_DB, temp_table=ANALYSIS_TABLE):
    """
    aggregate the meta_table with data_table, filter out some runs by controlling variable,
    write the final table to new table, new file --> ready for visualization.

    ARGUMENTS:
        meta_table      table storing the mata info, usually configurations of a run
        data_table      table storing the data produced by the run --> join with meta to form a complete run
        var_attr        configuration variabes to analyze. will keep all other attr in conf the same
                        e.g.: if you want to analyze effect of momentum in ANN training.
                            then: var_attr='momentum', this will create multiple sub-tables,
                            with configurations such as learning rate, batch, etc., the same.
        comm_key            join meta_table and data_table by comm_key
        db_path, db_name    tell me where to find the meta_table & data_table
        db_temp, temp_table tell me where to write the processed new tables
                            table_temp should specify reflex rule for indexing.
                            e.g.:
                                temp_table = 'whoa-{}|ann', then whoa-0|ann, whoa-1|ann, ..., will be produced

    NOTE:
        the db_temp is always open with 'w' (overwrite), as it is not raw data.
        For security of db containing your raw data, always enforce read-only policy to it.
        (default setting for function in db_util.basic)
    """
    db_fullpath = '{}/{}'.format(db_path, db_name)
    # not my duty to check file existence
    conn = sqlite3.connect(db_fullpath)
    c = conn.cursor()
    meta_table = '[{}]'.format(meta_table)
    data_table = '[{}]'.format(data_table)
    temp_table = '[{}]'.format(temp_table)
    if len(array(comm_key).shape) == 0:
        comm_key = [comm_key]
    comm_key = list(map(lambda x: '[{}]'.format(x), comm_key))
    comm_key = list(map(lambda x: '{}.{}={}.{}'.format(meta_table, x, data_table, x), comm_key))
    var_attr = list(map(lambda x: '[{}]'.format(x), var_attr))
    # get list of attributes
    # attr in the meta table
    l_attr_meta = list(c.execute('pragma table_info({})'.format(meta_table)))
    l_attr_meta = list(map(lambda x: '[{}]'.format(x[1]), l_attr_meta))
    l_attr_meta_flt = [item for item in l_attr_meta if item not in var_attr]
    l_attr_meta_flt = [item for item in l_attr_meta_flt if item not in ['[{}]'.format(TIME_ATTR)]]
    # attr in the data table
    l_attr_data = list(c.execute('pragma table_info({})'.format(data_table)))
    l_attr_data = list(map(lambda x: '[{}]'.format(x[1]), l_attr_data))
    l_attr_type = {}
    for t in list(c.execute('pragma table_info({})'.format(meta_table))):
        l_attr_type['[{}]'.format(t[1])] = t[2]
    # store the index of attr if it is of TEXT type --> append quote later
    text_idx = [l_attr_meta_flt.index(itm) for itm in l_attr_meta_flt if l_attr_type[itm]=='TEXT']
    # attr list in the joined table
    l_attr = set(var_attr + l_attr_data)
    l_attr = {(itm in var_attr) and '{}.{}'.format(meta_table, itm) \
                or '{}.{}'.format(data_table, itm) for itm in l_attr}
    #
    control_var = list(c.execute('SELECT DISTINCT {} FROM {}'.format(','.join(l_attr_meta_flt), meta_table)))
    temp_fullpath = '{}/{}'.format(db_path, db_temp)
    open(temp_fullpath, 'w').close()    # always overwrite
    db_temp = '[{}]'.format(db_temp)
    c.execute('ATTACH DATABASE \'{}\' AS {}'.format(temp_fullpath, db_temp))
    for flt in control_var:
        flt = [(flt.index(itm) in text_idx) and '\'{}\''.format(itm) or itm for itm in flt]
        flt_cond = ['{}.{}={}'.format(meta_table, l_attr_meta_flt[i], flt[i]) for i in range(len(flt))]
        temp_table_i = temp_table.format(','.join(flt_cond).replace('[','').replace(']','').replace('|', '.'))
        c.execute('CREATE TABLE {}.{} AS SELECT {} FROM {} JOIN {} ON {} WHERE {}'\
            .format(db_temp, temp_table_i, ','.join(l_attr),
                    meta_table, data_table, ' and '.join(comm_key),
                    ' and '.join(flt_cond))) 
    conn.commit()
    conn.close()



def sanity_last_commit():
    """
    delete the entries with the latest populate_time
    """
    pass
