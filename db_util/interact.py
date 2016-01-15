"""
some inteaction functions for analyzing data by proper operations
on the tables in the database
"""

import os
import sqlite3
from db_util.conf import *
from db_util.basic import *
from db_util.util import *
from logf.printf import *
from time import strftime
from numpy import *

import pdb

def db_control_dim(meta_table, data_table, *var_attr, comm_key=TIME_ATTR,
    db_path=DB_DIR_PARENT, db_name=DB_NAME, db_temp=ANALYSIS_DB, temp_table=ANALYSIS_TABLE):
    """
    TODO: filter on populate_time & bp-version
    aggregate the meta_table with data_table, filter out some runs by controlling variable,
    write the final table to new table, new file --> ready for visualization.

    ARGUMENTS:
        meta_table      table storing the mata info, usually configurations of a run
        data_table      table storing the data produced by the run --> join with meta to form a complete run
        var_attr        configuration variabes to analyze. will keep all other attr in conf the same
                        e.g.: if you want to analyze effect of momentum in ANN training.
                            then: var_attr='momentum', this will create multiple sub-tables,
                            with configurations such as learning rate, batch, etc., the same.
        comm_key            list or string: join meta_table and data_table by comm_key
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
    meta_table, data_table, temp_table = surround_by_brackets([meta_table, data_table, temp_table])
    if len(array(comm_key).shape) == 0:
        comm_key = [comm_key]
    comm_key = surround_by_brackets(comm_key)
    comm_key = list(map(lambda x: '{}.{}={}.{}'.format(meta_table, x, data_table, x), comm_key))
    var_attr = surround_by_brackets(var_attr)
    # get list of attributes
    # attr in the meta table
    l_attr_meta = list(get_attr_info(meta_table, db_fullpath=db_fullpath).keys())
    l_attr_meta_flt = [item for item in l_attr_meta if item not in var_attr]
    l_attr_meta_flt = [item for item in l_attr_meta_flt if item not in ['[{}]'.format(TIME_ATTR)]]
    # attr in the data table
    l_attr_data = list(get_attr_info(data_table, db_fullpath=db_fullpath).keys())
    l_attr_type = get_attr_info(meta_table, db_fullpath=db_fullpath)
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
    db_temp = surround_by_brackets(db_temp)
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



def sanity_last_n_commit(*table, num_run=1, db_name=DB_NAME, db_path=DB_DIR_PARENT, time_attr=TIME_ATTR):
    """
    delete the entries with the latest populate_time, for all tables with the time attr

    ARGUMENTS:
        table       if table=(), then delete entries for all tables, otherwise only delete for that in *table
        num_run     delete entries with the last (num_run) populate time
        time_attr   the name of the time attribute
    """
    db_fullpath = '{}/{}'.format(db_path, db_name)
    conn = sqlite3.connect(db_fullpath)
    c = conn.cursor()
    if len(table) == 0:
        table = list(c.execute('SELECT name FROM sqlite_master WHERE type=\'table\''))
        table = list(map(lambda x: '[{}]'.format(x[0]), table))
    else:
        table = list(map(lambda x: '[{}]'.format(x), table))
    # fliter table list to those actually contains the time_attr
    table_flt = []
    for tbl in table:
        tbl_attr = list(get_attr_info(tbl, enclosing=False, db_fullpath=db_fullpath).keys())
        if time_attr in tbl_attr:
            table_flt += [tbl]
    time_attr = surround_by_brackets(time_attr)
    time_set = set()
    for tbl in table_flt:
        cur_time_set = set(c.execute('SELECT DISTINCT {} FROM {}'.format(time_attr, tbl)))
        time_set |= set(map(lambda x: x[0], cur_time_set))
    conn.close()
    time_len = len(time_set)
    num_run = (num_run>time_len) and time_len or num_run
    time_list = sorted(list(time_set))[time_len-num_run:]
    for tbl in table_flt:
        for t in time_list:
            sanity_db(time_attr[1:-1], t, tbl[1:-1], db_name=db_name, db_path=db_path)
    
    printf('Done: cleared last {} commits for {}'.format(num_run, table_flt))
    bad_table = set(table) - set(table_flt)
    if bad_table:
        printf('tables {} don\'t have attr {}', bad_table, time_attr, type='WARN')
