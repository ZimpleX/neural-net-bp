"""
Don't need to configure AWS CLI here, cuz I assume user will call
spark launcher / submit scripts before this parser. 

In this case, CLI will have already been configured.
"""

import argparse
from bs4 import BeautifulSoup
import urllib.request as urlreq
import numpy as np
import db_util.basic as db

from ec2.cmd import CMD
from logf.printf import printf

from ec2.EmbedScript import *
import pdb



def parse_args():
    parser = argparse.ArgumentParser('parse from web UI')
    parser.add_argument('-n', '--cluster_name', type=str, required=True, help='name of the spark cluster')
    parser.add_argument('-s', '--start_idx', type=int, default=0, help='start from which app')
    parser.add_argument('-e', '--end_idx', type=int, default=-1, help='end to which app (-1 means that you want to parse to the end)')
    return parser.parse_args()



def time_conv(timeStr):
    """
    format time to be 'second':
        recognize 'ms' / 'mS' / 'min' / 'Min' / 'h'
    """
    unit = timeStr.split()[-1]
    val = float(timeStr.split()[0])
    if unit in ['ms', 'mS']:
        return val*0.001
    elif unit in ['min', 'Min']:
        return val*60.
    elif unit in ['h']:
        return val*3600.
    elif unit in ['s', 'S']:
        return val
    else:
        return timeStr

def unit_conv(unit_str):
    ret = time_conv(unit_str)
    if type(ret) == type(''):
        unit = ret.split()[-1]
        val = float(ret.split()[0])
        if unit[0] in ['k', 'K']:
            ret = 1000*val
        elif unit[0] in ['m']:
            ret = 0.001*val
        elif unit[0] in ['M']:
            ret = 1000000*val
        elif unit[0] in ['g', 'G']:
            ret = 1000000000*val
        else:
            ret = val
    return ret
    


_URL_FMT = 'http://{}:8080{}'
METRIC_NAME = [ 'Duration',         # computing time
                'Scheduler Delay',   # IO delay ??
                'Task Deserialization Time',
                'GC Time',
                'Result Serialization Time',
                'Getting Result Time',
                'Peak Execution Memory',
                'Input Size',
                'Shuffle Read Size',
                'Shuffle Read Blocked Time',
                'Shuffle Remote Reads']        # NOTE: i am not considering Record metric currently




class job_profile:
    def __init__(self, job_html_entry, dns_parent):
        td_list = job_html_entry.find_all('td')
        self.jID = int(td_list[0].get_text())
        self.description = {'name': td_list[1].get_text().replace('\n', ' ').strip(),
                            'href': _URL_FMT.format(dns_parent,td_list[1].a['href'])}
        self.duration = time_conv(td_list[3].get_text())
        self.stages = {'success': int(td_list[4].get_text().split('/')[0]),
                        'total':  int(td_list[4].get_text().split('/')[1])}

    def __str__(self):
        return ("id:           {id}\n"
                "description:  {desc}\n"
                "duration:     {dur} s\n"
                "stages:       {stg}")\
                    .format(id=self.jID, desc=self.description['name'].split(' ')[0],
                        dur=self.duration, stg='{}/{}'.format(self.stages['success'], self.stages['total']))



class stage_profile:
    def __init__(self, stage_html_entry, dns_parent):
        td_list = stage_html_entry.find_all('td')
        self.sID = int(td_list[0].get_text())
        self.description = {'name': td_list[1].a.get_text().replace('\n', ' ').strip(),
                            'href': _URL_FMT.format(dns_parent,td_list[1].a['href'])}
        self.duration = time_conv(td_list[3].get_text())
    
    def __str__(self):
        return ("id:           {id}\n"
                "description:  {desc}\n"
                "duration:     {dur} s")\
                    .format(id=self.sID, desc=self.description['name'].split(' ')[0], dur=self.duration)


class app_profile:
    def __init__(self, app_html_entry, dns_parent):
        td_list = app_html_entry.find_all('td')
        self.dns_parent = dns_parent
        self.id = { 'name': td_list[0].a.get_text(),
                    'href': _URL_FMT.format(dns_parent,td_list[0].a['href'])}
        self.name = {'name': app_html_entry.find_all('td')[1].a.get_text(),
                    'href': _URL_FMT.format(dns_parent,td_list[1].a['href'])}
        self.cores = int(td_list[2].get_text())
        self.mem_per_node = td_list[3]['sorttable_customkey']     # measured in terms of Mega bytes
        self.submitted_time = td_list[4].get_text()
        self.state = td_list[6].get_text()
        self.duration = time_conv(td_list[7].get_text())
        self.job_list = []
        self.stage_list = []
        # dic value is an numpy array --> index is the stage id
        self.data = {k: None for k in METRIC_NAME}

    def __str__(self):
        return ("appID:       {id}\n"
                "appName:     {name}\n"
                "numCores:    {cores}\n"
                "memPerNode:  {mem} MB\n"
                "submitTime:  {time}\n"
                "state:       {state}\n"
                "duration:    {dur} s\n"
                "jobs:\n"
                "* {j}\n"
                "stages:\n"
                "* {s}")\
                    .format(id=self.id['name'], name=self.name['name'], cores=self.cores,
                        mem=self.mem_per_node, time=self.submitted_time,
                        state=self.state, dur=self.duration,
                        j='\n* '.join(map(lambda _: '\n  '.join(str(_).split('\n')), self.job_list)),
                        s='\n* '.join(map(lambda _: '\n  '.join(str(_).split('\n')), self.stage_list)))

    def set_jobs(self):
        printf('cur job: {}'.format(self.name['href']))
        r_pub = urlreq.urlopen(self.name['href'])
        job_soup = BeautifulSoup(r_pub, 'html.parser')
        job_table = job_soup.find_all('tbody')[0]
        self.job_list = []
        for j in job_table.find_all('tr'):
            self.job_list += [job_profile(j, self.dns_parent)]
    
    def set_stages(self):
        r_pub = urlreq.urlopen(self.name['href'])
        job_soup = BeautifulSoup(r_pub, 'html.parser')
        jump = job_soup.find_all('li')[1].a['href']
        job_url = _URL_FMT.format(self.dns_parent, jump)
        r_pub = urlreq.urlopen(job_url)
        stage_soup = BeautifulSoup(r_pub, 'html.parser')
        stage_table = stage_soup.find_all('tbody')[0]
        self.stage_list = []
        for s in stage_table.find_all('tr'):
            self.stage_list += [stage_profile(s, self.dns_parent)]
        s_len = len(self.stage_list)
        self.data = {k:np.zeros(s_len) for k in self.data.keys()}
        
    def parse_stages(self):
        idx = 0
        for s in self.stage_list:
            r_pub = urlreq.urlopen(s.description['href'])
            s_soup = BeautifulSoup(r_pub, 'html.parser')
            d_table = s_soup.find_all('tbody')[0]
            for t in d_table.find_all('tr'):
                k = t.find_all('td')[0].get_text().strip()
                v = t.find_all('td')[3].get_text()
                self.data[k][idx] = unit_conv(v)
            idx += 1

    def print_data(self):
        for k in self.data.keys():
            printf('{}\n{}',k,self.data[k],type='',separator='-')



class clt_profile:
    def __init__(self, cluster_name):
        """
        set basic information by querying aws:
            master url
            instance type
            num workers
        """
        self.basic = {'dns': None, 'instance_type': None, 'num_workers': None}
        master_name = "{}-master-{}".format(cluster_name, '\S*')
        slave_name = "{}-slave-{}".format(cluster_name, '\S*')
        try:
            stdout, stderr = runScript(CMD['get_dns'].format(name=master_name), output_opt='pipe')
        except ScriptException as se:
            printf(se, type='ERROR')
            exit()
        self.basic['dns'] = stdout.decode('utf-8').split('\n')[0]
        try:
            stdout, stderr = runScript(CMD['get_instance_type'].format(name=slave_name), output_opt='pipe')
        except ScriptException as se:
            printf(se, type='ERROR')
            exit()
        self.basic['instance_type'] = stdout.decode('utf-8').split('\n')[0]
        master_web_UI = _URL_FMT.format(self.basic['dns'],'')
        r_pub = urlreq.urlopen(master_web_UI)
        self.master_soup = BeautifulSoup(r_pub, 'html.parser')
        t_worker = self.master_soup.find_all('tbody')[0]
        self.basic['num_workers'] = len(t_worker.find_all('tr'))
        self.app_list = []

    def __str__(self):
        return ("BASIC INFO:\n"
                "-----------\n"
                "  DNS:       {dns}\n"
                "  clt size:  {clt_size}\n"
                "  instance:  {ins_type}\n"
                "\n"
                "APP INFO:\n"
                "---------\n"
                "* {app_list}")\
                    .format(dns=self.basic['dns'], clt_size=self.basic['num_workers'],
                        ins_type=self.basic['instance_type'],
                        app_list='\n* '.join(map(lambda _: '\n  '.join(str(_).split('\n')), self.app_list)))
            
    def set_app_list(self, start_idx=0, end_idx=-1):
        """
        get app id by specifying the starting and end index
        """
        app_table = self.master_soup.find_all('tbody')[2]
        num_app = len(app_table.find_all('tr'))
        end_idx = (end_idx < 0) and (num_app-1) or end_idx
        count = 0
        self.app_list = []
        for app in app_table.find_all('tr'):
            if count > end_idx:
                break
            if count >= start_idx:
                a = app_profile(app, self.basic['dns'])
                a.set_jobs()
                a.set_stages()
                a.parse_stages()
                self.app_list += [a]
            count += 1

    def data_to_db(self, db_name='clt_profiling.db'):
        attr_name = ['clt_size', 'node_type', 'num_cores', 'mem_per_node', 
                'app_id', 'app_name', 'data_size', 'num_partition', 'itr', 'tot_dur', 'stage_id', 'descp'] + METRIC_NAME
        attr_type = ['INTEGER', 'TEXT', 'INTEGER', 'REAL', 
                'INTEGER', 'TEXT', 'INTEGER', 'INTEGER', 'INTEGER', 'REAL', 'INTEGER', 'TEXT'] + ['REAL']*len(METRIC_NAME)
        clt_data = [self.basic['num_workers'], self.basic['instance_type']]
        for a in self.app_list:
            name_decompose = a.name['name'].split('-')
            a_name = name_decompose[0]
            a_dsize = int(name_decompose[1].split('_')[1].split('.')[0])
            a_partition = int(name_decompose[2].split('_')[1])
            a_itr = int(name_decompose[3].split('_')[1])
            app_data = [a.cores, a.mem_per_node, a.id['name'], a_name, a_dsize, a_partition, a_itr, a.duration]
            stage_data = None
            for k in METRIC_NAME:
                assert a.data[k] is not None
                data_T = a.data[k].reshape(-1,1)
                stage_data = ((stage_data is None) and [data_T] \
                        or [np.concatenate((stage_data,data_T), axis=1)])[0]
            desp_data = np.array([s.description['name'] for s in a.stage_list]).reshape(-1,1)
            sID_data = np.arange(desp_data.shape[0]).reshape(-1,1)
            db.populate_db(attr_name, attr_type, clt_data, app_data, sID_data, desp_data, stage_data, db_name=db_name)
                



if __name__ == '__main__':
    args = parse_args()
    cluster = clt_profile(args.cluster_name)
    cluster.set_app_list(args.start_idx, args.end_idx)
    # printf(str(cluster), type='', separator='-')
    cluster.data_to_db()
