"""
Don't need to configure AWS CLI here, cuz I assume user will call
spark launcher / submit scripts before this parser. 

In this case, CLI will have already been configured.
"""

import argparse
from bs4 import BeautifulSoup
import urllib.request as urlreq

from ec2.cmd import CMD
from logf.printf import printf

from ec2.EmbedScript import *
import pdb



def parse_args():
    parser = argparse.ArgumentParser('parse from web UI')
    parser.add_argument('-n', '--cluster_name', type=str, required=True, help='name of the spark cluster')
    # parser.add_argument('')
    return parser.parse_args()



def time_conv(timeStr):
    """
    format time to be 'second':
        recognize 'ms' / 'mS' / 'min' / 'Min'
    """
    unit = timeStr.split()[-1]
    val = float(timeStr.split()[0])
    if unit == 'ms' or unit == 'mS':
        return val*0.001
    elif unit == 'min' or unit == 'Min':
        return val*60.
    elif unit == 'h':
        return val*3600.
    else:
        return val



class app_profile:
    def __init__(self, app_html_entry):
        self.id = app_html_entry.find_all('td')[0].a.get_text()
        self.name = app_html_entry.find_all('td')[1].a.get_text()
        self.cores = int(app_html_entry.find_all('td')[2].get_text())
        self.mem_per_node = app_html_entry.find_all('td')[3]['sorttable_customkey']     # measured in terms of Mega bytes
        self.submitted_time = app_html_entry.find_all('td')[4].get_text()
        self.state = app_html_entry.find_all('td')[6].get_text()
        self.duration = time_conv(app_html_entry.find_all('td')[7].get_text())

    def __str__(self):
        return ("appID:       {id}\n"
                "appName:     {name}\n"
                "numCores:    {cores}\n"
                "memPerNode:  {mem}\n"
                "submitTime:  {time}\n"
                "state:       {state}\n"
                "duration:    {dur}")\
                    .format(id=self.id, name=self.name, cores=self.cores,
                        mem=self.mem_per_node, time=self.submitted_time,
                        state=self.state, dur=self.duration)

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
        master_web_UI = 'http://{}:8080'.format(self.basic['dns'])
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
            if count >= start_idx:
                self.app_list += [app_profile(app)]
            if count > end_idx:
                break
            count += 1




if __name__ == '__main__':
    args = parse_args()
    cluster = clt_profile(args.cluster_name)
    cluster.set_app_list(0,1)
    printf(str(cluster), type='', separator='-')
