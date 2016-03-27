from ec2.EmbedScript import *
from logf.printf import printf
import re
import ec2.conf as conf
import os


from ec2.cmd import CMD

import pdb

_OUTPUT_FORMAT = 'json'
_CUS_BASHRC = 'ec2.bashrc'
_AWS_DIR_INFO = {
        'spark': '/root/spark/',
        'hdfs': '/root/ephemeral-hdfs/bin/',
        'log': '/root/neural-net-bp/{train_name}/ann.db',
        'data': ['bloodcell/3cat_smaller/*']
}
_APP_INFO = {
        'repo_url': 'https://github.com/ZimpleX/neural-net-bp',
        'name': 'blood_cell_classification_3cat',
        'submit_main': ['conv_unittest.py','main.py','sweep_conv_unittest.py', 'sweep_training.py', 'test/batch_eval.py']
}

def conf_AWS_CLI(credential_f, region):
    """
    setup default values for aws command line interface.
    """
    try:
        scriptGetKey = CMD['key_id_parse'].format(credential_f=credential_f)
        stdout, stderr = runScript(scriptGetKey, output_opt='pipe')
        key_id, secret_key= stdout.decode('utf-8').split('\n')[:-1]
        os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
        os.environ['AWS_ACCESS_KEY_ID'] = key_id
        stdout, stderr = runScript('aws configure', output_opt='display',
            input_opt='pipe', input_pipe=[key_id, secret_key, region, _OUTPUT_FORMAT])
        print()
        printf('AWS-CLI conf done')
        return key_id, secret_key
    except ScriptException as se:
        printf(se, type='ERROR')
        exit()


def get_DNS(name):
    """
    get the ID and public-DNS of node
    """
    try:
        stdout, stderr = runScript(CMD['get_dns'].format(name=name), output_opt='pipe')
        printf('get public dns for {}', name)
        return stdout.decode('utf-8').split('\n')[0]
    except ScriptException as se:
        printf(se, type='ERROR')
        exit()


def get_master_DNS(cluster_name):
    """
    get the ID and public-DNS of master node
    """
    master_dns = ''
    try:
        stdout, stderr = runScript('aws ec2 describe-instances', output_opt='pipe')
        printf('instance info got, target: {}-master', cluster_name)
        master_id_regex = '{}-master-{}'.format(cluster_name, '\S*')
        master_id = re.search(master_id_regex, stdout.decode('utf-8'))
        if not master_id:
            printf('failed to get master-id:\n        check your cluster name / region ...', type='ERROR')
            exit()
        master_id = master_id.group().split('master-')[-1][:-1]
        master_id = master_id.split('"')[0]
        stdout, stderr = runScript('aws ec2 describe-instances --instance-ids {}'.format(master_id), output_opt='pipe')
        master_dns_regex = '"{}": "{}",'.format('PublicDnsName', '\S*')
        master_dns = re.search(master_dns_regex, stdout.decode('utf-8'))\
                        .group().split("\"")[-2]
        if not master_dns:
            printf('You probably have multiple masters of the same cluster name: \nRENAME the obsolete one to avoid confusion', type='ERROR')
        printf("Get {}-master public DNS:\n       {}", cluster_name, master_dns)
        return master_dns
    except ScriptException as se:
        printf(se, type='ERROR')
        exit()


def prepare(id_f, master_dns, credential_f, key_id, secret_key, is_hdfs=True, is_clone=True, is_scp=True, is_s3=True, pipe_args=''):
    try:
        if is_scp:
            # ____
            for f in [credential_f, 'ec2/'+_CUS_BASHRC]:#, 'train_data/usps.npz', 'train_data/3cat_1000_scale.npz']:
                scpScript = CMD['scp'].format(id=id_f, f=f, dns=master_dns, to_dir='')
                stdout, stderr = runScript(scpScript, output_opt='display', input_opt='display')
                printf(scpScript, type='WARN')
        if is_clone:
            cmd  = [CMD['zip']]
            cmd += [CMD['tar_z']]
            cmd += [CMD['scp'].format(id=id_f, f='temp.ignore/temp.tar.gz', dns=master_dns, to_dir='')]
            cmd  = '\n'.join(cmd)
            stdout, stderr = runScript(cmd, output_opt='display', input_opt='display')

        app_root = _APP_INFO['repo_url'].split('/')[-1].split('.git')[0]
        combineCmd  = []
        combineCmd += [CMD['source_rc'].format(rc=_CUS_BASHRC)]
        combineCmd += [CMD['key_id_export'].format(key_id=key_id, secret_key=secret_key)]
        if is_hdfs: # NOTE: for cluster version FF
            # >>>>>
            # not putting it in aws cuz checkpoint is only used together with hdfs
            combineCmd += [CMD['aws_cp'].format(s3_data='spark-ec2-log/blood_cell_classification_3cat/3000/finish.chkpt.npz', loc_des='/root/checkpoint.npz')]
            combineCmd += [CMD['hdfs_conf']\
                .format(hdfs=_AWS_DIR_INFO['hdfs'], key_id=key_id, secret=secret_key)]
            for d in _AWS_DIR_INFO['data']:
                combineCmd += [CMD['hdfs_distcp'].format(f=d)]
        if is_s3:   # NOTE: for serial version training
            combineCmd += [CMD['dir_create'].format(dir='/root/data_part')]
            combineCmd += [CMD['dir_create'].format(dir='/root/data_part/train')]
            combineCmd += [CMD['aws_cp'].format(s3_data='bloodcell/3cat_part/0.npz',loc_des='/root/data_part/train')]
            combineCmd += [CMD['aws_cp'].format(s3_data='bloodcell/3cat_part/1500.npz',loc_des='/root/data_part')]
        combineCmd += [CMD['dir_create'].format(dir='/tmp/spark-events/')]
        if is_clone:    # extract on EC2
            combineCmd += [CMD['tar_x']]
        combineCmd += [CMD['py3_check']]
        combineCmd += [CMD['swap_space']]
        combineCmd += ['exit\n']
        combineCmd = '\n'.join(combineCmd)
        remoteScript = CMD['pipe_remote'].format(pipe_args=pipe_args)
        printf(remoteScript)
        
        stdout, stderr = runScript(remoteScript, output_opt='display', input_opt='pipe', input_pipe=[combineCmd, '.quit'])
        return app_root
    except ScriptException as se:
        printf(se, type='ERROR')
        exit()


def submit_application(name, master_dns, main, args_main, key_id='', secret_key='', pipe_args=''):
    printf('ENTER application submission', type='WARN')
    try:
        shot = [CMD['source_rc'].format(rc=_CUS_BASHRC)]
        if main in ['conv_unittest.py', 'test/batch_eval.py']:
            submit_cmd = CMD['submit_spark'].format(dns=master_dns, name=name, main=main, args=args_main)
        elif main in ['sweep_training.py', 'sweep_conv_unittest.py', 'main.py']:
            submit_cmd = CMD['submit_normal'].format(name=name, main=main, args=args_main)
        shot += [CMD['key_id_export'].format(key_id=key_id,secret_key=secret_key)]
        shot += [CMD['record_submit_cmd'].format(cmd=';'.join(submit_cmd.split('\n')[1:]))]
        shot += [submit_cmd]
        shot += ['exit\n']
        shot = '\n'.join(shot)
        remoteScript = CMD['pipe_remote'].format(pipe_args=pipe_args)
        stdout, stderr = runScript(remoteScript, output_opt='display', input_opt='pipe', input_pipe=[shot, '.quit'])
        printf('submit to spark: {}', shot, type='WARN')
    except ScriptException as se:
        printf(se, type='ERROR')
        exit()
        

def parse_cluster_performance():
    pass

def parse_cnn_result():
    pass       




############################################################
if __name__ == '__main__':
    args = conf.parse_args()
    assert args.main in _APP_INFO['submit_main']
    if args.main_h:
        try:
            stdout, stderr = runScript('python3 {} -h'.format(args.main), output_opt='display') 
        except ScriptException as se:
            printf(se, type='ERROR')
        exit()
    key_id, secret_key = conf_AWS_CLI(args.credential_file, args.region)
    if args.via_cli:
        master_dns = get_DNS(args.cluster_name)
    else:
        master_dns = get_master_DNS(args.cluster_name)
    pipe_args = '-n {}'.format(args.cluster_name)
    if args.via_cli:
        pipe_args += ' --via_cli'
    name = prepare(args.identity_file, master_dns, args.credential_file, key_id, secret_key, 
        is_hdfs=(args.hdfs), is_clone=(args.clone), is_scp=(args.scp), is_s3=(args.s3), pipe_args=pipe_args)
    submit_application(name, master_dns, args.main, args.args_main, 
        key_id=key_id, secret_key=secret_key, pipe_args=pipe_args)
