from ec2.EmbedScript import *
from logf.printf import printf
import re
import ec2.conf as conf
import os

import pdb

_OUTPUT_FORMAT = 'json'
_AWS_DIR_INFO = {
        'spark': '/root/spark/',
        'hdfs': '/root/ephemeral-hdfs/bin/',
        'log': '/root/neural-net-bp/{train_name}/ann.db',
        'data': 'bloodcell/3cat_900_scale.npz'
}
_APP_INFO = {
        'repo_url': 'https://github.com/ZimpleX/neural-net-bp',
        'name': 'blood_cell_classification_3cat',
        'submit_main': 'net_structure.py'
}
_CMD = {
    'source_rc': """
            . /root/{rc}
    """,
    'key_id_parse': """
            credential_f={credential_f}
            ACCESS_KEY_ID=$(cat $credential_f | awk 'NR==2' | awk -F ',' '{{print $(NF-1)}}')
            SECRET_ACCESS_KEY=$(cat $credential_f | awk 'NR==2' | awk -F ',' '{{print $NF}}')
            export AWS_SECRET_ACCESS_KEY=$SECRET_ACCESS_KEY
            export AWS_ACCESS_KEY_ID=$ACCESS_KEY_ID
            echo $ACCESS_KEY_ID
            echo $SECRET_ACCESS_KEY
    """,
    'scp': """
            scp -i {id} {f} root@{dns}:/root/
    """,
    'pipe_remote': """
            python3 -m ec2.ec2_spark_launcher --login {script} --pipe
    """,
    'hdfs_cp': """
            echo ......
            echo $(pwd)
            hdfs_dir={hdfs}
            $hdfs_dir/start-all.sh
            $hdfs_dir/hadoop distcp s3n://{f} hdfs://
    """,
    'hdfs_conf': """
            conf_f='/root/ephemeral-hdfs/conf/core-site.xml'
            line=$(grep -n awsAccessKeyId $conf_f | cut -d: -f1)
            sed -i "$((line+1))s/.*/<value>{key_id}</value>/" $conf_f
            line=$(grep -n awsSecretAccessKey $conf_f | cut -d: -f1)
            sed -i "$((line+1))s/.*/<value>{secret}</value>/" $conf_f
    """,
    'dir_create': """
            dir={dir}
            if [ ! -d $dir ]; then mkdir $dir; fi
    """,
    'dir_clone': """
            dir={dir}
            if [ -d $dir ]; then rm -rf $dir; fi
            git clone {dir_git}
    """,
    'py3_check': """
            . .bashrc   # set env var
            py3_path=$(which python3)
            if [[ -z $py3_path ]] || [[ $py3_path =~ '/which:' ]]
            then
                echo 'python3 is not installed! quit.'
                exit
            fi    
    """
}


def conf_AWS_CLI(credential_f, region):
    """
    setup default values for aws command line interface.
    """
    try:
        scriptGetKey = _CMD['key_id_parse'].format(credential_f=credential_f)
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
        printf("Get {}-master public DNS:\n       {}", cluster_name, master_dns)
        return master_dns
    except ScriptException as se:
        printf(se, type='ERROR')
        exit()


def prepare(id_f, master_dns, credential_f, key_id, secret_key):
    try:
        for f in [credential_f, 'ec2/ec2.bashrc']:
            # TODO: ec2.bashrc is not sourced
            scpScript = _CMD['scp'].format(id=id_f, f=f, dns=master_dns)
            stdout, stderr = runScript(scpScript, output_opt='display', input_opt='display')

        app_root = _APP_INFO['repo_url'].split('/')[-1].split('.git')[0]
        combineCmd  = []
        combineCmd += [_CMD['source_rc'].format(rc='ec2.bashrc')]
        combineCmd += [_CMD['key_id_parse'].format(credential_f=credential_f)]
        # TODO: hdfs set up aws credential for cp from S3
        combineCmd += [_CMD['hdfs_cp'].format(hdfs=_AWS_DIR_INFO['hdfs'], f=_AWS_DIR_INFO['data'])]
        combineCmd += [_CMD['hdfs_conf'].format(key_id=key_id, secret=secret_key)]
        combineCmd += [_CMD['dir_create'].format(dir='/tmp/spark-events/')]
        combineCmd += [_CMD['dir_clone'].format(dir=app_root, dir_git=_APP_INFO['repo_url'])]
        combineCmd += [_CMD['py3_check']]
        combineCmd = '\n'.join(combineCmd)
        remoteScript = _CMD['pipe_remote'].format(script=combineCmd)
        
        stdout, stderr = runScript(remoteScript, output_opt='display', input_opt='display')
    except ScriptException as se:
        printf(se, type='ERROR')
        exit()


def submit_application(master_dns):
    try:
        submit_main = '/root/{}/{}'.format(_APP_INFO['name'], _APP_INFO['submit_main'])
        log_dir = _AWS_DIR_INFO['log'].format(train_name=_APP_INFO['name'])
        spark_dir = _AWS_DIR_INFO['spark']
        shot = '/root/{name}/ec2/fire_and_leave {dns} {main} {args}'\
                .format(name=_APP_INFO['name'], dns=master_dns, main=_APP_INFO['submit_main'], args='')
        remoteScript = _CMD['pipe_remote'].format(script=shot)
        stdout, stderr = runScript(remoteScript, output_opt='display', input_opt='display')
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
    key_id, secret_key = conf_AWS_CLI(args.credential_file, args.region)
    master_dns = get_master_DNS(args.cluster_name)
    prepare(args.identity_file, master_dns, args.credential_file, key_id, secret_key)
    #submit_application(master_dns)
