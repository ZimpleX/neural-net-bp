from ec2.EmbedScript import *
from logf.printf import printf
import re
import ec2.conf as conf
import os

import pdb

_OUTPUT_FORMAT = 'json'
_CUS_BASHRC = 'ec2.bashrc'
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
    'key_id_export': """
            export AWS_SECRET_ACCESS_KEY={secret_key}
            export AWS_ACCESS_KEY_ID={key_id}
    """,
    'scp': """
            scp -i {id} {f} root@{dns}:/root/
    """,
    'pipe_remote': """
            python3 -m ec2.spark_launcher --login --pipe
    """,
    'hdfs_cp': """
            $hdfs_dir/hadoop distcp s3n://{f} hdfs://
    """,
    'hdfs_conf': """
            hdfs_dir={hdfs}
            $hdfs_dir/start-all.sh
            conf_f='/root/ephemeral-hdfs/conf/core-site.xml'
            line=$(grep -n awsAccessKeyId $conf_f | cut -d: -f1)
            sed -i "$((line+1))s/.*/<value>{key_id}<\/value>/" $conf_f
            line=$(grep -n awsSecretAccessKey $conf_f | cut -d: -f1)
            sed -i "$((line+1))s/.*/<value>{secret}<\/value>/" $conf_f
    """,
    'dir_create': """
            dir={dir}
            if [ ! -d $dir ]; then mkdir $dir; fi
    """,
    'dir_clone': """
            dir={dir}
            if [ -d $dir ]; then rm -rf $dir; fi
            git clone --depth=1 -b ec2_spark {dir_git}
    """,
    'py3_check': """
            py3_path=$(which python3)
            if [[ -z $py3_path ]] || [[ $py3_path =~ '/which:' ]]
            then
                echo 'python3 is not installed! quit.'
                exit
            fi    
    """,
    'submit': """
            master_dns={dns}
            app_home=/root/{name}/
            cd $app_home
            submit_main=$app_home/{main}
            args='{args}'
            PYSPARK_PYTHON=$(which python3) \
            /root/spark/bin/spark-submit --master spark://$master_dns:7077 \
                --conf spark.eventLog.enabled=true $submit_main $args
            #/root/spark/bin/spark-submit /root/spark/examples/src/main/python/pi.py 10
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
        if not master_dns:
            printf('You probably have multiple masters of the same cluster name: \nRENAME the obsolete one to avoid confusion', type='ERROR')
        printf("Get {}-master public DNS:\n       {}", cluster_name, master_dns)
        return master_dns
    except ScriptException as se:
        printf(se, type='ERROR')
        exit()


def prepare(id_f, master_dns, credential_f, key_id, secret_key, is_hdfs=True, is_clone=True, is_scp=True):
    try:
        if is_scp:
            for f in [credential_f, 'ec2/'+_CUS_BASHRC]:
                scpScript = _CMD['scp'].format(id=id_f, f=f, dns=master_dns)
                stdout, stderr = runScript(scpScript, output_opt='display', input_opt='display')
                printf(scpScript, type='WARN')

        app_root = _APP_INFO['repo_url'].split('/')[-1].split('.git')[0]
        combineCmd  = []
        combineCmd += [_CMD['source_rc'].format(rc=_CUS_BASHRC)]
        combineCmd += [_CMD['key_id_export'].format(key_id=key_id, secret_key=secret_key)]
        if is_hdfs:
            combineCmd += [_CMD['hdfs_conf']\
                .format(hdfs=_AWS_DIR_INFO['hdfs'], key_id=key_id, secret=secret_key)]
            combineCmd += [_CMD['hdfs_cp'].format(f=_AWS_DIR_INFO['data'])]
        combineCmd += [_CMD['dir_create'].format(dir='/tmp/spark-events/')]
        if is_clone:
            combineCmd += [_CMD['dir_clone'].format(dir=app_root, dir_git=_APP_INFO['repo_url'])]
        combineCmd += [_CMD['py3_check']]
        combineCmd += ['exit\n']
        combineCmd = '\n'.join(combineCmd)
        remoteScript = _CMD['pipe_remote']
        printf(remoteScript)
        
        stdout, stderr = runScript(remoteScript, output_opt='display', input_opt='pipe', input_pipe=[combineCmd, '.quit'])
        return app_root
    except ScriptException as se:
        printf(se, type='ERROR')
        exit()


def submit_application(name, master_dns, slide_method):
    printf('ENTER application submission', type='WARN')
    try:
        app_args = 'convnet_usps1 --slide_method {}'.format(slide_method)
        shot = [_CMD['source_rc'].format(rc=_CUS_BASHRC)]
        shot += [_CMD['submit'].format(dns=master_dns, name=name, main=_APP_INFO['submit_main'], args=app_args)]
        shot += ['exit\n']
        shot = '\n'.join(shot)
        remoteScript = _CMD['pipe_remote']
        stdout, stderr = runScript(remoteScript, output_opt='display', input_opt='pipe', input_pipe=[shot, '.quit'])
        printf('submit: {}', shot, type='WARN')
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
    name = prepare(args.identity_file, master_dns, args.credential_file, key_id, secret_key, 
        is_hdfs=(not args.no_hdfs), is_clone=(not args.no_clone), is_scp=(not args.no_scp))
    submit_application(name, master_dns, args.cnn_slide_method)
