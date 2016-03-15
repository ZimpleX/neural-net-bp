import ec2.conf as conf
from ec2.EmbedScript import *
from logf.printf import printf
import os

import pdb

_DEF_LAUNCH_ARGS = ['--instance-type', 
                       '--region',
                       '--ami']
_DEF_LOGIN_ARGS = ['--region']
_DEF_DESTROY_ARGS = ['--region']


_CMD = {
    'get_env': """
            credential_file={cred_f}
            ACCESS_KEY_ID=$(cat $credential_file | awk 'NR==2' | awk -F ',' '{{print $(NF-1)}}')
            SECRET_ACCESS_KEY=$(cat $credential_file | awk 'NR==2' | awk -F ',' '{{print $NF}}')
            echo $ACCESS_KEY_ID
            echo $SECRET_ACCESS_KEY
    """,
    'launch': """
        {wrap_script} {wrap_args} launch {name}
    """,
    'login': """
        {wrap_script} {wrap_args} login {name}
    """, 
    'destroy': """
        {wrap_script} {wrap_args} destroy {name}
    """,
    'cli-launch': """
        id=$(aws ec2 run-instances {args} | grep 'InstanceId' | cut -d '"' -f 4)
        aws ec2 create-tags --resources $id --tags Key=Name,Value={ins_name}
        echo $id
    """,
    'cli-login': """
        name='"{name}"'
        id=$(aws ec2 describe-instances | grep 'InstanceId' | awk '{{print $2}}')
        for i in $id
        do
            i=$(echo $i | cut -d'"' -f2)
            echo "searching: ----$i----"
            k=$(aws ec2 describe-instances --instance-ids $i | grep $name)
            if [ "$k" != '' ]
            then
                dns=$(aws ec2 describe-instances --instance-ids $i | grep 'PublicDnsName' | awk 'NR==1' | cut -d'"' -f4)
                echo $dns
                ssh -t -t -i {cred_f} root@$dns
                exit
            fi
        done
    """,
    'cli-destroy': """
        name='"{name}"'
        id=$(aws ec2 describe-instances | grep 'InstanceId' | awk '{{print $2}}')
        for i in $id
        do 
            i=$(echo $i | cut -d'"' -f2)
            echo "searching: ----$i----"
            k=$(aws ec2 describe-instances --instance-ids $i | grep $name)
            if [ "$k" != '' ]
            then
                aws ec2 terminate-instances --instance-ids $i
                exit
            fi
        done
        echo 'NO INSTANCE MATCHING THE NAME'
    """
}


def setup_spark_ec2_flag(args):
    second_lvl_arg_dict = {k:conf.DEFAULT_EC2_SPARK_ARGS[k] for k in mode_keys}
    second_lvl_arg_list = map(lambda i: '{} {}'.format(list(second_lvl_arg_dict.keys())[i], 
                        list(second_lvl_arg_dict.values())[i]), range(len(second_lvl_arg_dict)))
    # NOTE: don't use '+=', cuz if the same arg is overwritten in cmd line, 
    # you need to put the default value prior to the overwritten one
    spark_ec2_flag = ' {} {}'.format(' '.join(second_lvl_arg_list), args.spark_ec2_flag)
    spark_ec2_flag += ' -i {} -k {}' \
        .format(args.identity_file, args.identity_file.split('.pem')[0].split('/')[-1])
    printf('args to spark-ec2 script: \n\t{}',spark_ec2_flag)
    return spark_ec2_flag


def setup_cli_ec2_flag():
    second_lvl_arg_dict = conf.DEFAULT_EC2_CLI_ARGS
    second_lvl_arg_list = map(lambda i: '{} {}'.format(list(second_lvl_arg_dict.keys())[i], 
                        list(second_lvl_arg_dict.values())[i]), range(len(second_lvl_arg_dict)))
    return ' '.join(second_lvl_arg_list)
    


def setup_env(cred_f): 
    try:
        stdout, stderr = runScript(_CMD['get_env'].format(cred_f=cred_f))
        aws_access_key_id = stdout.decode('utf-8').split("\n")[0]
        aws_secret_access_key = stdout.decode('utf-8').split("\n")[1]
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
        printf('environment var set: {}, {}', aws_access_key_id, aws_secret_access_key)
        return aws_access_key_id, aws_secret_access_key
    except ScriptException as se:
        printf(se, type='ERROR')


def launch(wrap_script, wrap_args, name):
    try:
        stdout, stderr = runScript(_CMD['launch'].format(wrap_script=wrap_script, wrap_args=wrap_args, name=name), output_opt='display')
        printf('cluster successfully launched.')
    except ScriptException as se:
        printf(se, type='ERROR')


def _get_input_pipe(input_opt):
    input_pipe = []
    if input_opt == 'pipe':
        printf('enter cmds you want to send to ec2 cluster. type \'.quit\' to finish up.')
        while True:
            new_ip = input('>> ')
            print(new_ip)
            if new_ip != '.quit':
                input_pipe += [new_ip]
            else:
                break
    return input_pipe


def login(wrap_script, wrap_args, name, input_opt):
    """
    input_opt should be either 'pipe' or 'cmd'.
    """
    try:
        input_pipe = _get_input_pipe(input_opt)
        login_scpt = _CMD['login'].format(wrap_script=wrap_script, wrap_args=wrap_args, name=name)
        printf(login_scpt, type='WARN')
        stdout, stderr = runScript(login_scpt,
                output_opt='display', input_opt=input_opt, input_pipe=input_pipe) 
        printf('finish interaction with master node.')
    except ScriptException as se:
        printf(se, type='ERROR')


def destroy(wrap_script, wrap_args, name):
    try:
        stdout, stderr = runScript(_CMD['destroy'].format(wrap_script=wrap_script, wrap_args=wrap_args, name=name), output_opt='display', input_opt='cmd')
        printf('cluster successfully destroyed.')
    except ScriptException as se:
        printf(se, type='ERROR')




def cli_launch(name):
    """
    launch instances with aws-cli:
    i.e.: 
        don't start spark after launching 
    """
    try:
        fmt_args = setup_cli_ec2_flag()
        stdout, stderr = runScript(_CMD['cli-launch'].format(ins_name=name, args=fmt_args), output_opt='display')
        printf('instance(s) successfully launched.')
        return stdout   # should be instance id: i-xxxxxx
    except ScriptException as se:
        printf(se, type='ERROR')


def cli_login(name, cred_f, input_opt):
    try:
        input_pipe = _get_input_pipe(input_opt)
        stdout, stderr = runScript(_CMD['cli-login'].format(name=name, cred_f=cred_f), 
            output_opt='display', input_opt=input_opt, input_pipe=input_pipe)
        printf('finish interaction with instance.')
    except ScriptException as se:
        printf(se, type='ERROR')


def cli_destroy(name):
    """
    destroy the instance with the name
    """
    try:
        stdout, stderr = runScript(_CMD['cli-destroy'].format(name=name), output_opt='display')
        printf('instance successfully destroyed.')
    except ScriptException as se:
        printf(se, type='ERROR')





if __name__=='__main__':
    args = conf.parse_args()
    wrap_script = '{}/ec2/spark-ec2'.format(args.spark_dir)

    if args.spark_ec2_help:
        try:
            printf('\nhelp msg from spark-ec2 script:\n')
            stdout, stderr = runScript('{} -h'.format(wrap_script), output_opt='display')
        except ScriptException as se:
            printf(se, type='ERROR')
        exit()

    # mode_keys are for launching with Spark ONLY
    if args.launch: 
        mode_keys=_DEF_LAUNCH_ARGS
    elif args.login: 
        mode_keys=_DEF_LOGIN_ARGS
    elif args.destroy:
        mode_keys=_DEF_DESTROY_ARGS
    else:
        printf('unknown launch mode', type='ERROR')
        exit()
    

    spark_ec2_flag = setup_spark_ec2_flag(args)
    aws_access_key_id, aws_secret_access_key = setup_env(args.credential_file)

    if args.launch:
        if args.via_cli:
            cli_launch(args.cluster_name)
        else:
            launch(wrap_script, spark_ec2_flag, args.cluster_name)
    elif args.login:
        ip_opt = args.pipe and 'pipe' or 'cmd'
        if args.via_cli:
            cli_login(args.cluster_name, args.identity_file, ip_opt)
        else:
            login(wrap_script, spark_ec2_flag, args.cluster_name, ip_opt)
    elif args.destroy:
        if args.via_cli:
            cli_destroy(args.cluster_name)
        else:
            destroy(wrap_script, spark_ec2_flag, args.cluster_name)
