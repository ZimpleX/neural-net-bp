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
    """
}


def setup_spark_ec2_flag(args):
    second_lvl_arg_dict = {k:conf.DEFAULT_EC2_ARGS[k] for k in mode_keys}
    second_lvl_arg_list = map(lambda i: '{} {}'.format(list(second_lvl_arg_dict.keys())[i], 
                        list(second_lvl_arg_dict.values())[i]), range(len(second_lvl_arg_dict)))
    # NOTE: don't use '+=', cuz if the same arg is overwritten in cmd line, 
    # you need to put the default value prior to the overwritten one
    spark_ec2_flag = ' {} {}'.format(' '.join(second_lvl_arg_list), args.spark_ec2_flag)
    spark_ec2_flag += ' -i {} -k {}' \
        .format(args.identity_file, args.identity_file.split('.pem')[0].split('/')[-1])
    printf('args to spark-ec2 script: \n\t{}',spark_ec2_flag)
    return spark_ec2_flag


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


def login(wrap_script, wrap_args, name, input_opt):
    """
    input_opt should be either 'pipe' or 'cmd'.
    """
    try:
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
        stdout, stderr = runScript(_CMD['login'].format(wrap_script=wrap_script, wrap_args=wrap_args, name=name),
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
        launch(wrap_script, spark_ec2_flag, args.cluster_name)
    elif args.login:
        ip_opt = args.pipe and 'pipe' or 'cmd'
        login(wrap_script, spark_ec2_flag, args.cluster_name, ip_opt)
    elif args.destroy:
        destroy(wrap_script, spark_ec2_flag, args.cluster_name)
