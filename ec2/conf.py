DEFAULT_CREDENTIAL = '../EC2-credential/zimplex0-credentials.csv'
CHILD_SCRIPT = 'ec2/spark-ec2'
DEFAULT_IDENTITY = '../EC2-credential/zimplex0-key-pair-ap-southeast-1.pem'
DEFAULT_SPARK = '../spark-1.6.0-bin-hadoop2.6/'

DEFAULT_NAME = 'unnamed_cluster'

DEFAULT_EC2_ARGS = {'--instance-type': 't2.micro',
                    '--region': 'ap-southeast-1',
                    '--ami': 'ami-12a66771'}

import argparse

def parse_args():
    launch_script = 'spark_launcher.py'
    submit_script = 'spark_submit.py'
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][1]
    if launch_script in caller:
        parser = argparse.ArgumentParser('launch spark in EC2, wrapper of spark-ec2 script')
    elif submit_script in caller:
        parser = argparse.ArgumentParser('submit app to EC2 Spark')
    else:
        from logf.printf import printf
        printf('UNKNOWN caller!', type='ERROR')
        exit()

    parser.add_argument('-c', '--credential_file', type=str, metavar='CREDENTIALS', 
            default=DEFAULT_CREDENTIAL, help='file location ec2 user credential')
    parser.add_argument('-i', '--identity_file', type=str, metavar='IDENTITY',
            default=DEFAULT_IDENTITY, help='identity file to ec2, usually <identity>.pem')

    if launch_script in caller:
        parser.add_argument('-s', '--spark_dir', type=str, metavar='SPARK_DIR',
                default=DEFAULT_SPARK, help='location of spark dir')
        parser.add_argument('-seh', '--spark_ec2_help', action='store_true',
                help='help msg from the spark-ec2 script')
        parser.add_argument('-sef', '--spark_ec2_flag', type=str, metavar='SPARK_EC2_FLAG', default='',
                help='flags passed to spark-ec2 script (wrap by "" or \'\') \
                    \n[NOTE]: don\'t pass credential file and identity file using -sef; \
                              pass them with -c or -i \
                    \n[NOTE]: don\'t contain \'=\' in the arg string')
        parser.add_argument('--launch', action='store_true', help='launch a ec2 cluster of name <CLUSTER_NAME>')
        parser.add_argument('--login', action='store_true', help='login to a cluster')
        parser.add_argument('--destroy', action='store_true', help='destroy cluster, data UNRECOVERABLE!')
        parser.add_argument('--pipe', action='store_true', 
                help='[FOR LOGIN ONLY]: do you want to pipe the input to ec2 master node terminal? \
                        \nwill prompt out an interactive shell to record all cmds to be piped to ec2 shell')

    parser.add_argument('-r', '--region', type=str, metavar='REGION', 
            default=DEFAULT_EC2_ARGS['--region'], help='region where clusters located in')
    parser.add_argument('-n', '--cluster_name', type=str, metavar='CLUSTER_NAME', 
            default=DEFAULT_NAME, help='name of the ec2 cluster')

    if submit_script in caller:
        parser.add_argument('--hdfs', action='store_true', default=False, help='[FOR SUBMIT]: add this flag if no copying from S3')
        parser.add_argument('--clone', action='store_true', default=False, help='[FOR SUBMIT]: add this flag if no clone from git')
        parser.add_argument('--scp', action='store_true', default=False, help='[FOR SUBMIT]: add this flag if no scp of credentials/rc')
        # parser.add_argument('--cnn_slide_method',  type=str, choices=['slide_serial', 'slide_spark'], 
        # default='slide_spark', help='how would you like to perform the sliding window operation for your CNN?')
        parser.add_argument('--main', required=True, type=str, help='py file of main function')
        parser.add_argument('--args_main', type=str, help='args passed to main')
        parser.add_argument('--main_h', action='store_true', help='display help msg for the selected main python file')
    
    return parser.parse_args()


