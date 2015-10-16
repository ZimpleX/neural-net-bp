"""
this script is to generate training data set for LogReg benchmark
it uses a mix of shell script and python script, as i think shell script is
more concise with regard to directory formatting

also note that the policy regarding data set is:
    never overwrite a data set file again --> to ensure that different runs of benchmark use the same data set


adapted from original version:
    https://github.com/ZimpleX/spark-benchmark-python/blob/500b285aca981ccc4b817d0e5364d0bcd737735e/LogReg/DataGen.py
"""

# TODO: auto generate conf in data dir

from __future__ import print_function
from util.embed_shell_script import runScript, ScriptException
from util.training_data_func import trainingFunc
from random import uniform
import conf
from node_activity import activation_dict
from cost import cost_dict
import os
import argparse
from functools import reduce
import pdb

trainingDirName = conf.TRAINING_DIR
dataSetSizeStart = 3
dataSizeDefault = 12
inputSizeDefault = 3
outputSizeDefault = 1
funcDefault = 'Sigmoid'

inputSizeRangeDefault = range(1,11)
outputSizeRangeDefault = range(1, 11)
dataSizeRangeDefault = range(3,15)
funcChoices = ['Sigmoid',
               'AttenSin',
               'AttenSin-x0',
               'AttenSin-abs-x0',
               'Random',
               'ANN-bp']

def parseArg():
    parser = argparse.ArgumentParser("generating training data for ANN")
    parser.add_argument("-is", "--input_size", type=int, metavar='M', 
            choices=inputSizeRangeDefault, default=inputSizeDefault, 
            help="specify the num of input to the ANN")
    parser.add_argument("-os", "--output_size", type=int, metavar='S',
            choices=outputSizeRangeDefault, default=outputSizeDefault,
            help="specify the num of output to the ANN")
    parser.add_argument("-ds", "--data_size", type=int, metavar='N', 
            choices=dataSizeRangeDefault, default=dataSizeDefault, 
            help="specify the size of data set (in terms of 2^N)")
    parser.add_argument("-f", "--function", type=str, 
            choices=funcChoices, default=funcDefault, 
            help="Specify the training function to gen the data set")

    # following args are only for training func of ANN-bp
    parser.add_argument("--struct", type=int, metavar='NET_STRUCT', nargs='+',
            default=conf.STRUCT, help='[ANN-bp]: specify the structure of the ANN (num of nodes in each layer)')
    parser.add_argument('--activation', type=str, metavar='NET_ACTIVATION',
            default=conf.ACTIVATION, nargs='+', help='[ANN-bp]: specify the activation of each layer',
            choices=activation_dict.keys())
    parser.add_argument('--cost', type=str, metavar='COST', 
            default=conf.COST, help='[ANN-bp]: specify the cost function',
            choices=cost_dict.keys())

    return parser.parse_args()



def dataGeneratorMain(args):
    taskName = args.function
    inputSize = args.input_size
    outputSize = args.output_size
    if args.function == 'ANN-bp':
        inputSize = args.struct[0]
        outputSize = args.struct[-1]
    dataSetSize = args.data_size
    assert inputSize > 0 and inputSize <= 10
    assert dataSetSize <= 14 and dataSetSize >= 3
    #################################
    #    format the data set dir    #
    #################################
    try:
        # shell script to be run as subprocess
        scriptFormatDir = """
            set -eu
            # setup data set dir
            orig_dir="`pwd`/"
            dir_name={}
            task_name="$0"
            ip_size="$1"
            set_size_pow="$2"
            set_size_pow_start=0"$3"
            if [ ! -d $orig_dir$dir_name ]
            then
                mkdir $orig_dir$dir_name
                echo "created dir: $orig_dir$dir_name"
            fi
            if [ ! -d $orig_dir$dir_name$task_name ]
            then
                mkdir $orig_dir$dir_name$task_name
                echo "created dir: $orig_dir$dir_name$task_name"
            fi
            echo "input size: "$ip_size
            echo "set pow size: "$set_size_pow

            task_dir=$orig_dir$dir_name$task_name
            ##################
            cd $task_dir     #
            ##################
            existing_set="`ls`"
            for size_pow in $(eval echo "{{$set_size_pow_start..$set_size_pow}}")
            do
                file_name=$size_pow
                if [ "`find . -maxdepth 1 -type f -printf '%f\n' | grep $file_name`" ]
                then 
                    echo "data set already exists: "$file_name    
                else
                    touch $file_name
                    echo "created data set file: "$file_name
                fi
            done
            ##################
            cd $orig_dir     #
            ##################
        """.format(trainingDirName)
        # args to the script:
        #   $0: name for the training set (e.g.: sin: the data set should show characteristic of sin function)
        #   $1: number of input to the sigmoid neuron
        #   $2: indicate number of tuples in the data-set:
        #       range: 3 ~ 14
        #       e.g.:
        #           suppose user provide 10, then data set with size 2^3, 2^4, 2^5, ... 2^10 will be generated,
        #           and be stored as separate files in the corresponding folder
        #   $3: indicate min number of tuples in the data-set
        if args.function == 'ANN-bp':
            structStr = reduce(lambda a,b: '{}-{}'.format(a,b), args.struct)
            actStr = reduce(lambda a,b: '{}-{}'.format(a,b), args.activation)
            costStr = args.cost
            taskName += '_struct-{}_act-{}_cost-{}'.format(structStr, actStr, costStr)
        else:
            taskName += '_in-{}-out-{}'.format(inputSize, outputSize)
        stdout, stderr = runScript(scriptFormatDir, [taskName, str(inputSize), str(dataSetSize), str(dataSetSizeStart)])
        print("============================")
        print("script msg: \n{}".format(stdout.decode('ascii')))
        print("============================")
    except ScriptException as se:
        print(se)


    #########################################
    #    generate data and write to file    #
    #########################################
    genY = trainingFunc(args.function)
    for dFile in os.listdir(trainingDirName + taskName):
        dFileFull = trainingDirName + taskName + '/' + dFile
        # all old files should already be read-only
        if not os.access(dFileFull, os.W_OK):
            continue
        if 'conf' in dFile or 'ignore' in dFile:
            continue
        dSize = dFile
        dSize = int(dSize)
        assert dSize >= dataSetSizeStart and dSize <= dataSetSize
        assert os.stat(dFileFull).st_size == 0
        f = open(dFileFull, 'w')
        numEntry = pow(2, dSize)
        for i in range(0, numEntry):
            # randomly generate input list, within range 0 ~ 10
            xList = [uniform(0, 10) for k in range(0, inputSize)]
            yList = None
            if args.function == 'ANN-bp':
                yList = genY(xList, args.struct, args.activation, args.cost)
            else:
                yList = genY(xList)
            dataList = yList + xList
            dataStr = reduce(lambda x,y: str(x)+' '+str(y), dataList)
            print(dataStr, file=f)
        f.close()
    

    ##########################################################
    #    always enforce read-only policy for data set dir    #
    ##########################################################
    try:
        scriptChmod = """
            orig_dir="`pwd`/"
            dir_name={}
            task_name="$0"
            chmod 444 $orig_dir$dir_name$task_name/*
        """.format(trainingDirName)
        stdout, stderr = runScript(scriptChmod, [taskName])
    except ScriptException as se:
        print(se)


    ####################
    #    write conf    #
    ####################
    try:
        scriptEchoConf = """
            orig_dir="`pwd`/"
            dir_name={}
            task_name="$0"/
            conf_full=$orig_dir$dir_name$task_name/conf
            conf_str="$1"
    
            if [ ! -f $conf_full ]
            then
                echo $conf_str > $conf_full
            fi
        """.format(trainingDirName)
        confStr = [conf.TARGET]*outputSize + [conf.INPUT]*inputSize
        confStr = reduce(lambda a,b:'{} {}'.format(a,b), confStr)
        stdout, stderr = runScript(scriptEchoConf, [taskName, confStr])
    except ScriptException as se:
        print(se)
    
    return trainingDirName + taskName + '/'


if __name__ == '__main__':
    args = parseArg()
    dataGeneratorMain(args)
