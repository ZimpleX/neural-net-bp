CMD = {
    'get_dns': """
            name='"{name}"'
            id=$(aws ec2 describe-instances | grep 'InstanceId' | awk '{{print $2}}')
            for i in $id
            do
                i=$(echo $i | cut -d'"' -f2)
                k=$(aws ec2 describe-instances --instance-ids $i | grep $name)
                if [ "$k" != '' ]
                then
                    dns=$(aws ec2 describe-instances --instance-ids $i | grep 'PublicDnsName' | awk 'NR==1' | cut -d'"' -f4)
                    echo $dns
                    exit
                fi
            done

    """,
    'get_instance_type': """
            name='"{name}"'
            id=$(aws ec2 describe-instances | grep 'InstanceId' | awk '{{print $2}}')
            for i in $id
            do
                i=$(echo $i | cut -d'"' -f2)
                k=$(aws ec2 describe-instances --instance-ids $i | grep $name)
                if [ "$k" != '' ]
                then
                    dns=$(aws ec2 describe-instances --instance-ids $i | grep 'InstanceType' | awk 'NR==1' | cut -d'"' -f4)
                    echo $dns
                    exit
                fi
            done

    """,
    'tar_z': """
            tar -zcf temp.ignore/temp.tar.gz ../neural-net-bp/ --exclude='.git' --exclude='*ignore*' \
                --exclude='__pycache__' --exclude='*.db' --exclude='*.pyc' --exclude='*.npz'
    """,
    'tar_x': """
            tar -xzf temp.tar.gz
    """,
    'zip': """
            zip -R packed_module.zip "*.py" #--exclude="*.npz" --exclude="*.git*" --exclude="*.db" --exclude="*.pyc" --exclude="*__pycache__*" --exclude="*ignore*"
    """,
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
            scp -rp -i {id} {f} root@{dns}:/root/{to_dir}
    """,
    'pipe_remote': """
            python3 -m ec2.spark_launcher --login --pipe {pipe_args}
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
    'dir_clear': """
            dir=/root/{dir}
            if [ -d $dir ]; then rm -rf $dir; fi
    """,
    'py3_check': """
            py3_path=$(which python3)
            if [[ -z $py3_path ]] || [[ $py3_path =~ '/which:' ]]
            then
                echo 'python3 is not installed! quit.'
                exit
            fi    
    """,
    'submit_spark': """
            master_dns={dns}
            app_home=/root/{name}/
            cd $app_home
            submit_main=$app_home/{main}
            args="{args}"
            PYSPARK_PYTHON=$(which python3) \
            /root/spark/bin/spark-submit --master spark://$master_dns:7077 \
                --conf spark.eventLog.enabled=true --conf spark.executor.memory=2g --conf spark.driver.memory=2g --py-files packed_module.zip $submit_main $args
            #/root/spark/bin/spark-submit /root/spark/examples/src/main/python/pi.py 10
    """,
    'submit_normal': """
            app_home=/root/{name}/
            cd $app_home
            screen -d -m python3 {main} {args}
    """,
    'record_submit_cmd': """
            echo '{cmd}' > debug_submit.sh
            chmod 777 debug_submit.sh
    """,
    'swap_space': """
        cur_swap=$(free -m | grep 'Swap' | awk '{print $2}')
        if ! [[ $cur_swap =~ 40[0-9][0-9] ]]
        then
            sudo /bin/dd if=/dev/zero of=/var/swap.1 bs=1M count=4096
            sudo /sbin/mkswap /var/swap.1
            sudo swapon /var/swap.1
        fi
    """,
    '_': """
        #aws s3 cp s3://bloodcell/3cat_7500.npz /root/raw_bloodcell.npz
        cd neural-net-bp
        python3 -m util.npz_prepare
        #aws s3 cp /root/raw_bloodcell_part*.npz s3://bloodcell/3cat_7500_scaled.npz
    """,
    '__': """
        cd neural-net-bp
        python3 -m test.batch_eval /root/maybegood.npz /root/test_batch_dir/
    """
}   # _ & __ are temporary commands: for net training on non-spark env


