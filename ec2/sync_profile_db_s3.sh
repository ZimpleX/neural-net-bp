app_home='/root/neural-net-bp/'
timestamp_f='update_time'
cd $app_home
db_dir_wildcard='cell_kelvin_*'
db_dir='./profile_data/'$db_dir_wildcard
db_full_dir=$(ls profile_data/ | grep "^cell_kelvin_.*")
if ! [ -d $db_dir ]
then
    exit
fi

# set up env var
cred_f='/root/zimplex0-credentials.csv'
export AWS_ACCESS_KEY_ID=$(cat $cred_f | awk 'NR==2' | awk -F ',' '{print $(NF-1)}')
export AWS_SECRET_ACCESS_KEY=$(cat $cred_f | awk 'NR==2' | awk -F ',' '{print $NF}')


cur_t="$(ls -lt $db_dir | awk 'NR==2')"
prev_t=''

if [ -f $timestamp_f ]
then
    prev_t="$(cat $timestamp_f)"
fi

if [ "$cur_t" != "$prev_t" ]
then
    echo "$cur_t" > $timestamp_f
    echo update
    aws s3 cp 'profile_data/'$db_full_dir'/ann.db' s3://spark-ec2-log/temp/$db_full_dir/
fi
    
