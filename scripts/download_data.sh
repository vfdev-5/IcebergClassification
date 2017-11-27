#!/bin/sh

echo "Download data to path. Environment variables should be defined: USERNAME (kaggle username), PASSWD (kaggle password)"

# Test if there is kaggle-cli
retcode=$(kg --version >> /dev/null ; echo $?)

if [ $retcode != "0" ]; then
    echo "Kaggle-cli is not found. Probably it is not installed: pip install kaggle-cli"
    exit 1
fi

if [ -z "$1" ]; then
    echo "No output path supplied. run sh download_data.sh /path/to/where/to/download"    
    exit 1
fi

current_path=$PWD
output_path=$1

# check if exists
retcode=$(ls $output_path >> /dev/null ; echo $?)

if [ $retcode != "0" ]; then
    echo "Create output folder : ${output_path}"
    mkdir -p ${output_path}
fi

cd $output_path

kg download -u $USERNAME -p $PASSWD -c "statoil-iceberg-classifier-challenge"

cd $current_path
