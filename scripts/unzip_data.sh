#!/bin/sh

echo "Unzip data stored at ../input"

# Test if there is kaggle-cli
retcode=$(7z >> /dev/null ; echo $?)

if [ $retcode != "0" ]; then
    echo "7zip is not found. Probably it is not installed"
    exit 1
fi

current_path=$PWD
input_path=$PWD/../input

# check if exists
retcode=$(ls $input_path >> /dev/null ; echo $?)

if [ $retcode != "0" ]; then
    echo "Not found data at $input_path"
    exit 1
fi

cd $input_path

7z e sample_submission.csv.7z
7z e test.json.7z
7z e train.json.7z

cd $current_path
