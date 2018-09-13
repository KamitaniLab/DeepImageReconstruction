#!/bin/bash
#
# Download demo data for Deep Image Reconstruction
#

FILE_LIST=file_list.csv

for line in `cat $FILE_LIST`; do
    fname=$(echo $line | cut -d, -f1)
    fid=$(echo $line | cut -d, -f2)
    checksum=$(echo $line | cut -d, -f3)

    ext=${fname#*.}

    # Downloaded file check
    if [ -f $fname ]; then
        echo "$fname has already been downloaded."
        continue
    fi

    if [ "$ext" = "zip" ]; then
        unzipped_path=$(echo ${fname%.zip} | sed s%-%/%g)
        if [ -d $unzipped_path ]; then
            echo "$unzipped_path has already been downloaded."
            continue
        fi
    fi

    # Download file
    echo "Downloading $fname"

    dlurl=https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/${fid}/$(echo $fname | sed s/-//g)
    echo $dlurl
    curl -o $fname $dlurl

    # Validate the downloaded file
    if [ "$OSTYPE" = "linux-gnu" ]; then
        checksum_dl=$(md5sum $fname | awk '{print $1}')
    else
        checksum_dl=$(md5 -q $fname)
    fi

    if [ "$checksum" != "$checksum_dl" ]; then
        echo "Downloaded file is invalid!"
        exit 1
    fi

    # Unzip file
    if [ "$ext" = "zip" ]; then
        unzip $fname
        rm -f $fname
    fi

    echo ""
done
