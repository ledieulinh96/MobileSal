#!/bin/bash

# Define local machine information
LOCAL_HOST="linhle@163.180.117.34"
LOCAL_BASE_PATH="/D:/DL/MobileSal/"

# Define local destination directory
LOCAL_DEST="/path/to/local/directory"

# Generate a timestamp and create a new local directory with that timestamp
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
LOCAL_DEST="$LOCAL_BASE_PATH/$TIMESTAMP"
ssh $LOCAL_HOST "mkdir -p $LOCAL_DEST"  # Make directory on local machine

# Use scp with the -r flag to copy directories recursively to the local machine
scp -r ./maps $LOCAL_HOST:$LOCAL_DEST
scp -r ./models $LOCAL_HOST:$LOCAL_DEST
scp -r ./snapshot/train $LOCAL_HOST:$LOCAL_DEST

echo "Files have been pushed to local machine successfully."
