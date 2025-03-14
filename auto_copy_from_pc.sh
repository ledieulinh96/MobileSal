#!/bin/bash

# Define remote server information
REMOTE_HOST="root@163.180.117.216"
REMOTE_BASE_PATH="/root/linhle/DL/MobileSal/MobileSal/"  # Adjust this path to the base directory on the remote server

# Define local destination directory
LOCAL_DEST="/path/to/local/directory"

# Generate a timestamp and create a new local directory with that timestamp
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
LOCAL_DEST="/D:/DL/MobileSal/$TIMESTAMP"
mkdir -p $LOCAL_DEST

# Use scp with the -r flag to copy directories recursively
scp -r $REMOTE_HOST:$REMOTE_BASE_PATH/maps $LOCAL_DEST
scp -r $REMOTE_HOST:$REMOTE_BASE_PATH/models $LOCAL_DEST
scp -r $REMOTE_HOST:$REMOTE_BASE_PATH/snapshot/train $LOCAL_DEST

echo "Files have been copied to $LOCAL_DEST successfully.

# Use scp with the -r flag to copy directories recursively
scp -r  -P 7877 $REMOTE_HOST:$REMOTE_BASE_PATH/maps $LOCAL_DEST
scp -r  -P 7877 $REMOTE_HOST:$REMOTE_BASE_PATH/models $LOCAL_DEST
scp -r  -P 7877 $REMOTE_HOST:$REMOTE_BASE_PATH/snapshot/train $LOCAL_DEST

echo "Files have been copied successfully."
