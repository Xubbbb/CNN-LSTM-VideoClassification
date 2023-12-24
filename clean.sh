#!/bin/bash

# 删除 ./snapshots 下的所有文件和目录，除了 weightfile.txt
find ./snapshots -mindepth 1 -not -name 'weightfile.txt' -delete

# 清空 ./tf_logs 目录下的所有文件和目录
rm -rf ./tf_logs/*

echo "Cleanup completed."