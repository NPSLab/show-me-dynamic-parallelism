#!/bin/bash

BIN=../src/gpu/gpu-graph-color
DATA_FILE=../../../datasets/DIMACS10/coPapersCiteseer.graph
#DATA_FILE=../../../datasets/SLNDC/wiki-Vote.txt
LOG_FILE=sp_performance.log

EXE=$BIN
#$EXE -f 1 -i ${DATA_FILE} -v -s 0
#$EXE -f 1 -i ${DATA_FILE} -v -s 1
#$EXE -f 1 -i ${DATA_FILE} -v -s 2
$EXE -f 1 -i ${DATA_FILE} -v -s 3
$EXE -f 1 -i ${DATA_FILE} -v -s 4
#$EXE -f 1 -i ${DATA_FILE} -v -d -s 5
#$EXE -f 1 -i ${DATA_FILE} -v -d -s 6
#$EXE -f 1 -i ${DATA_FILE} -v -s 7
$EXE -f 1 -i ${DATA_FILE} -v -s 8
$EXE -f 1 -i ${DATA_FILE} -v -s 9
#$EXE -f 1 -i ${DATA_FILE} -v -s 10
#$EXE -f 1 -i ${DATA_FILE} -v -s 11
#$EXE -f 1 -i ${DATA_FILE} -v -s 12
