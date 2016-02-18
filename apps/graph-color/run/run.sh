#!/bin/bash

BIN=../src/gpu-graph-color
DATA_FILE=../../../datasets/DIMACS10/coPapersCiteseer.graph
#DATA_FILE=../../../data/soc-LiveJournal1.txt
LOG_FILE=sp_performance.log

EXE=$BIN
#$EXE -f 1 -i ${DATA_FILE} -v -s 0 &>> ${LOG_FILE}
#$EXE -f 1 -i ${DATA_FILE} -v -s 1 &>> ${LOG_FILE}
#$EXE -f 1 -i ${DATA_FILE} -v -s 2 &>> ${LOG_FILE}
#$EXE -f 1 -i ${DATA_FILE} -v -s 3 &>> ${LOG_FILE}
$EXE -f 1 -i ${DATA_FILE} -v -s 4 #&>>  ${LOG_FILE}

