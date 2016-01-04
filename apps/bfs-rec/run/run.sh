#!/bin/bash

BIN=../src/gpu/gpu-bfs-rec
#DATA_FILE=../../../datasets/DIMACS9/USA-road-d.COL.gr
#DATA_FILE=../../../datasets/DIMACS10/coPapersCiteseer.graph
DATA_FILE=../../../datasets/SLNDC/wiki-Vote.txt
#DATA_FILE=../../../datasets/SLNDC/amazon0505.txt
LOG_FILE=bfs_rec_performance.log

for i in 64
do
#	EXE=$BIN-t$i
	EXE=$BIN
	$EXE -f 2 -i ${DATA_FILE} -v -s 0 #>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 1 #>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 2 #>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 3 #>> ${LOG_FILE}
done
