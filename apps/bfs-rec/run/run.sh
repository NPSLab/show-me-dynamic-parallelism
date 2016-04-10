#!/bin/bash

BIN=../src/gpu-bfs-rec
#DATA_FILE=../../../datasets/DIMACS9/USA-road-d.COL.gr
#DATA_FILE=../../../datasets/DIMACS10/kron_g500-simple-logn18.graph
DATA_FILE=../../../datasets/DIMACS10/kron_g500-simple-logn16.graph
#DATA_FILE=../../../datasets/SLNDC/wiki-Vote.txt
#DATA_FILE=../../../datasets/SLNDC/amazon0505.txt
LOG_FILE=bfs_rec_performance.log

for i in 64
do
#	EXE=$BIN-t$i
	EXE=$BIN
	$EXE -f 1 -i ${DATA_FILE} -v -s 0 #>> ${LOG_FILE}
	$EXE -f 1 -i ${DATA_FILE} -v -s 4 #>> ${LOG_FILE}
	$EXE -f 1 -i ${DATA_FILE} -v -s 5 #>> ${LOG_FILE}
done
