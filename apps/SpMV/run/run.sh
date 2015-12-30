#!/bin/bash

BIN=../src/gpu/gpu-SpMV
DATA_FILE=../../../datasets/DIMACS10/coPapersCiteseer.graph
LOG_FILE=spmv_performance.log

#for i in 16 32 64 128 256 512 1024
#for i in 16 32 64 128
for i in 64
do
	EXE=$BIN-t$i
#	echo "Threshold: $i"  >> ${LOG_FILE}
#	$EXE -f 1 -i ${DATA_FILE} -v -s 0 &>> ${LOG_FILE}
#	$EXE -f 1 -i ${DATA_FILE} -v -s 1 &>> ${LOG_FILE}
#	$EXE -f 1 -i ${DATA_FILE} -v -s 2 &>> ${LOG_FILE}
#	$EXE -f 1 -i ${DATA_FILE} -v -s 3 &>> ${LOG_FILE}
#	$EXE -f 1 -i ${DATA_FILE} -v -s 4 &>>  ${LOG_FILE}
#	$EXE -f 1 -i ${DATA_FILE} -v -s 5 &>> ${LOG_FILE}
	$EXE -f 1 -i ${DATA_FILE} -v -s 6 &>> ${LOG_FILE}
done

#DATA_FILE=../../../data/soc-LiveJournal1.txt

#for i in 16	32 64 128 256 512 1024
#do
#	EXE=$BIN-t$i
#	echo "Threshold: $i"  >> ${LOG_FILE}
#	$EXE -f 2 -i ${DATA_FILE} -s 0 &>> ${LOG_FILE}
#	$EXE -f 2 -i ${DATA_FILE} -s 1 &>> ${LOG_FILE}
#	$EXE -f 2 -i ${DATA_FILE} -s 2 &>> ${LOG_FILE}
#	$EXE -f 2 -i ${DATA_FILE} -s 3 &>> ${LOG_FILE}
#	#$EXE -f 1 -i ${DATA_FILE} -s 4 &>>  ${LOG_FILE}
#	$EXE -f 2 -i ${DATA_FILE} -s 5 &>> ${LOG_FILE}
#done
