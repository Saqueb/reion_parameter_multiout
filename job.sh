#!/bin/bash
#$ -l hostname=compute-0-2
#$ -cwd
#$ -N ionz_main
#$ -o nohup.out
#$ -e error
#$ -j y
#$ -pe smp 64
nohup ./ionz_main



