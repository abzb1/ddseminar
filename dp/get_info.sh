#!/bin/bash

jobid=$1

echo "jobid: $jobid"

echo "******************* last output message *******************"
ls ./_log/$jobid/*.out
tail -n 5 ./_log/$jobid/*.out 
echo `cat ./_log/$jobid/*.out | wc -l` lines..

echo "******************* last error message *******************"
ls ./_log/$jobid/*.err
tail -n 1 ./_log/$jobid/*.err 
echo `cat ./_log/$jobid/*.err | wc -l` lines..

echo "******************* last gpu message *******************"
ls ./_log/$jobid/*.gpu
tail -n 30 ./_log/$jobid/*.gpu
echo `cat ./_log/$jobid/*.gpu | wc -l` lines..

