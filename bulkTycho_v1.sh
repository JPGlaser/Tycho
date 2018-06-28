#!/bin/bash

#read -p 'Path to Tycho Directory (Ex: /home/draco/jglaser/GitHub/Tycho): ' TYCHO_DIR
TYCHO_DIR=/home/draco/jglaser/GitHub/Tycho
#read -p 'ID of GPU (Ex: 0): ' GPU_ID
read -p 'Random Seed (Ex: Glaser): ' seed
IFS=', ' read -p 'List of the Number of CoM, N: ' -r -a NList
IFS=', ' read -p 'List of King Denisty Values, W: ' -r -a WList

echo "[ALERT] Starting First Run ..."

for N in ${NList[@]}; do
    for W in ${WList[@]}; do
        P=$(($N/2))
        name="${seed}_N${N}_W${W}"
        echo $name
        mkdir $name
        cd $name
        # Figure out what is up with GPU on Draco
        python ${TYCHO_DIR}/sim_cluster.py -g -p ${P} -b -s ${N} -w ${W} -T 10000 -t 0.2 -c ${name} -S ${seed}
# > /dev/null 2>&1
        cd ../
        echo "[ALERT] Moving to Next Run ..."
    done
done

echo "[ALERT] All Runs Finished ..."
