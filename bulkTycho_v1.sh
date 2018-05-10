#!/bin/bash

read -p 'Path to Tycho Directory (Ex: /home/draco/jglaser/GitHub/Tycho): ' TYCHO_DIR
read -p 'ID of GPU (Ex: 0): ' GPU_ID
read -p 'Random Seed (Ex: Glaser): ' seed

echo "[ALERT] Starting First Run ..."

for N in 100 1000; do
    for W in 3 6; do
        P=$(($N/2))
        name="${seed}_N${N}_W${W}"
        echo $name
        mkdir $name
        cd $name
        python ${TYCHO_DIR}/sim_cluster.py -i ${GPU_ID} -p ${P} -b -s ${N} -w ${W} -T 10000 -t 0.2 -c ${name} -S ${seed} > /dev/null 2>&1
        cd ../
        echo "[ALERT] Moving to Next Run ..."
    done
done

echo "[ALERT] All Runs Finished ..."
