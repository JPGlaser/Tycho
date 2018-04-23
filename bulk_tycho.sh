#!/bin/bash
echo "Welcome to Tycho 2.0's Bulk Run Script"
read -p 'Repeats in Each Bin (Integer): ' num_of_repeats
read -p 'King Parameters, W0 (Space-Seperayed Array): ' list_of_W
read -p 'Number of Stars (Space-Seperated Array): ' -a list_of_NStars
read -p 'List of Nodes (Space-Seperated Array): ' -a list_of_nodes

num_of_W=${#list_of_W[@]}
num_of_runs=$((${#list_of_NStars[@]}*$num_of_W*$num_of_repeats))
num_of_nodes_needed=$(($num_of_runs/4))
num_of_nodes_provided=${#list_of_nodes[@]}

if [ $num_of_nodes_needed -eq $num_of_nodes_provided] 
then
	echo "Correct Number of Nodes Provided!"
	echo "Starting ..."
else
	echo "Incorrect Number of Nodes Provided!"
	echo "Exiting ..."
	exit 1
fi

node_counter=0
w_counter=0
star_counter=0
for (( i=0; i<$num_of_runs; i++ )); do
	if [ $(($i%4)) == 0 ]
	then
		#Set the Node's Name
		node=${list_of_nodes[$node_counter]}
		export $node
		$node_counter++
	fi
	if [ $(($i%$num_of_W)) == 0]
	then
		#Set the W
		W=${list_of_W[$w_counter]}
		export $W
		$w_counter++
	fi
	if [ $(($i%$num_of_repeats)) == 0]
	then
		#Set the Number of Stars
		NStars=${list_of_NStars[%star_counter]}
		export $NStars
		$star_counter++
	fi
	# Start a Screen in Detached Mode
	# 
done

for Node in nodes; do
	for W in W_0; do
	done
done	
	





#100 500 1000 5000
3 6 9
for N in num_stars; do

    for W in 3 6 9; do

        for seed in Fabrycky; do

            P=$(($N/2))
            name="cluster_N=${N}_W0=${W}_s=${seed}"
            echo $name
            mkdir $name
            cd $name
            cp ../sim_cluster.py .
            python sim_cluster.py -i 0 -p ${P} -s ${N} -t 0.05 -w ${W} -c ${name} -T 500 > /dev/null 2>&1
            cd ../
        
        done
    done
done 
