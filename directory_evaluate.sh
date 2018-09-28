#!/bin/bash

# Directory
python=$1
ckpt_directory=$2
json_directory=$3
num_episodes=$4
cmdargs=$5
num_runs=$6

json_directory_replaced=${json_directory//\//-}
output_directory="evaluation-"$json_directory_replaced"-numep"$num_episodes
echo $output_directory

for f in `ls $ckpt_directory | grep '\.pth'`; do
	 # Skip invalid paths
	 if [ -d $ckpt_directory"/"$f ]; then
		  echo "directory"
		  continue
	 fi

	 # Build path to output of a finished run
	 output_file=$ckpt_directory"/"$output_directory"-"$f"/JSONWalker-"$json_directory_replaced"-evaluation-statistics-evalep"$num_episodes".pth"
	 
	 # Check if this run has already has been finished previously
	 if [ -f $output_file ]; then
		  echo "Skipping "$output_file
		  continue
	 fi
	 
	 echo "Running command"
	 cmd=$python" directory_evaluate.py --json-directory "$json_directory" --output-directory "$output_directory"-"$f" --num-episodes "$num_episodes" --load-file "$ckpt_directory"/"$f" "$cmdargs
	 sem -j $num_runs $cmd
	 #eval $cmd
	 #sem --wait --citation
	 #exit
done

sem --wait

#python directory_evaluate.py --json-directory datasets/bipedal-random-offcenter-hull-1-12-25-percent/valid --output-directory datasets-bipedal-random-offcenter-hull-1-12-25-percent-valid --num-episodes 100 --stack-frames 1 --model-name models.mlp --load-file saved/env-RandomJSONWalker-datasets-bipedal-random-offcenter-hull-1-12-25-percent-train-modelmlp-lr0.0001-nsteps20-optimAdam-nstack1-EXPIDIfhprMAx/model.25.pth
