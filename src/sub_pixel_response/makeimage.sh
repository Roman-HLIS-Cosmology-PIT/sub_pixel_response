#!/bin/bash
#SBATCH --job-name=makeimage
#SBATCH --account=PAS2340
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=testScriptImageSim_7_23.out
#SBATCH --error=testScriptImageSim_7_23.err
cd $SLURM_SUBMIT_DIR
python imagesim.py example_test.yaml 
# python optimizedStarSim.py optimizedConfig.yaml
