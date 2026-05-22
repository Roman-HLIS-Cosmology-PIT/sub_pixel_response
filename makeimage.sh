#!/bin/bash
#SBATCH --job-name=makeimage
#SBATCH --account=PAS2340
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=testScriptImageSim_5_20.out
#SBATCH --error=testScriptImageSim_5_20.err
cd $SLURM_SUBMIT_DIR
python imageSim.py example_test.yaml 
# python optimizedStarSim.py optimizedConfig.yaml
