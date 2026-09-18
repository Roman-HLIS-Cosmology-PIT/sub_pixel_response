#!/bin/bash
#SBATCH --job-name=runallscas
#SBATCH --account=PAS2340
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=run_all_scas_9_18.out
#SBATCH --error=run_all_scas_9_18.err
cd $SLURM_SUBMIT_DIR
python run_all_scas.py
# python optimizedStarSim.py optimizedConfig.yaml