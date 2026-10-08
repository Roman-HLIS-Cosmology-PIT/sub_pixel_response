#!/bin/bash
#SBATCH --job-name=runoffsets
#SBATCH --account=PAS2340
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=run_offsets_10_7.out
#SBATCH --error=run_offsets_10_7.err
cd $SLURM_SUBMIT_DIR
python -m sub_pixel_response.offsets.run_offsets example_test.yaml offsets/test_offset_map.fits offsets/final_image.fits
# python optimizedStarSim.py optimizedConfig.yaml