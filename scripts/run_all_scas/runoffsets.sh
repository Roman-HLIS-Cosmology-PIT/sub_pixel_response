#!/bin/bash
#SBATCH --job-name=runoffsets
#SBATCH --account=PAS2340
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=run_offsets_10_21.out
#SBATCH --error=run_offsets_10_21.err
cd /users/PAS2340/chloesmith/sub_pixel_response || exit 1

echo "Working directory: $(pwd)"
echo "Checking offset file..."
ls -lh src/sub_pixel_response/offsets/test_offset_map.fits

python -m sub_pixel_response.offsets.run_offsets \
    example_test.yaml \
    src/sub_pixel_response/offsets/test_offset_map.fits \
    src/sub_pixel_response/offsets/final_image.fits