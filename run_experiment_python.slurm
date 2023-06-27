#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=run_cpu_job
#SBATCH --time=8:00:00
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=20
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL


singularity exec --nv \
	    --overlay ~/zc2157/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "
source /ext3/env.sh
conda activate torch

python3 optimize_peds.py -m 6 --alpha-inc 0.1
"

# python3 optimize_peds.py -m 2 --alpha-inc 0.1 --lower-N 6