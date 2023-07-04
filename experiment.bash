#!/bin/bash

# Define the list of arguments
mlist=(1 10 20 30) # 40 50 60 70 80 90 100
test=("Rastrigin" "Ackley")
flag=("--independent" "")

for m in "${mlist[@]}"; do
    for t in "${test[@]}"; do
        for f in "${flag[@]}"; do
            # job_file=$(mktemp)
        
sbatch << EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=run_cpu_job
#SBATCH --time=12:00:00
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=20
#SBATCH --output=log/%x-%j.out
#SBATCH --mail-type=ALL


singularity exec --nv \
--overlay ~/zc2157/overlay-50G-10M.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "

source /ext3/env.sh
conda activate torch

python3 optimize_peds.py --yaml_config_path config.yml --folder exp --m $m --test_function $t $f
"
EOF

        done
    done

    # # Call the python function with each argument
    # sbatch run_experiment_python.slurm "$m" Rastrigin "--independent"
    # sbatch run_experiment_python.slurm "$m" Ackley "--independent"
    # sbatch run_experiment_python.slurm "$m" Rastrigin
    # sbatch run_experiment_python.slurm "$m" Ackley
done
