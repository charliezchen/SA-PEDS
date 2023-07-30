#!/bin/bash

# Define the list of arguments
mlist=(1 2 4 8 16 32 64 128)
test=("Ackley")
# test=("Rastrigin")
optimizer=("Adam" "SGD")
flag=("--rv" "" "--independent")


for m in "${mlist[@]}"; do
    for t in "${test[@]}"; do
        for optim in "${optimizer[@]}"; do
            for rv in "${flag[@]}"; do
            job_file=$(mktemp)

# Check how much memory needed #TODO:
cat << EOF > $job_file
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=run_cpu_job
#SBATCH --time=1:00:00
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=20
#SBATCH --output=log_larger_noise/%x-%j.out
#SBATCH --mail-type=ALL

N_list=(1 2 4 8 16 32 64 128)

for N in "\${N_list[@]}"; do

singularity exec --nv \
--overlay ~/zc2157/overlay-50G-10M.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate torch

python3 -u optimize_peds.py \
    --test_function $t \
    --m $m --N \$N --alpha 0 --alpha-inc 0.01 $rv \
    --class torch.optim.$optim 

"
done

EOF


sbatch $job_file
# cat $job_file
rm $job_file




# exit 0

done
done
done
done

