rsync -avz --include='mypython' --include='*.py' --include='*.ipynb' --include="*.yml" --include="*.slurm" --include="*.bash" --exclude='*' . greene:~/workspace/am_sure

