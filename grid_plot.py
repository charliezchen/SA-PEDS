import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re
import sys
import ast

dir = "log_larger_noise"

all_logs = os.listdir(dir)

logs = []

def read_one(path):
    with open(path, 'r') as f:
        last = []
        skip = False
        for line in f.readlines():
            if "[WARNING]: There" in line: continue

            if "WARNING: Could not find" in line or 'experiment' in line:
                if last:
                    try:
                        logs.append(ast.literal_eval("".join(last)))
                    except:
                        print("Parse error")
                        from IPython import embed
                        embed() or exit(0)
                last = []
                if 'experiment' in line:
                    last.append(line.replace("nan", '-1'))
            else:
                if "last_x" in line:
                    skip = True
                elif "mean_loss" in line:
                    skip = False
                if not skip:
                    last.append(line.replace("nan", '-1'))
        
        if last:
            try:
                logs.append(ast.literal_eval("".join(last)))
            except:
                print("Parse error")
                from IPython import embed
                embed() or exit(0)
for l in all_logs:
    read_one(os.path.join(dir, l))

# Filtering
def check_is_filtered(log, optimizer_fixed_fields, model_fixed_fields):
    for filter, value in optimizer_fixed_fields.items():
        if log['optimizer_config'][filter] != value:
            return True
    for filter, value in model_fixed_fields.items():
        if log['model_config'][filter] != value:
            return True
    return False
optimizer_fixed_fields = {
    "class": "torch.optim.SGD"
}
# CHANGE HERE: Change these settings to filter through results
model_fixed_fields = {
    "rv": False,
    "independent": False,
    # 'm': 100,
    'alpha_inc': 0.01,
    'lower': -10,
    'upper': 10,
    'init_noise': 10,
}
logs = [l for l in logs if not check_is_filtered(l, 
                                                 optimizer_fixed_fields, 
                                                 model_fixed_fields)]
for l in logs:
    print(l)

print("length of logs:", len(logs))

def get_m_and_N(m, N, logs):
    filtered_logs = [l for l in logs if \
                     l['model_config']['N']==N and \
                     l['model_config']['m']==m   ]
    if not filtered_logs:
        raise Exception(f"Cannot find m={m} and N={N}")
        return 0 # 
    try:
        assert len(filtered_logs) == 1, f"Should be unique, but get {len(filtered_logs)}"
    except:
        from IPython import embed
        embed() or exit(0)
    return filtered_logs[0]['result']['success_rate'] # CHANGE HERE: Variable of interest

# Plot the graph N and m
N_range, m_range = 8, 8
x = [2**i for i in range(m_range)]
y = [2**i for i in range(N_range)]

# y = [1, 20, 40, 60, 80, 100]
x_grid, y_grid = np.meshgrid(x, y)
Z = np.ones((m_range, N_range))
for i in range(m_range):
    for j in range(N_range):
        Z[i][j] = get_m_and_N(x[i], y[j], logs)

fig, ax = plt.subplots()
ax.matshow(Z, origin='lower')
for i in range(m_range):
    for j in range(N_range):
        ax.text(i, j, f"{Z[j][i]:.3f}", va='center', ha='center')
ax.set_xticklabels([0]+x)
ax.set_yticklabels([0]+y)
plt.gca().xaxis.tick_bottom()
plt.xlabel("N")
plt.ylabel("m")
plt.show()
# plt.tight_layout()
# plt.savefig("matrix_restart.png")

model_fixed_fields = {
    'm': 128,
}
logs = [l for l in logs if not check_is_filtered(l, {}, model_fixed_fields)]
succ = [l['result']['success_rate'] for l in logs]
mean_time = [l['result']['mean_time'] for l in logs]
x = [l['model_config']['N'] for l in logs]

plt.rcParams.update({'font.size': 22})

fig, ax1 = plt.subplots()
color='tab:red'
ax1.set_xlabel("N")
ax1.set_ylabel("success rate", color=color)
ax1.plot(x,succ,linestyle='--', marker='o', color=color, markersize=12, linewidth=3)
# ax1.set_ylim(0,1)
ax1.tick_params(axis ='y', labelcolor = color)

ax2=ax1.twinx()
color='tab:green'
ax2.set_ylabel("average time", color=color)
ax2.plot(x, mean_time,linestyle='--', marker='o', color=color, markersize=12, linewidth=3)
ax2.set_ylim(0,100)
ax2.tick_params(axis ='y', labelcolor = color)

# plt.title("Comparison between success rate and average time (m=128)\n")

# plt.ylim(0,100)
plt.show()
