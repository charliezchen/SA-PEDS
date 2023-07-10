import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re

m = 1
alpha_inc = 0.1

dir = 'exp_center_reset/Rastrigin_indep_False'

all_pkl = os.listdir(dir)

# pattern = r'.*m_10_.*_inc_0\.1.*'

matches = []
Ns = []
for s in all_pkl:
    match = re.match(r"N_(\d+)_m_20_.*inc_0.1.pkl", s)
    if match:
        matches.append((s))
        Ns.append(int(match.group(1)))

succ = []

# all_pkl = [pkl for pkl in all_pkl if re.search(pattern, pkl)]


interesting = ['success_rate']

for pkl in matches:
    path = os.path.join(dir, pkl)
    with open(path, 'rb') as f:
        res = pickle.load(f)
        succ.append(res['success_rate'])

pairs = sorted(zip(Ns, succ))
Ns_sorted, succ_sorted = zip(*pairs)

# plot the data

one = succ_sorted[0]
assert Ns_sorted[0] == 1
simulated = [1-(1-one)**N for N in Ns_sorted]

plt.plot(Ns_sorted, succ_sorted, 'o-', label='PEDS')
plt.plot(Ns_sorted, simulated, 'o-', label='simulated')
plt.legend()
plt.xlabel('N')
plt.ylabel('Success rate')
plt.title('Success rate vs N')
plt.savefig('m20.png')

from IPython import embed
embed() or exit(0)