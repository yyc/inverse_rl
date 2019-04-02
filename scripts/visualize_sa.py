import numpy as np
import matplotlib
matplotlib.use('tkagg')  # Or any other X11 back-end
import matplotlib.pyplot as plt

import pickle

results = pickle.load(open('results/airl_infl_poisoned_neg.pkl', 'rb'))

#expert_trajs, (diff_lz, traj_losses, hessian_val, hessian_inv), influences = results[0]

trajectories = [r[0] for r in results]
influences = [r[2] for r in results]

poisoned = [0 if r['poisoned'] else 1 for r in trajectories[0]]

traj_influences = np.swapaxes(influences, 0, 1)

averages = np.average(traj_influences, axis=1)

variances = np.var(traj_influences, axis=1)

#plt.hist(influences[])
# plt.plot(influences[1], poisoned, 'go', linewidth=0)

plt.plot(averages, poisoned, 'ro')
#plt.plot(variances)
plt.show()
