import os
import re
from collections import defaultdict
from glob import glob
from os import stat
import numpy as np
import matplotlib.pyplot as plt


idf1s = defaultdict(list)
times = defaultdict(list)
# for fn in glob("cachedir/logdir_?.*_??/eval_results_int.txt"):
for fn in glob("/usr/share/cognimatics/nobackup/hakan/ggdtrack/**/eval_results_int.txt", recursive=True):
    amount = float(fn.split('_')[2])
    snapshots = sorted(glob(os.path.dirname(fn) + "/snapshot_???.pyt"))
    ts0 = stat(snapshots[0]).st_mtime_ns / 1e9
    ts1 = stat(snapshots[-1]).st_mtime_ns / 1e9
    train_time = (ts1 - ts0) / (len(snapshots) - 1) * len(snapshots)
    res_int = open(fn).read()
    idf1 = float(re.split(r'\s+', res_int.split('\n')[-1])[10].replace('%', ''))
    idf1s[amount].append(idf1)
    times[amount].append(train_time)

times[0.1] = [95*60]
times[0.01] = [47*60]
times[0.001] = [34*60]

data = []
for amount in sorted(idf1s.keys()):
    mm = idf1s[amount]
    tt = times[amount]
    data.append((amount, np.mean(mm), np.std(mm), np.mean(tt),
                 np.median(mm), np.quantile(mm, 0.10), np.quantile(mm, 0.90)))
data = np.array(data)
print(data[:,3]/60)
print(data[:,4])

fig, ax1 = plt.subplots()
color1 = "#cb5b5a"
ax1.set_xscale('log')
ax1.bar(data[:, 0], data[:,3]/60, data[:, 0] * 0.5, color=color1)
ax1.set_ylabel('Mean train time (min)', color=color1)
ax1.tick_params('y', colors=color1)

ax2 = ax1.twinx()
color2 = '#6b406e'
# ax2.plot(data[:, 0], data[:, 1], '-', color=color2)
# ax2.plot(data[:, 0], data[:, 1] - 2 * data[:, 2], '--', color=color2)
# ax2.plot(data[:, 0], data[:, 1] + 2 * data[:, 2], '--', color=color2)
ax2.plot(data[:, 0], data[:, 4], '-', color=color2, label="Median IDF1 Score")
ax2.plot(data[:, 0], data[:, 5], '--', color=color2, label="10 % Quantile IDF1 Score")
ax2.plot(data[:, 0], data[:, 6], '--', color=color2, label="90 % Quantile IDF1 Score")
# ax2.plot([min(data[:,0]), 1], [83.4, 83.4], ':', color=color2, label="Ground truth tracks")
ax2.set_xlabel('Amount of training data used')
ax2.set_ylabel('IDF1', color=color2)
ax2.tick_params('y', colors=color2)
ax2.legend(loc='center')

fig.tight_layout()
plt.show()
