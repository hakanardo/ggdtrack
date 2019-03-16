import re
from collections import defaultdict
from glob import glob
from os import stat
import numpy as np
import matplotlib.pyplot as plt


motas = defaultdict(list)
times = defaultdict(list)
for fn in glob("cachedir/logdir_?.??_??/eval_results_int.txt"):
    amount = float(fn.split('_')[1])
    ts = stat(fn).st_mtime_ns / 1e9
    res_int = open(fn).read()
    mota = float(re.split(r'\s+', res_int.split('\n')[-1])[4].replace('%', ''))
    motas[amount].append(mota)
    times[amount].append(ts)

data = []
for amount in sorted(motas.keys()):
    mm = motas[amount]
    tt = times[amount]
    data.append((amount, np.mean(mm), np.std(mm), (max(tt) - min(tt)) / (len(tt) - 1),
                 np.median(mm), np.quantile(mm, 0.10), np.quantile(mm, 0.90)))
data = np.array(data)



fig, ax1 = plt.subplots()
color1 = "#cb5b5a"
ax1.bar(data[:, 0], data[:,3], 0.05, color=color1)
ax1.set_ylabel('Mean train time (s)', color=color1)
ax1.tick_params('y', colors=color1)

ax2 = ax1.twinx()
color2 = '#6b406e'
# ax2.plot(data[:, 0], data[:, 1], '-', color=color2)
# ax2.plot(data[:, 0], data[:, 1] - 2 * data[:, 2], '--', color=color2)
# ax2.plot(data[:, 0], data[:, 1] + 2 * data[:, 2], '--', color=color2)
ax2.plot(data[:, 0], data[:, 4], '-', color=color2)
ax2.plot(data[:, 0], data[:, 5], '--', color=color2)
ax2.plot(data[:, 0], data[:, 6], '--', color=color2)
ax2.set_xlabel('Amount of training data used')
ax2.set_ylabel('MOTA', color=color2)
ax2.tick_params('y', colors=color2)

fig.tight_layout()
plt.show()
