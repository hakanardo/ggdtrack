from matplotlib.pyplot import plot, show

from ggdtrack.utils import load_json

hammings = load_json("cachedir/logdir/hammings.json")

hammings.sort()
times = [t-hammings[0][0] for t, m in hammings]
plot(times, [m for t, m in hammings])
show()