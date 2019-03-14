from matplotlib.pyplot import plot, show

from ggdtrack.utils import load_json

motas = load_json("cachedir/logdir/motas.json")

motas.sort()
times = [t-motas[0][0] for t, m in motas]
plot(times, [m for t, m in motas])
show()