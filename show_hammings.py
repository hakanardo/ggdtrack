from matplotlib.pyplot import plot, show

from ggdtrack.utils import load_json

hammings = load_json("cachedir/logdir/hammings.json")
hammings = [[1552559741.2964199, 50222], [1552558857.6851532, 46158], [1552559251.9772584, 51524], [1552559009.4836545, 47194], [1552559374.6640458, 47516], [1552559611.1537073, 41856], [1552559181.913951, 48154], [1552559877.7630694, 48896], [1552560495.3409534, 45130], [1552559711.252717, 30026], [1552558827.6574497, 35650], [1552558677.5269315, 38186], [1552560275.0711355, 46234], [1552560690.3190207, 51104], [1552559344.6363428, 41732], [1552558697.534734, 44442], [1552559302.0367637, 42770], [1552558627.343429, 54798], [1552559221.9495552, 52668], [1552559049.523259, 43722], [1552559504.814759, 52038], [1552558797.617746, 57746]]


hammings.sort()
times = [t-hammings[0][0] for t, m in hammings]
plot(times, [m for t, m in hammings])
show()