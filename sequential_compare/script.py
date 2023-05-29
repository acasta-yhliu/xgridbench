import os
import matplotlib.pyplot as plt

plt.style.use("classic")

try:
    os.remove("time.log")
except:
    pass

for i in [51, 101, 201, 301, 401, 501]:
    os.system(f"python sequential.py {i} 0.001")

x = []
y_xgrid = []
y_numpy = []

with open("time.log", "r", encoding="utf-8") as f:
    for line in f.readlines():
        size, xgrid_time, numpy_time = line.strip().split(' ')
        x.append(f"{size},{size}")
        y_xgrid.append(float(xgrid_time))
        y_numpy.append(float(numpy_time))

plt.figure(dpi=200)
plt.plot(x, y_xgrid, marker="o", label="XGrid (Optimized, Sequential)")
plt.plot(x, y_numpy, marker="^", label="Numpy")
plt.xlabel("scale/x,y")
plt.ylabel("time/s")
plt.grid(linestyle="--", axis="y")
plt.legend(loc=2)
plt.title("Time Usage (Sequential)")
plt.savefig("benchmark_sequential_numpy.png")
plt.clf()