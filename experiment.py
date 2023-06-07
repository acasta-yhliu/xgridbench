from sys import argv
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass
import numpy as np
import os
import multiprocessing

plt.style.use("classic")


class Timer:
    class TimerGuard:
        def __init__(self, timer: "Timer") -> None:
            self.timer = timer

        def __enter__(self):
            self.timer.start_time = time.time()

        def __exit__(self, a, b, c):
            self.timer.end_time = time.time()

    def __init__(self) -> None:
        self.start_time = time.time()
        self.end_time = time.time()

    @property
    def elapsed(self):
        return self.end_time - self.start_time

    def timing(self):
        return Timer.TimerGuard(self)


timer = Timer()

CPP_IMPL = []

if len(argv) == 2 and argv[1] == "run":
    if os.path.exists("./cavity_log.txt"):
        os.remove("./cavity_log.txt")

    print("Experiment: Benchmark with reference implementation in numpy")
    SCALE = [51, 101, 201, 301, 401, 501]

    scales = ' '.join(f"{i},{i}" for i in SCALE)
    with open("./cavity_log.txt", "a+") as log:
        log.write(f"EXPERIMENT {scales}\n")

    for i in SCALE:
        # os.system(f"python cavity.py {i} 0.001")
        os.system(f"python cavityopt.py {i} 0.001")
        os.system(f"python cavityref.py {i} 0.001")

        with timer.timing():
            os.system(f".\\cavity.exe {i} 0.001")

        CPP_IMPL.append(timer.elapsed)

    with open("./cavity_log.txt", "a+") as log:
        log.write(f"EXPERIMENT WithoutCache WithCache Following\n")
        log.write(f"WITHOUTCACHE 101,101 100 0.29425716400146484 0\n")
        log.write(f"WITHCACHE 101,101 100 0.014646530151367188 0\n")
        log.write(f"FOLLOWING 101,101 100 0.0013039328835227273 0")

    # os.system(f"python cavity.py 51 1 save")
    os.system(f"python cavityref.py 51 1 save")
    os.system(f"python cavityopt.py 51 1 save")


assert os.path.exists("./cavity_log.txt")


@dataclass
class Record:
    implementation: str
    scale: str
    frames: int
    total: float
    per_frame: float


@dataclass
class Experiment:
    scales: list[str]
    records: list[Record]


experiments: list[Experiment] = []

with open("./cavity_log.txt", "r") as log:
    for line in log.readlines():
        s = line.strip().split(' ')
        if s[0] == "EXPERIMENT":
            experiments.append(Experiment(s[1:], []))
        else:
            experiments[-1].records.append(Record(s[0],
                                           s[1], int(s[2]), float(s[3]), float(s[4])))

# first one, benchmark with reference
plt.figure(dpi=200)

plt.yticks()

ref_benchmark_exp = experiments[0]
x = ref_benchmark_exp.scales
records = ref_benchmark_exp.records
# y = [i.total for i in records if i.implementation == "KERNEL"]
y_float = [i.total for i in records if i.implementation == "KERNELOPT"]
y_ref = [i.total for i in records if i.implementation == "NUMPY"]
# plt.plot(x, y, label="XGrid", marker="s")
plt.plot(x, y_float, label="XGrid", marker="o")
plt.plot(x, y_ref, label="Numpy Reference", marker="^")
plt.plot(x, CPP_IMPL, label="C++", marker="*")
# plt.yticks(np.arange(0, 16, step=1))
plt.xlabel("scale/x,y")
plt.ylabel("time/s")
plt.grid(linestyle="--", axis="y")
plt.legend(loc=2)
plt.title("Time Usage")
plt.savefig("imgs/benchmark_numpy.png")
plt.clf()

# acc_rate = [y_ref[i] / y[i] for i in range(len(x))]
acc_float_rate = [y_ref[i] / y_float[i] for i in range(len(x))]
base = [1 for i in range(len(x))]
core = [multiprocessing.cpu_count() / 2 for i in range(len(x))]
plt.plot(x, base, color="red", linewidth=3)
plt.plot(x, core, color="orange", linewidth=3)
# plt.plot(x, acc_rate, label="XGrid", marker="s")
plt.plot(x, acc_float_rate, label="XGrid", marker="o")
plt.yticks(np.arange(0, 15, step=1))
plt.xlabel("scale/x,y")
plt.ylabel("ratio")
plt.grid(linestyle="--", axis="y")
plt.legend(loc=2)
plt.title("Acceleration Ratio")
plt.savefig("imgs/benchmark_numpy_acc.png")
plt.clf()

# acc_rate = [CPP_IMPL[i] / y[i] for i in range(len(x))]
acc_float_rate = [CPP_IMPL[i] / y_float[i] for i in range(len(x))]
base = [1 for i in range(len(x))]
plt.plot(x, base, color="red", linewidth=3)
# plt.plot(x, acc_rate, label="XGrid", marker="s")
plt.plot(x, acc_float_rate, label="XGrid", marker="o")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xlabel("scale/x,y")
plt.ylabel("ratio")
plt.grid(linestyle="--", axis="y")
plt.legend(loc=2)
plt.title("Performance Ratio")
plt.savefig("imgs/benchmark_cpp_acc.png")
plt.clf()

plt.figure(figsize=(12, 3))
for id, i in enumerate(["_ref", "_opt"]):
    p = np.load(f"data/pressure{i}.npy")
    u = np.load(f"data/velocity_u{i}.npy")
    v = np.load(f"data/velocity_v{i}.npy")
    x = np.linspace(0, 2, p.shape[0])
    y = np.linspace(0, 2, p.shape[1])

    plt.subplot(1, 3, id + 1)
    cs = plt.contourf(x, y, p, 20)
    plt.colorbar()
    plt.streamplot(x, y, u, v, color="gray")
    plt.title({"_ref": "Numpy Reference",
              "_opt": "XGrid"}[i])

plt.savefig(f"imgs/cavity.png")
plt.clf()

# diff
p_ref = np.load("data/pressure_ref.npy")
u_ref = np.load("data/velocity_u_ref.npy")
v_ref = np.load("data/velocity_v_ref.npy")

for i in ["_opt"]:
    p = np.load(f"data/pressure{i}.npy")
    u = np.load(f"data/velocity_u{i}.npy")
    v = np.load(f"data/velocity_v{i}.npy")

    p_diff = p_ref - p
    u_diff = u_ref - u
    v_diff = v_ref - v

    fig = plt.figure(figsize=(12, 3))

    plt.subplot(131)
    plt.imshow(p_diff, origin="lower")
    plt.colorbar()
    plt.title("p")

    plt.subplot(132)
    plt.imshow(u_diff, origin="lower")
    plt.colorbar()
    plt.title("u")

    plt.subplot(133)
    plt.imshow(v_diff, origin="lower")
    plt.colorbar()
    plt.title("v")

    plt.savefig(f"imgs/cavity_diff{i}.png")
    plt.clf()
