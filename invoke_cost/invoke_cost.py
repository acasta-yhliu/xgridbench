from dataclasses import dataclass
from sys import argv

import tqdm
import xgrid
import numpy
import time


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

xgrid.init(parallel=True, precision="double")

float2d = xgrid.grid[float, 2]  # type: ignore


@dataclass
class Config:
    rho: float
    nu: float
    dt: float
    dx: float
    dy: float


SIZE_X = SIZE_Y = int(argv[1])

u = xgrid.Grid((SIZE_X, SIZE_Y), float)
v = xgrid.Grid((SIZE_X, SIZE_Y), float)
p = xgrid.Grid((SIZE_X, SIZE_Y), float)
pt = xgrid.Grid((SIZE_X, SIZE_Y), float)
b = xgrid.Grid((SIZE_X, SIZE_Y), float)


u.boundary[0, :] = u.boundary[:, 0] = u.boundary[:, -1] = 1
u.boundary[-1, :] = 2

v.boundary.fill(1)
v.boundary[1:-1, 1:-1] = 0

p.boundary[:, -1] = 1
p.boundary[0, :] = 2
p.boundary[:, 0] = 3
p.boundary[-1, :] = 4

pt.boundary[:, -1] = 1
pt.boundary[0, :] = 2
pt.boundary[:, 0] = 3
pt.boundary[-1, :] = 4

b.boundary.fill(1)
b.boundary[1:-1, 1:-1] = 0

TIME = float(argv[2])
FRAMES = 1000


config = Config(1.0, 0.1, TIME / FRAMES, 2 / (SIZE_X - 1), 2 / (SIZE_Y - 1))


@xgrid.kernel(includes=["time.h"], macro=["clock_t start, end;"])
def cavity_kernel(b: float2d, p: float2d, pt: float2d, u: float2d, v: float2d, cfg: Config) -> float:
    with xgrid.c():
        r"""start = clock();"""

    b[0, 0] = (cfg.rho * (1.0 / cfg.dt *
                          ((u[0, 1] - u[0, -1]) /
                           (2.0 * cfg.dx) + (v[1, 0] - v[-1, 0]) / (2.0 * cfg.dy)) -
                          ((u[0, 1] - u[0, -1]) / (2.0 * cfg.dx))**2.0 -
                          2.0 * ((u[1, 0] - u[-1, 0]) / (2.0 * cfg.dy) *
                                 (v[0, 1] - v[0, -1]) / (2.0 * cfg.dx)) -
                          ((v[1, 0] - v[-1, 0]) / (2.0 * cfg.dy))**2.0))

    p[0, 0] = (((p[0, 1] + p[0, -1]) * cfg.dy**2.0 +
                (p[1, 0] + p[-1, 0]) * cfg.dx**2.0) /
               (2.0 * (cfg.dx**2.0 + cfg.dy**2.0)) -
               cfg.dx**2.0 * cfg.dy**2.0 / (2.0 * (cfg.dx**2.0 + cfg.dy**2.0)) *
               b[0, 0][0])

    with xgrid.boundary(1):
        p[0, 0] = p[0, -1][0]  # dp/dx = 0 at x = 2
    with xgrid.boundary(2):
        p[0, 0] = p[1, 0][0]   # dp/dy = 0 at y = 0
    with xgrid.boundary(3):
        p[0, 0] = p[0, 1][0]   # dp/dx = 0 at x = 0
    with xgrid.boundary(4):
        p[0, 0] = 0.0

    for _ in range(0, 25):
        pt[0, 0] = (((p[0, 1][0] + p[0, -1][0]) * cfg.dy**2.0 +
                    (p[1, 0][0] + p[-1, 0][0]) * cfg.dx**2.0) /
                    (2.0 * (cfg.dx**2.0 + cfg.dy**2.0)) -
                    cfg.dx**2.0 * cfg.dy**2.0 / (2.0 * (cfg.dx**2.0 + cfg.dy**2.0)) *
                    b[0, 0][0])

        with xgrid.boundary(1):
            pt[0, 0] = pt[0, -1][0]  # dp/dx = 0 at x = 2
        with xgrid.boundary(2):
            pt[0, 0] = pt[1, 0][0]   # dp/dy = 0 at y = 0
        with xgrid.boundary(3):
            pt[0, 0] = pt[0, 1][0]   # dp/dx = 0 at x = 0
        with xgrid.boundary(4):
            pt[0, 0] = 0.0

        p[0, 0] = (((pt[0, 1][0] + pt[0, -1][0]) * cfg.dy**2.0 +
                    (pt[1, 0][0] + pt[-1, 0][0]) * cfg.dx**2.0) /
                   (2.0 * (cfg.dx**2.0 + cfg.dy**2.0)) -
                   cfg.dx**2.0 * cfg.dy**2.0 / (2.0 * (cfg.dx**2.0 + cfg.dy**2.0)) *
                   b[0, 0][0])

        with xgrid.boundary(1):
            p[0, 0] = p[0, -1][0]  # dp/dx = 0 at x = 2
        with xgrid.boundary(2):
            p[0, 0] = p[1, 0][0]   # dp/dy = 0 at y = 0
        with xgrid.boundary(3):
            p[0, 0] = p[0, 1][0]   # dp/dx = 0 at x = 0
        with xgrid.boundary(4):
            pt[0, 0] = 0.0

    u[0, 0] = (u[0, 0] -
               u[0, 0] * cfg.dt / cfg.dx *
               (u[0, 0] - u[0, -1]) -
               v[0, 0] * cfg.dt / cfg.dy *
               (u[0, 0] - u[-1, 0]) -
               cfg.dt / (2.0 * cfg.rho * cfg.dx) * (p[0, 1][0] - p[0, -1][0]) +
               cfg.nu * (cfg.dt / cfg.dx**2.0 *
                         (u[0, 1] - 2.0 * u[0, 0] + u[0, -1]) +
                         cfg.dt / cfg.dy**2.0 *
                         (u[1, 0] - 2.0 * u[0, 0] + u[-1, 0])))

    v[0, 0] = (v[0, 0] -
               u[0, 0] * cfg.dt / cfg.dx *
               (v[0, 0] - v[0, -1]) -
               v[0, 0] * cfg.dt / cfg.dy *
               (v[0, 0] - v[-1, 0]) -
               cfg.dt / (2.0 * cfg.rho * cfg.dy) * (p[1, 0][0] - p[-1, 0][0]) +
               cfg.nu * (cfg.dt / cfg.dx**2.0 *
                         (v[0, 1] - 2.0 * v[0, 0] + v[0, -1]) +
                         cfg.dt / cfg.dy**2.0 *
                         (v[1, 0] - 2.0 * v[0, 0] + v[-1, 0])))

    with xgrid.boundary(1):
        u[0, 0] = 0.0
        v[0, 0] = 0.0

    with xgrid.boundary(2):
        u[0, 0] = 1.0
    
    with xgrid.c():
        r"""
        end = clock();
        return (end - start) / (double)CLK_TCK;
        """

acc = 0
with timer.timing():
    for i in tqdm.tqdm(range(FRAMES)):
        acc += cavity_kernel(b, p, pt, u, v, config)

import matplotlib.pyplot as plt
plt.style.use("classic")
plt.figure(dpi=200)
plt.bar(["Total", "Kernel", "Cost"], [timer.elapsed, acc, timer.elapsed - acc], align="center", color=["blue", "blue", "red"])
plt.ylabel("time/s")
plt.title("True Kernel Invoke Time")
plt.savefig("invoke_cost.png")

print((timer.elapsed - acc) / timer.elapsed)