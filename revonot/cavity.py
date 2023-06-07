from dataclasses import dataclass
from sys import argv

import tqdm
import xgrid
import numpy as np
import matplotlib.pyplot as plt

xgrid.init(precision="double", opt_level=3)

float2d = xgrid.grid[float, 2]  # type: ignore


@dataclass
class Config:
    rho: float
    nu: float
    dt: float
    dx: float
    dy: float


SIZE_X = SIZE_Y = 41

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

FRAMES = 100000


config = Config(1000, float(argv[1]), 0.000001, 2 / (SIZE_X - 1), 2 / (SIZE_Y - 1))


@xgrid.kernel()
def cavity_kernel(b: float2d, p: float2d, pt: float2d, u: float2d, v: float2d, cfg: Config) -> None:
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


for i in tqdm.tqdm(range(FRAMES)):
    cavity_kernel(b, p, pt, u, v, config)

x = np.linspace(0, 2, p.shape[0])
y = np.linspace(0, 2, p.shape[1])

plt.style.use("classic")
plt.figure(dpi=200)
plt.contourf(x, y, p.now, 20)
plt.streamplot(x, y, u.now, v.now, color="gray")

RE = config.rho * max(np.max(u.now), np.max(v.now)) * 2 / config.nu
print(f"re = {RE}")
plt.savefig(f"{RE}_cavity.png")