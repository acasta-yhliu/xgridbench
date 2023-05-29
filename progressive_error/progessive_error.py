from dataclasses import dataclass
from sys import argv

import tqdm
import xgrid
import numpy
import matplotlib.pyplot as plt

xgrid.init(cacheroot=".xgridtest", parallel=True,
           precision="float", opt_level=2)

float2d = xgrid.grid[float, 2]  # type: ignore


@dataclass
class Config:
    rho: float
    nu: float
    dt: float
    dx: float
    dy: float


SIZE_X = SIZE_Y = int(argv[1])

ux = xgrid.Grid((SIZE_X, SIZE_Y), float)
vx = xgrid.Grid((SIZE_X, SIZE_Y), float)
px = xgrid.Grid((SIZE_X, SIZE_Y), float)
ptx = xgrid.Grid((SIZE_X, SIZE_Y), float)
bx = xgrid.Grid((SIZE_X, SIZE_Y), float)


ux.boundary[0, :] = ux.boundary[:, 0] = ux.boundary[:, -1] = 1
ux.boundary[-1, :] = 2

vx.boundary.fill(1)
vx.boundary[1:-1, 1:-1] = 0

px.boundary[:, -1] = 1
px.boundary[0, :] = 2
px.boundary[:, 0] = 3
px.boundary[-1, :] = 4

ptx.boundary[:, -1] = 1
ptx.boundary[0, :] = 2
ptx.boundary[:, 0] = 3
ptx.boundary[-1, :] = 4

bx.boundary.fill(1)
bx.boundary[1:-1, 1:-1] = 0

TIME = float(argv[2])
FRAMES = 1000
nit = 50


def rmse(a):
    return numpy.mean(numpy.sqrt(a ** 2))


config = Config(1.0, 0.1, numpy.float32(TIME / FRAMES), numpy.float32(2 / (SIZE_X - 1)), numpy.float32(2 / (SIZE_Y - 1)))


@xgrid.kernel()
def cavity_kernel(b: float2d, p: float2d, u: float2d, v: float2d, cfg: Config) -> None:
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

    for _ in range(0, 49):
        p[0, 0] = (((p[0, 1][0] + p[0, -1][0]) * cfg.dy**2.0 +
                    (p[1, 0][0] + p[-1, 0][0]) * cfg.dx**2.0) /
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


def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt *
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) /
                             (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return b


def pressure_poisson(p, dx, dy, b):
    pn = numpy.empty_like(p, dtype=numpy.float32)
    pn = p.copy()
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) *
                         b[1:-1, 1:-1])
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        p[-1, :] = 0        # p = 0 at y = 2
    return p


def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    un = numpy.empty_like(u, dtype=numpy.float32)
    vn = numpy.empty_like(v, dtype=numpy.float32)
    b = numpy.zeros((SIZE_Y, SIZE_X), dtype=numpy.float32)

    u_err = []
    v_err = []
    p_err = []

    u_err_max = []
    v_err_max = []
    p_err_max = []

    u_final = v_final = p_final = None

    for n in tqdm.tqdm(range(nt)):
        un = u.copy()
        vn = v.copy()

        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                         (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                         (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                               (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                               dt / dy**2 *
                               (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                         (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                         (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         nu * (dt / dx**2 *
                               (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                               dt / dy**2 *
                               (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1    # set velocity on cavity lid equal to 1
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0

        cavity_kernel(bx, px, ux, vx, config)

        # calculate error
        u_err.append(rmse(u - ux.now))
        v_err.append(rmse(v - vx.now))
        p_err.append(rmse(p - px.now))

        u_err_max.append(numpy.max(numpy.abs(u - ux.now)))
        v_err_max.append(numpy.max(numpy.abs(v - vx.now)))
        p_err_max.append(numpy.max(numpy.abs(p - px.now)))

        if n == nt - 1:
            u_final = u - ux.now
            v_final = v - vx.now
            p_final = p - px.now

    return u_err, u_err_max, v_err, v_err_max, p_err, p_err_max, u_final, v_final, p_final


uref = numpy.zeros((SIZE_Y, SIZE_X), dtype=numpy.float32)
vref = numpy.zeros((SIZE_Y, SIZE_X), dtype=numpy.float32)
pref = numpy.zeros((SIZE_Y, SIZE_X), dtype=numpy.float32)
bref = numpy.zeros((SIZE_Y, SIZE_X), dtype=numpy.float32)

x = [i for i in range(FRAMES)]
uerr, uerrmax, verr, verrmax, perr, perrmax, uf, vf, pf = cavity_flow(
    FRAMES, uref, vref, config.dt, config.dx, config.dy, pref, config.rho, config.nu)

plt.style.use("classic")
plt.figure(dpi=200)

plt.plot(x, uerr, label="$u$ RMSE")
plt.plot(x, uerrmax, label="$u$ Max")
plt.plot(x, verr, label="$v$ RMSE")
plt.plot(x, verrmax, label="$v$ Max")
plt.legend(loc=2)
plt.grid()
plt.title("$u,v$ Error")
plt.savefig("error_u_v.png")
plt.clf()

plt.plot(x, perr, label="RMSE")
plt.plot(x, perrmax, label="Max")
plt.legend()
plt.grid()
plt.title("$p$ Error")
plt.savefig("error_p.png")
plt.clf()

fig = plt.figure(figsize=(14, 3))

plt.subplot(131)
plt.imshow(pf, origin="lower")
plt.colorbar()
plt.title("p")

plt.subplot(132)
plt.imshow(uf, origin="lower")
plt.colorbar()
plt.title("u")

plt.subplot(133)
plt.imshow(vf, origin="lower")
plt.colorbar()
plt.title("v")

plt.savefig(f"cavity_diff_float.png")
plt.clf()