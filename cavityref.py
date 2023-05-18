from dataclasses import dataclass
from sys import argv
import numpy
import tqdm
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

SIZE_Y = SIZE_X = int(argv[1])
nit = 50


@dataclass
class Config:
    rho: float
    nu: float
    dt: float
    dx: float
    dy: float

TIME = float(argv[2])
FRAMES = 1000

config = Config(1.0, 0.1, TIME / FRAMES, 2 / (SIZE_X - 1), 2 / (SIZE_Y - 1))


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
    pn = numpy.empty_like(p)
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
    un = numpy.empty_like(u)
    vn = numpy.empty_like(v)
    b = numpy.zeros((SIZE_Y, SIZE_X))

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

    return u, v, p


uref = numpy.zeros((SIZE_Y, SIZE_X))
vref = numpy.zeros((SIZE_Y, SIZE_X))
pref = numpy.zeros((SIZE_Y, SIZE_X))
bref = numpy.zeros((SIZE_Y, SIZE_X))

with timer.timing():
    uref, vref, pref = cavity_flow(
        FRAMES, uref, vref, config.dt, config.dx, config.dy, pref, config.rho, config.nu)


if len(argv) == 4 and argv[3] == "save":
    numpy.save("data/pressure_ref.npy", pref)
    numpy.save("data/velocity_u_ref.npy", uref)
    numpy.save("data/velocity_v_ref.npy", vref)
else:
    with open("cavity_log.txt", "a+") as f:
        f.write(
            f"NUMPY {SIZE_X},{SIZE_Y} {FRAMES} {timer.elapsed} {timer.elapsed / FRAMES}\n")
