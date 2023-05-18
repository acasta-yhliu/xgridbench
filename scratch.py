STR = r"""v[0, 0] = (v[0, 0] -
               u[0, 0] * cfg.dt / cfg.dx *
               (v[0, 0] - v[0, -1]) -
               v[0, 0] * cfg.dt / cfg.dy *
               (v[0, 0] - v[-1, 0]) -
               cfg.dt / (2.0 * cfg.rho * cfg.dy) * (p[1, 0][0] - p[-1, 0][0]) +
               cfg.nu * (cfg.dt / cfg.dx**2.0 *
                         (v[0, 1] - 2.0 * v[0, 0] + v[0, -1]) +
                         cfg.dt / cfg.dy**2.0 *
                         (v[1, 0] - 2.0 * v[0, 0] + v[-1, 0])))""".replace('[', '.last(').replace(']', ')')
print(STR)
