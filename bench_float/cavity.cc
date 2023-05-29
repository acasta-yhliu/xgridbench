#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <vector>

template <typename T> struct ArbitaryGrid {
  std::vector<T> _data[2];
  int _time = 0;
  std::vector<int> _mask;

  const int shape[2];

  ArbitaryGrid(int sx, int sy) : shape{sx, sy} {
#define RESIZE(x) x.resize(sx *sy)
    RESIZE(_data[0]);
    RESIZE(_data[1]);
    RESIZE(_mask);
#undef RESIZE
  }

  T &now(int x, int y) { return _data[_time][x + y * shape[0]]; }

  T now(int x, int y) const { return _data[_time][x + y * shape[0]]; }

  T last(int x, int y) const { return _data[1 - _time][x + y * shape[0]]; }

  void tick() { _time = 1 - _time; }

  int mask(int x, int y) const { return _mask[x + y * shape[1]]; }

  int &mask(int x, int y) { return _mask[x + y * shape[1]]; }
};

template <typename T> struct ArbitaryConfig {
  T rho, nu, dt, dx, dy;
};

using Precision = float;

using Grid = ArbitaryGrid<Precision>;

using Config = ArbitaryConfig<Precision>;

constexpr int FRAMES = 1000;

template <typename T> void parallel_for(int SIZE_X, int SIZE_Y, T action) {
#pragma omp parallel for collapse(2)
  for (int y = 0; y < SIZE_Y; ++y) {
    for (int x = 0; x < SIZE_X; ++x) {
      action(x, y);
    }
  }
}

void cavity(Grid &b, Grid &p, Grid &u, Grid &v, int SIZE_X, int SIZE_Y,
            const Config &cfg) {

  // initialize b
  parallel_for(SIZE_X, SIZE_Y, [&](int x, int y) {
    if (b.mask(x, y) == 0) {
      b.now(x, y) =
          (cfg.rho *
           (1.0 / cfg.dt *
                ((u.last(x, y + 1) - u.last(x, y - 1)) / (2.0 * cfg.dx) +
                 (v.last(x + 1, y) - v.last(x - 1, y)) / (2.0 * cfg.dy)) -
            std::pow((u.last(x, y + 1) - u.last(x, y - 1)) / (2.0 * cfg.dx),
                     2.0) -
            2.0 * ((u.last(x + 1, y) - u.last(x - 1, y)) / (2.0 * cfg.dy) *
                   (v.last(x, y + 1) - v.last(x, y - 1)) / (2.0 * cfg.dx)) -
            std::pow((v.last(x + 1, y) - v.last(x - 1, y)) / (2.0 * cfg.dy),
                     2.0)));
    }
  });

  // solve possion equation
  for (int i = 0; i < 50; ++i) {
    parallel_for(SIZE_X, SIZE_Y, [&](int x, int y) {
      if (p.mask(x, y) == 0) {
        p.now(x, y) =
            (((p.last(x, y + 1) + p.last(x, y - 1)) * cfg.dy * cfg.dy +
              (p.last(x + 1, y) + p.last(x - 1, y)) * cfg.dx * cfg.dx) /
                 (2.0 * (cfg.dx * cfg.dx + cfg.dy * cfg.dy)) -
             cfg.dx * cfg.dx * cfg.dy * cfg.dy /
                 (2.0 * (cfg.dx * cfg.dx + cfg.dy * cfg.dy)) * b.now(x, y));
      }
    });

    parallel_for(SIZE_X, SIZE_Y, [&](int x, int y) {
      switch (p.mask(x, y)) {
      case 1:
        p.now(x, y) = p.now(x, y - 1);
        break;
      case 2:
        p.now(x, y) = p.now(x + 1, y);
        break;
      case 3:
        p.now(x, y) = p.now(x, y + 1);
        break;
      case 4:
        p.now(x, y) = 0;
        break;
      }
    });
    if (i != 49)
      p.tick();
  }

  // solve u and v
  parallel_for(SIZE_X, SIZE_Y, [&](int x, int y) {
    switch (u.mask(x, y)) {
    case 0:
      u.now(x, y) =
          (u.last(x, y) -
           u.last(x, y) * cfg.dt / cfg.dx * (u.last(x, y) - u.last(x, y - 1)) -
           v.last(x, y) * cfg.dt / cfg.dy * (u.last(x, y) - u.last(x - 1, y)) -
           cfg.dt / (2.0 * cfg.rho * cfg.dx) *
               (p.now(x, y + 1) - p.now(x, y - 1)) +
           cfg.nu *
               (cfg.dt / (cfg.dx * cfg.dx) *
                    (u.last(x, y + 1) - 2.0 * u.last(x, y) + u.last(x, y - 1)) +
                cfg.dt / (cfg.dy * cfg.dy) *
                    (u.last(x + 1, y) - 2.0 * u.last(x, y) +
                     u.last(x - 1, y))));
      break;
    case 1:
      u.now(x, y) = 0;
      break;
    case 2:
      u.now(x, y) = 1;
      break;
    }

    switch (v.mask(x, y)) {
    case 0:
      v.now(x, y) =
          (v.last(x, y) -
           u.last(x, y) * cfg.dt / cfg.dx * (v.last(x, y) - v.last(x, y - 1)) -
           v.last(x, y) * cfg.dt / cfg.dy * (v.last(x, y) - v.last(x - 1, y)) -
           cfg.dt / (2.0 * cfg.rho * cfg.dy) *
               (p.now(x + 1, y) - p.now(x - 1, y)) +
           cfg.nu *
               (cfg.dt / (cfg.dx * cfg.dx) *
                    (v.last(x, y + 1) - 2.0 * v.last(x, y) + v.last(x, y - 1)) +
                cfg.dt / (cfg.dy * cfg.dy) *
                    (v.last(x + 1, y) - 2.0 * v.last(x, y) +
                     v.last(x - 1, y))));
      break;
    case 1:
      v.now(x, y) = 0;
      break;
    }
  });
}

int main([[maybe_unused]] int argc, char **argv) {
  // extract the size from argument
  int SIZE_X, SIZE_Y;
  SIZE_X = SIZE_Y = std::atoi(argv[1]);

  std::printf("SIZE_X = %d, SIZE_Y = %d\n", SIZE_X, SIZE_Y);

  float TIME;
  TIME = std::atof(argv[2]);

  std::printf("TIME = %g\n", TIME);

  // initialize parameter
  Grid u{SIZE_X, SIZE_Y}, v{SIZE_X, SIZE_Y}, p{SIZE_X, SIZE_Y},
      b{SIZE_X, SIZE_Y};
  Config cfg{1.0, 0.1, TIME / FRAMES, 2.0f / (SIZE_X - 1), 2.0f / (SIZE_Y - 1)};

  // initialize boundary condition
  for (int y = 0; y < SIZE_Y; ++y) {
    for (int x = 0; x < SIZE_X; ++x) {
      b.mask(x, y) = p.mask(x, y) = u.mask(x, y) = v.mask(x, y) = 0;

      if (x == 0 || y == 0 || y == SIZE_Y - 1)
        u.mask(x, y) = 1;

      if (x == SIZE_X - 1)
        u.mask(x, y) = 2;

      if (x == 0 || y == 0 || x == SIZE_X - 1 || y == SIZE_Y - 1) {
        v.mask(x, y) = 1;
        b.mask(x, y) = 1;
      }

      if (y == SIZE_Y - 1)
        p.mask(x, y) = 1;

      if (x == 0)
        p.mask(x, y) = 2;

      if (y == 0)
        p.mask(x, y) = 3;

      if (x == SIZE_X - 1)
        p.mask(x, y) = 4;
    }
  }

  std::printf("boundary condition initialized\n");

  for (int i = 0; i < FRAMES; ++i) {
    // std::printf("\rcurrent at %d / %d", i, FRAMES);
    cavity(b, p, u, v, SIZE_X, SIZE_Y, cfg);
    b.tick();
    p.tick();
    u.tick();
    v.tick();
  }

  std::printf("finished\n");

  // exit(0);
}