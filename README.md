# XGrid Benchmark

Benchmark and test cases for XGrid

## Requirements

```bash
pip3 install xgrid numpy matplotlib tqdm
```

Compile `cavity.cc`:

```bash
cc -O2 -fopenmp cavity.cc -o cavity.exe
```

## Run

```bash
python3 experiment.py run
```