# Voxelyze3 (Voxelyze that runs on DeepGreen)

[A basic manual of DeepGreen](https://wiki.uvm.edu/w/DeepGreenDocs)

## Usage:

This happens in DeepGreen Server environment:

1. Store .vxa files in a new folder, e.g. `~/data/generation_n/*.vxa`;

2. Call Voxelyze3 to process those .vxa files, e.g. `vx_run -i ~/data/generation_n/ -o report.xml`; (`vx_run` will divide .vxa files into batches and send them to different nodes and collect results, and every nodes have 8 GPUs, so we can utilize as many GPUs as we can.)

3. After the process exits, `report.xml` is ready for read.

## Usage (if you don't have access to DeepGreen):

1. Store .vxa files in a new folder, e.g. `~/data/generation_n/*.vxa`;

2. Call Voxelyze3, e.g. `Voxelyze3 -i ~/data/generation_n/ -o report.xml`;

3. After the process exits, `report.xml` is ready for read.

## Installation

This happens in DeepGreen Server environment:

```bash
git clone git@github.com:liusida/gpuVoxels.git
cd gpuVoxels/Voxelyze3
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 10
```

## Updates

### version 0.1

The idea of version 0.1 is to prove the basic idea of GPUlization can work.

Known issue: no collision.

