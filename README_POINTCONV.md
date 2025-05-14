# PointConv Layer

This repository provides a high-performance implementation of the PointConv layer, leveraging optimized CUDA kernels via [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) for improved memory efficiency and runtime performance on 3D point clouds.

## 🚀 Installation

**Clone the repo**

```
git clone https://github.com/Deep-Machine-Vision/ml-pointconvformer.git
cd ml-pointconvformer
```

**Install dependencies.** This project was tested with:
- PyTorch 2.7
- CUDA 11.8
- GCC 9.5

You can install PyTorch from [here](https://pytorch.org/get-started/previous-versions/).


**Clone CUTLASS**. This implementation depends on [CUTLASS](https://github.com/NVIDIA/cutlass) for optimized CUDA kernels. Clone it in the project root directory:
```
git clone https://github.com/NVIDIA/cutlass.git
```
🔧 Note: The relative path to CUTLASS is specified in [cpp_wrappers/cpp_pcf_kernel/setup.py](./cpp_wrappers/cpp_pcf_kernel/setup.py)


**Build CUDA Wrappers**

Install the custom CUDA C++ extensions:
```
sh setup.sh
```

This should install the `PCFcuda` repo that includes the optimized kernels for `PointConv` using CUTLASS.

## ⚡ Quickstart

## Running Tests

To run tests for PointConv, use the test files in `tests_pointconv/`:

**Simple PointConv Layer on a Random point cloud**


```bash
python tests_pointconv/test_pointconv_single.py 
```

**📦  PointConv layer with packed representations**

```bash
python tests_pointconv/test_pointconv_packed.py 
```

## Running PointConv Encoder

You can look into [encoder.py](./tests_pointconv/encoder.py) to see an example where a batch of random point clouds in (packed representation form)[https://pytorch3d.org/docs/batching] are encoded via two PointConv layers. You can run this by:

```bash
python tests_pointconv/encoder.py 
```

### PointConv optimizations
To enable optimized PointConv CUDA kernels for saving peak GPU memory usage, set `USE_CUDA_KERNEL: True` and `PCONV_OPT: True` in the configuration file. Refer to ['pointconv_packed.yaml'](./test_configs/pointconv_packed.yaml). Use the fused PointConv and Linear layer forward and backward kernels, Compute kNN Inverse mapping indices for all the edges, use the inverse indices in the backward kernel to compute gradients.


## 📬 Feedback & Contributions
Feel free to open issues or pull requests! Contributions to enhance usability, performance, or documentation are welcome.