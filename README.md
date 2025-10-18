## Machine Learning + Stats Library 

---
Baylor ECS - BearAI

All associated papers + derivations included (where applicable)

Educational ML/DL framework built from scratch

Accessible Machine Learning

A living framework that evolves with current ML breakthroughs

blog imfindingithard.com

---

## Update: 
The plan is to continue working on this to make it more robust, I've taken a bit of a break. Roadmap is subject to change and I just write whatever is the most fun.
Current workings:
- [ ] More Stats 
- [ ] Working on C++ header only library might use the bindings here but not sure yet [Dailia](https://github.com/lovechants/dalia)
- [ ] Whatever is in Scikit-Learn is going to be implemented here 
- [ ] polynomial chaos expansion (cool name)
- [ ] advance algo calls
- [ ] model zoo 
- [ ] quantization 
- [ ] more kernels 
- [ ] working on easier metal library 
- [ ] proper docs 

---

## Features 

- AutoGrad
- First order + second order operations.
    - Check examples
- Multiple Backends
    - LLVM 
    - CUDA
    - Metal
    - Clang 
    - Triton + Native Julia Tensors without PyTorch dependancy
- Onnx Support 
- Terminal vis support through [Aliyah](https://github.com/lovechants/Aliyah/tree/main)

## Contributing 
Any welcome.
```bash
Fork the repo
git clone https://github.com/your-username/your-fork
Create a branch: git checkout -b your-feature-branch
Make changes
Test 
Commit your changes: git commit -m "Add feature-that-you-added"
Push to the branch: git push origin your-feature-branch
Submit a pr
```
Please add tests.

#### Requirements
1. Python3
2. Numpy

Optional

3. llvmlite
4. pycuda
5. onnx 
6. clang 

## Testing 

> All tests are run with pytest

```bash
~/julia pytest -vvs [additional args]
```

## Roadmap
- [ ] Finish full python autograd impl 
- [ ] More optimization passes 
- [ ] Model quantization 
- [ ] Full model zoo 
- [X] Full in depth summary
- [ ] Rust or zig autograd engine (Long term)
- [ ] Remove numpy dependancy (Long term)
- [ ] Meta-Learning WASM UX (Long term)


## Current TODO
- [X] Refactor conv2d im2col by vectorization
- [ ] Optimizer state management
- [X] Fix backward profiling 
- [X] Memory pooling (finish)
- [X] Auto-registration system for operations
- [x] Finish IR 
    - [X] Fix test `shape_test::matmul`
    - [ ] Zig bindings -> and optimize function 
- [x] Finish LLVM -> Actually optimize it  
- [ ] Fix CLANG setup
- [x] Set up ONNX 
- [ ] Backend support 
    - [ ] CUDA 
    - [ ] OpenCL 
    - [ ] Metal 
    - [x] CLANG 
    - [ ] ZIG 
    - [ ] WASM 
    - [ ] TPU/NPU?
    - [ ] x86
- [ ] Pure NN primitives  
    - [X] Recurrent autograd 
    - [X] Attention blocks
    - [ ] RL components
    - [x] Optimizers 
        - [x] Muon 
        - [x] Adam 
        - [x] AdamW 
        - [x] RMSProp 
        - [x] LAMB 
        - [x] AdaGrad
        - [x] ReLu + Derivations 
        - [x] Sigmoid 
- [ ] Profiling tools 
    - [ ] Easy Vis + Metric tracking 
    - [x] Profiler
- [ ] Model Zoo + Stats
- [X] Datasets
- [ ] Benchmarking util
- [ ] Testing suite util
