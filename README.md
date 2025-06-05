## Machine Learning + Stats Library 

---
Skeleton development environment for BearAI machine learning and stats library

All associated papers + derivations included (where applicable)

Educational ML/DL framework built from scratch 

---

## Features 

- AutoGrad
- Multiple Backends
    - LLVM 
    - CUDA
    - Metal
    - Clang 
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
- [ ] In depth summary 
- [ ] Rust or zig autograd engine (Long term)
- [ ] Remove numpy dependancy (Long term)


## Current TODO
- [ ] Refactor conv2d im2col by vectorization
- [ ] Fix backward profiling 
- [ ] Memory pooling (finish)
- [ ] Auto-registration system for operations
- [x] Finish IR 
    - [ ] Fix test `shape_test::matmul`
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
    - [ ] Recurrent autograd 
    - [ ] Attention blocks
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
- [ ] Datasets
- [ ] Benchmarking util
- [ ] Testing suite util
