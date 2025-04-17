## Machine Learning + Stats Library 

---
Skeleton development environment for BearAI machine learning and stats library

All associated papers + derivations included (where applicable)

LLVM compilation + ONNX runtime + Primitives

## Testing 

> All tests are run with pytest

```bash
~/julia pytest -vvs [additional args]
```
## Current TODO 
- [x] Finish IR 
    - [ ] Fix test `shape_test::matmul`
    - [ ] Zig bindings -> and optimize function 
- [x] Finish LLVM -> Actually optimize it  
- [ ] Fix CLANG setup
- [x] Set up ONNX 
- [ ] Backend support 
    - [ ] CUDA 
    - [ ] OpenCL 
    - [ ] Metal (arm) -> Same shader (swift/obj c) dynamic lib needed (afaik)
    - [x] CLANG 
    - [ ] ZIG 
    - [ ] WASM 
    - [ ] TPU/NPU?
    - [ ] x86
- [ ] Pure NN primitives 
    - [ ] Layers 
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
- [ ] Model Zoo + Stats
