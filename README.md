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
- [ ] Finish LLVM -> Actually optimize it  
- [x] Set up ONNX 
- [ ] Backend support 
    - [ ] CUDA 
    - [ ] OpenCL 
    - [ ] Metal (arm) -> Same shader (swift/obj c) dynamic lib needed (afaik)
    - [ ] CLANG 
    - [ ] ZIG 
    - [ ] WASM 
    - [ ] TPU/NPU?
    - [ ] x86
- [ ] Pure NN primitives 
    - [ ] Layers 
    - [ ] Optimizers 
        - [ ] Muon 
        - [ ] Adam 
        - [ ] AdamW 
        - [ ] RMSProp 
        - [ ] LAMB 
        - [ ] AdaGrad
        - [ ] ReLu + Derivations 
        - [ ] Sigmoid 
- [ ] Profiling tools 
    - [ ] Easy Vis + Metric tracking 
- [ ] Model Zoo + Stats 
