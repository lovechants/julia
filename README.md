## Machine Learning + Stats Library 

---
Skeleton development environment for BearAI machine learning and stats library
All associated papers + derivations included (where applicable)
LLVM compilation + ONNX runtime + Primitives

## Testing 

> All tests are run with pytest

```bash
~/julia pytest [args]
```
## Current TODO 
- [ ] Finish IR + LLVM compilation
- [ ] Set up ONNX 
- [ ] Backend support 
    - [ ] CUDA 
    - [ ] OpenCL 
    - [ ] Metal -> Same shader (swift/obj c) dynamic lib needed (afaik)
- [ ] Pure NN primitives 
    - [ ] Layers 
    - [ ] More optimizers
