# Custom 3D Finite Element Solver

This repository contains a from-scratch linear-elasticity finite element solver for
3D tetrahedral meshes. It uses NumPy/SciPy for arrays, sparse matrices, and sparse
linear solves; no finite element libraries are used.

Implemented features:

- linear isotropic elasticity on first-order tetrahedra
- structured tetrahedral mesh generation for boxes/beams
- sparse global stiffness assembly
- body-force and boundary-traction loading
- Dirichlet displacement boundary conditions
- legacy VTK output for ParaView
- validation tests including a constant-strain patch test and a manufactured
  mesh-convergence study

Run tests:

```bash
python -m pytest
```
