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

Run the manufactured-solution convergence study:

```bash
python examples/convergence_study.py --levels 2 4 8 --vtk-dir results --csv results/convergence.csv
```

The study solves a quadratic analytical displacement field with matching body
force and exact Dirichlet boundary conditions, then reports L2 and H1-seminorm
rates. For first-order tetrahedra the expected rates are approximately second
order in L2 and first order in H1.

Generate a simple loaded beam result for ParaView:

```bash
python examples/beam_traction.py --output results/beam_traction.vtk
```

After installing the package, the same workflows are available as CLI commands:

```bash
fem3d convergence --levels 2 4 8 --csv results/convergence.csv
fem3d beam --output results/beam_traction.vtk
```
