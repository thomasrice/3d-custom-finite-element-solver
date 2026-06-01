# Custom 3D Finite Element Solver

> **Note:** This project is just a demo of LLM-driven coding — it was written almost
> entirely by an AI coding agent to see how far that approach can get on a non-trivial
> numerical problem. It is not intended for production engineering use.

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

## Setup

Create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The core solver depends only on NumPy and SciPy. The PNG-rendering examples also
need Matplotlib installed in the environment:

```bash
python -m pip install matplotlib
```

Run tests:

```bash
python -m pytest
```

## Examples

Run the manufactured-solution convergence study:

```bash
python examples/convergence_study.py --levels 2 4 8
```

The study solves a quadratic analytical displacement field with matching body
force and exact Dirichlet boundary conditions, then reports L2 and H1-seminorm
rates. For first-order tetrahedra the expected rates are approximately second
order in L2 and first order in H1.

Generate VTK results for ParaView:

```bash
python examples/beam_traction.py
python examples/uniaxial_tension.py
python examples/self_weight_beam.py
python examples/bending_refinement.py
```

Generate PNGs that can be inspected without ParaView:

```bash
python examples/render_deformed_mesh.py
python examples/convergence_plot.py
```

Artifacts are written under `results/` by default:

- `beam_traction.vtk`: clamped beam with end traction
- `uniaxial_tension.vtk`: uniaxial tension with Poisson contraction
- `self_weight_beam.vtk`: clamped beam sag from body-force loading
- `bending_refinement.csv` and `bending_refinement/*.vtk`: tip deflection vs mesh density
- `deformed_von_mises.png` and `deformed_von_mises.vtk`: deformed mesh colored by von Mises stress
- `convergence.csv`, `convergence_error.png`, and `convergence_vtk/*.vtk`: manufactured-solution convergence outputs

After installing the package, the beam and convergence workflows are also
available as CLI commands:

```bash
fem3d convergence --levels 2 4 8 --csv results/convergence.csv
fem3d beam --output results/beam_traction.vtk
```

## Validation

The test suite covers:

- global equilibrium, linearity/superposition, and strain-energy consistency
- stiffness null-space rigid-body modes and rigid-body invariance
- uniaxial tension and simple shear physical benchmarks
- structured and distorted-mesh patch tests
- von Mises invariants
- direct vs CG solver agreement
- conflicting constraints, under-constrained systems, and invalid tetrahedra
- VTK tensor-output semantics, including engineering strain shear conversion
- manufactured-solution convergence rates

## Limitations

- First-order tetrahedra can shear-lock in bending; the bending example is a
  refinement trend demo, not a strict Euler-Bernoulli equality check.
- Mesh generation is limited to simple structured box meshes.
- This is an educational implementation, not production engineering software.

## License

Released under the [MIT License](LICENSE).
