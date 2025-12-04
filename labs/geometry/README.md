# SphereLab - Geometry Lab for OpenDleto

This directory contains the SphereLab notebook, which demonstrates how to use Dleto to recover surface structures from randomized tensor data.

## Quick Start

### Prerequisites

1. **Julia** (version 1.7 or later)
   - Download from: https://julialang.org/downloads/

2. **Required Julia Packages**:
   - IJulia (for Jupyter notebook support)
   - Arpack (for eigenvalue computations)
   - PlotlyJS (for 3D visualizations)

### Installation

1. Clone the OpenDleto repository:
   ```bash
   git clone https://github.com/thetensor-space/OpenDleto.git
   cd OpenDleto
   ```

2. Install required Julia packages:
   ```julia
   using Pkg
   Pkg.add("IJulia")
   Pkg.add("Arpack")
   Pkg.add("PlotlyJS")
   ```

3. Register the Julia kernel with Jupyter (first time only):
   ```julia
   using IJulia
   installkernel("Julia")
   ```

### Running the Notebook

#### Option 1: Using Jupyter

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Navigate to `labs/geometry/SphereLab.ipynb`

3. Select the Julia kernel

4. Run the cells sequentially

#### Option 2: Using VS Code

1. Open the repository in VS Code

2. Install the Jupyter extension

3. Open `labs/geometry/SphereLab.ipynb`

4. Select the Julia kernel

5. Run the cells sequentially

### Verification

To verify that your environment is correctly set up, run the test script:

```bash
cd labs/geometry
julia test_spherelab.jl
```

This will test all the core functionality used in the notebook.

## What the Notebook Demonstrates

The SphereLab notebook walks you through:

1. **Loading the Dleto library** - Setting up the Julia environment with IJulia and loading OpenDleto

2. **Creating a sphere tensor** - Building a 3D tensor with values clustered near a spherical surface

3. **Randomization** - Applying random orthogonal transformations to hide the structure

4. **Recovery** - Using Dleto's algorithms to recover the surface structure from randomized data

5. **Visualization** - Creating interactive 3D plots to visualize the tensors

6. **Validation** - Measuring how well the recovered structure matches the original

## Key Functions

- `randomSurfaceTensor(u, v, w, t)` - Creates a tensor with surface structure
- `randomizeTensor(tensor)` - Applies random orthogonal transformations
- `toSurfaceTensor(tensor)` - Recovers surface structure
- `testSurfaceTensor(...)` - Validates recovery quality
- `plotTensor(tensor, threshold)` - Creates 3D visualization

## Troubleshooting

### "Package not found" errors

Make sure you've installed all required packages:
```julia
using Pkg
Pkg.add(["IJulia", "Arpack", "PlotlyJS"])
```

### "Julia kernel not found"

Register the Julia kernel:
```julia
using IJulia
installkernel("Julia")
```

Then restart Jupyter.

### WebIO errors

If you encounter WebIO-related errors, set the environment variable:
```julia
ENV["WEBIO_JUPYTER_DETECTED"] = "true"
```

### Plotting issues

PlotlyJS requires a working display. In headless environments (like CI/CD), plots may not render but the code will still execute.

## Additional Resources

- [OpenDleto Main Repository](https://github.com/thetensor-space/OpenDleto)
- [TheTensor.Space](https://TheTensor.Space/) - Main project website
- [Julia Documentation](https://docs.julialang.org/)
- [IJulia Documentation](https://github.com/JuliaLang/IJulia.jl)

## License

See the main repository LICENSE file for details.
