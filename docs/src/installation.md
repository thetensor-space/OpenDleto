# Installation

## Requirements

- Julia 1.7 or later
- ITensors.jl for tensor operations

## Installing from Source

Since Dleto.jl is currently in development, install directly from the repository:

```julia
using Pkg
Pkg.add(url="https://github.com/your-username/OpenDleto")
```

Or for development:

```julia
Pkg.develop(url="https://github.com/your-username/OpenDleto")
```

## Dependencies

The package automatically installs required dependencies:

- ITensors.jl - Core tensor operations
- LinearAlgebra.jl - Linear algebra routines  
- DocStringExtensions.jl - Documentation utilities
- And other standard Julia packages

## Testing the Installation

Verify your installation by running the test suite:

```julia
using Pkg
Pkg.test("Dleto")
```

## Getting Help

- Check the [API Reference](@ref) for detailed function documentation
- See [Examples](@ref) for usage patterns
- Review the test files in `test/` for comprehensive usage examples