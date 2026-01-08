# Contributing

## Development Setup

1. Fork and clone the repository
2. Set up development environment:

```julia
using Pkg
Pkg.develop(path="/path/to/OpenDleto")
Pkg.instantiate()
```

## Code Style

- Follow standard Julia conventions
- Add docstrings to all exported functions
- Include type annotations where helpful
- Write comprehensive tests for new features

## Documentation

Documentation is automatically generated using Documenter.jl. To build locally:

```bash
cd docs/
julia --project make.jl
```

## Adding New Operators

When implementing new operator types:

1. Implement the core interface functions: `apply`, `unsafe_apply`, `member`
2. For linear operators, also implement: `coapply`, `unsafe_coapply`  
3. Add comprehensive tests in `test/`
4. Document with proper docstrings
5. Verify all operator laws are satisfied

## Testing

Run the full test suite:

```julia
using Pkg
Pkg.test("Dleto")
```

Run specific test files:

```julia
include("test/TestOperators.jl")
testAllOperators()
```