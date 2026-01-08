# Dleto.jl

*Strata Dleto: Local Operators Abstract - Creation and application of local operators for tensors.*

Dleto.jl is a Julia package for tensor operations, operator applications, and advanced tensor decomposition algorithms. It provides a comprehensive framework for working with tensor operators, chiseling operations, and stratified analysis of tensor data.

## Features

- **Tensor Operators**: Complete interface for applying linear operators to tensors
- **Chiseling**: Advanced tensor decomposition and analysis tools  
- **ITensor Integration**: Seamless compatibility with the ITensors.jl ecosystem
- **Stratification**: Tools for analyzing tensor structure and patterns
- **Detection Algorithms**: Specialized algorithms for pattern detection in tensor data

## Quick Start

```julia
using Dleto, ITensors

# Create an operator
a = Index(3, "a")
M = [1.0 0.5 0.0; 0.5 2.0 0.3; 0.0 0.3 1.5]
ω = asoperator(M, a)

# Apply to a tensor
Γ = randomITensor(a, Index(2, "b"))
result = apply(ω, Γ, a)
```

## Package Components

```@contents
Pages = ["api/core.md", "api/operators.md", "api/chisels.md"]
Depth = 1
```

## Mathematical Foundation

Dleto.jl implements tensor operators that satisfy rigorous mathematical laws:

- **Linearity**: `apply(ω, Γ + Δ, a) ≈ apply(ω, Γ, a) + apply(ω, Δ, a)`
- **Scalar compatibility**: `apply(ω, Γ * s, a) ≈ apply(ω, Γ, a) * s`
- **Index preservation**: Operations maintain tensor structure
- **Operator composition**: Support for combining and composing operators

## Index

```@index
```