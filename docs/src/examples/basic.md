# Basic Usage

This page demonstrates basic usage patterns for Dleto.jl.

## Creating and Using Operators

```julia
using Dleto, ITensors

# Create an index
a = Index(3, "a")

# Create an operator from a matrix
M = [2.0 0.5 0.0; 0.5 1.8 0.2; 0.0 0.2 1.5]
ω = asoperator(M, a)

# Verify it's a valid operator
@assert member(ω)

# Create a tensor to operate on
b = Index(2, "b")
Γ = randomITensor(a, b)

# Apply the operator
result = apply(ω, Γ, a)
```

## Testing Operator Laws

Dleto.jl operators satisfy mathematical laws that you can verify:

```julia
# Create test data
Γ = randomITensor(a, b)
Δ = randomITensor(a, b)
s = 2.5

# Test linearity
@assert isapprox(apply(ω, Γ + Δ, a), apply(ω, Γ, a) + apply(ω, Δ, a))
@assert isapprox(apply(ω, Γ * s, a), apply(ω, Γ, a) * s)
```

## Working with Different Operator Types

```julia
# Identity operator
I_op = asoperator(Matrix(I, 3, 3), a)

# Diagonal operator  
D = Matrix(Diagonal([1.0, 2.0, 3.0]))
diag_op = asoperator(D, a)

# Symmetric operator
S = [1.0 0.5 0.2; 0.5 2.0 0.3; 0.2 0.3 1.5]
sym_op = asoperator(S, a)
```