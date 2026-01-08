# Tensor Operations

Advanced tensor manipulation examples.

## Tensor Synthesis

```julia
using Dleto, ITensors

# Create a random tensor and randomize it
Γ = randn(2, 3, 4)
Ξ, Xs = randomize_tensor(Γ)

# Verify the relationship
@assert isapprox(Γ * Xs, Ξ)
```

## Index Collision Handling

When working with operators, Dleto.jl automatically handles index conflicts:

```julia
a = Index(3, "a")
ω = asoperator(randn(3, 3), a)

# Create tensor with both a and a' indices (potential conflict)
Γ = randomITensor(a, a', Index(2, "b"))

# Apply operator - collision is handled automatically
result = apply(ω, Γ, a)
```

## Coapplication

Create operators from tensor contractions:

```julia
# Create two tensors with same indices
a, b = Index(3, "a"), Index(2, "b")
Λ = randomITensor(a, b)
Δ = randomITensor(a, b)

# Create operator via coapplication
ω = unsafe_coapply(Λ, Δ, a)

# Use the resulting operator
Γ = randomITensor(a, b)
result = apply(ω, Γ, a)
```