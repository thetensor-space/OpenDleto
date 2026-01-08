# Operators

Comprehensive documentation for the operator interface and implementations.

## Operator Interface

The fundamental operator interface that all tensor operators must implement.

```@docs
Dleto.Operators
```

## Core Operator Functions

```@autodocs
Modules = [Dleto.Operators]
```

## ITensor Operators

Specific implementations for ITensor objects.

```@autodocs
Modules = [Dleto]
Pages = ["ops/ITensorOps.jl"]
```

## Operator Implementations

```@autodocs
Modules = [Dleto]
Pages = ["ops/OperatorImpls.jl"]
```

## Mathematical Laws

All operators in Dleto.jl satisfy these fundamental laws:

### Basic Tensor Operator Laws

For any operator `ω`, scalars `s`, tensors `Γ, Δ` with equal indices, and index `a`:

- **Additivity**: `apply(ω, Γ + Δ, a) ≈ apply(ω, Γ, a) + apply(ω, Δ, a)`
- **Scalar Compatibility**: `apply(ω, Γ * s, a) ≈ apply(ω, Γ, a) * s`  
- **Consistency**: `apply(ω, Γ, a) == unsafe_apply(ω, Γ, Δ, a)` if `member(ω) == true`
- **Index Preservation**: `inds(apply(ω, Γ, a)) == inds(Γ)` if `member(ω) == true`

### Linear Tensor Operator Laws

For linear operators `ω, λ` and scalar `s`:

- **Operator Additivity**: `apply(ω + λ, Γ, a) ≈ apply(ω, Γ, a) + apply(λ, Γ, a)`
- **Operator Scaling**: `apply(s * ω, Γ, a) ≈ s * apply(ω, Γ, a)`

### Coapplication Laws

- **Bilinearity**: `coapply(Λ₁ + Λ₂, Δ, a) ≈ coapply(Λ₁, Δ, a) + coapply(Λ₂, Δ, a)`
- **Duality**: `L ↦ coapply(L)` is dual to `ω ↦ apply(ω)`