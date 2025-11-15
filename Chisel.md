# Chisel

Fix $\mathbb{K}$ a number type with an addition and a distributive multiplication. Fix a finite set $A$ called **axes** and a function $d:A\to \mathbb{N}$ of dimensions for each axis.  An **$(A,d)$-tensor** is a function $\Gamma:\prod_{a:A}[d_a]\to \mathbb{K}$.  Finally a **chisel** is a function (a matrix) $C:B\times A\to \mathbb{K}$ with rows indexed by a finite set $B$.

The chisel equation is the following system of linear equations in variables $X_a\in \mathbb{K}^{[d_a]\times [d_a]}$ for each $a\in A$.
\[
(\forall b\in B)\qquad
\sum_{a\in A} \sum_{l=1}^{d(a)} C(b,a)X_a(i_a,l)\Gamma(l_a,i_{\bar{a}})=0
\]

```julia
using LinearAlgebra

# Example data
A = [:x, :y]                # axes
d = Dict(:x => 2, :y => 3)  # dimensions for each axis
B = [:b1, :b2]              # rows

# Example chisel matrix C: B × A → ℝ
C = Dict((:b1, :x) => 1.0, (:b1, :y) => 2.0,
         (:b2, :x) => 0.5, (:b2, :y) => -1.0)

# Example tensor Γ: ([dₓ], [dᵧ]) → ℝ
Γ = randn(d[:x], d[:y])

# Variables Xₐ: for each a ∈ A, a dₐ × dₐ matrix
X = Dict(a => randn(d[a], d[a]) for a in A)

# Build the system of equations
eqs = []
for b in B
    eq = 0.0
    for a in A
        for i in 1:d[a], l in 1:d[a]
            # For simplicity, assume iₐ = i, lₐ = l, i_{bar{a}} = 1 (fix indices)
            eq += C[(b, a)] * X[a][i, l] * Γ[l, 1]
        end
    end
    push!(eqs, eq == 0)
end

# Print the equations
for (i, eq) in enumerate(eqs)
    println("Equation $(i): ", eq)
end
```