#
# Strata Dleto: Utils
#   Extending ITensors multiplication to AbstractArrays for convenience
#
# -----------------------------------------------------------------------------
# Copyright 2022-2026 Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the “Software”), 
# to deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.
#-----------------------------------------------------------------------------
using LinearAlgebra: I
using ITensors: ITensor, Index, inds, replaceinds, addtags, store, norm

"""
    Make ITensor out of AbstractArray without indices, 
    don't export this function is dangerous and makes 
    no promise to keep working in future versions of Dleto.jl.
    It is mainly here to conveniently fix a few places where 
    ITensors but 1691 is blocking proper use of AbstractArrays.
"""
function __ITensor(Γ::AbstractArray)::ITensor 
    frame = [ Index(size(Γ,a), "a$a") for a in 1:ndims(Γ) ]
    iΓ = try 
        ITensor(Γ, frame...) 
    catch e
        # a temporary fix for AbstractArray inputs like ReshapedArray 
        # which has a low-grade bug(?) in ITensors (Bug #1691)
        ITensor( Array(Γ), frame...)
    end
    return iΓ
end


"""
    match_idx[!](A::ITensor, a::Index,
               E::ITensor, e::Index, 
               tag::String
              )::NamedTuple{(:ae, :A, :E),Tuple{Index, ITensor, ITensor}}

    Given an ITensor `A` with index `a` and another ITensor `E` with index `e`,
    create a compound label `ae` and a shallow copy of both tensors with the index relabeled to `ae`.
    The function with `!` modifies the tensors in place.
""" 
function match_idx(
        A::ITensor, a::Index,
        E::ITensor, e::Index,
        tag::String="matched_idx"
    )::NamedTuple{(:ae, :A, :E),Tuple{Index, ITensor, ITensor}}
    @assert dim(a)==dim(e) "Indices must have the same dimension to be relabeled"

    ae = Index( dim(a), tag, tags(a))
    At = replaceind(A, a, ae )
    Et = replaceind(E, e, ae )
    return (;ae=ae,A= At, E= Et)
end
function match_idx!(
        A::ITensor, a::Index,
        E::ITensor, e::Index,
        tag::String="matched_idx"
    )::NamedTuple{(:ae, :A, :E),Tuple{Index, ITensor, ITensor}}
    ae = Index( dim(a), tag, tags(a))
    At = replaceind!(A, a, ae )
    Et = replaceind!(E, e, ae )
    return (;ae=ae,A= At, E= Et)
end


# -- Action of a list of operators on a tensor, one per axis --------------------
#
# These were `Base.:*(::ITensor, ::Vector{ITensor})`, `Base.:*(::AbstractArray,
# ::Vector{ITensor})`, `Base.:*(::ITensor, ::Vector{<:AbstractMatrix})`, their
# argument-swapped twins, a scalar chain, and `Base.:+(::ITensor, ::AbstractArray)`.
# None of those argument types is owned by Dleto, so every one of them was type
# piracy: a definition that can change what `*` and `+` mean for ITensors.jl and
# for every other package loaded next to it, and an invalidation magnet on every
# `using`.  They are now a named verb, `act`, with the same semantics.  Nothing in
# the package, the tests or the labs used the swapped forms, the scalar chain or
# the `+` methods, so those are gone without replacement.

"""
    act(Γ::ITensor, X::Vector{ITensor}) -> ITensor
    act(Γ::AbstractArray, X::Vector{ITensor}) -> ITensor
    act(Γ::ITensor, X::Vector{<:AbstractMatrix}) -> ITensor

The tensor `Γ` acted on by one operator per axis: `Γ ×₁ X₁ ×₂ X₂ ⋯`, each `X_a`
a two-index ITensor sharing one index with `Γ` (the contraction) and carrying the
new index the result lives on.  This is the change of frame `stratify` and
`randomize_tensor` apply; `act(Γ, Xs)` with the `Xs` those return reproduces
their `Σ` / `Δ`.

An `AbstractArray` `Γ` is read into the frame the `X` carry (one matrix per
axis, so the axis count must match).  A vector of plain matrices is embedded
against `Γ`'s own frame, with a fresh index per axis.
"""
function act(Γ::ITensor, X::Vector{ITensor})
    Σ = Γ
    for x in X
        Σ = Σ * x
    end
    return Σ
end

function act(Γ::AbstractArray, X::Vector{ITensor})
    length(X) == ndims(Γ) ||
        throw(DimensionMismatch("act: $(length(X)) operators for a tensor with $(ndims(Γ)) axes; one operator per axis is needed to read the array into the operators' frame."))
    fr = [ ITensors.inds(x)[1] for x in X ]
    iΓ = Γ isa Array ? ITensor(Γ, fr...) : ITensor(Array(Γ), fr...)
    return act(iΓ, X)
end

function act(Γ::ITensor, X::Vector{<:AbstractMatrix})
    length(X) == ndims(Γ) ||
        throw(DimensionMismatch("act: $(length(X)) matrices for a tensor with $(ndims(Γ)) axes; one matrix per axis is needed."))
    fr = inds(Γ)
    iX = [ ITensor(Array(X[i]), fr[i], __new_index_for_change_of_basis(fr[i])) for i in 1:length(X) ]
    return act(Γ, iX)
    # MDK This does not work if Γ = random_itensor(i,i',i'')!!!!!
end


# Define ⊕ as a new operator (not extending Base since it doesn't exist there)
⊕(Γ::ITensor, Δ::ITensor) = begin
    first_frame = ITensors.inds(Γ)
    second_frame = ITensors.inds(Δ)
    Σ, fr = ITensors.directsum(Γ=>first_frame, Δ=>second_frame)
    return Σ
end

⊕(Γ::AbstractArray, Δ::AbstractArray) = begin
    iΓ = __ITensor(Γ)
    iΔ = __ITensor(Δ)
    return iΓ ⊕ iΔ
end


⊕(Γ::ITensor, Δ::AbstractArray) = begin
    iΔ = __ITensor(Δ)
    return Γ ⊕ iΔ
end
⊕(Γ::AbstractArray, Δ::ITensor) = begin
    iΓ = __ITensor(Γ)
    return iΓ ⊕ Δ
end

# --- Utiliity functions ---

__isapproxzero(x::Number)::Bool = isapprox(x,0.0);


# to be moved into Utils.jl
"""
    realCanonicalForm(M::AbstractMatrix; tol = 1e-10) -> (; D, T)

Real canonical form of a real square matrix: `M * T == T * D` with `T` real and
invertible and `D` real block-diagonal -- a `1x1` block `λ` for every real
eigenvalue and a `2x2` block `[a b; -b a]` for every conjugate pair `a ± ib`,
whose two columns of `T` are the real and imaginary parts of the eigenvector of
`a + ib`.  A symmetric `M` is diagonalised directly.

Pairs are recognised BY THE EIGENVALUE: `|imag(λ)| > tol * max(1, |λ|)`.  The
previous implementation looked at the eigenvectors instead -- two consecutive
eigenvectors whose real parts nearly coincided were taken for a conjugate
pair -- so a nearly defective matrix with two REAL, nearly parallel eigenvectors
(`[1 1; 0 1 + 1e-6]`) was written as a complex block with a zero column in `T`,
singular, and the law above failed.  (docs/review/OpenDleto-vs-Magma.md, RISK3.)

LAPACK returns a conjugate pair as adjacent eigenvalues `λ, conj(λ)`; that is
checked rather than assumed, and the partner is found and moved next to its
mate if it is not.

```julia
res = realCanonicalForm(M); isapprox(M * res.T, res.T * res.D)
```
"""
function realCanonicalForm(M::AbstractMatrix; tol::Real=1e-10)::NamedTuple{(:D, :T), Tuple{AbstractMatrix,AbstractMatrix}}
    n = size(M, 1)
    n == size(M, 2) || throw(DimensionMismatch("realCanonicalForm: the matrix must be square, got $(size(M))."))
    eltype(M) <: Real || throw(ArgumentError("realCanonicalForm: a REAL canonical form needs a real matrix, got eltype $(eltype(M))."))
    RT = float(eltype(M))
    if all(x -> abs(x) < tol, M)
        # The zero matrix: identity frame.
        return (; D = zeros(RT, n, n), T = LinearAlgebra.Diagonal(ones(RT, n)))
    end
    if M isa LinearAlgebra.Symmetric || LinearAlgebra.issymmetric(M)
        eig = LinearAlgebra.eigen(LinearAlgebra.Symmetric(Matrix(M)))
        return (; D = LinearAlgebra.Diagonal(eig.values), T = eig.vectors)
    end
    eig = LinearAlgebra.eigen(Matrix(M))
    λ = collect(eig.values)
    V = Matrix(eig.vectors)
    isreal_ev(z) = abs(imag(z)) <= tol * max(one(RT), abs(z))

    D = zeros(RT, n, n)
    T = zeros(RT, n, n)
    i = 1
    found_complex = false
    while i <= n
        if isreal_ev(λ[i])
            D[i, i] = real(λ[i])
            T[:, i] = real.(V[:, i])
            i += 1
            continue
        end
        i < n || error("realCanonicalForm: the eigenvalue $(λ[i]) has no conjugate partner; the input is not a real matrix with a real spectrum structure.")
        # The partner should be next; if LAPACK put it elsewhere, bring it here.
        j = i + 1
        if !isapprox(λ[j], conj(λ[i]); atol = tol * max(one(RT), abs(λ[i])))
            k = argmin([abs(λ[m] - conj(λ[i])) for m in (i + 1):n]) + i
            λ[j], λ[k] = λ[k], λ[j]
            V[:, j], V[:, k] = V[:, k], V[:, j]
        end
        a, b = real(λ[i]), imag(λ[i])
        # M (u + iv) = (a + ib)(u + iv)  =>  M u = a u - b v,  M v = b u + a v,
        # i.e. M [u v] = [u v] [a b; -b a].
        T[:, i]     = real.(V[:, i])
        T[:, i + 1] = imag.(V[:, i])
        D[i, i] = a;      D[i, i + 1]     = b
        D[i + 1, i] = -b; D[i + 1, i + 1] = a
        found_complex = true
        i += 2
    end
    return (; D = found_complex ? D : LinearAlgebra.Diagonal(LinearAlgebra.diag(D)), T = T)
end
