#
# DletoMetalSylver -- `sylvesterLM(...; backend = :metal)`.
#
# Included by ext/DletoMetalExt.jl, so `Metal`, `LinearAlgebra` and `Dleto` are
# already in scope.  Everything Metal-specific in the SylverLining kernel lives
# here; src/SylverLining/SylverLining.jl knows only that `:metal` asks for a
# device kernel and hands it `Dleto._sylver_plan`.
#
# WHAT IS ON THE DEVICE AND WHAT IS NOT.  The expensive objects are the mode-`p`
# unfoldings of Γ and the residual, all O(d^n); they are uploaded once at
# construction and never come back.  The cheap objects are the transverse
# operator coordinates, dimension Σ_a d_a² (3·10⁴ for a 100³×3 colour video
# against 3·10⁶ tensor entries), and those are the only thing that crosses the
# bus per apply of the SQUARE map -- which is the map an eigensolver applies
# 500-2000 times.  So:
#
#   sylvester!  host coords -> Σ d_a² up  ... all O(d^n) work on device ...
#                              Σ d_a² down -> host coords         (the hot path)
#   ester!      host coords up, the m·d^n residual DOWN            (m·d^n bus)
#   sylve!      the m·d^n residual UP, host coords down            (m·d^n bus)
#
# `ester!`/`sylve!` cannot avoid moving the big vector -- it is their argument.
# They exist for the transpose law and for `den`; the derivation solve does not
# use them.  Both are still on the device for the arithmetic.
#
# The maps stay CPU-FACING: Float32 (or Float64) host vectors in and out, so
# every null solver in src/solvers/NullSolvers.jl works against `:metal`
# unchanged and unaware.  Metal has no fp64, so the arithmetic is Float32
# whatever `eltype(Γ)` is; a Float64 Γ therefore gets fp32 answers in a
# Float64-typed map (the map's eltype has to match what the solver holds).
#
# NO SCALAR INDEXING.  `MtlArray` getindex/setindex! on the host is disallowed
# (Metal.jl throws) and would be catastrophic anyway.  Every device line below
# is a `mul!`, a `permutedims!`, a `fill!` or a broadcast; the two places the
# CPU kernel writes an element loop -- embedding the operator coordinates and
# scattering a term across the chisel rows -- are handled respectively on the
# host (the matrices are small) and by a rank-1 broadcast.
#

using ITensors: ITensor
using Dleto: TransverseOps

# --- device-side helpers, dispatching on MtlArray ----------------------------

# Upload/download of the small per-axis matrices, into buffers allocated once.
# `Dleto.to_gpu` would be the documented way in, but it ALLOCATES a new device
# buffer per call, and this runs `engsize` times per apply inside an
# eigensolver's inner loop; `copyto!` into a preallocated buffer is the same
# transfer with no allocation.  `to_gpu` is still what does the one-time
# construction-time uploads below.
_sylver_up!(dst::MtlArray{Float32}, src::Array{Float32}) = copyto!(dst, src)
_sylver_down!(dst::Array{Float32}, src::MtlArray{Float32}) = copyto!(dst, src)

# The device is asynchronous: `mul!` and friends only enqueue.  Host reads must
# wait.  Metal.jl's `copyto!` to host does synchronize, but saying so here is
# what makes the ordering a property of this file rather than of a Metal.jl
# implementation detail.
_sylver_sync() = Metal.synchronize()

"""
    _sylverlm_metal(Ω, P, Γ)

`sylvesterLM`'s array kernel with the unfoldings and the residual resident on
an Apple GPU.  Same algebra, same layout, same vectors (to fp32 rounding) as
`Dleto._sylvesterLM_array`; see the block comment in
src/SylverLining/SylverLining.jl for both.

Dense only.  The CPU kernel's `SparseMatrixCSC` branch has no device twin --
Metal.jl has no sparse `mul!` -- and a sphere-octant Γ belongs on the CPU
anyway, where the sparse branch already beats dense by 2-3x.
"""
function _sylverlm_metal(Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor)
    Metal.functional() ||
        error("sylvesterLM: backend = :metal -- Metal is loaded but not " *
              "functional on this machine (`Metal.functional()` is false).")

    pl = Dleto._sylver_plan(Ω, P, Γ)
    T = pl.T
    T <: Union{Float32,Float64} ||
        error("sylvesterLM: backend = :metal supports Float32 and Float64 " *
              "tensors (Float64 computed in fp32); got eltype(Γ) = $T.")
    S = Float32                     # the device element type, always

    engsize, n, m, N = pl.engsize, pl.n, pl.m, pl.N
    dims, da, Ka, flat = pl.dims, pl.da, pl.Ka, pl.flat
    perms, iperms, Pm = pl.perms, pl.iperms, pl.Pm

    # THE PERMUTE IS THE BOTTLENECK, so two of the axes dodge it.  Measured on
    # an M4 Max at (200,200,100,3), one axis of the mode product:
    #
    #     permutedims! (48 MB in, 48 MB out)   3.11 ms   (~31 GB/s)
    #     mul! (K x d)·(d x d)                 1.44 ms
    #     mul! (d x K)ᵗ·(K x d)                1.24 ms
    #     x .+= α .* y over N                   0.52 ms
    #
    # GPUArrays' generic `permutedims!` runs at under a tenth of this device's
    # bandwidth, so it costs more than the GEMM it feeds.  Two axes need none:
    #
    #   axis n ("flat"):  the unfolding is `reshape(Γ, N/d, d)` and the product
    #     `U·M` lands in natural order.  (The CPU kernel already uses this.)
    #   axis 1 ("head"):  `A = reshape(Γ, d, N/d)` is the axis-1-first
    #     unfolding, and `Mᵗ·A` ALSO lands in natural order -- axis 1 is the
    #     fastest-varying index of both.  Its adjoint is `A·Zᵗ`.  Same algebra,
    #     same numbers; the mode product just happens to be written on the
    #     other side.  (The CPU kernel cannot use this one: it is the
    #     `dense × sparse` orientation that SparseArrays has no kernel for --
    #     see the block comment in SylverLining.jl.  On a dense device GEMM
    #     the orientation is free.)
    #
    # So valence 3 pays 1 permute per apply instead of 2, and valence 4 pays 2
    # instead of 3 -- and axis 1 needs no permuted copy of Γ on the device
    # either, which is N floats of device memory saved as well.
    head = [pl.axpos[a] == 1 for a in 1:engsize]
    mid = [!head[a] && !flat[a] for a in 1:engsize]

    # --- one-time uploads ----------------------------------------------------
    # Γ in natural axis order, then one unfolding per MIDDLE axis.  Those are
    # permuted ON THE DEVICE, so the host never holds more than the single
    # `pl.GA` copy however many axes there are.
    Gdev = Dleto.to_gpu(pl.GA)::MtlArray{S,n}
    permuted_shape(a) = ntuple(k -> dims[perms[a][k]], n)
    zmat = MtlArray{S}(undef, 0, 0)
    zten = MtlArray{S}(undef, ntuple(_ -> 0, n))
    # `Udev[a]` -- the (N/d_a) x d_a unfolding, axis `a` in the COLUMNS.
    Udev = [begin
                if head[a]
                    zmat                        # `Ahead[a]` is used instead
                elseif flat[a]
                    reshape(Gdev, Ka[a], da[a])
                else
                    tmp = MtlArray{S}(undef, permuted_shape(a))
                    permutedims!(tmp, Gdev, perms[a])
                    reshape(tmp, Ka[a], da[a])
                end
            end for a in 1:engsize]
    # `Ahead[a]` -- the d_a x (N/d_a) unfolding, axis `a` in the ROWS; only
    # meaningful, and only free, for axis 1.
    Ahead = [head[a] ? reshape(Gdev, da[a], Ka[a]) : zmat for a in 1:engsize]

    # --- device scratch ------------------------------------------------------

    # ONE shared scratch buffer for the unfolded term, not one per axis: every
    # unfolding has exactly `N` entries (`Ka[a] · da[a] == N`), only a different
    # shape, and the axes are visited strictly one at a time in both closures.
    # The CPU kernel keeps `engsize` separate buffers; at 300×300×300×3 that is
    # 972 MB against 324, which on a unified-memory device is worth having.
    need_W = any(a -> mid[a], 1:engsize)
    Wbuf = MtlArray{S}(undef, need_W ? N : 0)
    Wmat = [mid[a] ? reshape(Wbuf, Ka[a], da[a]) : zmat for a in 1:engsize]
    Wten = [mid[a] ? reshape(Wbuf, permuted_shape(a)) : zten for a in 1:engsize]

    Fold = MtlArray{S}(undef, dims)         # one term, natural axis order
    Foldv = reshape(Fold, N)
    Foldkd = [reshape(Fold, Ka[a], da[a]) for a in 1:engsize]
    Foldhd = [reshape(Fold, da[a], Ka[a]) for a in 1:engsize]

    Rdev = MtlArray{S}(undef, m, N)         # the residual, chisel axis FIRST
    Rvec = reshape(Rdev, m * N)
    # `m == 1` -- the universal and adjoint chisels, i.e. every derivation solve
    # that actually runs -- is its own chisel collapse up to the scalar P[1,a],
    # and that scalar rides in the GEMM's α.  See the CPU kernel.
    Rnat = m == 1 ? reshape(Rdev, dims) : zten
    Rkd = [m == 1 ? reshape(Rdev, Ka[a], da[a]) : zmat for a in 1:engsize]
    Rhd = [m == 1 ? reshape(Rdev, da[a], Ka[a]) : zmat for a in 1:engsize]
    Zt = m == 1 ? zten : MtlArray{S}(undef, dims)
    Ztv = reshape(Zt, length(Zt))
    Znat = m == 1 ? Rnat : Zt
    Znatkd = [reshape(Znat, Ka[a], da[a]) for a in 1:engsize]
    Znathd = [reshape(Znat, da[a], Ka[a]) for a in 1:engsize]

    Ms_d = [MtlArray{S}(undef, da[a], da[a]) for a in 1:engsize]
    Ys_d = [MtlArray{S}(undef, da[a], da[a]) for a in 1:engsize]
    # Chisel columns: host scalars when m == 1, device vectors otherwise.
    pa1 = [S(Pm[1, a]) for a in 1:engsize]
    Pdev = [m == 1 ? MtlArray{S}(undef, 0) : Dleto.to_gpu(Vector{S}(Pm[:, a]))
            for a in 1:engsize]

    # --- host staging --------------------------------------------------------
    Ms_h = [Matrix{S}(undef, da[a], da[a]) for a in 1:engsize]
    Ys_h = [Matrix{S}(undef, da[a], da[a]) for a in 1:engsize]
    # The operator coordinates are staged in Float32 unconditionally: the
    # in-place embeddings in SylverLining.jl are element loops over host
    # matrices, and they have to write the type the device wants.  Σ_a d_a²
    # elements, so the copy is free next to anything else here.
    op_dim = pl.op_dim
    xh = Vector{S}(undef, op_dim)
    # The m·d^n staging vector is allocated only if `ester!`/`sylve!` are ever
    # called -- 324 MB at 300×300×300×3, and the square map never needs it.
    Rh = Ref(Vector{S}(undef, 0))
    function rhost()
        length(Rh[]) == m * N || (Rh[] = Vector{S}(undef, m * N))
        return Rh[]
    end

    # --- the two cores, mirroring `_sylvesterLM_array` -----------------------

    # ester:  R <- Σ_a P[:,a] ⊗ vec(Γ ×_p M_aᵗ),  the mode product being the
    # single GEMM `U_a · M_a` in this unfolding.
    function ester_core!(x::AbstractVector)
        copyto!(xh, x)                        # host, Σ d_a² elements
        Dleto._embed_all!(Ms_h, Ω, xh)        # host, small matrices
        for a in 1:engsize
            _sylver_up!(Ms_d[a], Ms_h[a])
        end
        fill!(Rdev, zero(S))
        for a in 1:engsize
            if m == 1 && (head[a] || flat[a])
                # No permute, no scratch, no scatter: the term accumulates onto
                # the residual in place, with P[1,a] as the GEMM's α.
                head[a] ? mul!(Rhd[a], transpose(Ms_d[a]), Ahead[a],
                               pa1[a], one(S)) :
                          mul!(Rkd[a], Udev[a], Ms_d[a], pa1[a], one(S))
            else
                if head[a]
                    mul!(Foldhd[a], transpose(Ms_d[a]), Ahead[a],
                         one(S), zero(S))
                else
                    mul!(flat[a] ? Foldkd[a] : Wmat[a], Udev[a], Ms_d[a],
                         one(S), zero(S))
                    mid[a] && permutedims!(Fold, Wten[a], iperms[a])
                end
                if m == 1
                    Rvec .+= pa1[a] .* Foldv
                else
                    # Rank-1 update as a broadcast.  The CPU kernel's
                    # `R[c,i] += pa[c]*t` loop is exactly what an MtlArray may
                    # not do from the host.
                    Rdev .+= Pdev[a] .* transpose(Foldv)
                end
            end
        end
        return Rdev
    end

    # sylve:  Y_a <- U_aᵗ · (Σ_c P[c,a] R[c,·])_(p), then the adjoint of the
    # embedding on the host.
    function sylve_core!(x::AbstractVector)
        for a in 1:engsize
            α = one(S)
            if m == 1
                α = pa1[a]
            else
                mul!(Ztv, transpose(Rdev), Pdev[a])
            end
            if head[a]
                # The adjoint of `Mᵗ·A` in M is `A·Zᵗ`; no permute either.
                mul!(Ys_d[a], Ahead[a], transpose(Znathd[a]), α, zero(S))
            else
                mid[a] && permutedims!(Wten[a], Znat, perms[a])
                Zu = flat[a] ? Znatkd[a] : Wmat[a]
                mul!(Ys_d[a], transpose(Udev[a]), Zu, α, zero(S))
            end
        end
        _sylver_sync()
        for a in 1:engsize
            _sylver_down!(Ys_h[a], Ys_d[a])   # Σ d_a² elements back
        end
        Dleto._transverse_embed_all!(xh, Ω, Ys_h)
        copyto!(x, xh)
        return x
    end

    # --- the LinearMaps' CPU-facing wrappers ---------------------------------
    #
    # `y` here is whatever the solver holds -- often a view -- so the residual
    # is copied through a host buffer rather than reshaped in place, exactly as
    # the CPU kernel does, with the extra step that the host buffer is Float32
    # even when the map is Float64.
    function down_R!(y::AbstractVector)
        _sylver_sync()
        if y isa Vector{S}
            copyto!(y, Rvec)
        else
            buf = rhost()
            copyto!(buf, Rvec)
            copyto!(y, buf)
        end
        return y
    end
    function up_R!(y::AbstractVector)
        if y isa Vector{S}
            copyto!(Rvec, y)
        else
            buf = rhost()
            copyto!(buf, y)
            copyto!(Rvec, buf)
        end
        return Rdev
    end

    ester!(y, x) = (ester_core!(x); down_R!(y); y)
    sylve!(x, y) = (up_R!(y); sylve_core!(x); x)
    sylvester!(z, x) = (ester_core!(x); sylve_core!(z); z)

    densor_map = Dleto.LinearMaps.LinearMap{T}(ester!, sylve!, pl.densor_dim,
                                               op_dim; ismutating=true)
    derdensor_map = Dleto.LinearMaps.LinearMap{T}(sylvester!, sylvester!,
                                                  op_dim, op_dim;
                                                  ismutating=true,
                                                  issymmetric=true,
                                                  isposdef=false)
    return derdensor_map, densor_map
end

# The hook src/SylverLining/SylverLining.jl calls for `backend = :metal`.
# `Val{:metal}` is what makes this an ADDED method rather than an overwritten
# one -- the core package defines only the `::Val` fallback, so this extension
# precompiles.  (A same-signature redefinition is "Method overwriting is not
# permitted during Module precompilation" on Julia 1.12, and costs every
# `using Metal` a full recompile of the extension.)
Dleto._sylvesterLM_device(::Val{:metal}, Ω::TransverseOps, P::AbstractMatrix,
                          Γ::ITensor) = _sylverlm_metal(Ω, P, Γ)
