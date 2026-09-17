#!/usr/bin/env julia

"""
    SymmetricSphereDemo.jl

A small, reproducible command-line experiment for the symmetric-operator
recovery of a scrambled sphere-octant tensor.  The input support is the exact
zero-based `i+j+k=d-1` densor octant; its physical densor coordinates are
`sqrt((i-1)/(d-1))`, so exported densor plots are labelled as sphere
coordinates rather than as a plane of array indices.

Examples:

    julia --project=. labs/SymmetricSphereDemo.jl --dim=50 --seed=17 \
        --noise=0 --tol=1e-6 --method=QuickDer --sampling=densor --out=/tmp/sphere.csv

For a programmatic comparison grid, include this file and call `run_grid`.
Each attempted case appends a checkpoint row; errors are recorded before the
CLI exits nonzero.

`--method=SymmetricGram` uses Dleto's registered dense symmetric normal-matrix
method.  It requires a positive `--nd`; `--tol` controls inverse-iteration
convergence and does not choose the eigenvector count.
"""

using Dleto
using ITensors
using LinearAlgebra
using Printf
using Random

# Provides `sphere_octant`, `axis_chain`, and the transformation-chain score.
# We deliberately construct the noisy input below instead of `build_sphere`,
# because the score must retain the noiseless octant as its reference support.
include(joinpath(@__DIR__, "..", "bench", "SphereHarness.jl"))

const DEFAULTS = (; dim = 50, seed = 50, noise = 0.0, tol = 1e-6,
                  method = :QuickDer, sampling = :densor, cutoff = 1.5,
                  nd = -1, solver = :AutoSolver, nondeg_tol = 1e-10,
                  amplitudes = :gaussian,
                  gram_precision = :full, dense_solver = :auto,
                  symmetric_solver = :inverse,
                  out = joinpath(@__DIR__, "symmetric_sphere.csv"))

unwrap_tensor(x) = x isa Dleto.TensorSpace.TensorElement ? Dleto.TensorSpace.unwrap(x) : x

"""Build the exact reference octant, then add relative Gaussian noise before
the orthogonal scramble and nondegenerate reduction."""
function sphere_input(; dim::Int, seed::Int, noise::Real, sampling::Symbol, cutoff::Real,
                      nondeg_tol::Real, amplitudes::Symbol)
    dim >= 3 || error("--dim must be at least 3")
    noise >= 0 || error("--noise must be nonnegative")
    Random.seed!(seed)
    S = unwrap_tensor(sphere_octant(dim; valence = 3, mode = sampling, cutoff = cutoff))
    fr = collect(inds(S))
    A = Array(S, fr...)
    if amplitudes === :bounded
        # Keep the exact support, sign diversity, and continuous generic
        # values, while ruling out a nearly-zero populated slice by design.
        rng = MersenneTwister(seed + 2)
        for I in CartesianIndices(A)
            iszero(A[I]) || (A[I] = sign(A[I]) * (0.5 + rand(rng)))
        end
        S = ITensor(A, fr...)
    elseif amplitudes !== :gaussian
        error("--amplitudes must be gaussian or bounded")
    end
    G = if iszero(noise)
        S
    else
        # Independent stream: every grid noise level gets the same support and
        # the same later orthogonal scramble for this seed.
        η = randn(MersenneTwister(seed + 1), size(A)...)
        η .*= float(noise) * norm(A) / max(norm(η), eps(Float64))
        ITensor(A + η, fr...)
    end
    Random.seed!(seed + 10_000)
    rn = randomize_tensor(G; type = :orthogonal)
    nd = nondeg(unwrap_tensor(rn.Δ); tol = float(nondeg_tol))
    Γ = unwrap_tensor(nd.Δ)
    fr_nd = collect(inds(Γ))
    Ω = IndTransverseOps(fr_nd, SymmetricOp())
    # Noise is injected before both transformations, so this measures the
    # signal after exactly the same scramble/nondeg chain as Γ.
    signal_nd = act(act(S, rn.Xs), nd.Es)
    signal_residual = norm(Γ - signal_nd) / max(norm(signal_nd), eps(Float64))
    orthogonality_error = maximum(norm(Matrix(Array(X, inds(X)...))' * Matrix(Array(X, inds(X)...)) - I) /
                                  size(Array(X, inds(X)...), 1) for X in rn.Xs)
    return (; S, fr, scrambled = unwrap_tensor(rn.Δ), Xs = rn.Xs, Es = nd.Es, Γ, Ω, ch = UniversalChisel(3),
              dims = ITensors.dim.(fr_nd), nnz = count(!iszero, A), lean = false,
              signal_residual, orthogonality_error)
end

"""A residual against the full symmetric Sylvester map, when its construction fits."""
function full_residual(Ω, ch, Γ, ders)
    size(ders, 2) == 0 && return (NaN, "no derivations")
    try
        _, E = sylvesterLM(Ω, ch, Γ)
        # E is the rectangular densor map.  The first result is E' * E and
        # would square the residual being reported here.
        rs = [norm(E * ders[:, j]) / max(norm(Γ) * norm(ders[:, j]), eps(Float64))
              for j in axes(ders, 2)]
        return (maximum(rs), "ok")
    catch err
        return (NaN, "unavailable: " * first(split(sprint(showerror, err), '\n')))
    end
end

function csv_quote(x)
    s = string(x)
    return '"' * replace(s, '"' => "\"\"") * '"'
end

function ensure_csv(path::AbstractString)
    mkpath(dirname(abspath(path)))
    header = "dim,seed,noise,tol,method,actual_method,solver,actual_report_solver,nd,sampling,amplitudes,gram_precision,dense_solver,symmetric_solver,cutoff,nondeg_tol,active_dims,orthogonality_error,signal_residual,seconds,bytes,nullity,solver_nullity,scalar_dim,class,lsq_err,support,perm_ok,full_residual,full_residual_status,status"
    if isfile(path)
        readline(path) == header || error("CSV schema differs; select a new --out path: $path")
        return
    end
    open(path, "w") do io
        println(io, header)
    end
end

function append_csv(path, r)
    ensure_csv(path)
    open(path, "a") do io
        @printf(io, "%d,%d,%.17g,%.17g,%s,%s,%s,%s,%d,%s,%s,%s,%s,%s,%.17g,%.17g,%s,%.17g,%.17g,%.6f,%d,%d,%d,%d,%s,%.17g,%.17g,%s,%.17g,%s,%s\n",
                r.dim, r.seed, r.noise, r.tol, r.method, r.actual_method, r.solver, r.actual_report_solver, r.nd,
                r.sampling, r.amplitudes, r.gram_precision, r.dense_solver, r.symmetric_solver, r.cutoff, r.nondeg_tol, csv_quote(r.active_dims), r.orthogonality_error,
                r.signal_residual, r.seconds, r.bytes, r.nullity, r.solver_nullity,
                r.scalar_dim, r.class,
                r.lsq_err, r.support, r.perm_ok, r.full_residual,
                csv_quote(r.full_residual_status), csv_quote(r.status))
    end
end

artifact_stem(out::AbstractString, r) = begin
    base, _ = splitext(abspath(out))
    "$(base)_$(lowercase(r.method))_$(lowercase(r.actual_method))_$(lowercase(r.sampling))_$(lowercase(r.amplitudes))_$(lowercase(r.gram_precision))_$(lowercase(r.dense_solver))_$(lowercase(r.symmetric_solver))_$(lowercase(r.solver))_nd$(r.nd)_d$(r.dim)_seed$(r.seed)_noise$(@sprintf("%.3g", r.noise))_tol$(@sprintf("%.3g", r.tol))"
end

function recovered_alignment(inp, strat)
    perms = Vector{Vector{Int}}(undef, 3)
    outinds = Vector{Index}(undef, 3)
    for a in 1:3
        T, i_nd = axis_chain(inp.fr[a], inp.Xs, inp.Es)
        Z = only(filter(t -> hasind(t, i_nd), strat.Xs))
        i_out = only(filter(j -> j != i_nd, collect(inds(Z))))
        Zf = ITensor(Float64.(Array(Z, i_nd, i_out)), i_nd, i_out)
        Ta = Array(T * Zf, inp.fr[a], i_out)
        perms[a] = [argmax(abs.(Ta[:, c])) for c in axes(Ta, 2)]
        outinds[a] = i_nd # stratify retags its output back to the input frame
    end
    return perms, outinds
end

"""Write original, scrambled, and recovered projections.

Only original and aligned recovered densor points use physical sphere
coordinates.  The scrambled panel explicitly uses array coordinates, because
an orthogonal scramble has no pointwise sphere-coordinate interpretation.
"""
function write_artifacts(inp, r, out::AbstractString; strat = nothing)
    stem = artifact_stem(out, r)
    points = stem * "_points.csv"
    svg = stem * "_isometric.svg"
    Sarr = Array(inp.S, inp.fr...)
    d = size(Sarr, 1)
    original = NTuple{4,Float64}[]
    for I in CartesianIndices(Sarr)
        iszero(Sarr[I]) && continue
        if r.sampling == "densor"
            push!(original, ((I[1] - 1) / (d - 1) |> sqrt,
                             (I[2] - 1) / (d - 1) |> sqrt,
                             (I[3] - 1) / (d - 1) |> sqrt, Sarr[I]))
        else
            push!(original, (Float64(I[1] - 1), Float64(I[2] - 1), Float64(I[3] - 1), Sarr[I]))
        end
    end
    Garr = Array(inp.scrambled, inds(inp.scrambled)...)
    scrambled = NTuple{4,Float64}[]
    cutoff = 0.20 * maximum(abs, Garr)
    for I in CartesianIndices(Garr)
        abs(Garr[I]) >= cutoff || continue
        push!(scrambled, (Float64(I[1] - 1), Float64(I[2] - 1), Float64(I[3] - 1), Garr[I]))
    end
    recovered = NTuple{4,Float64}[]
    if strat !== nothing
        perms, outinds = recovered_alignment(inp, strat)
        Rarr = Array(unwrap_tensor(strat.Σ), outinds...)
        rcut = 0.20 * maximum(abs, Rarr)
        for I in CartesianIndices(Rarr)
            abs(Rarr[I]) >= rcut || continue
            if r.sampling == "densor"
                push!(recovered, (sqrt((perms[1][I[1]] - 1) / (d - 1)),
                                  sqrt((perms[2][I[2]] - 1) / (d - 1)),
                                  sqrt((perms[3][I[3]] - 1) / (d - 1)), Rarr[I]))
            else
                push!(recovered, (Float64(perms[1][I[1]] - 1), Float64(perms[2][I[2]] - 1),
                                  Float64(perms[3][I[3]] - 1), Rarr[I]))
            end
        end
    end
    open(points, "w") do io
        println(io, "series,x,y,z,value,coordinate_system")
        original_label = r.sampling == "densor" ? "physical_sphere_coordinate" : "literal_lattice_coordinate"
        for (series, pts, label) in (("original", original, original_label),
                                     ("scrambled", scrambled, "scrambled_array_coordinate"),
                                     ("recovered", recovered, r.sampling == "densor" ? "physical_sphere_coordinate_aligned_by_transform_argmax" : "literal_lattice_coordinate_aligned_by_transform_argmax"))
            for p in pts
                @printf(io, "%s,%.17g,%.17g,%.17g,%.17g,%s\n", series, p[1], p[2], p[3], p[4], label)
            end
        end
    end
    open(svg, "w") do io
        title = r.sampling == "densor" ? "Densor: isometric physical sphere coordinates; recovered axes aligned by composed-transform argmax." : "Literal lattice: isometric coordinates; recovered axes aligned by composed-transform argmax."
        println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"960\" height=\"380\" viewBox=\"0 0 960 380\">")
        println(io, "<rect width=\"960\" height=\"380\" fill=\"white\"/><text x=\"16\" y=\"22\" font-size=\"13\">$(title)</text>")
        for (panel, (name, pts, color)) in enumerate((("original", original, "#1769aa"), ("scrambled (array coordinates)", scrambled, "#b45309"), ("recovered (aligned)", recovered, "#15803d")))
            if name == "original" && !isempty(pts)
                display_cut = 0.20 * maximum(p -> abs(p[4]), pts)
                pts = filter(p -> abs(p[4]) >= display_cut, pts)
            end
            x0 = 20 + 310 * (panel - 1)
            println(io, "<text x=\"$(x0)\" y=\"48\" font-size=\"12\">$(name)</text><rect x=\"$(x0)\" y=\"60\" width=\"280\" height=\"280\" fill=\"none\" stroke=\"black\"/>")
            # Isometric projection preserves all three displayed coordinates:
            # horizontal is x-y, vertical is x+y-2z.  Per-panel normalization
            # makes the sparse support legible without calling array indices a
            # sphere in the scrambled panel.
            iso(p) = ((p[1] - p[2]) / sqrt(2), (p[1] + p[2] - 2p[3]) / sqrt(6))
            uv = iso.(pts)
            umin, umax = isempty(uv) ? (-1.0, 1.0) : extrema(first.(uv))
            vmin, vmax = isempty(uv) ? (-1.0, 1.0) : extrema(last.(uv))
            for p in pts
                u, v = iso(p)
                x = x0 + 12 + 256 * (u - umin) / max(umax - umin, eps())
                y = 328 - 256 * (v - vmin) / max(vmax - vmin, eps())
                @printf(io, "<circle cx=\"%.3f\" cy=\"%.3f\" r=\"1.1\" fill=\"%s\"/>\n", x, y, color)
            end
        end
        println(io, "</svg>")
    end
    return (; points, svg)
end

"""Run one case.  A non-scalar basis is only called recovery after the harness
reports a genuine signed-permutation support reconstruction."""
function run_case(; dim::Int = DEFAULTS.dim, seed::Int = DEFAULTS.seed,
                  noise::Real = DEFAULTS.noise, tol::Real = DEFAULTS.tol,
                  method::Symbol = DEFAULTS.method, sampling::Symbol = DEFAULTS.sampling,
                  cutoff::Real = DEFAULTS.cutoff, nd::Int = DEFAULTS.nd,
                  solver::Symbol = DEFAULTS.solver, out::AbstractString = DEFAULTS.out,
                  nondeg_tol::Real = DEFAULTS.nondeg_tol,
                  amplitudes::Symbol = DEFAULTS.amplitudes,
                  gram_precision::Symbol = DEFAULTS.gram_precision,
                  dense_solver::Symbol = DEFAULTS.dense_solver,
                  symmetric_solver::Symbol = DEFAULTS.symmetric_solver, artifacts::Bool = true)
    ensure_csv(out)
    solve_start = nothing
    seconds = 0.0; bytes = 0; nullity = 0; solver_nullity = 0; scalar_dim = 2
    lsq_err = NaN; support = NaN; perm_ok = false; fres = NaN; fres_status = "not run"
    status = "ok"; class = "error"; actual_method = String(method); actual_report_solver = ""
    inp = nothing; strat = nothing
    active_dims = ""; orthogonality_error = NaN; signal_residual = NaN
    try
        method === :SymmetricGram && !(isfinite(tol) && tol > 0) &&
            error("--method=SymmetricGram requires a finite positive --tol; it controls inverse-iteration convergence")
        inp = sphere_input(; dim, seed, noise, sampling, cutoff, nondeg_tol, amplitudes)
        active_dims = join(inp.dims, "x"); orthogonality_error = inp.orthogonality_error
        signal_residual = inp.signal_residual
        m = method === :SymmetricGram ?
            get_derivation_method(:SymmetricGram; eigensolver = symmetric_solver, seed = seed) :
            method === :QuickDer ? get_derivation_method(method; seed = seed, solver = solver) :
            method === :SylverLining ? get_derivation_method(method; solver = solver) :
            method === :Auto ? get_derivation_method(method; seed = seed, solver = solver, fallback_solver = solver) :
            get_derivation_method(method)
        gram_precision in (:default, :full, :mixed) || error("--gram-precision must be default, full, or mixed")
        dense_solver in (:auto, :svd, :gram) || error("--dense-solver must be auto, svd, or gram")
        symmetric_solver in (:inverse, :eigen) || error("--symmetric-solver must be inverse or eigen")
        solve_start = time()
        if method === :SymmetricGram
            nd > 0 || error("--method=SymmetricGram requires a positive --nd (for example --nd=3); --tol does not select its count")
            solved = @timed derTrOpsReduced(m, inp.Ω, inp.ch, inp.Γ;
                                             tol = float(tol), nd = nd,
                                             return_diagnostics = true)
            seconds, bytes = solved.time, solved.bytes
            rΩ, expand_map, ders, report = solved.value
            nullity = size(ders, 2); solver_nullity = report.nullity
            scalar_dim = report.scalar_dim; actual_method = String(report.method)
            actual_report_solver = report.solver === nothing ? "" : String(report.solver)
        else
            # QuickDer's dense branch selects Gram/SVD from these module Refs.
            # They are restored even if the solve errors, keeping this override
            # local to a demo invocation (and avoiding any source change).
            old_mixed = Dleto.QDN_GRAM_MIXED_PRECISION[]
            old_mincols = Dleto.QDN_GRAM_MIN_COLS[]
            solved = try
                gram_precision === :full && (Dleto.QDN_GRAM_MIXED_PRECISION[] = false)
                gram_precision === :mixed && (Dleto.QDN_GRAM_MIXED_PRECISION[] = true)
                dense_solver === :svd && (Dleto.QDN_GRAM_MIN_COLS[] = typemax(Int))
                dense_solver === :gram && (Dleto.QDN_GRAM_MIN_COLS[] = 0)
                @timed (method === :QuickDer3 ?
                        derTrOpsReduced(m, inp.Ω, inp.ch, inp.Γ; tol = float(tol), nd = nd) :
                        derTrOpsReduced(m, inp.Ω, inp.ch, inp.Γ;
                                         tol = float(tol), nd = nd, return_diagnostics = true))
            finally
                Dleto.QDN_GRAM_MIXED_PRECISION[] = old_mixed
                Dleto.QDN_GRAM_MIN_COLS[] = old_mincols
            end
            seconds, bytes = solved.time, solved.bytes
            if method === :QuickDer3
            rΩ, expand_map, ders = solved.value
            nullity = size(ders, 2); solver_nullity = nullity
            actual_method = "QuickDer3"
            else
            rΩ, expand_map, ders, report = solved.value
            nullity = size(ders, 2); solver_nullity = report.nullity
            scalar_dim = report.scalar_dim; actual_method = String(report.method)
            actual_report_solver = report.solver === nothing ? "" : String(report.solver)
            end
        end
        fres, fres_status = full_residual(rΩ, inp.ch, inp.Γ, ders)
        if nullity == 0
            class = "no-directions"
        elseif nullity <= scalar_dim
            class = "scalar-only"
        else
            # Fixed local RNG: changing unrelated random calls cannot change the score.
            coeffs = randn(MersenneTwister(seed + 10_000), eltype(ders), nullity)
            δ = embedITensors(inp.Ω, expand_map(ders * coeffs))
            strat = stratify(inp.Γ, δ)
            sc = reconstruction(inp, unwrap_tensor(strat.Σ), strat.Xs)
            lsq_err, support, perm_ok = sc.lsq_err, sc.support, sc.perm_ok
            class = perm_ok && support >= 0.99 ? "recovery" : "non-scalar-unrecovered"
        end
    catch err
        solve_start === nothing || (seconds = time() - solve_start)
        status = "error: " * first(split(sprint(showerror, err), '\n'))
    end
    r = (; dim, seed, noise = float(noise), tol = float(tol), method = String(method),
         actual_method, solver = String(solver), actual_report_solver, nd, sampling = String(sampling), amplitudes = String(amplitudes),
         gram_precision = String(gram_precision), dense_solver = String(dense_solver), symmetric_solver = String(symmetric_solver), cutoff = float(cutoff), nondeg_tol = float(nondeg_tol),
         active_dims, orthogonality_error, signal_residual,
         seconds, bytes, nullity, solver_nullity, scalar_dim, class, lsq_err, support, perm_ok, full_residual = fres,
         full_residual_status = fres_status, status)
    append_csv(out, r)
    art = artifacts && inp !== nothing ? write_artifacts(inp, r, out; strat) : nothing
    return (; r..., artifacts = art)
end

"""Run an externally callable noise/tolerance comparison grid, checkpointing each case."""
function run_grid(; dims = [DEFAULTS.dim], seeds = [DEFAULTS.seed], noises = [DEFAULTS.noise],
                  tols = [DEFAULTS.tol], method::Symbol = DEFAULTS.method,
                  sampling::Symbol = DEFAULTS.sampling, cutoff::Real = DEFAULTS.cutoff,
                  nd::Int = DEFAULTS.nd, solver::Symbol = DEFAULTS.solver,
                  nondeg_tol::Real = DEFAULTS.nondeg_tol,
                  amplitudes::Symbol = DEFAULTS.amplitudes,
                  gram_precision::Symbol = DEFAULTS.gram_precision,
                  dense_solver::Symbol = DEFAULTS.dense_solver,
                  symmetric_solver::Symbol = DEFAULTS.symmetric_solver,
                  out::AbstractString = DEFAULTS.out, artifacts::Bool = false)
    results = NamedTuple[]
    for d in dims, s in seeds, n in noises, t in tols
        push!(results, run_case(; dim = Int(d), seed = Int(s), noise = n, tol = t,
                                method, sampling, cutoff, nd, solver, out, nondeg_tol, amplitudes, gram_precision, dense_solver, symmetric_solver, artifacts))
    end
    return results
end

function parse_cli(args)
    opts = Dict{String,String}()
    for arg in args
        startswith(arg, "--") && occursin('=', arg) || error("expected --name=value, got $arg")
        k, v = split(arg[3:end], '='; limit = 2); opts[k] = v
    end
    allowed = Set(["dim", "seed", "noise", "tol", "method", "sampling", "amplitudes", "gram-precision", "dense-solver", "symmetric-solver", "cutoff", "nd", "solver", "nondeg-tol", "out"])
    all(k in allowed for k in keys(opts)) || error("unknown option; allowed: $(join(sort!(collect(allowed)), ", "))")
    method_names = Dict("quickder" => :QuickDer, "sylverlining" => :SylverLining,
                        "auto" => :Auto, "quickder3" => :QuickDer3, "symmetricgram" => :SymmetricGram)
    method_key = lowercase(get(opts, "method", String(DEFAULTS.method)))
    haskey(method_names, method_key) || error("--method must be QuickDer, SylverLining, Auto, QuickDer3, or SymmetricGram")
    sampling = Symbol(lowercase(get(opts, "sampling", String(DEFAULTS.sampling))))
    sampling in (:densor, :lattice) || error("--sampling must be densor or lattice")
    amplitudes = Symbol(lowercase(get(opts, "amplitudes", String(DEFAULTS.amplitudes))))
    amplitudes in (:gaussian, :bounded) || error("--amplitudes must be gaussian or bounded")
    gram_precision = Symbol(lowercase(get(opts, "gram-precision", String(DEFAULTS.gram_precision))))
    gram_precision in (:default, :full, :mixed) || error("--gram-precision must be default, full, or mixed")
    dense_solver = Symbol(lowercase(get(opts, "dense-solver", String(DEFAULTS.dense_solver))))
    dense_solver in (:auto, :svd, :gram) || error("--dense-solver must be auto, svd, or gram")
    symmetric_solver = Symbol(lowercase(get(opts, "symmetric-solver", String(DEFAULTS.symmetric_solver))))
    symmetric_solver in (:inverse, :eigen) || error("--symmetric-solver must be inverse or eigen")
    solver_names = Dict(lowercase(String(s)) => s for s in available_solvers())
    solver_key = lowercase(get(opts, "solver", String(DEFAULTS.solver)))
    haskey(solver_names, solver_key) || error("--solver must name an available solver: $(join(string.(available_solvers()), ", "))")
    tol = parse(Float64, get(opts, "tol", string(DEFAULTS.tol)))
    method_names[method_key] === :SymmetricGram && !(isfinite(tol) && tol > 0) &&
        error("--method=SymmetricGram requires a finite positive --tol; it controls inverse-iteration convergence")
    return (; dim = parse(Int, get(opts, "dim", string(DEFAULTS.dim))),
            seed = parse(Int, get(opts, "seed", string(DEFAULTS.seed))),
            noise = parse(Float64, get(opts, "noise", string(DEFAULTS.noise))),
            tol, method = method_names[method_key],
            sampling, cutoff = parse(Float64, get(opts, "cutoff", string(DEFAULTS.cutoff))),
            nd = parse(Int, get(opts, "nd", string(DEFAULTS.nd))), solver = solver_names[solver_key],
            nondeg_tol = parse(Float64, get(opts, "nondeg-tol", string(DEFAULTS.nondeg_tol))),
            amplitudes, gram_precision, dense_solver, symmetric_solver,
            out = get(opts, "out", DEFAULTS.out))
end

function main(args = ARGS)
    if args == ["--help"] || args == ["-h"]
        println("""
        Symmetric sphere recovery (run with julia --project=. --threads=4).
        Exact: --dim=50 --seed=50 --amplitudes=bounded --gram-precision=full
        Noisy: --method=SymmetricGram --nd=3 --tol=1e-6 --noise=0.001
        Reproduce default precision issue: --gram-precision=mixed --seed=17
        Reproduce sparse notebook shell: --sampling=lattice

        Options use --name=value:
          dim, seed, noise (relative Frobenius noise), tol, nondeg-tol,
          amplitudes=gaussian|bounded, sampling=densor|lattice, cutoff,
          method=QuickDer|QuickDer3|Auto|SylverLining|SymmetricGram,
          nd, solver, gram-precision=default|full|mixed,
          dense-solver=auto|svd|gram, symmetric-solver=inverse|eigen,
          out (CSV path).
        SymmetricGram selects nd smallest modes, including two scalar modes;
        tol controls solver convergence and must be finite and positive. This
        is a known-model approximation.
        CSV metrics use the full tensor; figures are thresholded for display.
        """)
        return nothing
    end
    cfg = parse_cli(args)
    BLAS.set_num_threads(4)
    result = run_case(; cfg...)
    @printf("d=%d seed=%d noise=%.3g tol=%.3g method=%s/%s nd=%d: %s; returned=%d (solver=%d, scalar=%d), support=%.4g, residual=%.3e\n",
            result.dim, result.seed, result.noise, result.tol, result.method, result.actual_method, result.nd, result.class,
            result.nullity, result.solver_nullity, result.scalar_dim, result.support, result.full_residual)
    result.artifacts !== nothing && println("artifacts: $(result.artifacts.points), $(result.artifacts.svg)")
    if startswith(result.status, "error:")
        println(stderr, result.status)
        exit(1)
    end
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
