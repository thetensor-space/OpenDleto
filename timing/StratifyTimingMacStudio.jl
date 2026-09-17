using Dleto
using ITensors
using LinearAlgebra
using Printf
using Random
using Dates
using Statistics

# Register weak-dependency solver extensions when available.
try
    @eval using Arpack
catch
    @warn "Arpack not available; ArpackSolver entries may fail and be retired early."
end

include(joinpath(@__DIR__, "..", "bench", "SphereHarness.jl"))

const VALENCE = 3
const D_MIN = parse(Int, get(ENV, "D_MIN", "10"))
const D_MAX = parse(Int, get(ENV, "D_MAX", "100"))
const D_STEP = parse(Int, get(ENV, "D_STEP", "5"))
const DS = collect(D_MIN:D_STEP:D_MAX)
const NTRIALS = parse(Int, get(ENV, "REPS", "3"))
const OPS = (:universal, :symmetric)
const TARGET = Dict(Float32 => 1e-8, Float64 => 1e-16)
const OP_SPACE = Dict(:universal => UniversalOp(), :symmetric => SymmetricOp())
const DEFAULT_CSV_PATH = joinpath(@__DIR__, "stratify-timing-MacStudio.csv")
const HEADER = "d,valence,eltype,solve_tol,target,ops,method,solver,threads,heap_hint_gb,timeout_s,der_seconds," *
               "strat_seconds,total_seconds,bytes,nullity,residual,meets_target,lsq_err,support,perm_ok,dims,nnz,status"

# Every solver type we want to sweep, matching the full repo family of methods.
const BASE_CONFIGS = [
    (; name = "CPU/Auto", method = :Auto, solver = :none, nd = -1, ops = (:universal, :symmetric), kwargs = (;)),
    (; name = "CPU/QuickDer/Auto", method = :QuickDer, solver = :AutoSolver, nd = -1, ops = (:universal,), kwargs = (;)),
    (; name = "CPU/QuickDer/Arpack", method = :QuickDer, solver = :ArpackSolver, nd = -1, ops = (:universal,), kwargs = (;)),
    (; name = "CPU/QuickDer/Gram", method = :QuickDer, solver = :GramSolver, nd = -1, ops = (:universal,), kwargs = (;)),
    (; name = "CPU/QuickDer3", method = :QuickDer3, solver = :none, nd = -1, ops = (:universal,), kwargs = (;)),
    (; name = "CPU/SylverLining/Auto", method = :SylverLining, solver = :AutoSolver, nd = -1, ops = (:universal, :symmetric), kwargs = (;)),
    (; name = "CPU/SylverLining/Gram", method = :SylverLining, solver = :GramSolver, nd = -1, ops = (:universal, :symmetric), kwargs = (;)),
    (; name = "CPU/SylverLining/LU", method = :SylverLining, solver = :LUSolver, nd = -1, ops = (:universal, :symmetric), kwargs = (;)),
    (; name = "CPU/SylverLining/Arpack", method = :SylverLining, solver = :ArpackSolver, nd = -1, ops = (:universal, :symmetric), kwargs = (;)),
    (; name = "CPU/SylverLining/Krylov", method = :SylverLining, solver = :KrylovSolver, nd = -1, ops = (:universal, :symmetric), kwargs = (;)),
    (; name = "CPU/SylverLining/Lanczos", method = :SylverLining, solver = :LanczosSolver, nd = -1, ops = (:universal, :symmetric), kwargs = (;)),
    (; name = "CPU/SylverLining/ShiftInvert", method = :SylverLining, solver = :ShiftInvertSolver, nd = -1, ops = (:universal, :symmetric), kwargs = (;)),
    (; name = "CPU/SylverLining/SVD", method = :SylverLining, solver = :SVDSolver, nd = -1, ops = (:universal, :symmetric), kwargs = (;)),
    (; name = "CPU/SymmetricGram", method = :SymmetricGram, solver = :none, nd = 3, ops = (:symmetric,), kwargs = (;)),
]

const GPU_CONFIGS = [
    (; name = "GPU/QuickDer/Gram", method = :QuickDer, solver = :GramSolver, nd = -1, ops = (:universal,), kwargs = (; device = :gpu)),
    (; name = "GPU/SylverLining/Gram", method = :SylverLining, solver = :GramSolver, nd = -1, ops = (:universal, :symmetric), kwargs = (; backend = :metal)),
    (; name = "GPU/SylverLining/Auto", method = :SylverLining, solver = :AutoSolver, nd = -1, ops = (:universal, :symmetric), kwargs = (; backend = :metal)),
]

const PROFILES = [
    (; label = "float32", eltype = Float32, solve_tol = 1e-8, threads = 8, heap_hint_gb = 10.0, timeout_s = 60.0),
    (; label = "float64", eltype = Float64, solve_tol = 1e-16, threads = 32, heap_hint_gb = 40.0, timeout_s = 30.0),
]

function prepare_csv(csv_path::String; reset::Bool = false)
    if reset
        isfile(csv_path) && rm(csv_path; force = true)
    end
    if !isfile(csv_path) || filesize(csv_path) == 0
        open(csv_path, "w") do io
            println(io, HEADER)
        end
    end
end

function timed_stratify(inp, cfg, tol::Float64)
    Ω, ch = inp.Ω, inp.ch
    Γ = inp.Γ isa Dleto.TensorSpace.TensorElement ? inp.Γ.value : inp.Γ
    der_s = strat_s = 0.0
    bytes = 0
    nullity = 0
    residual = NaN
    res = nothing
    status = "ok"

    try
        quietly() do
            m = cfg.solver === :none ?
                Dleto.get_derivation_method(cfg.method; cfg.kwargs...) :
                Dleto.get_derivation_method(cfg.method; solver = cfg.solver, cfg.kwargs...)
            d1 = @timed derTrOpsReduced(m, Ω, ch, Γ; tol = tol, nd = cfg.nd)
            der_s = d1.time
            bytes += d1.bytes
            (_, expand_map, ders) = d1.value
            nullity = size(ders, 2)
            nullity == 0 && error("no nontrivial derivations at tol = $tol")
            local δ
            d2 = @timed begin
                δ = embedITensors(Ω, expand_map(ders * randn(eltype(ders), nullity)))
                stratify(Γ, δ)
            end
            strat_s = d2.time
            bytes += d2.bytes
            res = d2.value
            residual = Float64(Dleto.der_residual(Γ, δ, ch))
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
        res = nothing
    end

    Σ = res === nothing ? nothing :
        (res.Σ isa Dleto.TensorSpace.TensorElement ? res.Σ.value : res.Σ)
    sc = (Σ !== nothing && inp.S !== nothing) ? reconstruction(inp, Σ, res.Xs) :
        (; lsq_err = NaN, support = NaN, perm_ok = false)

    return (; der_s, strat_s, total_s = der_s + strat_s, bytes, nullity, residual,
              lsq_err = sc.lsq_err, support = sc.support, perm_ok = sc.perm_ok, status)
end

function fmt_num(x::Real)
    isnan(x) ? "NaN" : string(x)
end

function emit_row(d::Int, valence::Int, eltype::Type, solve_tol::Float64,
                 target::Float64, ops_name::Symbol, method_name::String,
                 solver_name::String, threads::Int, heap_hint_gb::Float64,
                 timeout_s::Float64, result, inp, csv_path::String)
    line = @sprintf("%d,%d,%s,%.0e,%.0e,%s,%s,%s,%d,%.1f,%.1f,%.4f,%.4f,%.4f,%d,%d,%s,%s,%s,%s,%s,%s,%d,%s",
                    d, valence, string(eltype), solve_tol, target, String(ops_name),
                    method_name, solver_name, threads, heap_hint_gb, timeout_s,
                    result.der_s, result.strat_s, result.total_s, result.bytes,
                    result.nullity, fmt_num(result.residual),
                    isnan(result.residual) ? "false" : string(result.residual <= target),
                    fmt_num(result.lsq_err), fmt_num(result.support),
                    string(result.perm_ok), join(inp.dims, ' '), inp.nnz,
                    replace(result.status, ',' => ';'))
        # Build row parts and join with comma (safer than relying on @sprintf)
        parts = [
            string(d),
            string(valence),
            string(eltype),
            @sprintf("%.0e", solve_tol),
            @sprintf("%.0e", target),
            String(ops_name),
            method_name,
            solver_name,
            string(threads),
            @sprintf("%.1f", heap_hint_gb),
            @sprintf("%.1f", timeout_s),
            @sprintf("%.4f", result.der_s),
            @sprintf("%.4f", result.strat_s),
            @sprintf("%.4f", result.total_s),
            string(result.bytes),
            string(result.nullity),
            fmt_num(result.residual),
            isnan(result.residual) ? "false" : string(result.residual <= target),
            fmt_num(result.lsq_err),
            fmt_num(result.support),
            string(result.perm_ok),
            join(inp.dims, ' '),
            string(inp.nnz),
            replace(result.status, ',' => ';')
        ]
        line = join(parts, ',')
        open(csv_path, "a") do io
            println(io, line)
    end
end

function bar_string(done::Int, total::Int; width::Int = 24)
    frac = total == 0 ? 1.0 : done / total
    filled = clamp(round(Int, frac * width), 0, width)
    return "[" * repeat("█", filled) * repeat(" ", width - filled) * "]"
end

function print_progress_line(done::Int, total::Int, dim_done::Int, dim_total::Int,
                             d::Int, ops_name::Symbol, cfg_name::String,
                             trial::Int, status::String, timeout_s::Float64)
    g_pct = round((total == 0 ? 1.0 : done / total) * 100; digits = 1)
    d_pct = round((dim_total == 0 ? 1.0 : dim_done / dim_total) * 100; digits = 1)
    global_bar = bar_string(done, total)
    dim_bar = bar_string(dim_done, dim_total; width = 18)
    status_show = length(status) > 80 ? status[1:80] * "..." : status
    msg = @sprintf("\r%s %4d/%4d %5.1f%% | d=%3d %s %3d/%3d %5.1f%% | %s %-20s t%d timeout=%.0fs",
                   global_bar, done, total, g_pct,
                   d, dim_bar, dim_done, dim_total, d_pct,
                   String(ops_name), cfg_name, trial, timeout_s)
    print(msg * " status=" * status_show)
    flush(stdout)
end

function warmup_profile(profile, configs; warmup_ds::Vector{Int} = [10, 15])
    println("\n=== warmup for $(profile.label) ===")
    println("  warmup dims: ", join(warmup_ds, ", "))
    # Warm each selected method on representative dimensions before timed rows are emitted.
    for d in warmup_ds
        for ops_name in OPS
            inp = build_sphere(d; valence = VALENCE, T = profile.eltype, ops = OP_SPACE[ops_name])
            for cfg in configs
                ops_name in cfg.ops || continue
                r = timed_stratify(inp, cfg, profile.solve_tol)
                @printf("  warmup d=%d ops=%s method=%s solver=%s time=%.3f s\n",
                        d, String(ops_name), cfg.name, String(cfg.solver), r.total_s)
            end
        end
    end
end

function timeout_for_dimension(profile, d::Int)
    d < 50 ? min(profile.timeout_s, 30.0) : profile.timeout_s
end

function run_profile(profile, configs; start_d::Int = first(DS), csv_path::String = DEFAULT_CSV_PATH)
    println("\n=== profile $(profile.label) ===")
    println("  eltype = $(profile.eltype), solve_tol = $(profile.solve_tol), threads = $(profile.threads), heap = $(profile.heap_hint_gb) GB, timeout = $(profile.timeout_s) s")
    println("  start_d = $(start_d)")

    run_ds = filter(d -> d >= start_d, DS)
    isempty(run_ds) && error("No dimensions to run: start_d=$(start_d) is larger than max d=$(last(DS)).")

    valid_cfgs = filter(cfg -> any(ops_name -> ops_name in cfg.ops, OPS), configs)
    total = length(run_ds) * length(OPS) * length(valid_cfgs) * NTRIALS
    done = 0
    retired = Dict{Tuple{Symbol, String, String}, Int}()

    for d in run_ds
        dim_total = count(cfg -> :universal in cfg.ops, configs) * NTRIALS +
            count(cfg -> :symmetric in cfg.ops, configs) * NTRIALS
        dim_done = 0
        dim_retired = String[]

        for ops_name in OPS
            # Build the input once per (d, ops) and reuse it for all methods/trials.
            print("\rpreparing d=$(d) ops=$(String(ops_name))..." * repeat(" ", 20))
            flush(stdout)
            inp = build_sphere(d; valence = VALENCE, T = profile.eltype, ops = OP_SPACE[ops_name])

            for cfg in configs
                ops_name in cfg.ops || continue
                key = (ops_name, cfg.name, String(cfg.solver))
                if haskey(retired, key)
                    done += NTRIALS
                    dim_done += NTRIALS
                    timeout_s = timeout_for_dimension(profile, d)
                    print_progress_line(done, total, dim_done, dim_total, d,
                                        ops_name, cfg.name, NTRIALS,
                                        @sprintf("skipped (retired@d=%d)", retired[key]),
                                        timeout_s)
                    continue
                end

                timeout_s = timeout_for_dimension(profile, d)
                trial_totals = Float64[]
                force_retire_reason = ""

                for trial in 1:NTRIALS
                    done += 1
                    dim_done += 1
                    print_progress_line(done, total, dim_done, dim_total, d,
                                        ops_name, cfg.name, trial, "running", timeout_s)
                    start = time()
                    result = timed_stratify(inp, cfg, profile.solve_tol)
                    elapsed = time() - start
                    if elapsed > timeout_s
                        result = (; der_s = NaN, strat_s = NaN, total_s = NaN,
                                    bytes = 0, nullity = 0, residual = NaN,
                                    lsq_err = NaN, support = NaN, perm_ok = false,
                                    status = "timeout")
                        force_retire_reason = @sprintf("trial %d exceeded timeout %.1f s (wall %.2f s)", trial, timeout_s, elapsed)
                    elseif result.status == "ok" && isfinite(result.total_s)
                        push!(trial_totals, result.total_s)
                    elseif occursin("PosDefException", result.status)
                        force_retire_reason = "PosDefException"
                    end
                    emit_row(d, VALENCE, profile.eltype, profile.solve_tol,
                             TARGET[profile.eltype], ops_name, cfg.name, String(cfg.solver),
                             profile.threads, profile.heap_hint_gb, timeout_s,
                             result, inp, csv_path)
                    shown_status = isnan(result.total_s) ?
                        @sprintf("%s (wall %.2fs)", result.status, elapsed) :
                        @sprintf("%s (solver %.3fs | wall %.3fs)", result.status, result.total_s, elapsed)
                    print_progress_line(done, total, dim_done, dim_total, d,
                                        ops_name, cfg.name, trial, shown_status, timeout_s)

                    if !isempty(force_retire_reason)
                        # Fill any remaining trial slots without burning more wall-time.
                        if trial < NTRIALS
                            for skipped_trial in (trial + 1):NTRIALS
                                done += 1
                                dim_done += 1
                                skipped = (; der_s = NaN, strat_s = NaN, total_s = NaN,
                                             bytes = 0, nullity = 0, residual = NaN,
                                             lsq_err = NaN, support = NaN, perm_ok = false,
                                             status = "skipped_after_retire")
                                emit_row(d, VALENCE, profile.eltype, profile.solve_tol,
                                         TARGET[profile.eltype], ops_name, cfg.name, String(cfg.solver),
                                         profile.threads, profile.heap_hint_gb, timeout_s,
                                         skipped, inp, csv_path)
                                print_progress_line(done, total, dim_done, dim_total, d,
                                                    ops_name, cfg.name, skipped_trial,
                                                    "skipped_after_retire", timeout_s)
                            end
                        end
                        break
                    end
                end

                avg_total = isempty(trial_totals) ? NaN : mean(trial_totals)
                if !isempty(force_retire_reason)
                    retired[key] = d
                    push!(dim_retired,
                          @sprintf("%s/%s (%s)", String(ops_name), cfg.name, force_retire_reason))
                elseif isnan(avg_total) || avg_total > timeout_s
                    retired[key] = d
                    reason = isnan(avg_total) ?
                        "no successful returns" :
                        @sprintf("avg %.3f s > timeout %.1f s", avg_total, timeout_s)
                    push!(dim_retired,
                          @sprintf("%s/%s (%s)", String(ops_name), cfg.name, reason))
                end
            end
        end

        println()
        if isempty(dim_retired)
            @printf("  d=%d complete: no new retirements\n", d)
        else
            @printf("  d=%d complete: retired %d solver paths\n", d, length(dim_retired))
        end
    end

    println("\nprofile $(profile.label) complete")
end

function usage()
    println("Usage:")
    println("  JULIA_NUM_THREADS=8 julia --project=. --heap-size-hint=10G timing/StratifyTimingMacStudio.jl float32")
    println("  JULIA_NUM_THREADS=32 julia --project=. --heap-size-hint=40G timing/StratifyTimingMacStudio.jl float64")
    println("  JULIA_NUM_THREADS=8 julia --project=. --heap-size-hint=10G timing/StratifyTimingMacStudio.jl float32 30")
    println("  JULIA_NUM_THREADS=8 REPS=5 julia --project=. --heap-size-hint=10G timing/StratifyTimingMacStudio.jl float32 30")
    println("  JULIA_NUM_THREADS=8 julia --project=. --heap-size-hint=10G timing/StratifyTimingMacStudio.jl float32 30 --reset")
    println("  JULIA_NUM_THREADS=8 julia --project=. timing/StratifyTimingMacStudio.jl float32 --shard=1/4 --csv=timing/stratify-b1.csv --reset")
    println("  JULIA_NUM_THREADS=8 julia --project=. timing/StratifyTimingMacStudio.jl float32 --shard=2/4 --csv=timing/stratify-b2.csv --no-warmup")
    println("  D_MIN=125 D_MAX=500 D_STEP=25 REPS=1 THREADS=16 HEAP_HINT_GB=40 TIMEOUT_S=30 JULIA_NUM_THREADS=16 julia --project=. --heap-size-hint=40G timing/StratifyTimingMacStudio.jl float32 --gpu --csv=timing/stratify-cpu-gpu-30s.csv --reset")
    println("  D_MIN=125 D_MAX=500 D_STEP=25 REPS=1 THREADS=16 HEAP_HINT_GB=40 TIMEOUT_S=60 JULIA_NUM_THREADS=16 julia --project=. --heap-size-hint=40G timing/StratifyTimingMacStudio.jl float32 --gpu --csv=timing/stratify-cpu-gpu-60s.csv --reset")
    println("  BLAS_THREADS=16 D_MIN=125 D_MAX=500 D_STEP=25 JULIA_NUM_THREADS=16 julia --project=. timing/StratifyTimingMacStudio.jl float32 --include=CPU/Auto,CPU/QuickDer --csv=timing/stratify-fast-30s.csv")
    println("  julia timing/StratifyTimingMacStudio.jl --help")
end

function _pattern_tokens(spec::String)
    toks = String[]
    for tok in split(spec, ",")
        s = strip(tok)
        isempty(s) || push!(toks, s)
    end
    return toks
end

function _matches_any(name::String, pats::Vector{String})
    isempty(pats) && return true
    for p in pats
        if startswith(p, "re:")
            occursin(Regex(p[4:end]), name) && return true
        elseif occursin(p, name)
            return true
        end
    end
    return false
end

function filter_configs(configs; include_spec::String = "", exclude_spec::String = "")
    includes = _pattern_tokens(include_spec)
    excludes = _pattern_tokens(exclude_spec)
    filtered = [cfg for cfg in configs if _matches_any(cfg.name, includes) && (isempty(excludes) || !_matches_any(cfg.name, excludes))]
    isempty(filtered) && error("Method filter selected no configs. include='$(include_spec)' exclude='$(exclude_spec)'.")
    return filtered
end

function shard_configs(configs, shard_idx::Int, shard_count::Int)
    (1 <= shard_idx <= shard_count) ||
        error("Invalid --shard index: $(shard_idx)/$(shard_count).")
    shard = [cfg for (i, cfg) in enumerate(configs) if mod(i - 1, shard_count) + 1 == shard_idx]
    isempty(shard) && error("Shard $(shard_idx)/$(shard_count) selected no configs.")
    return shard
end

function parse_shard(spec::String)
    m = match(r"^(\d+)/(\d+)$", spec)
    m === nothing && error("Invalid --shard format: $(spec). Expected i/n, e.g. 1/4")
    idx = parse(Int, m.captures[1])
    cnt = parse(Int, m.captures[2])
    cnt < 1 && error("Invalid --shard count: $(cnt)")
    return idx, cnt
end

function parse_warmup_dims(spec::String, start_d::Int)
    dims = Int[]
    seen = Set{Int}()
    for raw in split(spec, ",")
        tok = strip(raw)
        isempty(tok) && continue
        d = if lowercase(tok) == "start"
            start_d
        else
            parse(Int, tok)
        end
        d < 1 && continue
        if !(d in seen)
            push!(dims, d)
            push!(seen, d)
        end
    end
    isempty(dims) && error("Warmup dimension list is empty. Use --warmup-dims=10,15,start or set WARMUP_DIMS.")
    return dims
end

function main()
    if length(ARGS) == 0 || ARGS[1] in ("-h", "--help")
        usage(); return
    end

    mode = lowercase(String(ARGS[1]))
    idx = findfirst(p -> p.label == mode, PROFILES)
    if idx === nothing
        error("Unknown mode: $(ARGS[1]). Expected one of: $(join([p.label for p in PROFILES], ", "))")
    end

    start_d = first(DS)
    reset = false
    do_warmup = true
    include_gpu = false
    include_spec = ""
    exclude_spec = ""
    csv_path = DEFAULT_CSV_PATH
    shard_idx = 1
    shard_count = 1
    warmup_spec = get(ENV, "WARMUP_DIMS", "10,15,start")
    for arg in ARGS[2:end]
        if arg == "--reset"
            reset = true
        elseif arg == "--no-warmup"
            do_warmup = false
        elseif arg == "--gpu"
            include_gpu = true
        elseif startswith(arg, "--include=")
            include_spec = String(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--exclude=")
            exclude_spec = String(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--csv=")
            csv_arg = split(arg, "=", limit = 2)[2]
            csv_path = isabspath(csv_arg) ? csv_arg : joinpath(@__DIR__, "..", csv_arg)
        elseif startswith(arg, "--shard=")
            shard_idx, shard_count = parse_shard(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--warmup-dims=")
            warmup_spec = String(split(arg, "=", limit = 2)[2])
        else
            try
                start_d = parse(Int, arg)
            catch
                error("Unknown argument: $(arg). Use start_d and flags: --reset, --no-warmup, --gpu, --include=..., --exclude=..., --csv=..., --shard=i/n, --warmup-dims=...")
            end
        end
    end

    warmup_dims = parse_warmup_dims(warmup_spec, start_d)

    profile_base = PROFILES[idx]
    profile = (; profile_base...,
        threads = parse(Int, get(ENV, "THREADS", string(profile_base.threads))),
        heap_hint_gb = parse(Float64, get(ENV, "HEAP_HINT_GB", string(profile_base.heap_hint_gb))),
        timeout_s = parse(Float64, get(ENV, "TIMEOUT_S", string(profile_base.timeout_s))))

    blas_threads = parse(Int, get(ENV, "BLAS_THREADS", string(profile.threads)))
    try
        BLAS.set_num_threads(blas_threads)
    catch e
        @warn "Could not set BLAS threads" exception = (e, catch_backtrace())
    end

    configs = copy(BASE_CONFIGS)
    if include_gpu
        try
            @eval using Metal
            append!(configs, GPU_CONFIGS)
        catch e
            @warn "GPU requested but Metal could not be loaded; continuing with CPU configs only." exception = (e, catch_backtrace())
        end
    end

    configs = shard_count == 1 ? configs : shard_configs(configs, shard_idx, shard_count)
    configs = filter_configs(configs; include_spec = include_spec, exclude_spec = exclude_spec)
    prepare_csv(csv_path; reset = reset)
    println("=== MacStudio timing sweep: $(profile.label) ===")
    println("CSV: $(csv_path)")
    println("Start: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("Sweep start_d: $(start_d)")
    println("Trials per config: $(NTRIALS)")
    println("BLAS threads: $(BLAS.get_num_threads())")
    println("CSV reset: $(reset)")
    println("Warmup: $(do_warmup)")
    do_warmup && println("Warmup dims: ", join(warmup_dims, ", "))
    println("Shard: $(shard_idx)/$(shard_count)")
    println("GPU configs enabled: $(include_gpu)")
    println("Method include filter: '$(include_spec)'")
    println("Method exclude filter: '$(exclude_spec)'")
    println("Dimension range: $(first(DS)):$(D_STEP):$(last(DS))")
    println("Configs in this shard: ", join([cfg.name for cfg in configs], ", "))
    do_warmup && warmup_profile(profile, configs; warmup_ds = warmup_dims)
    run_profile(profile, configs; start_d = start_d, csv_path = csv_path)
    println("Finish: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
end

main()
