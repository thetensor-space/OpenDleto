# Complete, guarded Julia CPU sweep for one graded Float64 cubic tensor.
# Usage: julia --compiled-modules=existing --project=. large_sweep.jl FOLDER N [DENSE_MIB] [universal|adjoint] [METHODS]
using Dleto, ITensors, LinearAlgebra, JSON, Random

BLAS.set_num_threads(1)
length(ARGS) in (2, 3, 4, 5) || error("usage: large_sweep.jl FOLDER N [DENSE_MIB] [universal|adjoint] [METHODS]")
folder = ARGS[1]
n = parse(Int, ARGS[2])
dense_mib = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1024.0
chisel_name = length(ARGS) >= 4 ? ARGS[4] : "universal"
method_filter = length(ARGS) == 5 ? Set(Symbol.(split(ARGS[5], ','))) : nothing
n >= 2 && dense_mib >= 0 || error("invalid size or memory budget")
chisel_name in ("universal", "adjoint") || error("unknown chisel")

function emit(stage, status; method=nothing, seconds=nothing, dimension=nothing,
              residual=nothing, extra...)
    row = Dict{String,Any}("language" => "julia", "case" => string(n),
        "size" => n, "stage" => stage, "status" => status,
        "chisel" => chisel_name,
        "method" => method, "seconds" => seconds,
        "dimension" => dimension, "residual" => residual)
    for (key, value) in pairs(extra)
        row[string(key)] = value
    end
    println(JSON.json(row))
    flush(stdout)
end

function as_tensor(data)
    frame = [Index(size(data, a), "axis$a") for a in 1:3]
    return ITensor(data, frame...)
end

raw_data = read(joinpath(folder, "case_$(n).bin"))
length(raw_data) == 8n^3 || error("input size mismatch")
raw = reshape(collect(reinterpret(Float64, raw_data)), n, n, n)
Γraw = as_tensor(raw)

# Compile the reduction at a small size, then time the target input once.
nondeg(as_tensor(randn(MersenneTwister(1), 4, 4, 4)); tol=1e-10)
GC.gc()
start = time_ns()
reduced = nondeg(Γraw; mode=:trunc, tol=1e-10)
nondeg_seconds = (time_ns() - start) / 1e9
Γ = Dleto.TensorSpace.unwrap(reduced.Δ)
fr = collect(inds(Γ))
dims = [ITensors.dim(i) for i in fr]
emit("nondeg", "ok"; seconds=nondeg_seconds, reduced_shape=dims)

P = chisel_name == "universal" ? UniversalChisel(3) : AdjointChisel(3, 1, 2)
Ω = IndTransverseOps(fr, UniversalOp())
Ωeffective = chisel_name == "universal" ? Ω :
             IndTransverseOps(fr, [UniversalOp(), UniversalOp(), EmptyOp()])
rows, cols = prod(dims), chisel_name == "universal" ? sum(d^2 for d in dims) :
                                           sum(dims[a]^2 for a in 1:2)
estimate = 8 * (4 * rows * cols + 4 * cols * cols)
emit("preflight", estimate <= dense_mib * 2^20 ? "dense" : "skipped_memory";
     map_shape=[rows, cols], dense_estimated_bytes=estimate,
     dense_memory_limit_bytes=round(Int, dense_mib * 2^20))
if estimate <= dense_mib * 2^20
    function dense_dimension()
        _, dmap = sylvesterLM(Ωeffective, P, Γ)
        values = svdvals(Matrix(dmap))
        return size(dmap, 2) - count(>(1e-8 * maximum(values)), values)
    end
    dense_dimension() # compile
    GC.gc()
    start = time_ns()
    dim = dense_dimension()
    emit("svdvals", "exact_numeric"; seconds=(time_ns() - start) / 1e9,
         dimension=dim)
else
    emit("svdvals", "skipped_memory"; dense_estimated_bytes=estimate)
end

function non_scalar_score(D)
    total = 0.0
    for X in D
        A = Array(X, inds(X)...)
        scalar = tr(A) / size(A, 1)
        total += norm(A - scalar * I)^2
    end
    return sqrt(total)
end

reference = chisel_name == "universal" ? :QuickDer3 : :QuickSylver
methods = n >= 40 ? (:QuickDer, :QuickDerFullGram, :Auto, reference, :SylverLining) :
                    (:QuickDer, :Auto, reference, :SylverLining)
method_filter === nothing || (methods = filter(name -> name in method_filter, methods))
for name in methods
    fullgram = name == :QuickDerFullGram
    actual_name = fullgram ? :QuickDer : name
    method = actual_name == :QuickDer ? get_derivation_method(actual_name; seed=4242, verify=:full) :
             get_derivation_method(name)
    diagnostic = !(actual_name in (:QuickDer3, :QuickSylver))
    function solve_target()
        answer = if diagnostic
            Dleto.derTrOpsReduced(method, Ω, P, Γ; tol=1e-8, nd=-1,
                                  return_diagnostics=true)
        else
            Dleto.derTrOpsReduced(method, Ω, P, Γ; tol=1e-8, nd=-1)
        end
        _, expand, basis = answer[1:3]
        report = diagnostic ? answer[4] : nothing
        vectors = [embedITensors(Ω, expand * basis[:, j]) for j in 1:size(basis, 2)]
        residuals = report === nothing || report.residuals === nothing ?
                    [Dleto.der_residual(Γ, vector, P) for vector in vectors] :
                    report.residuals
        return vectors, residuals, report
    end
    prior_gram_precision = Dleto.QDN_GRAM_MIXED_PRECISION[]
    fullgram && (Dleto.QDN_GRAM_MIXED_PRECISION[] = false)
    try
        # A small call compiles this route. Failures on the warm tensor do not
        # decide the target result; the target has a different support pattern.
        try
            warm = as_tensor(randn(MersenneTwister(7), 4, 4, 4))
            warmfr = collect(inds(warm))
            warmΩ = IndTransverseOps(warmfr, UniversalOp())
            Dleto.derTrOpsReduced(method, warmΩ, P, warm; tol=1e-8, nd=-1,
                                  return_diagnostics=diagnostic)
        catch
        end
        target_warmed = actual_name != :SylverLining
        if target_warmed
            solve_target()
        end
        GC.gc()
        local target_start = time_ns()
        vectors, residuals, report = solve_target()
        seconds = (time_ns() - target_start) / 1e9
        maxres = isempty(residuals) ? nothing : maximum(residuals)
        certified = report === nothing ? nothing : report.certified
        verdict = report === nothing ? "unreported" : string(report.status)
        scalar_minimum = chisel_name == "universal" ? 2 : 1
        if length(vectors) < scalar_minimum
            verdict = "invalid_count"
        elseif certified === false
            verdict = "uncertified_" * verdict
        end
        emit("derivations", verdict; method=string(name), seconds=seconds,
             dimension=length(vectors), residual=maxres, certified=certified,
             solver=report === nothing ? nothing : string(report.solver),
             selected_method=report === nothing ? nothing : string(report.method),
             returned_vectors=length(vectors),
             scalar_dimension=chisel_name == "universal" ? 2 : 1,
             target_warmed=target_warmed)
        scores = non_scalar_score.(vectors)
        if isempty(scores) || maximum(scores) < 1e-7
            emit("stratify", "skipped_scalar_only"; method=string(name))
            continue
        end
        chosen = vectors[argmax(scores)]
        stratify(Γ, chosen) # compile the transformation
        GC.gc()
        local strat_start = time_ns()
        result = stratify(Γ, chosen)
        emit("stratify", "complete"; method=string(name),
             seconds=(time_ns() - strat_start) / 1e9,
             transformed_shape=collect(size(result.Σ)))
    catch err
        emit("derivations", "error"; method=string(name),
             error=sprint(showerror, err))
        emit("stratify", "skipped_solve_failed"; method=string(name))
    finally
        Dleto.QDN_GRAM_MIXED_PRECISION[] = prior_gram_precision
    end
end
