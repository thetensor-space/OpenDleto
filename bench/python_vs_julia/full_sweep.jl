# Time complete Julia derivation stages on a shared Fortran-order Float64 tensor.
# Usage: julia --project=. full_sweep.jl FOLDER CASE CHISEL REPEATS DIM...
# Each output line is JSON. Compilation is excluded by one warmup per stage.
using Dleto, ITensors, LinearAlgebra, Statistics, JSON

BLAS.set_num_threads(1)
length(ARGS) >= 7 || error("Usage: full_sweep.jl FOLDER CASE universal|adjoint REPEATS DIM...")
folder, case, chisel_name = ARGS[1:3]
repeats = parse(Int, ARGS[4])
dims = parse.(Int, ARGS[5:end])
repeats > 0 && all(>(0), dims) || error("repeats and dimensions must be positive")
length(dims) == 3 || error("this sweep currently uses order-three tensors")
chisel_name in ("universal", "adjoint") || error("unknown chisel")
P = chisel_name == "universal" ? UniversalChisel(3) : AdjointChisel(3, 1, 2)
raw = reshape(collect(reinterpret(Float64, read(joinpath(folder, "case_$(case).bin")))), dims...)
length(raw) == prod(dims) || error("input file size does not match dimensions")

function timed(f)
    f() # compile and warm the complete operation
    samples = Float64[]
    answer = nothing
    for _ in 1:repeats
        start = time_ns()
        answer = f()
        push!(samples, (time_ns() - start) / 1e9)
    end
    return answer, median(samples)
end

reduced_shape = nothing
function emit(method, stage, seconds; dimension=nothing, residual=nothing,
              status="ok", extra...)
    row = Dict{String,Any}(
        "case" => case, "chisel" => chisel_name, "method" => method,
        "stage" => stage, "seconds" => seconds, "dimension" => dimension,
        "residual" => residual, "status" => status,
        "raw_shape" => dims, "reduced_shape" => reduced_shape)
    for (key, value) in pairs(extra)
        row[string(key)] = value
    end
    println(JSON.json(row))
    flush(stdout)
end

function as_tensor(data)
    frame = [Index(size(data, a), "axis$a") for a in 1:ndims(data)]
    return ITensor(data, frame...)
end

function dense_dimension(data)
    Γ = as_tensor(data)
    frame = collect(inds(Γ))
    Ω = chisel_name == "universal" ?
        IndTransverseOps(frame, UniversalOp()) :
        IndTransverseOps(frame, [UniversalOp(), UniversalOp(), EmptyOp()])
    _, dmap = sylvesterLM(Ω, P, Γ)
    values = svdvals(Matrix(dmap))
    threshold = 1e-8 * (isempty(values) ? 0.0 : maximum(values))
    return size(dmap, 2) - count(>(threshold), values)
end

reduction, t = timed(() -> nondeg(as_tensor(raw); mode=:trunc, tol=1e-10))
core_tensor = Dleto.TensorSpace.unwrap(reduction.Δ)
core = Array(core_tensor, inds(core_tensor)...)
reduced_shape = collect(size(core))
emit("hosvd", "nondeg", t; extra_ranks=reduced_shape)
for (stage, data) in (("raw", raw), ("reduced", core))
    dimension, seconds = timed(() -> dense_dimension(data))
    emit("dense_svdvals", stage, seconds; dimension=dimension,
         operator_space="engaged_axes")
end

Γ = as_tensor(core)
frame = collect(inds(Γ))
Ω = IndTransverseOps(frame, UniversalOp())
methods = chisel_name == "universal" ?
          (:Auto, :SylverLining, :QuickDer, :QuickDer3) :
          (:Auto, :SylverLining, :QuickDer, :QuickSylver)
for name in methods
    method = name == :QuickDer ? get_derivation_method(name; seed=4242, verify=:full) :
             get_derivation_method(name)
    diagnostic = name in (:Auto, :SylverLining, :QuickDer)
    solve() = diagnostic ?
        Dleto.derTrOpsReduced(method, Ω, P, Γ; tol=1e-8, nd=-1,
                              return_diagnostics=true) :
        Dleto.derTrOpsReduced(method, Ω, P, Γ; tol=1e-8, nd=-1)
    local answer, seconds
    try
        answer, seconds = timed(solve)
    catch err
        emit(string(name), "reduced_solve", nothing;
             status="error", error=sprint(showerror, err))
        emit(string(name), "reduced_stratification", nothing;
             status="skipped: solve failed")
        continue
    end
    rΩ, expand, basis = answer[1:3]
    report = diagnostic ? answer[4] : nothing
    vectors = [embedITensors(Ω, expand * basis[:, j]) for j in 1:size(basis, 2)]
    residuals = report === nothing || report.residuals === nothing ?
                [Dleto.der_residual(Γ, vector, P) for vector in vectors] : report.residuals
    maximum_residual = isempty(residuals) ? nothing : maximum(residuals)
    status = report === nothing ? "unreported" : string(report.status)
    emit(string(name), "reduced_solve", seconds;
         dimension=size(basis, 2), residual=maximum_residual,
         status=status, returned_vectors=length(vectors),
         certified=report === nothing ? nothing : report.certified,
         scalar_dimension=chisel_name == "universal" ? 2 : 1,
         reported_scalar_dimension=report === nothing ? nothing : report.scalar_dim,
         solver=report === nothing ? nothing : string(report.solver))
    if isempty(vectors)
        emit(string(name), "reduced_stratification", 0.0;
             status="skipped: no derivation")
        continue
    end
    result, st = timed(() -> stratify(Γ, vectors[1]))
    emit(string(name), "reduced_stratification", st;
         status="complete", transformed_shape=collect(size(result.Σ)))
end
