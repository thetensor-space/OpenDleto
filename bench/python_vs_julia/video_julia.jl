# julia --project=<environment> video_julia.jl FOLDER cpu|metal DIMS... WARMUPS REPEATS
using Dleto, ITensors, LinearAlgebra, Statistics
length(ARGS) in (7, 8) || error("expected FOLDER cpu|metal DIMS... WARMUPS REPEATS")
folder, backend = ARGS[1:2]
backend in ("cpu", "metal") || error("backend must be cpu or metal")
dims = Tuple(parse.(Int, ARGS[3:end-2]))
warmups, repeats = parse.(Int, ARGS[end-1:end])
all(>(0), dims) && warmups >= 0 && repeats > 0 || error("invalid shape or counts")
if backend == "cpu"
    occursin("Accelerate", string(BLAS.get_config())) || error("BLAS is not configured with Accelerate")
else
    using Metal
    Metal.functional() || error("Metal is unavailable")
end
tensor = reinterpret(Float32, read(joinpath(folder, "tensor.f32")))
x = collect(reinterpret(Float32, read(joinpath(folder, "x.f32"))))
length(tensor) == prod(dims) && length(x) == sum(dims .^ 2) || error("input size mismatch")
println("START backend=$backend shape=$dims warmups=$warmups repeats=$repeats")
flush(stdout)
frame = [Index(n, "axis$a") for (a, n) in enumerate(dims)]
Γ = ITensor(reshape(tensor, dims), frame...)
Ω = IndTransverseOps(frame, UniversalOp())
_, dmap = sylvesterLM(Ω, UniversalChisel(length(dims)), Γ;
                     backend=backend == "cpu" ? :array : :metal)
function apply()
    start = time_ns()
    result = backend == "metal" ? Dleto.gpu_sync(() -> dmap * x) : dmap * x
    return (time_ns() - start) / 1e9, result
end
for _ in 1:warmups
    apply()
end
println("WARMUP_COMPLETE backend=$backend shape=$dims")
flush(stdout)
samples = Float64[]
for i in 1:repeats
    elapsed, result = apply()
    length(result) == prod(dims) || error("unexpected output length")
    push!(samples, elapsed)
    indices = [1 + fld(i * (length(result)-1), 15) for i in 0:15]
    global values = result[indices]
    global output_norm = norm(result)
    println("SAMPLE $i seconds=$elapsed")
    flush(stdout)
end
println("RESULT backend=$backend shape=$dims coordinate_length=$(length(x)) output_shape=$((1, dims...)) warmups=$warmups repeats=$repeats seconds=$samples median_seconds=$(median(samples)) sample=$values output_norm=$output_norm clip_seconds=10 fps=30 clip_equivalent_fps=$(300 / median(samples)) realtime_factor=$(10 / median(samples)) julia_version=$VERSION julia_threads=$(Threads.nthreads()) lbt_reported_blas_threads=$(BLAS.get_num_threads()) blas_config=$(repr(BLAS.get_config())) VECLIB_MAXIMUM_THREADS=$(get(ENV, "VECLIB_MAXIMUM_THREADS", "unset")) OPENBLAS_NUM_THREADS=$(get(ENV, "OPENBLAS_NUM_THREADS", "unset")) OMP_NUM_THREADS=$(get(ENV, "OMP_NUM_THREADS", "unset"))")
