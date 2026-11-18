# Run from repository root with --compiled-modules=existing.
# Set DLETO_BENCH_APPLE=1 for the uncapped Apple Accelerate sweep.
using Dleto, ITensors, LinearAlgebra, Statistics

apple_bench = get(ENV, "DLETO_BENCH_APPLE", "0") == "1"
if apple_bench
    occursin("Accelerate", string(BLAS.get_config())) || error("Apple Accelerate BLAS is not configured: $(BLAS.get_config())")
else
    BLAS.set_num_threads(1)
end
folder = ARGS[1]
dtype = length(ARGS) > 2 && ARGS[2] == "float32" ? Float32 : Float64
sizes = parse.(Int, ARGS[3:end])

function median_seconds(f; repeats=11)
    for _ in 1:5
        f()
    end # JIT and allocation warmup
    samples = Float64[]
    for _ in 1:repeats
        start = time_ns()
        f()
        push!(samples, (time_ns() - start) / 1e9)
    end
    return median(samples)
end

for n in sizes
    apple_bench && n > 128 && error("Apple CPU benchmark is limited to n <= 128")
    tensor = dtype.(reshape(collect(reinterpret(Float64, read(joinpath(folder, "tensor_$n.bin")))), n, n, n))
    x = dtype.(collect(reinterpret(Float64, read(joinpath(folder, "x_$n.bin")))))
    y = dtype.(collect(reinterpret(Float64, read(joinpath(folder, "y_$n.bin")))))
    frame = [Index(n, "axis$a") for a in 1:3]
    Γ = ITensor(tensor, frame...)
    Ω = IndTransverseOps(frame, UniversalOp())
    _, dmap = sylvesterLM(Ω, UniversalChisel(3), Γ; backend=:array)
    forward = median_seconds(() -> dmap * x)
    adjoint = median_seconds(() -> dmap' * y)
    tag = dtype == Float32 ? "f32" : "f64"
    write(joinpath(folder, "julia_forward_$(tag)_$n.bin"), dmap * x)
    write(joinpath(folder, "julia_adjoint_$(tag)_$n.bin"), dmap' * y)
    println("RESULT n=$n dtype=$dtype forward_seconds=$forward adjoint_seconds=$adjoint julia_threads=$(Threads.nthreads()) lbt_reported_blas_threads=$(BLAS.get_num_threads()) blas_config=$(repr(BLAS.get_config())) julia_version=$(VERSION) VECLIB_MAXIMUM_THREADS=$(get(ENV, "VECLIB_MAXIMUM_THREADS", "unset")) OPENBLAS_NUM_THREADS=$(get(ENV, "OPENBLAS_NUM_THREADS", "unset")) OMP_NUM_THREADS=$(get(ENV, "OMP_NUM_THREADS", "unset"))")
    flush(stdout)
end
