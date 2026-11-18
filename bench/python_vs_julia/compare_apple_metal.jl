# Compare Julia's Apple Metal SylverLining backend using the same binary inputs.
using Dleto, ITensors, LinearAlgebra, Statistics, Metal
Metal.functional() || error("Metal is not functional on this machine")
folder = ARGS[1]
sizes = parse.(Int, ARGS[2:end])

function median_seconds(f; repeats=11)
    for _ in 1:5
        Dleto.gpu_sync(f)
    end
    samples = Float64[]
    for _ in 1:repeats
        start = time_ns()
        Dleto.gpu_sync(f)
        push!(samples, (time_ns() - start) / 1e9)
    end
    return median(samples)
end

for n in sizes
    tensor = Float32.(reshape(collect(reinterpret(Float64, read(joinpath(folder, "tensor_$n.bin")))), n,n,n))
    x = Float32.(collect(reinterpret(Float64, read(joinpath(folder, "x_$n.bin")))))
    y = Float32.(collect(reinterpret(Float64, read(joinpath(folder, "y_$n.bin")))))
    frame = [Index(n, "axis$a") for a in 1:3]
    Γ = ITensor(tensor, frame...)
    Ω = IndTransverseOps(frame, UniversalOp())
    _, dmap = sylvesterLM(Ω, UniversalChisel(3), Γ; backend=:metal)
    forward = median_seconds(() -> dmap * x)
    adjoint = median_seconds(() -> dmap' * y)
    result = dmap * x
    indices = [1 + fld(i * (length(result) - 1), 15) for i in 0:15]
    write(joinpath(folder, "julia_metal_sample_$n.bin"), result[indices])
    output_norm = norm(result)
    println("RESULT n=$n backend=JuliaMetal dtype=Float32 forward_host_seconds=$forward adjoint_host_seconds=$adjoint output_norm=$output_norm sample_count=16 julia_threads=$(Threads.nthreads()) lbt_reported_blas_threads=$(BLAS.get_num_threads()) blas_config=$(repr(BLAS.get_config())) julia_version=$(VERSION) VECLIB_MAXIMUM_THREADS=$(get(ENV, "VECLIB_MAXIMUM_THREADS", "unset")) OPENBLAS_NUM_THREADS=$(get(ENV, "OPENBLAS_NUM_THREADS", "unset")) OMP_NUM_THREADS=$(get(ENV, "OMP_NUM_THREADS", "unset"))")
    flush(stdout)
end
