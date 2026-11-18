# Run from repository root with JULIA_NUM_THREADS=1 and --compiled-modules=existing.
using Dleto, ITensors, LinearAlgebra, Statistics

BLAS.set_num_threads(1)
folder = ARGS[1]
dtype = length(ARGS) > 2 && ARGS[2] == "float32" ? Float32 : Float64
sizes = parse.(Int, ARGS[3:end])

function median_seconds(f; repeats=7)
    f() # JIT and allocation warmup
    samples = Float64[]
    for _ in 1:repeats
        start = time_ns()
        f()
        push!(samples, (time_ns() - start) / 1e9)
    end
    return median(samples)
end

for n in sizes
    tensor = dtype.(reshape(collect(reinterpret(Float64, read(joinpath(folder, "tensor_$n.bin")))), n, n, n))
    x = dtype.(collect(reinterpret(Float64, read(joinpath(folder, "x_$n.bin")))))
    y = dtype.(collect(reinterpret(Float64, read(joinpath(folder, "y_$n.bin")))))
    frame = [Index(n, "axis$a") for a in 1:3]
    Γ = ITensor(tensor, frame...)
    Ω = IndTransverseOps(frame, UniversalOp())
    _, dmap = sylvesterLM(Ω, UniversalChisel(3), Γ)
    forward = median_seconds(() -> dmap * x)
    adjoint = median_seconds(() -> dmap' * y)
    tag = dtype == Float32 ? "f32" : "f64"
    write(joinpath(folder, "julia_forward_$(tag)_$n.bin"), dmap * x)
    write(joinpath(folder, "julia_adjoint_$(tag)_$n.bin"), dmap' * y)
    println("RESULT n=$n dtype=$dtype forward_seconds=$forward adjoint_seconds=$adjoint")
end
