# Compare Julia's Apple Metal SylverLining backend using the same binary inputs.
using Dleto, ITensors, LinearAlgebra, Statistics, Metal
BLAS.set_num_threads(1)
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
    write(joinpath(folder, "julia_metal_forward_$n.bin"), dmap * x)
    write(joinpath(folder, "julia_metal_adjoint_$n.bin"), dmap' * y)
    println("RESULT n=$n backend=JuliaMetal dtype=Float32 forward_seconds=$forward adjoint_seconds=$adjoint")
    flush(stdout)
end
