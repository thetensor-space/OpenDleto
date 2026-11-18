using Dleto, ITensors, LinearAlgebra, Statistics
BLAS.set_num_threads(1)
folder = ARGS[1]
n = 4
tensor = reshape(collect(reinterpret(Float64, read(joinpath(folder, "tensor_4.bin")))), n,n,n)
frame = [Index(n, "axis$a") for a in 1:3]
Γ = ITensor(tensor, frame...)
Ω = IndTransverseOps(frame, UniversalOp())
_, dmap = sylvesterLM(Ω, UniversalChisel(3), Γ)
solve() = svdvals(Matrix(dmap))
solve()
samples = Float64[]
for _ in 1:7
    start = time_ns()
    global values = solve()
    push!(samples, (time_ns() - start)/1e9)
end
println("JULIA_SVD seconds=", median(samples), " nullity=", size(dmap,2)-count(>(1e-8*values[1]), values))
