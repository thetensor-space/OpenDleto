# Development-only fixture generator. Run from repository root with Julia 1.12:
# julia --compiled-modules=existing --project=. python/examples/generate_julia_fixture.jl | tail -n 1 > python/tests/fixtures/julia_diagonal_222.json
using Dleto, ITensors, LinearAlgebra
frames = [Index(2, "axis$i") for i in 1:3]
tensor = zeros(2, 2, 2)
tensor[1, 1, 1] = 1
tensor[2, 2, 2] = 2
itensor = ITensor(tensor, frames...)
operators = IndTransverseOps(frames, UniversalOp())
chisel = UniversalChisel(3)
normal, dmap = sylvesterLM(operators, chisel, itensor)
x = zeros(12)
x[1] = 1
values = svdvals(Matrix(dmap))
nullity = size(dmap, 2) - sum(values .> 1e-9)
println("{\"reference_commit\":\"4635b39bf4837bb99919ff887e467f2bc329c528\",\"tensor_shape\":[2,2,2],\"tensor_nonzero\":[[0,0,0,1],[1,1,1,2]],\"chisel\":[[1,1,1]],\"forward_e1\":[", join(dmap * x, ","), "],\"singular_values\":[", join(values, ","), "],\"nullity\":", nullity, "}")
