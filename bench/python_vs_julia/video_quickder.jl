# Actual QuickDer solve on a 3-D grayscale video stored as Fortran-order Float32.
# julia --project=<environment> video_quickder.jl FOLDER NX NY NZ [SEED [TOL [ND]]]
# Use a Metal-enabled project; enforce a wall-time limit externally. `nd > 0`
# requests near-derivations too and is not a certificate of a full nullspace.
# The matrix-free restriction path is forced to avoid multi-gigabyte dense solves.
using Dleto, ITensors, LinearAlgebra, Metal, Random, Statistics
length(ARGS) in 4:7 || error("expected FOLDER NX NY NZ [SEED [TOL [ND]]]")
folder = ARGS[1]
dims = Tuple(parse.(Int, ARGS[2:4]))
seed = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 20261118
tol = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 1e-4
nd = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : -1
all(>(0), dims) && tol > 0 || error("invalid shape or tolerance")
Metal.functional() || error("Metal unavailable")
Dleto.QDN_DENSE_BUDGET_BYTES[] = 0.0
Dleto.QDN_GPU_DENSE_BUDGET_BYTES[] = 0.0
# Separate deterministic warmup, with the same method and matrix-free budget.
function warmup()
    dims = (4, 5, 6)
    frame = [Index(n, "warm$a") for (a, n) in enumerate(dims)]
    data = randn(MersenneTwister(20261118), Float32, dims...)
    Γ = ITensor(data, frame...)
    Ω = IndTransverseOps(frame, UniversalOp())
    method = get_derivation_method(:QuickDer; device=:gpu, verify=:random, seed=20261118)
    derTrOpsReduced(method, Ω, UniversalChisel(3), Γ; tol=1e-4, return_diagnostics=true)
    nothing
end
warm_start = time_ns()
warmup()
println("WARMUP shape=(4,5,6) seconds=$((time_ns()-warm_start)/1e9)")
flush(stdout)
setup_start = time_ns()
tensor = reinterpret(Float32, read(joinpath(folder, "tensor.f32")))
length(tensor) == prod(dims) || error("input size mismatch")
frame = [Index(n, "axis$a") for (a, n) in enumerate(dims)]
Γ = ITensor(reshape(tensor, dims), frame...)
Ω = IndTransverseOps(frame, UniversalOp())
P = UniversalChisel(3)
method = get_derivation_method(:QuickDer; device=:gpu, verify=:random, seed=seed)
setup_seconds = (time_ns()-setup_start) / 1e9
println("START shape=$dims dtype=Float32 requested_device=gpu verify=random seed=$seed tol=$tol nd=$nd dense_budget_bytes=0 setup_seconds=$setup_seconds julia_version=$VERSION julia_threads=$(Threads.nthreads()) blas_threads=$(BLAS.get_num_threads()) blas_config=$(repr(BLAS.get_config()))")
flush(stdout)
# Diagnostics include per-direction verification; timing includes those passes and
# any just-in-time compilation, so it is end-to-end rather than warmed throughput.
start = time_ns()
_, expand_map, ders, rep = derTrOpsReduced(method, Ω, P, Γ; tol=tol, nd=nd,
                                  progress=true, return_diagnostics=true)
seconds = (time_ns()-start) / 1e9
# The report's checks may use slices. Independently check the first two
# returned directions against the full host tensor in bounded blocks.
full_start = time_ns()
G = reshape(tensor, dims)
full_residuals = [begin
    coords = Vector{Float32}(expand_map * ders[:, j])
    matrices = [Matrix{Float32}(M) for M in Dleto.embedMatrices(Ω, coords)]
    Float64(Dleto.der_residual(G, matrices, P))
end for j in 1:min(2, size(ders, 2))]
full_residual_seconds = (time_ns()-full_start) / 1e9
summary(v) = v === nothing ? nothing : isempty(v) ? (count=0,) :
    (count=length(v), min=minimum(v), median=median(v), max=maximum(v))
println("RESULT solve_and_diagnostics_seconds=$seconds returned=$(rep.returned) nullity=$(rep.nullity) scalar_dim=$(rep.scalar_dim) status=$(rep.status) certified=$(rep.certified) policy=$(rep.policy) rule=$(rep.rule) requested_nd=$(rep.requested_nd) selected_summary=$(summary(rep.selected)) next_value=$(rep.next_value) threshold=$(rep.threshold) precision_floor=$(rep.precision_floor) residual_summary=$(summary(rep.residuals)) lift_residual_summary=$(summary(rep.lift_residuals)) full_residuals_first_two=$full_residuals full_residual_seconds=$full_residual_seconds restriction=$(rep.restriction) restricted_size=$(rep.restricted_size) solver=$(rep.solver) device=$(rep.device) stage_times=$(rep.stage_times) coordinates_shape=$(size(ders))")
flush(stdout)
