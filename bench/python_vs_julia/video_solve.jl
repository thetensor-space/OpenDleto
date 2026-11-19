# Full matrix-free null-space solve, with one input tensor and a seeded Krylov start.
# julia --project=<environment> video_solve.jl FOLDER NX NY NZ NV0 MAXITER TOL SEED [ND]
# Use a Metal-enabled environment and JULIA_DEPOT_PATH containing its dependencies.
# maxiter bounds each Krylov attempt, not the total adaptive solve; use an external
# time limit and capture stdout when a strict wall-clock limit is required.
using Dleto, ITensors, LinearAlgebra, Metal, Random
:KrylovSolver in available_solvers() || error("KrylovSolver is not loaded; use an environment that loads KrylovKit")
length(ARGS) in (8, 9) || error("expected FOLDER NX NY NZ NV0 MAXITER TOL SEED [ND]")
folder = ARGS[1]
dims = Tuple(parse.(Int, ARGS[2:4]))
nv0, maxiter = parse.(Int, ARGS[5:6])
tol = parse(Float64, ARGS[7])
seed = parse(Int, ARGS[8])
nd = length(ARGS) == 9 ? parse(Int, ARGS[9]) : -1
all(>(0), dims) && nv0 > 0 && maxiter > 0 && tol > 0 || error("invalid arguments")
Metal.functional() || error("Metal unavailable")
# A separate deterministic 4×5×6 tensor has 77 coordinates, above KrylovSolver's
# Float32 dense gate for nv=2 (32), and warms the iterative GPU code path.
# Its map and results are discarded; the production solve uses its own seed.
function warmup()
    wdims = (4, 5, 6)
    wframe = [Index(n, "warm$a") for (a, n) in enumerate(wdims)]
    wdata = randn(MersenneTwister(20261118), Float32, wdims...)
    wnormal, _ = sylvesterLM(IndTransverseOps(wframe, UniversalOp()),
                            UniversalChisel(3), ITensor(wdata, wframe...); backend=:metal)
    Dleto.solve(wnormal, :KrylovSolver; nv=2, maxiter=1, krylovdim=8, seed=20261118)
    nothing
end
warm_start = time_ns()
warmup()
println("WARMUP shape=(4,5,6) coordinates=77 nv=2 maxiter=1 seconds=$((time_ns()-warm_start)/1e9)")
flush(stdout)
setup_start = time_ns()
tensor = reinterpret(Float32, read(joinpath(folder, "tensor.f32")))
length(tensor) == prod(dims) || error("input size mismatch")
frame = [Index(n, "axis$a") for (a, n) in enumerate(dims)]
Γ = ITensor(reshape(tensor, dims), frame...)
Ω = IndTransverseOps(frame, UniversalOp())
normal, dmap = sylvesterLM(Ω, UniversalChisel(3), Γ; backend=:metal)
setup_seconds = (time_ns() - setup_start) / 1e9
println("START setup_seconds=$setup_seconds backend=metal shape=$dims coordinates=$(size(dmap,2)) solver=KrylovSolver nv0=$nv0 nd=$nd maxiter=$maxiter tol=$tol seed=$seed julia_version=$VERSION julia_threads=$(Threads.nthreads()) blas_threads=$(BLAS.get_num_threads()) blas_config=$(repr(BLAS.get_config())) VECLIB_MAXIMUM_THREADS=$(get(ENV, "VECLIB_MAXIMUM_THREADS", "unset"))")
flush(stdout)
# Includes all solver iterations and automatic request escalation, but not map setup.
start = time_ns()
result = solve_nullspace(normal, :KrylovSolver; squared=true, nv0=nv0, nd=nd,
                         maxiter=maxiter, tol=tol, seed=seed, store_eltype=Float32, progress=true,
                         label="video derivation solve")
seconds = (time_ns() - start) / 1e9
v = result.verdict
# Compute actual residuals outside the timed solve, one vector at a time.
residuals = [norm(dmap * result.vecs[:, i]) for i in axes(result.vecs, 2)]
println("RESULT solve_seconds=$seconds nd=$nd nullity=$(size(result.vecs,2)) status=$(v.status) certified=$(v.certified) rule=$(v.rule) requested=$(v.requested) values=$(result.vals) residual_norms=$residuals clip_seconds=10 fps=30 realtime_factor=$(10 / seconds)")
flush(stdout)
