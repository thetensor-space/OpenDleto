# Independent derivation-law and timing audit.
#
# Examples:
#   julia --project=. bench/DerivationSolverAudit.jl correctness SVDSolver
#   julia --project=. bench/DerivationSolverAudit.jl performance QuickDer 80
#
# The caller is expected to enforce the wall-clock limit externally, one process
# per solver. This keeps a stalled backend from concealing the results of the
# others.
using Dleto
using ITensors
using LinearAlgebra
using Random

mode = only(ARGS[1:1])

# Activate every optional solver that is installed in this project.
if mode == "correctness"
    for package in (:KrylovKit, :IterativeSolvers)
        try
            @eval using $(package)
        catch err
            @warn "Optional solver package unavailable" package exception = (err, catch_backtrace())
        end
    end
end

function residual(Γ, D, P)
    R = applyDerivation(Γ, D, Chisel(P, collect(inds(Γ))))
    norm(R) / max(norm(Γ) * maximum(norm.(D)), eps())
end

function tensor_case(name, n)
    if name == "random"
        return randn(n, n, n)
    elseif name == "rank1"
        return [i * j * k for i in 1:n, j in 1:n, k in 1:n] .+ 0.0
    elseif name == "smooth_noise"
        return [i + j + k for i in 1:n, j in 1:n, k in 1:n] .+ 1e-3 .* randn(n, n, n)
    end
    error("unknown tensor case: $name")
end

function audit_correctness(solver)
    P = UniversalChisel(3)
    for name in ("random", "rank1", "smooth_noise")
        n = 5
        frame = [Index(n, "a_$i") for i in 1:3]
        Γ = ITensor(tensor_case(name, n), frame...)
        local basis, elapsed
        try
            elapsed = @elapsed basis = der(:SylverLining, Γ; solver, tol = 1e-6)
            rs = isempty(basis) ? Float64[] : residual.(Ref(Γ), basis, Ref(P))
            maxres = isempty(rs) ? Inf : maximum(rs)
            # UniversalChisel has a two-dimensional scalar-derivation subspace.
            verdict = length(basis) >= 2 && maxres < 1e-6 ? "PASS" : "FAIL"
            println("mode=correctness solver=$solver case=$name seconds=$(round(elapsed; digits=3)) basis=$(length(basis)) max_relative_residual=$maxres verdict=$verdict")
        catch err
            println("mode=correctness solver=$solver case=$name verdict=ERROR error=$(sprint(showerror, err))")
        end
    end
end

function audit_performance(method, n)
    P = UniversalChisel(3)
    frame = [Index(n, "a_$i") for i in 1:3]
    Γ = ITensor(tensor_case("random", n), frame...)
    kwargs = method in available_solvers() ? (; solver = method) : (;)
    derivation_method = method in available_solvers() ? :SylverLining : method
    elapsed = @elapsed basis = der(derivation_method, Γ; tol = 1e-6, kwargs...)
    rs = isempty(basis) ? Float64[] : residual.(Ref(Γ), basis, Ref(P))
    maxres = isempty(rs) ? Inf : maximum(rs)
    verdict = !isempty(basis) && maxres < 1e-6 ? "PASS" : "FAIL"
    println("mode=performance method=$method n=$n seconds=$(round(elapsed; digits=3)) basis=$(length(basis)) max_relative_residual=$maxres verdict=$verdict")
end

Random.seed!(0x0D1E70)
if mode == "correctness"
    solver = Symbol(only(ARGS[2:2]))
    solver in available_solvers() || error("solver $solver is not registered; available: $(available_solvers())")
    audit_correctness(solver)
elseif mode == "performance"
    audit_performance(Symbol(only(ARGS[2:2])), parse(Int, only(ARGS[3:3])))
else
    error("mode must be correctness or performance")
end
