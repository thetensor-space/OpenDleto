using ITensors
using Dleto
using Dleto: ⊕
using KrylovKit

function timing_sylverlining(d)
    Γ = random_itensor(
        Index(d, "x"), 
        Index(d, "y"), 
        Index(d, "z")
    )
    ch = UniversalChisel(3)
    fr = collect(inds(Γ))
    Ω = IndTransverseOps(fr, UniversalOp())    
    @time sylvester, ester = sylvesterLM(Ω, ch, Γ)
    
    println("Matrix of size: ", size(sylvester))
    println("Testing evaluation time...")
    u = randn(size(sylvester, 2))
    sparse_times = Float64[]
    for i in 1:10
        u = randn(size(sylvester, 2))
        t = @elapsed v = sylvester * u
        push!(sparse_times, t)
    end
    avg_sparse_time = sum(sparse_times) / length(sparse_times)
    println("Average time: ", avg_sparse_time, " seconds")
    println("Min time: ", minimum(sparse_times), " seconds")
    println("Max time: ", maximum(sparse_times), " seconds")

    println("\r\nTesting matrix conversion time...")
    @time M = Matrix(sylvester);
    println()
    println("\r\nDense matrix of size: ", size(M))
    println("Testing evaluation time...")
    u = randn(size(sylvester, 2))
    times = Float64[]
    for i in 1:10
        u = randn(size(sylvester, 2))
        t = @elapsed v = M * u
        push!(times, t)
    end
    avg_time = sum(times) / length(times)
    println("Average time: ", avg_time, " seconds")
    println("Min time: ", minimum(times), " seconds")
    println("Max time: ", maximum(times), " seconds")

    println()
    println("Computing smallest real eigenvalues with KrylovKit...")
    nev = min(10, size(sylvester, 1))  # Number of eigenvalues to compute
    x0 = randn(size(sylvester, 2))     # Initial guess vector
    @time λ, vecs, info = eigsolve(sylvester, x0, nev, :SR)  # :SR = smallest real
    println("Converged: ", info.converged, " eigenvalues")
    println("Smallest eigenvalues: ", real.(λ))
    
    # Verify eigenpairs
    println("\nVerifying eigenpairs...")
    max_residual = 0.0
    for i in 1:length(λ)
        residual = norm(sylvester * vecs[i] - λ[i] * vecs[i])
        max_residual = max(max_residual, residual)
        if residual > 1e-6
            println("  λ[$i] = $(λ[i]), residual = $residual ⚠️")
        else
            println("  λ[$i] = $(λ[i]), residual = $residual ✓")
        end
    end
    println("Maximum residual: $max_residual")
    
    return (eigenvalues=λ, eigenvectors=vecs, info=info)   
   
end

function isder(Ω::TransverseOps, 
            P::AbstractMatrix,
            Γ::ITensor; 
            δ ::Vector{ITensor}
            ) :: Bool where (N,K) 
    frame = inds(Γ)
    frames = Vector{ITensor}(undef, length(frame))
    for i in 1:length(frame)
        # inds(δ[i]) = (e, Der.e)
        e_der = first(setdiff(inds(δ[i]), [frame[i]]))
        frames[i] = replaceind(Γ, frame[i], e_der)
    end
    frames = [ ( a for a in frame if a != e) for e in frame]
    Γs = [ replaceind(Γ, e, δ[e]) for e in keys(δ) ]
    member = true; # (coordinates(Ω, δ) !== nothing) 
    isder = mapreduce(a-> P[a]*δ[a]*Γs[a], +, engaged(Ω)) ≈ zero(Γ) 
    return member && isder
end


"""
    der(method::DerivationMethod, 
            Ω::TransverseOps, 
            P::LinearChisel, 
            Γ ::ITensor;
            tol::Float64=1e-6,
            nd::Integer=10
        ) :: Matrix{K} where K

    Computes up to `nd` many `P`-derivations of `Γ` for the to the given chisel `P` and transverse operators `Ω`.
    If `nd` is negative or exceeds the dimension of the derivation space
    then the full derivation space is returned.
    - `method`: An instance of a subtype of `DerivationMethod` defining the solving method.
    - `Ω`: The transverse operators.
    - `P`: a linear chisel
    - `Γ`: The input tensor
    - `nd`: (optional) Maximum number of singular vectors to compute (default: 10)
    - `tol`: (optional) Tolerance for the solver (default: 1e-6).
    
    Returns a matrix whose columns are vectorized derivations.
"""
function time_der(dims)
    for d in dims
        ds = [0,0,0] 
        blocks = []
        while sum(ds) < d
            push!(blocks, [rand(3:7),rand(3:7),rand(3:7)])
            ds .+= blocks[end]
        end

        Δ = zeros(Float64, 0,0,0)
        for b in blocks
            Δ = Δ ⊕ randn(b...)
        end

        ch = UniversalChisel(3)
        fr = collect(inds(Δ))
        Ω = IndTransverseOps(fr, UniversalOp())    
        
        println("Timing der with size $(size(Δ))")
        @time ders = der(SylverLiningMethod(), Ω, ch, Δ; tol=1e-6, nd=-1)
        println("Computed ", length(ders), " derivatons of tensor.")
    end
    return nothing
end