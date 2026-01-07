using ITensors
using Dleto
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

function time_der(dims)
    for d in dims
        Γ = random_itensor(
            Index(d, "x"), 
            Index(d, "y"), 
            Index(d, "z")
        )
        ch = UniversalChisel(3)
        fr = collect(inds(Γ))
        Ω = IndTransverseOps(fr, UniversalOp())    
        
        println("Timing der with d = $d")
        @time ders = der(SylverLiningMethod(), Ω, ch, Γ; tol=1e-6, nd=-1)
        println("Computed ", length(ders), " derivative tensors.")
    end
    return nothing
end