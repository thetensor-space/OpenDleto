"""
    Solver based on LinearAlgebra SVD
"""
struct SVDSolver <: NullSolver end
    
function solve_extra(::SVDSolver, L::LinearMaps.LinearMap; nd::Integer = 10, tol::Float64 = 1e-8, kwargs...)
    M = Matrix(L)
    svds = LinearAlgebra.svd(M;full=true, kwargs...)
    nvals = nd > 0 ? min(nd, size(M,1)) : size(M,1) 
    return (;vals=svds.S[end:-1:(end-nvals+1)], vecs=svds.V[:, end:-1:(end-nvals+1)])
end

NullSolversDict[:SVDSolver] = SVDSolver();
NullSolversDict[:default ] = SVDSolver();

"""
    Solver based on LinearAlgebra Complete Eigen 
"""
struct EigenSolver <: NullSolver end;
#assumes that the matrix is symmetric    

function solve_extra(::EigenSolver, L::LinearMaps.LinearMap; nd::Integer = 10, tol::Float64 = 1e-8, kwargs...)
    M = Matrix(L)
    eigens = LinearAlgebra.eigen( LinearAlgebra.Symmetric(M) )
    return (;vals = eigens.values, vecs=eigens.vectors) 
end

NullSolversDict[:EigenSolver] = EigenSolver();

"""
    Solver based on LinearAlgebra Patrial Eigen 
"""
struct PartialEigenSolver <: NullSolver end;
#assumes that the matrix is symmetric    

function solve_extra(::PartialEigenSolver, L::LinearMaps.LinearMap; nd::Integer = 10, tol::Float64 = 1e-8, kwargs...)
    M = Matrix(L)
    eigens = LinearAlgebra.eigen( LinearAlgebra.Symmetric(M), 1:min(nd,size(M,1)) )
    return (;vals = eigens.values, vecs=eigens.vectors) 
end

NullSolversDict[:PartialEigenSolver] = PartialEigenSolver();



"""
    Solver based on HouseHolder tridiagonalization Followed by Eigen 
"""
struct HouseHolderEigenSolver <: NullSolver end;

function solve_extra(::HouseHolderEigenSolver, L::LinearMaps.LinearMap; nd::Integer = 10, tol::Float64 = 1e-8, kwargs...)
    M = Matrix(L)    
    HD = LinearAlgebra.hessenberg(LinearAlgebra.Symmetric(M))
    eigens = LinearAlgebra.eigen( HD.H, 1:min(nd,size(M,1)) )
    return (;vals = eigens.values, vecs=HD.Q*eigens.vectors) 
end

NullSolversDict[:HouseHolderEigenSolver] = HouseHolderEigenSolver();

## LU solver might not be wokring!!!

"""
    Solver based on LU factorization of matrices 
"""
struct LUSolver <: NullSolver end


function solve_extra(::LUSolver, L::LinearMaps.LinearMap; nd::Integer = 10, tol = 1e-8, kwargs...)
#function lusolve(L::AbstractMatrix; nd::Integer = 10, tol = 1e-8, kwargs...)
    # println("Using LUSolver on Matrix...")
    M = Matrix(L)
    # println("Performing LU Factorization...")
    F = LinearAlgebra.lu(M;kwargs...)
    U = F.U
    # L = F.L
    # p = F.p  # permutation

    # Determine rank and free variables
    rank = sum(abs.(diag(U)) .> tol)
    # dimkernel = sum(abs.(diag(U)) .< tol)
    n = size(M, 2)
    output = min(nd, n-rank)
    # free_vars = rank+1:n
    free_vars = (n-output+1):n
# println("Matrix rank: $rank, free variables: ", free_vars, " with tolerance $tol")
    # Build null space basis
    vecs = zeros(n,output)
    vals = zeros(output)
    for i = 1:output
        v = zeros(output)
        v[i] = 1.0
        # Solve U * x = -U[:, free_vars]*v_free for pivot variables
        rhs = -U[1:rank, free_vars] * v
        x = U[1:rank, 1:rank] \ rhs
        vecs[1:rank,i] = x
        vecs[n-output + i,i] = 1.0
        vals[i] = U[n-output+i,n-output+i]
    end
    return (;vals = vals, vecs=vecs) 
end;


NullSolversDict[:LUSolver] = LUSolver();
