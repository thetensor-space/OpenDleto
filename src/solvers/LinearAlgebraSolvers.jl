struct SVDSolver <: NullSolver end
    
function solve_extra(::SVDSolver, L::LinearMaps.LinearMap; nd::Integer = 10, tol::Float64 = 1e-8, kwargs...)
    M = Matrix(L)
    svds = LinearAlgebra.svd(M;full=true, kwargs...)
    nvals = nd > 0 ? min(nd, size(M,1)) : size(M,1) 
    return (;vals=svds.S[end:-1:(end-nvals+1)], vecs=svds.V[:, end:-1:(end-nvals+1)])
end

NullSolversDict[:SVDSolver] = SVDSolver();
NullSolversDict[:default ] = SVDSolver();

struct EigenSolver <: NullSolver end;
#assumes that the matrix is symmetric    

function solve_extra(::EigenSolver, L::LinearMaps.LinearMap; nd::Integer = 10, tol::Float64 = 1e-8, kwargs...)
    M = Matrix(L)
    eigens = LinearAlgebra.eigen( LinearAlgebra.Symmetric(M) )
    return (;vals = eigens.values, vecs=eigens.vectors) 
end

NullSolversDict[:EigenSolver] = EigenSolver();

struct PartialEigenSolver <: NullSolver end;
#assumes that the matrix is symmetric    

function solve_extra(::PartialEigenSolver, L::LinearMaps.LinearMap; nd::Integer = 10, tol::Float64 = 1e-8, kwargs...)
    M = Matrix(L)
    eigens = LinearAlgebra.eigen( LinearAlgebra.Symmetric(M), 1:min(nd,size(M,1)) )
    return (;vals = eigens.values, vecs=eigens.vectors) 
end

NullSolversDict[:PartialEigenSolver] = PartialEigenSolver();


## LU solver might not be wokring!!!

# struct LUSolver <: NullSolver end

# function solve(::LUSolver, L::LinearMaps.LinearMap; nv::Integer = 10, tol::Float64 = 1e-8,kwargs...)
#     println("Using LUSolver on Matrix...")
#     M = Matrix(L)
#     println("Performing LU Factorization...")
#     F = LinearAlgebra.lu(M;kwargs...)
#     U = F.U
#     L = F.L
#     p = F.p  # permutation

#     # Determine rank and free variables
#     rank = sum(abs.(diag(F.U)) .> tol)
#     n = size(M, 2)
#     # free_vars = rank+1:n
#     free_vars = (n-nv+1):n
# println("Matrix rank: $rank, free variables: ", free_vars, " with tolerance $tol")
#     # Build null space basis
#     null_basis = []
#     for j in free_vars
#         v = zeros(n)
#         v[j] = 1
#         # Solve U * x = -U[:, free_vars]*v_free for pivot variables
#         rhs = -U[:, free_vars] * v[free_vars]
#         x = U[1:rank, 1:rank] \ rhs[1:rank]
#         v[1:rank] = x
#         push!(null_basis, v)
#     end

#     return null_basis #(;vals=svds.S[end:-1:(end-nvals+1)], vecs=svds.V[:, end:-1:(end-nvals+1)])
# end;

# NullSolversDict[:LUSolver] = LUSolver();
