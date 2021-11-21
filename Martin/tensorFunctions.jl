########################################################################
#  CC-BY 2021 
#  Peter A Brooksbank
#  Martin Kassabov
#  James B. Wilson
#
#    Distributed under MIT License.
########################################################################

using LinearAlgebra
using SparseArrays
using Dates
using Random

#using Pkg
#Pkg.add("JLD")
#using JLD

#
# Given a 3-tensor t, 
# Return the matrix whose left nullspace is the flattened Derivation algebra of t
function buildDerivationMatrix(t)
    F = eltype(t)
    a = size(t)[1]; b = size(t)[2]; c = size(t)[3]
    step = a^2+b^2+c^2;
    M = zeros(F, a*b*c*step )
    for i = axes(t,1)
        for j = axes(t,2) 
            for k = axes(t,3)
                row = i-1+a*((j-1)+b*(k-1))
                # ΣX_{il}Γ_{ljk}
                copyto!(M, step*row+a*(i-1)+1, t[1:a,j,k], 1)
                # ΣΓ_{ilk}Y_{jl}
                copyto!(M, step*row+a^2+b*(j-1)+1, t[i,1:b,k], 1)
                # ΣΓ_{ijl}Z_{lk}
                copyto!(M, step*row+ a^2+b^2+c*(k-1)+1, t[i,j,1:c], 1)
            end
        end
    end
    return reshape( M, (step,a*b*c) )
end 


#
# Given a 3-tensor t, 
# Return the matrix whose left nullspace is the flattened centroid algebra of t
# MK: This needs to be checked!
function buildCentroidMatrix(t)
    F = eltype(t)
    a = size(t)[1]; b = size(t)[2]; c = size(t)[3]
    step = a^2+b^2+c^2;
    M = zeros(F, 3*a*b*c*step )
    for i = axes(t,1)
        for j = axes(t,2) 
            for k = axes(t,3)
                row = i-1+a*((j-1)+b*(k-1))

                # ΣX_{il}Γ_{ljk}
                copyto!(M, step*row + a*(i-1) + 1, t[1:a,j,k], 1)
                # -ΣΓ_{ilk}Y_{jl}
                copyto!(M, step*row + a^2 + b*(j-1) + 1, -t[i,1:b,k], 1)

                # ΣΓ_{ilk}Y_{jl}
                copyto!(M, a*b*c*step + step*row + a^2 + b*(j-1) + 1, t[i,1:b,k], 1)
                # -ΣΓ_{ijl}Z_{lk}
                copyto!(M, a*b*c*step + step*row + a^2 + b^2 + c*(k-1) + 1, -t[i,j,1:c], 1)

                # ΣΓ_{ijl}Z_{lk}
                copyto!(M, 2*a*b*c*step + step*row + a^2 + b^2 + c*(k-1) + 1, t[i,j,1:c], 1)
                # -ΣX_{il}Γ_{ljk}
                copyto!(M, 2*a*b*c*step + step*row + a*(i-1) + 1, -t[1:a,j,k], 1)
            end
        end
    end
    return reshape( M, (step,3*a*b*c) )
end 


#
# Given a 3-tensor t, 
# Return the matrix whose left nullspace is the flattened 1-2 adjoint algebra of t
# MK: This needs to be checked!
function buildAdjoint12Matrix(t)
    F = eltype(t)
    a = size(t)[1]; b = size(t)[2]; c = size(t)[3]
    step = a^2+b^2;
    M = zeros(F, a*b*c*step )
    for i = axes(t,1)
        for j = axes(t,2) 
            for k = axes(t,3)
                row = i-1+a*((j-1)+b*(k-1))
                # ΣX_{il}Γ_{ljk}
                copyto!(M, step*row + a*(i-1) + 1, t[1:a,j,k], 1)
                # -ΣΓ_{ilk}Y_{jl}
                copyto!(M, step*row + a^2 + b*(j-1) + 1, -t[i,1:b,k], 1)
            end
        end
    end
    return reshape( M, (step,a*b*c) )
end 

#
# Given a 3-tensor t, 
# Return the matrix whose left nullspace is the flattened 2-3 adjoint algebra of t
# MK: This needs to be checked!
function buildAdjoint23Matrix(t)
    F = eltype(t)
    a = size(t)[1]; b = size(t)[2]; c = size(t)[3]
    step = b^2+c^2;
    M = zeros(F, a*b*c*step )
    for i = axes(t,1)
        for j = axes(t,2) 
            for k = axes(t,3)
                row = i-1+a*((j-1)+b*(k-1))
                # ΣΓ_{ilk}Y_{jl}
                copyto!(M, a*b*c*step + step*row + b*(j-1) + 1, t[i,1:b,k], 1)
                # -ΣΓ_{ijl}Z_{lk}
                copyto!(M, a*b*c*step + step*row + b^2 + c*(k-1) + 1, -t[i,j,1:c], 1)
            end
        end
    end
    return reshape( M, (step,a*b*c) )
end 

#
# Given a 3-tensor t, 
# Return the matrix whose left nullspace is the flattened 1-3 adjoint algebra of t
# MK: This needs to be checked!
function buildAdjoint23Matrix(t)
    F = eltype(t)
    a = size(t)[1]; b = size(t)[2]; c = size(t)[3]
    step = a^2+c^2;
    M = zeros(F, a*b*c*step )
    for i = axes(t,1)
        for j = axes(t,2) 
            for k = axes(t,3)
                row = i-1+a*((j-1)+b*(k-1))
                # ΣΓ_{ijl}Z_{lk}
                copyto!(M, 2*a*b*c*step + step*row + a^2 + c*(k-1) + 1, t[i,j,1:c], 1)
                # -ΣX_{il}Γ_{ljk}
                copyto!(M, 2*a*b*c*step + step*row + a*(i-1) + 1, -t[1:a,j,k], 1)
            end
        end
    end
    return reshape( M, (step,a*b*c) )
end 



# Given a flattened triple (a^2,b^2,c^2) matrices,
# Return them as a triple of matrices
#
function inflateToTripleOfMatrices(u,a,b,c)
    x = reshape(u[1:(a^2)],(a,a));
    y = reshape(u[(a^2+1):(a^2+b^2)],(b,b));
    z = reshape(u[(a^2+b^2+1):(a^2+b^2+c^2)],(c,c));
    return x,y,z
end

#
# Given a flattened pair (a^2,b^2) matrices,
# Return them as a pair of matrices
#
function inflateToPairOfMatrices(u,a,b)
    x = reshape(u[1:(a^2)],(a,a));
    y = reshape(u[(a^2+1):(a^2+b^2)],(b,b));
    return x,y
end



# Action of matrices on the tensor
function actFirst(ten,mat)
	# the result is a tensor with complex coefficients to avoid problems when multiplying with matix with complex coeffficients
#    ten2 = zeros(Complex{Float32},size(ten))
    ten2 = zeros(Float32,size(ten))
    for i = 1:size(ten)[1]
        for j = 1:size(ten)[2]
            for k = 1:size(ten)[3]
                for m = 1:size(ten)[1]
                    ten2[i,j,k] += ten[m,j,k]*mat[m,i]
                end
            end
        end
    end
    return ten2
end


function actSecond(ten,mat)
	# the result is a tensor with complex coefficients to avoid problems when multiplying with matix with complex coeffficients
#    ten2 = zeros(Complex{Float32},size(ten))
    ten2 = zeros(Float32,size(ten))
    for i = 1:size(ten)[1]
        for j = 1:size(ten)[2]
            for k = 1:size(ten)[3]
                for m = 1:size(ten)[2]
                    ten2[i,j,k] += ten[i,m,k]*mat[m,j]
                end
            end
        end
    end
    return ten2
end


function actThird(ten,mat)
	# the result is a tensor with complex coefficients to avoid problems when multiplying with matix with complex coeffficients
#    ten2 = zeros(Complex{Float32},size(ten))
    ten2 = zeros(Float32,size(ten))
    for i = 1:size(ten)[1]
        for j = 1:size(ten)[2]
            for k = 1:size(ten)[3]
                for m = 1:size(ten)[3]
                    ten2[i,j,k] += ten[i,j,m]*mat[m,k]
                end
            end
        end
    end
    return ten2
end





