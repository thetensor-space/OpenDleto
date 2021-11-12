using LinearAlgebra
using SparseArrays

#
# Given a 3-tensor r, 
# Return the matrix whose left nullspace is the flattened Derivation algebra of t
function der(t)
    F = eltype(t)
    a = size(t)[1]; b = size(t)[2]; c = size(t)[3]
    step = a^2+b^2+c^2;
    M = zeros(F, a*b*c*step )
    for i = axes(t,1)
        for j = axes(t,2) 
            for k = axes(t,3)
                row = i-1+a*((j-1)+b*(k-1))
                # ΣZ_{il}Γ_{ljk}
                copyto!(M, step*row+a*(i-1)+1, t[1:a,j,k], 1)
                # ΣΓ_{ilk}Y_{jl}
                copyto!(M, step*row+a^2+b*(j-1)+1, -t[i,1:b,k], 1)
                # -ΣΓ_{ijl}Z_{lk}
                copyto!(M, step*row+ a^2+b^2+c*(k-1)+1, -t[i,j,1:c], 1)
                #M = vcat(M,derRow(t, i,j,k))
            end
        end
    end
    return reshape( M, (step,a*b*c) )
end 

#
# Given a flattened triple (a^2,b^2,c^2) matrices,
# Return them as a triple of matrices
#
function inflate(u,a,b,c)
    x = reshape(u[1:(a^2)],(a,a));
    y = reshape(u[(a^2+1):(a^2+b^2)],(b,b));
    z = reshape(u[(a^2+b^2+1):(a^2+b^2+c^2)],(c,c));
    return x,y,z
end

function stratify(t)
    a = size(t)[1]; b = size(t)[2]; c = size(t)[3]
    u,s,v = svd(der(t))
    if round( s[a*b*c-2], digits=4) == 0 
        return true, inflate(u[:,a*b*c-2], a,b,c)
    else
        return false, []
    end
end 

###############################################################

# Return a tensor with sparse randome values
function sprandten(F, dims, density)
    return reshape( sprand(F,prod(dims), density), dims)
end



d = 10;
t = sprandten(Float32,(d,d,d), 0.1);
pass, mats = stratify(t)