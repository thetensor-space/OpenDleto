########################################################################
#  CC-BY 2021 
#  Peter A Brooksbank
#  Martin Kassabov
#  James B. Wilson
#
#    Distributed under MIT License.
########################################################################


using LinearAlgebra  # For svd.

#
# Given a 3-tensor r, 
# Return the matrix whose left nullspace is the flattened Derivation algebra of t
#
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
# Given a 3-tensor r, 
# Return the matrix whose left nullspace is the flattened Derivation algebra of t
#
function adj(t)
    F = eltype(t)
    a = size(t)[1]; b = size(t)[2]; c = size(t)[3]
    step = a^2+b^2;
    M = zeros(F, a*b*c*step )
    for i = axes(t,1)
        for j = axes(t,2) 
            for k = axes(t,3)
                row = i-1+a*((j-1)+b*(k-1))
                # ΣZ_{il}Γ_{ljk}
                copyto!(M, step*row+a*(i-1)+1, t[1:a,j,k], 1)
                # ΣΓ_{ilk}Y_{jl}
                copyto!(M, step*row+a^2+b*(j-1)+1, -t[i,1:b,k], 1)
                # # -ΣΓ_{ijl}Z_{lk}
                # copyto!(M, step*row+ a^2+b^2+c*(k-1)+1, -t[i,j,1:c], 1)
                # #M = vcat(M,derRow(t, i,j,k))
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
    x = reshape(u[1:(a^2)],(a,a))
    y = reshape(u[(a^2+1):(a^2+b^2)],(b,b))
    z = reshape(u[(a^2+b^2+1):(a^2+b^2+c^2)],(c,c))
    return x,y,z
end

#
# Given a flattened triple (a^2,b^2) matrices,
# Return them as a double of matrices
#
function inflate12(u,a,b)
    x = reshape(u[1:(a^2)],(a,a))
    y = reshape(u[(a^2+1):(a^2+b^2)],(b,b))
    return x,y
end



###############################################################################
#  Actions on Tensors
###############################################################################

function actLeft(ten,mat)
    ## PENDING CONFIRM EQUAL TO
#    foliate = reshape(t,(a,b*c))
#   return reshape( m*foliate, (a,b,c))

    ten2 = zeros(eltype(ten),size(ten))
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

function actRight(ten,mat)
    ten2 = zeros(eltype(ten),size(ten))
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

function actOut(ten,mat)
    ## PENDING SEE IF EQUAL
    #foliate = reshape(t,(a*b,c))
    #return reshape( foliate*m, (a,b,c))
    ten2 = zeros(eltype(ten),size(ten))
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



###############################################################################
## Structure Algorithms.
###############################################################################

#
# Stratify: given a tensor t and an optional spectral gap
# Detect if a there is a startification and return a 
# change of basis to demonstrate it.
#
function stratify(t, gap=2)
    a = size(t)[1]; b = size(t)[2]; c = size(t)[3]
    u,s,v = svd(der(t))
    print( "\tFinal singular values ... " )
    print( string(round( s[(a^2+b^2+c^2)-3], digits=5) )* ",\t" )
    print( string(round( s[(a^2+b^2+c^2)-2], digits=5) )* ",\t" )
    print( string(round( s[(a^2+b^2+c^2)-1], digits=5) )* ",\t" )
    print( string(round( s[(a^2+b^2+c^2)], digits=5)) * "\n" )

    signal = (a^2+b^2+c^2)-2
    if round( s[signal], digits=4) < gap 
        x, y, z = inflate(u[:,signal], a,b,c)
        xvals,xvecs = eigen(x)
        yvals,yvecs = eigen(y)
        zvals,zvecs = eigen(z)
        t2 = actLeft(t,xvecs)
        t2 = actRight(t2,yvecs)
        t2 = actOut(t2,zvecs)
        return true, t2, [x,y,z], [xvals, yvals, zvals]
    else
        return false, [], [], []
    end
end 

#
# Blaock 12: given a tensor t and an optional spectral gap
# Detect if a there is a block decomposition on the 12-face
# and change the basis to match.
#
function block12(t, gap=2)
    a = size(t)[1]; b = size(t)[2]
    u,s,v = svd(adj(t))
    print( "\tFinal singular values ... " )
    print( string(round( s[(a^2+b^2)-3], digits=4) )* ",\t" )
    print( string(round( s[(a^2+b^2)-2], digits=4) )* ",\t" )
    print( string(round( s[(a^2+b^2)-1], digits=4) )* ",\t" )
    print( string(round( s[(a^2+b^2)], digits=4)) * "\n" )

    signal = (a^2+b^2)-1
    if round( s[signal], digits=4) < gap 
        x, y = inflate12(u[:,signal], a,b)
        xvals,xvecs = eigen(x)
        yvals,yvecs = eigen(y)
        t2 = actLeft(t,xvecs)
        t2 = actRight(t2,yvecs)
        return true, t2, [x,y], [xvals, yvals]
    else
        return false, [], [], []
    end
end 
