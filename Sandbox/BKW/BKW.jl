using LinearAlgebra
using SparseArrays
using Dates

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


###############################################################

function randTrans(d)
    m = Matrix{Complex{Float16}}(I,d,d)
    i = rand(1:d)
    j = rand(1:d)
    v = complex(rand(Float16))
    if v != 0 
        m[i,j] = complex(rand(Float16))
    end 
    return m
end 


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

function matnorm( m)
    return map((x)->norm(x),m)
end

function matround( m)
    return map((x)->round(norm(x),digits=4),m)
end

function stratify(t)
    a = size(t)[1]; b = size(t)[2]; c = size(t)[3]
    u,s,v = svd(der(t))
    if round( s[(a^2+b^2+c^2)-2], digits=4) < 0.5 
        x, y, z = inflate(u[:,(a^2+b^2+c^2)-2], a,b,c)
        xvecs = eigvecs(x)
        yvecs = eigvecs(y)
        zvecs = eigvecs(z)
        t2 = actLeft(t,transpose(xvecs))
        t2 = actRight(t,yvecs)
        t2 = actOut(t2,zvecs)
        return true, t2, [x,y,z]
    else
        return false, [], []
    end
end 


function randomize(t,rounds)
    ## 1 randomizedx
    d = size(t)[1]
    mat = Matrix{Complex{Float16}}(I,d,d)
    for i = 1:rounds
        mat *= randTrans(size(t)[1])
    end
    t = actLeft( t, mat )

    ## 1 randomizedx
    d = size(t)[2]
    mat = Matrix{Complex{Float16}}(I,d,d)
    for i = 1:rounds
        mat *= randTrans(size(t)[2])
    end
    t = actRight( t, mat )

    ## 3 randomized
    d = size(t)[3]
    mat = Matrix{Complex{Float16}}(I,d,d)
    for i = 1:rounds 
        mat *= randTrans(size(t)[3])
    end
    t = actOut(t,mat)

    return t
end 


# Return a tensor with sparse randome values
function sprandten(F, dims, density)
    return reshape( sprand(F,prod(dims), density), dims)
end


function test(d,param1, param2)
    t = MartiniT(d,d,d,param)
    save3D( "images/plot-"*string(d)*"-org.ply", matround(t))
    pass, nt, mats = stratify(t)
    save3D( "images/plot-"*string(d)*"-org-recons.ply", matround(nt))

    rt = randomize(t, param2)
    save3D( "images/plot-"*string(d)*"-rand.ply", matround(rt))
    pass, nrt, mats = stratify(rt)
    save3D( "images/plot-"*string(d)*"-rand-recons.ply", matround(nrt))
    return pass
end 
