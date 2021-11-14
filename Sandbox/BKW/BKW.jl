using LinearAlgebra
using SparseArrays
using Dates
using Random
#using Basic.FileSystem

### Used to save data
## Reload using 
## load("mydata.jld")["data"]
using Pkg
Pkg.add("JLD")
using JLD


include( "Tensor3D.jl")
include("Laplace.jl")

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

function matround( m, dig)
    return map((x)->round(norm(x),digits=dig),m)
end

function stratify(t)
    a = size(t)[1]; b = size(t)[2]; c = size(t)[3]
    u,s,v = svd(der(t))
    print( "\tFinal singular values ... " )
    print( string(round( s[(a^2+b^2+c^2)-3], digits=5) )* ",\t" )
    print( string(round( s[(a^2+b^2+c^2)-2], digits=5) )* ",\t" )
    print( string(round( s[(a^2+b^2+c^2)-1], digits=5) )* ",\t" )
    print( string(round( s[(a^2+b^2+c^2)], digits=5)) * "\n" )

    if round( s[(a^2+b^2+c^2)-2], digits=4) < 0.5 
        x, y, z = inflate(u[:,(a^2+b^2+c^2)-2], a,b,c)
        xvals,xvecs = eigen(x)
        yvals,yvecs = eigen(y)
        zvals,zvecs = eigen(z)
		# print("\tEigenvalues")
		# print(xvals)
		# print("\n")
		# print(yvals)
		# print("\n")
		# print(zvals)
		# print("\n\n")
        t2 = actLeft(t,xvecs)
        t2 = actRight(t2,yvecs)
        t2 = actOut(t2,zvecs)
        return true, t2, [x,y,z], [xvals, yvals, zvals]
    else
        return false, [], [], []
    end
end 

function randomperm(ten)
    pi1 = randperm(size(ten)[1])
    pi2 = randperm(size(ten)[2])
    pi3 = randperm(size(ten)[3])

    ten2 = zeros(eltype(ten),size(ten))
    for i = 1:size(ten)[1]
        for j = 1:size(ten)[2]
            for k = 1:size(ten)[3]
                for m = 1:size(ten)[3]
                    ten2[i,j,k] = ten[pi1[i],pi2[j],pi3[k]]
                end
            end
        end
    end
    return ten2
end

function randomize(t,rounds)
    ## 1 randomizedx
    d = size(t)[1]
#    mat = rand(Complex{Float16}, (d,d))
    mat = Matrix{Complex{Float16}}(I,d,d)
    for i = 1:rounds
        mat *= randTrans(size(t)[1])
    end
    t = actLeft( t, mat )

    ## 1 randomizedx
    d = size(t)[2]
#    mat = rand(Complex{Float16}, (d,d))
    mat = Matrix{Complex{Float16}}(I,d,d)
    for i = 1:rounds
        mat *= randTrans(size(t)[2])
    end
    t = actRight( t, mat )

    ## 3 randomized
    d = size(t)[3]
#    mat = rand(Complex{Float16}, (d,d))
    mat = Matrix{Complex{Float16}}(I,d,d)
    for i = 1:rounds 
        mat *= randTrans(size(t)[3])
    end
    t = actOut(t,mat)

    return randomperm(t)
end 


# Return a tensor with sparse randome values
function sprandten(F, dims, density)
    return reshape( sprand(F,prod(dims), density), dims)
end

function test(d,param1, param2, control=false)
    date = now()
    date = "" * string(year(date)) * "-" * string(month(date)) * "-" * string(day(date)) * "-time-" * string(hour(date)) * "-" * string(minute(date)) * "-" * string(second(date))
    mkdir(date)
    mkdir( date * "/data")
    mkdir( date * "/images")

    print("Creating original\n")
    t = MartiniT(d,d,d,param1)    
    print("Saving original\n")
    save( date * "/data/original.jld", "data", t)
    
    if control 
        print("Startifying original.\n")
        @time pass, nt, mats = stratify(t)
        print("Saving original stratification.\n" )
        save( date * "/data/original-strat.jld", "data", nt)
    end 

    print( "Randomizing original.\n")
    @time rt = randomize(t, param2)
    print( "Saving randomized version.\n")
    save( date * "/data/randomized.jld", "data", rt)

    print( "Stratifying randomized version.\n")
    @time pass, nrt, mats = stratify(rt)
    print( "Saving stratification of randomized.\n")
    save( date * "/data/randomized-start.jld", "data", nrt)

    print( "Rending in 3D...")
    save3D( date * "/images/plot-"*string(d)*"-org.ply", matround(t,3))
    if control
        save3D( date * "/images/plot-"*string(d)*"-org-recons.ply", matround(nt,3))
    end 
    save3D( date * "/images/plot-"*string(d)*"-rand.ply", matround(rt,3))
    save3D( date * "/images/plot-"*string(d)*"-rand-recons.ply", matround(nrt,3))

    return pass
end 
