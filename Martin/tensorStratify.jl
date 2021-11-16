using LinearAlgebra
using SparseArrays
using Dates
using Random

using Pkg
Pkg.add("JLD")
using JLD

include("tensor3D.jl")

#
# Given a 3-tensor t, 
# Return the matrix whose left nullspace is the flattened Derivation algebra of t
function derivationMatrix(t)
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
function centroidMatrix(t)
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
# Given a flattened triple (a^2,b^2,c^2) matrices,
# Return them as a triple of matrices
#
function inflate(u,a,b,c)
    x = reshape(u[1:(a^2)],(a,a));
    y = reshape(u[(a^2+1):(a^2+b^2)],(b,b));
    z = reshape(u[(a^2+b^2+1):(a^2+b^2+c^2)],(c,c));
    return x,y,z
end




# Action of matrices on the tensor
function actFirst(ten,mat)
    ten2 = zeros(Complex{Float32},size(ten))
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
    ten2 = zeros(Complex{Float32},size(ten))
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
    ten2 = zeros(Complex{Float32},size(ten))
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





#random transovection -- might also generate diagonal matrix
function randTrans(d)
    m = Matrix{Float32}(I,d,d)
    i = rand(1:d)
    j = rand(1:d)
    v = rand(Float32)
    if v != 0 
        m[i,j] = rand(Float32)
    end 
    return m
end 

#random permutation matrix
function randomPermMarix(d)
    pi = randperm(d)
    m = Matrix{Float32}(I,d,d)
    for i = 1:d
		m[i,i] = 0
		m[i,pi[i]] = 1
    end
	return m
end 

#random matrix
function randomMarix(d,rounds)
    mat = randomPermMarix(d)
    for i = 1:rounds
        mat *= randTrans(d)
    end
	return mat
end


function tensorRandomize(t,rounds )
    d1= size(t)[1]
	m1 = randomMarix(d1, rounds)
    d2= size(t)[2]
	m2 = randomMarix(d2, rounds)
    d3= size(t)[3]
	m3 = randomMarix(d3, rounds)
    randt = actFirst( t, m1 )
    randt = actSecond( randt, m2 )
    randt = actThird( randt, m3 )
	return randt
end




###############################################################
# retunrs new tensor, trnasformations, derivation as 3 matrices, all singular value, all singular vectors

function transofromTensor(t,M,offset,toprint)
    a = size(t)[1]; b = size(t)[2]; c = size(t)[3]
    u,s,v = svd(M)
    print( "\tFinal singular values for the system ... " )
	for j= 0:toprint
		print( string(round( s[(a^2+b^2+c^2)-j], digits=4) )* ",\t" )	
	end 
	print("\n")
	x, y, z = inflate(u[:,(a^2+b^2+c^2)-offset], a,b,c)
	xvals,xvecs = eigen(x)
	yvals,yvecs = eigen(y)
	zvals,zvecs = eigen(z)
	print("\tEignevalues Xmatrix\n\t\t")
	print(xvals)
	print("\n\t Eignevalues Ymatrix\n\t\t")
	print(yvals)
	print("\n\t Eignevalues Zmatrix\n\t\t")
	print(zvals)
	print("\n")
	t2 = actFirst(t,xvecs)
	t2 = actSecond(t2,yvecs)
	t2 = actThird(t2,zvecs)
	return t2, [xvecs, yvecs, zvecs], [x,y,z], s, u
end 


function stratify(t,toprint)
	t2, mats, der, s, u = transofromTensor(t, derivationMatrix(t) , 2,toprint)
	return t2, mats, der, s, u 
end 



function curvify(t,toprint)
	t2, mats, der, s, u = transofromTensor(t, centroidMatrix(t) , 1,toprint)
	return t2, mats, der, s, u 
end 


###############################################################
# test for stratification

function stratificationTest(t,rounds)
    date = now()
    date = "" * string(year(date)) * "-" * string(month(date)) * "-" * string(day(date)) * "-time-" * string(hour(date)) * "-" * string(minute(date)) * "-" * string(second(date))
    mkdir(date)
    mkdir( date * "/data")
    mkdir( date * "/images")

    print("Saving original\n")
    save( date * "/data/original.jld", "data", t)

    print("Startifying original.\n")
    @time st, matrices, derivation, singularValues, singularVectors  = stratify(t,10)
    print("Saving original stratification.\n" )
    save( date * "/data/original-strat.jld", "data", st)
    save( date * "/data/original-strat-singularvalues.jld", "data", singularValues)

    print( "Randomizing original.\n")
    @time rt = tensorRandomize(t, rounds)
    print( "Saving randomized version.\n")
    save( date * "/data/randomized.jld", "data", rt)

    print( "Stratifying randomized version.\n")
    @time srt, matrices, derivation, singularValues, singularVectors= stratify(rt,10)
    print( "Saving stratification of randomized.\n")
    save( date * "/data/randomized-start.jld", "data", srt)
    save( date * "/data/randomized-strat-singularvalues.jld", "data", singularValues)

    print( "Generating images\n")
    save3D( date * "/images/plot-org.ply", t )
    save3D( date * "/images/plot-org-recons.ply", st)
    save3D( date * "/images/plot-rand.ply", rt)
    save3D( date * "/images/plot-rand-recons.ply", srt)

	return true
end


function curvificationTest(t,rounds)
    date = now()
    date = "" * string(year(date)) * "-" * string(month(date)) * "-" * string(day(date)) * "-time-" * string(hour(date)) * "-" * string(minute(date)) * "-" * string(second(date))
    mkdir(date)
    mkdir( date * "/data")
    mkdir( date * "/images")

    print("Saving original\n")
    save( date * "/data/original.jld", "data", t)

    print("Curvifying original.\n")
    @time st, matrices, derivation, singularValues, singularVectors  = curvify(t,10)
    print("Saving original curvification.\n" )
    save( date * "/data/original-strat.jld", "data", st)
    save( date * "/data/original-strat-singularvalues.jld", "data", singularValues)

    print( "Randomizing original.\n")
    @time rt = tensorRandomize(t, rounds)
    print( "Saving randomized version.\n")
    save( date * "/data/randomized.jld", "data", rt)

    print( "Curvifying randomized version.\n")
    @time srt, matrices, derivation, singularValues, singularVectors = curvify(rt,10)
    print( "Saving curvification of randomized.\n")
    save( date * "/data/randomized-start.jld", "data", srt)
    save( date * "/data/randomized-strat-singularvalues.jld", "data", singularValues)

    print( "Generating images\n")
    save3D( date * "/images/plot-org.ply", t )
    save3D( date * "/images/plot-org-recons.ply", st)
    save3D( date * "/images/plot-rand.ply", rt)
    save3D( date * "/images/plot-rand-recons.ply", srt)

	return true
end




