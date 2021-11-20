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

using Pkg
Pkg.add("JLD")
using JLD

include("tensorRandomize.jl")
include("tensor3D.jl")

const VERBOSE = false

# take a matrix of eigenvectors and break it into "real eigenvectors" 
# some mess about normalization....

function realEigenVectors(m)
	n = real.(m)
	nn =real.(m)

	u = real.(m[:,1])
	nn[:,1] = (1/norm(u))*u 
	
	for i = 2: size(m)[2]
		if norm(n[:,i] - n[:,i-1]) > 0.001
			u = real.(m[:,i])
		else
			u = imag.(m[:,i])
		end
		nn[:,i] = (1/norm(u))*u 
	end
	return nn
end



###############################################################
# retunrs new tensor, trnasformations, derivation as 3 matrices, all singular value, all singular vectors
# M is a matrix used to compute derivarions
# offset which singular value to use as to build eigenvalues/eigenvectors
# toprint debugging info how many singular values to print

function transofromTensorByDerivation(t,M,offset,toprint)
    a = size(t)[1]; b = size(t)[2]; c = size(t)[3]
    print( "\tComputing SVD.  This may take a while...")
    @times u,s,v = svd(M)
    print( "\tFinal singular values for the system ...\n\t\t" )
	for j= 0:toprint
		print( string(round( s[(a^2+b^2+c^2)-j], digits=4) )* ",\t" )	
	end 
	print("\n")
	x, y, z = inflateToTripleOfMatrices(u[:,(a^2+b^2+c^2)-offset], a,b,c)
	xvals,xvecs = eigen(x)
	yvals,yvecs = eigen(y)
	zvals,zvecs = eigen(z)
    if VERBOSE
        print("\tEignevalues Xmatrix\n\t\t")
        print(xvals)
        print("\n\t Eignevalues Ymatrix\n\t\t")
        print(yvals)
        print("\n\t Eignevalues Zmatrix\n\t\t")
        print(zvals)
        print("\n")
    end

	realxvecs= realEigenVectors(xvecs)
	realyvecs= realEigenVectors(yvecs)
	realzvecs= realEigenVectors(zvecs)

	t2 = actFirst(t,realxvecs)
	t2 = actSecond(t2,realyvecs)
	t2 = actThird(t2,realzvecs)

#	return t2, [xvecs, yvecs, zvecs], [x,y,z], s, u
	return t2, [realxvecs, realyvecs, realzvecs], [x,y,z], s, u
end 


function stratify(t,toprint)
	t2, mats, der, s, u = transofromTensorByDerivation(t, buildDerivationMatrix(t) , 2,toprint)
	return t2, mats, der, s, u 
end 



function curvify(t,toprint)
	t2, mats, der, s, u = transofromTensorByDerivation(t, buildCentroidMatrix(t) , 1,toprint)
	return t2, mats, der, s, u 
end 


###############################################################
# test for stratification

function stratificationTest(t,rounds,filename,ratio)
    date = replace(string(now()), ':' => '.')
#    date = "" * string(year(date)) * "-" * string(month(date)) * "-" * string(day(date)) * "-time-" * string(hour(date)) * "-" * string(minute(date)) * "-" * string(second(date))

    print("Working on " * filename * "    dir: " * date *"\n")

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
    rt,rm = tensorRandomize(t, rounds)
    print( "Saving randomized version.\n")
    save( date * "/data/randomized.jld", "data", rt)

    print( "Stratifying randomized version.\n")
    @time srt, matrices, derivation, singularValues, singularVectors= stratify(rt,10)
    print( "Saving stratification of randomized.\n")
    save( date * "/data/randomized-start.jld", "data", srt)
    save( date * "/data/randomized-strat-singularvalues.jld", "data", singularValues)
	
    print( "Product of the X matrices\n")
	display( round.(rm[1]*matrices[1], digits=3) )
    print( "\n")

    print( "Product of the Y matrices\n")
	display( round.(rm[2]*matrices[2], digits=3) )
    print( "\n")

    print( "Product of the Z matrices\n")
	display( round.(rm[3]*matrices[3], digits=3) )
    print( "\n")


    print( "Generating images\n")
    save3D(t,   date * "/images/" * filename * "-org.ply", date * "/images/" * filename * "-org.dat"  ,ratio,1)
    save3D(st,  date * "/images/" * filename * "-org-recons.ply", date * "/images/" * filename * "-org-recons.dat",ratio,2)
    save3D(rt,  date * "/images/" * filename * "-rand.ply", date * "/images/" * filename * "-rand.dat",ratio,1)
    save3D(rt,  date * "/images/" * filename * "-rand10.ply", date * "/images/" * filename * "-rand10.dat",ratio/10,1)
    save3D(rt,  date * "/images/" * filename * "-rand100.ply", date * "/images/" * filename * "-rand100.dat",ratio/100,1)
    save3D(srt, date * "/images/" * filename * "-rand-recons.ply", date * "/images/" * filename * "-rand-recons.dat",ratio,2)

	return true
end


#changes names
function curvificationTest(t,rounds,filename,ratio)
    date = replace(string(now()), ':' => '.')
#    date = "" * string(year(date)) * "-" * string(month(date)) * "-" * string(day(date)) * "-time-" * string(hour(date)) * "-" * string(minute(date)) * "-" * string(second(date))

    print("Working on " * filename * "    dir: " * date *"\n")

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
    rt, rm = tensorRandomize(t, rounds)
    print( "Saving randomized version.\n")
    save( date * "/data/randomized.jld", "data", rt)

    print( "Curvifying randomized version.\n")
    @time srt, matrices, derivation, singularValues, singularVectors = curvify(rt,10)
    print( "Saving curvification of randomized.\n")
    save( date * "/data/randomized-start.jld", "data", srt)
    save( date * "/data/randomized-strat-singularvalues.jld", "data", singularValues)

    print( "Product of the X matrices\n")
	display( round.(rm[1]*matrices[1], digits=3) )
    print( "\n")

    print( "Product of the Y matrices\n")
	display( round.(rm[2]*matrices[2], digits=3) )
    print( "\n")

    print( "Product of the Z matrices\n")
	display( round.(rm[3]*matrices[3], digits=3) )
    print( "\n")


    print( "Generating images\n")
    save3D(t,   date * "/images/" * filename * "-org.ply", date * "/images/" * filename * "-org.dat",ratio,1  )
    save3D(st,  date * "/images/" * filename * "-org-recons.ply", date * "/images/" * filename * "-org-recons.dat",ratio,2)
    save3D(rt,  date * "/images/" * filename * "-rand.ply", date * "/images/" * filename * "-rand.dat",ratio,1)
    save3D(rt,  date * "/images/" * filename * "-rand10.ply", date * "/images/" * filename * "-rand10.dat",ratio/10,1)
    save3D(rt,  date * "/images/" * filename * "-rand100.ply", date * "/images/" * filename * "-rand100.dat",ratio/100,1)
    save3D(srt, date * "/images/" * filename * "-rand-recons.ply", date * "/images/" * filename * "-rand-recons.dat",ratio,2)

	return true
end




