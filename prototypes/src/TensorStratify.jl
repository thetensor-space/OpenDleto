########################################################################
#  CC-BY 2021 
#  Peter A Brooksbank
#  Martin Kassabov
#  James B. Wilson
#
#    Distributed under MIT License.
########################################################################


include("tensorRandomize.jl")
include("Tensor3D.jl")



# take a matrix of eigenvectors and break it into "real eigenvectors" 
# some mess about normalization....

function realEigenVectors(m)
	n  = real.(m)
	nn = real.(m)
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

function transofromTensorByDerivation(t,M,offset,inflatefunc;verbose=false,toprint=10,sizecorrection=0)
    a = size(t)[1] 
	b = size(t)[2] 
	c = size(t)[3]
	timings=Dict{String,Any}()
    print( "\tComputing SVD.  This may take a while...")
    svdtime = @timed (svd(M))
	u,s,v = svdtime.value
	timings["svd"] = svdtime.time
	
    print( "\tFinal singular values for the system ...\n\t\t" )
	for j= 0:toprint
		print( string(round( s[(a^2+b^2+c^2)-sizecorrection-j], digits=4) )* ",\t" )	
	end 
	print("\n")
	x, y, z = inflatefunc(u[:,(a^2+b^2+c^2)-sizecorrection-offset], a,b,c)
	eigentime = @timed([eigen(x),eigen(y),eigen(z)])	
	xvals,xvecs = eigentime.value[1] 
	yvals,yvecs = eigentime.value[2]
	zvals,zvecs = eigentime.value[3]	
	timings["eigenvalues"] = eigentime.time
    if verbose
        print("\tEignevalues Xmatrix\n\t\t")
        print(xvals)
        print("\n\t Eignevalues Ymatrix\n\t\t")
        print(yvals)
        print("\n\t Eignevalues Zmatrix\n\t\t")
        print(zvals)
        print("\n")
    end

	realeigentime = @timed([realEigenVectors(xvecs),realEigenVectors(yvecs),realEigenVectors(zvecs)])
	realxvecs = realeigentime.value[1]
	realyvecs = realeigentime.value[2]
	realzvecs = realeigentime.value[3]
	timings["realeigenvalues"] = realeigentime.time
	
	actiontime = @timed (t2 = actAll(t, [realxvecs, realyvecs, realzvecs]); nothing)
	timings["action"] =actiontime.time
	return t2, Dict( "matrices" => [realxvecs, realyvecs, realzvecs], "derivation" => [x,y,z], "singularValues" => s, "singularVectors" => u, "timings" => timings) 
end 


function stratify(t;verbose=false,toprint=10)
	dermatrixtime = @timed(buildDerivationMatrix(t))
	M = dermatrixtime.value
	t2, D = transofromTensorByDerivation(t, M, 2, inflateToTripleOfMatrices;toprint,verbose)
	D["timings"]["derviation"] = dermatrixtime.time
	return t2, D 
end 



function curvify(t;verbose=false,toprint=10)
	dermatrixtime = @timed(buildCentroidMatrix(t)) 
	M = dermatrixtime.value
	t2, D = transofromTensorByDerivation(t, M, 1, inflateToTripleOfMatrices; toprint,verbose)
	D["timings"]["centroid"] = dermatrixtime.time
	return t2, D 
end 



function stratify12face(t;verbose=false,toprint=10)
	dermatrixtime = @timed(buildAdjoint12Matrix(t))
	M = dermatrixtime.value
	t2, D = transofromTensorByDerivation(t, M, 1, inflateToTripleOfMatrices12;toprint,verbose,sizecorrection=size(t)[3]*size(t)[3])
	D["timings"]["adjoint12"] = dermatrixtime.time
	return t2, D 
end 

function stratify13face(t;verbose=false,toprint=10)
	dermatrixtime = @timed(buildAdjoint13Matrix(t))
	M = dermatrixtime.value
	t2, D = transofromTensorByDerivation(t, M, 1, inflateToTripleOfMatrices13;toprint,verbose,sizecorrection=size(t)[2]*size(t)[2])
	D["timings"]["adjoint13"] = dermatrixtime.time
	return t2, D 
end 

function stratify23face(t;verbose=false,toprint=10)
	dermatrixtime = @timed(buildAdjoint23Matrix(t))
	M = dermatrixtime.value
	t2, D = transofromTensorByDerivation(t, M, 1, inflateToTripleOfMatrices23;toprint,verbose,sizecorrection=size(t)[1]*size(t)[1])
	D["timings"]["adjoint23"] = dermatrixtime.time
	return t2, D 
end 


nothing

