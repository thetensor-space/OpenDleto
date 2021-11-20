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


# random number generator with values -1 to 1
function randomNumber()
  return rand(-1000:1000) * 0.0001;
end 


# get info for an array use to noralization
function getNormalizationInfo(x)
	xsize = size(x,1)
	xmin = min(x...)
	xmax = max(x...)
	xavg = (xmin + xmax)/2
	xrange = (xmax-xmin)
    return [xsize, xmin, xmax, xavg, xrange]
end

# normalize element 
# the second argument is the normalization info from the previous function
function normalizeElement(i,n)
	return 2*(i - n[4])/n[5]
end

# computes the equation of the surface
# tests if a point is on the surface 
# nn is a a triple of normalization info for each coordinate
function surfaceEquation(i, nn)
	return abs( normalizeElement(i[1],nn[1]) + normalizeElement(i[2],nn[2])  + normalizeElement(i[3],nn[3]))  
end 

# computes the equation of the curve
# tests if a point is on the curve 
# nn is a a triple of normalization info for each coordinate
function curveEquation(i, nn)
	return abs(normalizeElement(i[1],nn[1])- normalizeElement(i[2],nn[2])) + abs(normalizeElement(i[2],nn[2])- normalizeElement(i[3],nn[3])) + abs(normalizeElement(i[3],nn[3])- normalizeElement(i[1],nn[1]))  
end 


# generate entries which are random numbers with norm ~ exp( - s^2)
function entry(s)
	if abs(s) < 5
		return randomNumber()*  exp( 5 -s*s )
	else
		return 0
	end
end


# construct a tensor supported on a surface
# x,y,z are vectors whose size determine the size of tensor
# the support is restrcted to the "surface" x[i] + y[j] + z[k] =0, however we normalize first to ensure that the surface cuts through the middle of the cube
# width measure the thickness of the surface
function generateSurfaceTensor(x, y, z, width)      
	# normalize the x, y, z
	nx = getNormalizationInfo(x)  
	ny = getNormalizationInfo(y)  
	nz = getNormalizationInfo(z)
	nn = [nx,ny,nz]
	
	# builds the tensor
	t = zeros(Float32, (size(x,1),size(y,1),size(z,1)))
	for i = 1:size(x,1)
		for j = 1:size(y,1) 
			for k = 1:size(z,1)
				s = surfaceEquation([x[i],y[j],z[k]] , nn)*( nx[1] + ny[1] + nz[1]) / width
				t[i,j,k] = entry(s)
			end 
		end 
	end 
	return t
end

# construct a tensor supported on a curve
# x,y,z are vectors whose size determine the size of tensor
# the support is restrcted to the "curve" x[i] = y[j] = z[k], however we normalize first to ensure that the surface cuts through the middle of the cube
# width measure the thickness of the curve
function generateCurveTensor(x, y, z, width)      
	# normalize the x, y, z
	nx = getNormalizationInfo(x)  
	ny = getNormalizationInfo(y)  
	nz = getNormalizationInfo(z)
	nn = [nx,ny,nz]
	
	# builds the tensor
	t = zeros(Float32, (size(x,1),size(y,1),size(z,1)))
	for i = 1:size(x,1)
		for j = 1:size(y,1) 
			for k = 1:size(z,1)
				s = curveEquation([x[i],y[j],z[k]],nn)*( nx[1] + ny[1] + nz[1]) / (10* width)
				t[i,j,k] = entry(s)
			end 
		end 
	end 
	return t
end

