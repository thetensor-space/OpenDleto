using LinearAlgebra
using SparseArrays
using Dates
using Random


# random
function random()
  return rand(-1000:1000) * 0.0001;
end 


# get info for an array
function getnormalization(x)
	xsize = size(x,1)
	xmin = min(x...)
	xmax = max(x...)
	xavg = (xmin + xmax)/2
	xrange = (xmax-xmin)
    return [xsize, xmin, xmax, xavg, xrange]
end

# normalize element 
function normal(i,n)
	return 2*(i - n[4])/n[5]
end

# computes the equation of the surface
function surface_eq(i,j,k, nn)
	return abs( normal(i,nn[1]) + normal(j,nn[2])  + normal(k,nn[3]))  
end 

# computes the equation of the curve
function curve_eq(i,j,k, nn)
	return abs(normal(i,nn[1])- normal(j,nn[2])) + abs(normal(j,nn[2])- normal(k,nn[3])) + abs(normal(k,nn[3])- normal(i,nn[1]))  
end 


# generate entries which are random numbers with norm ~ exp( - s^2)
function entry(s)
	if abs(s) < 5
		return random()*  exp( 5 -s*s )
	else
		return 0
	end
end


function surfaceT(x, y, z, width)      
	# normalize the x, y, z
	nx = getnormalization(x)  
	ny = getnormalization(y)  
	nz = getnormalization(z)
	nn = [nx,ny,nz]
	
	# builds the tensor
	t = zeros(Float32, (size(x,1),size(y,1),size(z,1)))
	for i = 1:size(x,1)
		for j = 1:size(y,1) 
			for k = 1:size(z,1)
				s = surface_eq(x[i],y[j],z[k],nn)*( nx[1] + ny[1] + nz[1]) / width
				t[i,j,k] = entry(s)
			end 
		end 
	end 
	return t
end

function curveT(x, y, z, width)      
	# normalize the x, y, z
	nx = getnormalization(x)  
	ny = getnormalization(y)  
	nz = getnormalization(z)
	nn = [nx,ny,nz]
	
	# builds the tensor
	t = zeros(Float32, (size(x,1),size(y,1),size(z,1)))
	for i = 1:size(x,1)
		for j = 1:size(y,1) 
			for k = 1:size(z,1)
				s = curve_eq(x[i],y[j],z[k],nn)*( nx[1] + ny[1] + nz[1]) / (10* width)
				t[i,j,k] = entry(s)
			end 
		end 
	end 
	return t
end

