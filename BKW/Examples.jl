

#############################################
## Examples.
#############################################
using SparseArrays

## Flat Genus 2 Indecomposable
function FlatGenus2(F,d)
    t = zeros(F, (2*d+1,2*d+1,2))
    for i = 1:d
        t[i,d+i,1] = 1;   t[d+i,i,1] = -1
        t[i,d+i+1,2] = 1; t[d+i+1,i,2] = -1
    end
    return t
end 

# Return a tensor with sparse randome values
function sprandten(F, dims, density)
    return reshape( sprand(F,prod(dims), density), dims)
end


## ---- Martin's stratificaton construction ----
## ---- this is for in valancy 3, everythign can be generlized to higher dimension
## ---- the input are 3 functions: xscale, yscale, zscale
## ---- it is a good idea these functions to be increasing, and relatively smooth

function genNoise(epsilon)
  return epsilon * rand(-1000:1000) * 0.0001;
end 

# x function
function xscale(i)
  return (30+i)^2
end 

# y function
function yscale(i)
  return (40+i)^2
end 

# z function
function zscale(i)
  return (50+i)^2
end

# scale the functions to be between -1 and 1
function xnorm(xsize)
  xdif = abs(xscale(xsize) - xscale(1))/2
  xavg = (xscale(xsize) + xscale(1)) /2 
  x = [xsize,xavg,xdif]
  return x;
end 

function ynorm(ysize)
  ydif = abs(yscale(ysize) - yscale(1))/2
  yavg = (yscale(ysize) + yscale(1)) /2 
  y = [ysize,yavg,ydif]
  return y;
end 

function znorm(zsize)
  zdif = abs(zscale(zsize) - zscale(1))/2
  zavg = (zscale(zsize) + zscale(1)) /2 
  z = [zsize,zavg,zdif]
  return z;
end 



# computes the equation of the surface
function surface_eq(x,y,z, i,j,k)
  xn =  x[1]*(xscale(i) - x[2])/x[3]
  yn =  y[1]*(yscale(j) - y[2])/y[3]
  zn =  z[1]*(yscale(k) - z[2])/z[3]  
  return abs(xn + yn + zn); 
end 

# computes the equation of the curve
function curve_eq(x,y,z, i,j,k)
  xn =  x[1]*(xscale(i) - x[2])/x[3]
  yn =  y[1]*(yscale(j) - y[2])/y[3]
  zn =  z[1]*(yscale(k) - z[2])/z[3]  
  return abs(xn - yn) + abs( yn- zn) + abs(yn- zn); 
end 


# generate entries which are random numbers with norm ~ exp( - C (surface_eq)^2)
function surface_entry(x,y,z, width,i,j,k, amp)
	s = 2* surface_eq(x,y,z, i,j,k)/width
	r = 0
	# no need to randomly generate tiny numbers
	if abs(s) < 5
		r = genNoise(exp( -s*s ) );
	end
	return amp*r;
end

# generate entries which are random numbers with norm ~ exp( - C (curve_eq)^2)
function surface_entry(x,y,z, width,i,j,k, amp)
	s = 2* curve_eq(x,y,z, i,j,k)/width
	r = 0
	# no need to randomly generate tiny numbers
	if abs(s) < 5
		r = genNoise(exp( -s*s ) );
	end
	return amp*r;
end

# constructs a tensor where the support is close a surface
# the entries are gaussian with parameter the distance to the surface 
# width is measures the tichness of the support
# amp measures the size of the entries

# this tensor show have 3 dimensional devivation algebra
# the non-trivial derivation should consists of almost diagonal matrices, where the eigenvalues/diagonal entries are some normalization of the functions xscale....

function surfaceTensor(xsize, ysize, zsize, width, amp)
 #scale the functions
  x = xnorm(xsize)
  y = ynorm(ysize)
  z = znorm(zsize)
  # builds the tensor
  t = zeros(Float32, (xsize,ysize,zsize))
  for i = 1:xsize
    for j = 1:ysize 
      for k = 1:zsize
        t[i,j,k] = surface_entry(x,y,z,width,i,j,k,amp)
      end 
    end 
  end 
  return t
end


# constructs a tensor where the support is close a curve
# the entries are gaussian with parameter the distance to the curve 
# width is measures the tichness of the support
# amp measures the size of the entries

# this tensor show have 2 dimensional centroid
# the non-trivial centroid should consists of almost diagonal matrices, where the eigenvalues/diagonal entries are some normalization of the functions xscale....

# this tensor show have a relatively large derivation algebra -- the dimension should be around xsize
# the derivations should consists of almost diagonal matrices

function curveTensor(xsize, ysize, zsize, width, amp)
 #scale the functions
  x = xnorm(xsize)
  y = ynorm(ysize)
  z = znorm(zsize)
  # builds the tensor
  t = zeros(Float32, (xsize,ysize,zsize))
  for i = 1:xsize
    for j = 1:ysize 
      for k = 1:zsize
        t[i,j,k] = curve_entry(x,y,z,width,i,j,k,amp)
      end 
    end 
  end 
  return t
end

