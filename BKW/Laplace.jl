## ---- Martin's first construction ----

function addNoise(a, epsilon)
  return a + epsilon * rand(-1000:1000) * 0.0001;
end 

# x function for the surface
function xscale(i)
  return (30+i)^2
end 

# y function for the surface
function yscale(i)
  return (40+i)^2
end 

# z function for the surface
function zscale(i)
  return (50+i)^2
end

# computes the equation of the surface
function surface_eq(x,y,z, i,j,k)
 return x[1]*(xscale(i) - x[2])/x[3] + y[1]*(yscale(j) - y[2])/y[3] + z[1]*(zscale(k) - z[2])/z[3]; 
end 


# generate entries which are random numbers with norm ~ exp( - C (surface_eq)^2)
function entry(x,y,z, width,i,j,k)
	s = 2* surface_eq(x,y,z, i,j,k)/width
	r = 0
	# no need to randomly generate tiny numbers
	if abs(s) < 5
		r = addNoise(0, exp( -s*s ) );
	end
	return 100*r;
end

function MartiniT(xsize, ysize, zsize, width)      
  # normalize the x, y, z functions 
  xnorm = abs(xscale(xsize) - xscale(1))
  xavg = (xscale(xsize) + xscale(1)) /2 
  x = [xsize,xavg,xnorm]

  ynorm = abs(yscale(ysize) - yscale(1))
  yavg = (yscale(ysize) + yscale(1)) /2 
  y = [ysize,yavg,ynorm]

  znorm = abs(zscale(zsize) - zscale(1))
  zavg = (zscale(zsize) + zscale(1)) /2 
  z = [zsize,zavg,znorm]
  # builds the tensor
  t = zeros(Complex{Float32}, (xsize,ysize,zsize))
  for i = 1:xsize
    for j = 1:ysize 
      for k = 1:zsize
        t[i,j,k] = entry(x,y,z,width,i,j,k)
      end 
    end 
  end 
  return t
end
