include("TensorStratify.jl")
include("HeatEqTensor.jl")



function downsizeTensor(t,n)
	nx = n[1]
	ny = n[2]
	nz = n[3]
	rx = trunc(Int,size(t)[1]/nx) - 1
	ry = trunc(Int,size(t)[2]/ny) - 1
	rz = trunc(Int,size(t)[3]/nz) - 1
	res = zeros((rx,ry,rz))

	for i = 1:rx
		for j = 1:ry
			for k = 1:rz
				res[i,j,k] = t[i*nx + 1, j*ny + 1, k*nz + 1]
			end
		end
	end
	return res
end

function computeFourierCoef(u,a,b)
	c = 0
	n = size(u)[1] -1
	m = size(u)[2] -1
	for i = 1:n
		for j = 1:m
			c += u[i,j] * sin((i-1) *pi* a/ n) * sin((j-1) *pi* b/ m)
		end
	end
	return 4* c / (n * m)
end

function removeFourierCoef(u,a,b)
	c = computeFourierCoef(u,a,b)
	n = size(u)[1] -1
	m = size(u)[2] -1
	for i = 1:n
		for j = 1:m
			u[i,j] -= c * sin((i-1) *pi* a/ n) * sin((j-1) *pi* b/ m)
		end
	end
	return u
end

function matrixFourierCoef(u,a,b)
	m= zeros((a,b))
	for i= 1:a
		for j=1:b
			m[i,j] = computeFourierCoef(u,i,j)
		end
	end
	return m
end

function generateRandomFourierCoef(a,b,lmax,lmin)
	m= zeros((a,b))
	for i= 1:a
		for j=1:b
			if ((i*i + j*j) < lmin) || ((i*i + j*j) > lmax)
				m[i,j] = randn(Float32)
			end
		end
	end
	return m
end

function generateHeatEqTensor(m,a,b,c,r)
	t= zeros((a,b,c))
	for i= 1:a
		for j=1:b
			for k= 1:c
				for p = 1:size(m)[1]
					for q = 1:size(m)[2]
						t[i,j,k] += m[p,q]*sin(p*i *pi /(a+1)) * sin(q*j *pi /(b+1)) * exp( - (p^2 + q^2)*k *r)  
					end
				end
			end
		end
	end
	return t
end


#xsize=311
#ysize=311
xsize=11
ysize=21


function base_func(x,y)
	r= randn(Float32)
	return 10000*r
end 


initial_func(x,y) = x*y*(1-x)*(1-y)*base_func(x,y)


initial =  zeros((xsize,ysize))

for j=1:ysize
	for i = 1:xsize 
		initial[i,j] = initial_func((i-1)/(xsize-1),(j-1)/(ysize-1)) 
	end
end

#display(initial)

#display(matrixFourierCoef(initial,10,10))
for p = 1:30
	for q = 1:30
		if ((p*p + q*q) < 90) || ((p*p + q*q) > 300) 
			global initial = removeFourierCoef(initial,p,q)
		end
	end
end


#display(initial)
#display(matrixFourierCoef(initial, 10,10))

#heatEqSolution =downsizeTensor(solveHeat(initial, 0.0000005, 0.00011),[15,15,10])
#display(heatEqSolution)

#@timed stratificationTest(heatEqSolution,50,"HeatEq",1000)


FC = generateRandomFourierCoef(40,40,1000,1050)
t = generateHeatEqTensor(FC,30,30,7,0.00005)
@timed stratify(t,50)
#@timed stratificationTest(t,50,"HeatEq",1000)
true