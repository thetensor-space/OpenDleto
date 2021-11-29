include("tensorStratify.jl")
include("tensorGenerator.jl")

## testting for trtratification and curvification

square(x) = x^2
cube(x)=x^3
fifth(x)=x^5

function runDiagonalTest()
	m = zeros(30,7)
	for n=5:20

		index= 1:n
		m[n,1] = n

		dTensor = generateCurveTensor(index,index,index,0.1)
		t= @timed curvificationTest(dTensor,50,"DiagonalTensor-" * string(n),1000)
		m[n,2] =t[2]

		dTensorN = generateCurveTensor(index,index,index,1)
		t= @timed curvificationTest(dTensorN,50,"DiagonalTensorNoise-" * string(n),1000)
		m[n,3] =t[2]
		
		dTensorLN = generateCurveTensor(index,index,index,2)
		t= @timed curvificationTest(dTensorLN,50,"DiagonalTensorLargeNoise-" * string(n),1000)
		m[n,4] =t[2]

		sTensor = generateSurfaceTensor(index,index,index,0.1)
		t= @timed stratificationTest(sTensor,50,"SurfaceTensor-" * string(n),1000)
		m[n,5] =t[2]

		sTensorN = generateSurfaceTensor(index,index,index,1)
		t= @timed stratificationTest(sTensorN,50,"SurfaceTensorNoise-" * string(n),1000)
		m[n,6] =t[2]

		sTensorLN = generateSurfaceTensor(index,index,index,2)
		t= @timed stratificationTest(sTensorLN,50,"SurfaceTensorLargeNoise-" * string(n),1000)
		m[n,7] =t[2]

	end
	return m
end

function runBlockTests()
	#Blocky Tests

	blockvectorX = [1,1,1,1,1,1,1, 2,2,2,2, 3,3,3, 4,4, 5,5]
	blockvectorY = [1,1,1,1, 2,2,2,2,2,2, 3,3,3,3,3, 4,4,4,4, 5,5,5]
	blockvectorZ = [1,1,1, 2,2,2,2, 3,3,3,3,3,3, 4,4,4,4,4,4, 5,5,5,5]


#	blockvectorX = [1,1,1,1,1,1,1,2,2,2,2,3,3,3]
#	blockvectorY = [1,1,1,1,2,2,2,2,2,2,3,3,3,3]
#	blockvectorZ = [1,1,1,2,2,2,2,3,3,3,3]
	
	#simple block diagonal tensor
    print( "\n\nBlock Digonal Small Noise...\n" )
	blockDiagonalTensor = generateCurveTensor(blockvectorX,blockvectorY,blockvectorZ,0.5)
	curvificationTest(blockDiagonalTensor,100,"blockDiagonalTensor",1000)

	#block diagonal tensor plus noise
    print( "\n\nBlock Digonal Meduim Noise...\n" )
	blockDiagonalTensorNoise = generateCurveTensor(blockvectorX,blockvectorY,blockvectorZ,3)
	curvificationTest(blockDiagonalTensorNoise,100,"blockDiagonalTensorNoise",1000)

	#block diagonal tensor plus noise
    print( "\n\nBlock Digonal Large Noise...\n" )
	blockDiagonalTensorLargeNoise = generateCurveTensor(blockvectorX,blockvectorY,blockvectorZ,5)
	curvificationTest(blockDiagonalTensorLargeNoise,100,"blockDiagonalTensorLargeNoise",1000)


	#block planner tensor
    print( "\n\nBlock Plannar Small Noise...\n" )
	blockPlannarTensor = generateSurfaceTensor(blockvectorX,blockvectorY,blockvectorZ,1)
	stratificationTest(blockPlannarTensor,100,"blockPlannarTensor",1000)

	#block planner tensor noise
	print( "\n\nBlock Pannar Medium Noise...\n" )
	blockPlannarTensorNoise = generateSurfaceTensor(blockvectorX,blockvectorY,blockvectorZ,20)
	stratificationTest(blockPlannarTensorNoise,100,"blockPlannarTensorNoise",1000)

	#block planner tensor noise
	print( "\n\nBlock Pannar Large Noise...\n" )
	blockPlannarTensorLargeNoise = generateSurfaceTensor(blockvectorX,blockvectorY,blockvectorZ,30)
	stratificationTest(blockPlannarTensorLargeNoise,100,"blockPlannarTensorLargeNoise",1000)
end


function runSmoothTests()

	#smooth tests
	indexX = 1:30
	indexY = 1:25
	indexZ = 1:20

	# diagonal line tensor small noise
    print( "\n\nLine Small Noise...\n" )
	lineTensor = generateCurveTensor(indexX,indexY,indexZ,1)
	curvificationTest(lineTensor,150,"lineTensor",1000)

	# diagonal line tensor big noise
    print( "\n\nLine Medium Noise...\n" )
	lineTensorNoise = generateCurveTensor(indexX,indexY,indexZ, 2)
	curvificationTest(lineTensorNoise,150,"lineTensorNoise",1000)

	# diagonal line tensor big noise
    print( "\n\nLine Large Noise...\n" )
#	lineTensorLargeNoise = generateCurveTensor(indexX,indexY,indexZ, 3)
#	curvificationTest(lineTensorLargeNoise,150,"lineTensorLargeNoise",1000)


	# plane tensor small noise
    print( "\n\nPlane Small Noise...\n" )
	planeTensor = generateSurfaceTensor(indexX,indexY,indexZ,3)
#	stratificationTest(planeTensor,150,"planeTensor",1000)

	# plane tensor big noise
    print( "\n\nPlane Medium Noise...\n" )
#	planeTensorNoise = generateSurfaceTensor(indexX,indexY,indexZ,10)
#	stratificationTest(planeTensorNoise,150,"planeTensorNoise",1000)

	# plane tensor big noise
    print( "\n\nPlane Large Noise...\n" )
#	planeTensorLargeNoise = generateSurfaceTensor(indexX,indexY,indexZ,25)
#	stratificationTest(planeTensorLargeNoise,150,"planeTensorLargeNoise",1000)
end




function runCurvyTests()
	#smooth tests
	indexX = 1:40
	indexY = 6:30
	indexZ = 11:30
	indexX2= square.(indexX)
	indexX3= cube.(indexX)
	indexY2= square.(indexY)
	indexY3= cube.(indexY)
	indexZ2= square.(indexZ)
	indexZ3= cube.(indexZ)

	# diagonal curve tensor small noise
    print( "\n\nCurve Small Noise...\n" )
	curveTensor = generateCurveTensor(indexX,indexY2,indexZ3,2)
	curvificationTest(curveTensor,90,"curveTensor",1000)

	# diagonal curbe tensor big noise
    print( "\n\nCurve Medium Noise...\n" )
#	curveTensorNoise = generateCurveTensor(indexX,indexY2,indexZ3, 3)
#	curvificationTest(curveTensorNoise,90,"curveTensorNoise",100)

	# diagonal curbe tensor big noise
    print( "\n\nCurve Large Noise...\n" )
#	curveTensorLargeNoise = generateCurveTensor(indexX,indexY2,indexZ3, 4)
#	curvificationTest(curveTensorLargeNoise,90,"curveTensorLargeNoise",100)

	# surface tensor small noise
    print( "\n\nSurface Small Noise...\n" )
#	surfaceTensor = generateSurfaceTensor(indexX,indexY2,indexZ3,3)
#	stratificationTest(surfaceTensor,90,"surfaceTensor",100)

	# surface tensor big noise
    print( "\n\nSurface Medium Noise...\n" )
#	surfaceTensorNoise = generateSurfaceTensor(indexX,indexY2,indexZ3,10)
#	stratificationTest(surfaceTensorNoise,90,"surfaceTensorNoise",100)

	# surface tensor big noise
    print( "\n\nSurface Large Noise...\n" )
#	surfaceTensorLargeNoise = generateSurfaceTensor(indexX,indexY2,indexZ3,20)
#	stratificationTest(surfaceTensorLargeNoise,90,"surfaceTensorLargeNoise",100)
end


# produce some cubic curves tensors
function runCubicCurveTests(n=15)
#	n=15

	index = -n:n
	power(x) = x* (abs(x)^ 0.5)
	index3 = power.(index)
#	index3 = cube.(index)
	index5 = fifth.(index)

	cubicCurveTensor = generateCurveTensor(index,index,index3,1.25)
	curvificationTest(cubicCurveTensor,90,"cubicCurveTensor",1000)

#	cubicCurveTensorNoise = generateCurveTensor(index,index,index3,2.5)
#	curvificationTest(cubicCurveTensorNoise,90,"cubicCurveTensorNoise",100)

#	cubicCurveDTensor = generateCurveTensor(index,index3,index3,1.5)
#	curvificationTest(cubicCurveDTensor,90,"cubicDCurveTensor",100)

#	cubicCurveDTensorNoise = generateCurveTensor(index,index3,index3,2.5)
#	curvificationTest(cubicCurveDTensorNoise,90,"cubicDCurveTensorNoise",100)

#	fifthCurveTensor = generateCurveTensor(index,index3,index5,1.5)
#	curvificationTest(fifthCurveTensor,90,"fifthCurveTensor",100)

#	fifthCurveTensorNoise = generateCurveTensor(index,index3,index5,2.5)
#	curvificationTest(fifthCurveTensorNoise,90,"fifthCurveTensorNoise",100)

 end


# produce some cubic curves tensors
function runTwistedCubicTests(n,noise,ratio)
#	n=15

# for larger exponent use more noise....
#	power(x) = x* (abs(x)^ 1)
	power(x) = x* (abs(x)^0.4)
	indexL = (-2*n):0
	indexM = -n:n
#	indexM = 0:(2*n)
	indexR = 0:(2*n)

	cubicCurveTensor = generateCurveTensor(power.(indexL),power.(indexM),power.(indexR),noise)
	curvificationTest(cubicCurveTensor,90,"twistedCubicCurveTensor",ratio)

#	cubicCurveTensorNoise = generateCurveTensor(power.(indexL),power.(indexM),power.(indexR),2.5)
#	curvificationTest(cubicCurveTensorNoise,90,"twistedCubicCurveTensorNoise",1000)

#	cubicSurfaceTensor = generateSurfaceTensor(power.(indexL),power.(indexM),power.(indexR),1)
#	cubicSurfaceTensor = generateSurfaceTensor(power.(indexL),power.(indexM),power.(indexR),3)
#	stratificationTest(cubicSurfaceTensor,90,"twistedCubicSurfaceTensor",100)

#	cubicSurfaceTensorNoise = generateSurfaceTensor(power.(indexL),power.(indexM),power.(indexR),3)
#	cubicSurfaceTensorNoise = generateSurfaceTensor(power.(indexL),power.(indexM),power.(indexR),5)
#	stratificationTest(cubicSurfaceTensorNoise,90,"twistedCubicSurfaceTensorNoise",100)

 end



# produce some cubic surfaces tensors
function runCubicSurfaceTests(n=20)
#	n=20

	index = -n:n
	index3 = cube.(index)
	index5 = fifth.(index)


	cubicSurfaceTensor = generateSurfaceTensor(index,index,index3,2)
	stratificationTest(cubicSurfaceTensor,100,"cubicSurfaceTensor",1000)

	cubicSurfaceTensorNoise = generateSurfaceTensor(index,index,index3,7)
	stratificationTest(cubicSurfaceTensorNoise,100,"cubicSurfaceTensorNoise",1000)

	cubicSurfaceDTensor = generateSurfaceTensor(index,index3,index3,2)
	stratificationTest(cubicSurfaceDTensor,100,"cubicSurfaceDTensor",1000)

	cubicSurfaceDTensorNoise = generateSurfaceTensor(index,index3,index3,7)
	stratificationTest(cubicSurfaceDTensorNoise,100,"cubicSurfaceDTensorNoise",1000)

	cubicSurfaceTTensor = generateSurfaceTensor(index3,index3,index3,2)
	stratificationTest(cubicSurfaceTTensor,100,"cubicSurfaceTTensor",1000)

	cubicSurfaceTTensorNoise = generateSurfaceTensor(index3,index3,index3,7)
	stratificationTest(cubicSurfaceTTensorNoise,100,"cubicSurfaceTTensorNoise",1000)

	fifthSurfaceTensor = generateSurfaceTensor(index,index3,index5,2)
	stratificationTest(fifthSurfaceTensor,100,"fifthSurfaceTensor",1000)

	fifthSurfaceTensorNoise = generateSurfaceTensor(index,index3,index5,7)
	stratificationTest(fifthSurfaceTensorNoise,100,"fifthSurfaceTensorNoise",1000)

	fifthSurfaceDTensor = generateSurfaceTensor(index3,index3,index5,2)
	stratificationTest(fifthSurfaceDTensor,100,"fifthSurfaceDTensor",1000)

	fifthSurfaceDTensorNoise = generateSurfaceTensor(index3,index3,index5,7)
	stratificationTest(fifthSurfaceDTensorNoise,100,"fifthSurfaceDTensorNoise",1000)
end

# produce some cubic surfaces tensors
function runSinSurfaceTests(n=15)
#	n=15

	wave1(x) = 10*x - 15*sin(x/3)
	wave2(x) = 20*x - 19*sin(x/3)

	index = -n:n
	index3 = wave1.(index)
	index5 = wave2.(index)


	cubicSurfaceDTensor = generateSurfaceTensor(index,index3,index3,2)
	stratificationTest(cubicSurfaceDTensor,100,"cubicSurfaceDTensor",1000)

	cubicSurfaceDTensorNoise = generateSurfaceTensor(index,index3,index3,7)
	stratificationTest(cubicSurfaceDTensorNoise,100,"cubicSurfaceDTensorNoise",1000)

	cubicSurfaceTTensor = generateSurfaceTensor(index3,index3,index3,2)
	stratificationTest(cubicSurfaceTTensor,100,"cubicSurfaceTTensor",1000)

	cubicSurfaceTTensorNoise = generateSurfaceTensor(index3,index3,index3,7)
	stratificationTest(cubicSurfaceTTensorNoise,100,"cubicSurfaceTTensorNoise",1000)

	fifthSurfaceTensor = generateSurfaceTensor(index,index3,index5,2)
	stratificationTest(fifthSurfaceTensor,100,"fifthSurfaceTensor",1000)

	fifthSurfaceTensorNoise = generateSurfaceTensor(index,index3,index5,7)
	stratificationTest(fifthSurfaceTensorNoise,100,"fifthSurfaceTensorNoise",1000)

	fifthSurfaceDTensor = generateSurfaceTensor(index3,index3,index5,2)
	stratificationTest(fifthSurfaceDTensor,100,"fifthSurfaceDTensor",1000)

	fifthSurfaceDTensorNoise = generateSurfaceTensor(index3,index3,index5,7)
	stratificationTest(fifthSurfaceDTensorNoise,100,"fifthSurfaceDTensorNoise",1000)
end


function runTests(n=20)
	runBlockTests(n)
	runSmoothTests(n)
	runCurvyTests(n)
end