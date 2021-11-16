include("tensorStratify.jl")
include("tensorGenerator.jl")

## testting for trtratification and curvification

square(x) = x^2
cube(x)=x^3


function runBlockTests()
	#Blocky Tests

	blockvectorX = [1,1,1,1,1,1,1, 2,2,2,2, 3,3,3, 4,4, 5,5]
	blockvectorY = [1,1,1,1, 2,2,2,2,2,2, 3,3,3,3,3, 4,4,4,4, 5,5,5]
	blockvectorZ = [1,1,1, 2,2,2,2, 3,3,3,3,3,3, 4,4,4,4,4,4, 5,5,5,5]
	
	#simple block diagonal tensor
	blockDiagonalTensor = curveT(blockvectorX,blockvectorY,blockvectorZ,0.1)
	curvificationTest(blockDiagonalTensor,20)

	#block diagonal tensor plus noise
	blockDiagonalTensorNoise = curveT(blockvectorX,blockvectorY,blockvectorZ,4)
	curvificationTest(blockDiagonalTensorNoise,20)

	#block planner tensor
	blockPlannarTensor = surfaceT(blockvectorX,blockvectorY,blockvectorZ,0.1)
	stratificationTest(blockPlannarTensor,20)

	#block planner tensor noise
	blockPlannarTensorNoise = surfaceT(blockvectorX,blockvectorY,blockvectorZ,20)
	stratificationTest(blockPlannarTensorNoise,20)
end


function runSmoothTests()

	#smooth tests
	indexX = 10:30
	indexY = 20:30
	indexZ = 15:30

	# diagonal line tensor small noise
	lineTensor = curveT(indexX,indexY,indexZ,1)
	curvificationTest(lineTensor,20)

	# diagonal line tensor big noise
	lineTensorNoise = curveT(indexX,indexY,indexZ, 2)
	curvificationTest(lineTensorNoise,20)


	# plane tensor small noise
	planeTensor = surfaceT(indexX,indexY,indexZ,3)
	stratificationTest(planeTensor,20)

	# plane tensor big noise
	planeTensorNoise = surfaceT(indexX,indexY,indexZ,15)
	stratificationTest(planeTensorNoise,20)

end

function runCurvyTests()
	#smooth tests
	indexX = 10:50
	indexY = 20:50
	indexZ = 15:30
	indexX2= square.(indexX)
	indexX3= cube.(indexX)
	indexY2= square.(indexY)
	indexY3= cube.(indexY)
	indexZ2= square.(indexZ)
	indexZ3= cube.(indexZ)

	# diagonal curve tensor small noise
	curveTensor = curveT(indexX,indexY2,indexZ3,1)
	curvificationTest(curveTensor,20)

	# diagonal curbe tensor big noise
	curveTensorNoise = curveT(indexX,indexY2,indexZ3, 3)
	curvificationTest(curveTensorNoise,20)

	# surface tensor small noise
	surfaceTensor = surfaceT(indexX,indexY2,indexZ3,3)
	stratificationTest(surfaceTensor,20)

	# surface tensor big noise
	surfaceTensorNoise = surfaceT(indexX,indexY2,indexZ3,10)
	stratificationTest(surfaceTensorNoise,20)

end

function runTests()
	runBlockTests()
	runSmoothTests()
	runCurvyTests()
end