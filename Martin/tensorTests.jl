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


#	blockvectorX = [1,1,1,1,1,1,1, 2,2,2,2, 3,3,3]
#	blockvectorY = [1,1,1,1, 2,2,2,2,2,2, 3,3,3,3]
#	blockvectorZ = [1,1,1, 2,2,2,2, 3,3,3,3]
	
	#simple block diagonal tensor
    print( "\n\nBlock Digonal Small Noise...\n" )
	blockDiagonalTensor = generateCurveTensor(blockvectorX,blockvectorY,blockvectorZ,0.5)
	curvificationTest(blockDiagonalTensor,50)

	#block diagonal tensor plus noise
    print( "\n\nBlock Digonal Meduim Noise...\n" )
	blockDiagonalTensorNoise = generateCurveTensor(blockvectorX,blockvectorY,blockvectorZ,3)
	curvificationTest(blockDiagonalTensorNoise,50)

	#block diagonal tensor plus noise
    print( "\n\nBlock Digonal Large Noise...\n" )
	blockDiagonalTensorLargeNoise = generateCurveTensor(blockvectorX,blockvectorY,blockvectorZ,5)
	curvificationTest(blockDiagonalTensorLargeNoise,50)


	#block planner tensor
    print( "\n\nBlock Plannar Small Noise...\n" )
	blockPlannarTensor = generateSurfaceTensor(blockvectorX,blockvectorY,blockvectorZ,1)
	stratificationTest(blockPlannarTensor,50)

	#block planner tensor noise
	print( "\n\nBlock Pannar Medium Noise...\n" )
	blockPlannarTensorNoise = generateSurfaceTensor(blockvectorX,blockvectorY,blockvectorZ,20)
	stratificationTest(blockPlannarTensorNoise,50)

	#block planner tensor noise
	print( "\n\nBlock Pannar Large Noise...\n" )
	blockPlannarTensorLargeNoise = generateSurfaceTensor(blockvectorX,blockvectorY,blockvectorZ,30)
	stratificationTest(blockPlannarTensorLargeNoise,50)
end


function runSmoothTests()

	#smooth tests
	indexX = 1:30
	indexY = 1:25
	indexZ = 1:20

	# diagonal line tensor small noise
    print( "\n\nLine Small Noise...\n" )
	lineTensor = generateCurveTensor(indexX,indexY,indexZ,1)
	curvificationTest(lineTensor,70)

	# diagonal line tensor big noise
    print( "\n\nLine Medium Noise...\n" )
	lineTensorNoise = generateCurveTensor(indexX,indexY,indexZ, 2)
	curvificationTest(lineTensorNoise,70)

	# diagonal line tensor big noise
    print( "\n\nLine Large Noise...\n" )
	lineTensorLargeNoise = generateCurveTensor(indexX,indexY,indexZ, 3)
	curvificationTest(lineTensorLargeNoise,70)


	# plane tensor small noise
    print( "\n\nPlane Small Noise...\n" )
	planeTensor = generateSurfaceTensor(indexX,indexY,indexZ,3)
	stratificationTest(planeTensor,70)

	# plane tensor big noise
    print( "\n\nPlane Medium Noise...\n" )
	planeTensorNoise = generateSurfaceTensor(indexX,indexY,indexZ,10)
	stratificationTest(planeTensorNoise,70)

	# plane tensor big noise
    print( "\n\nPlane Large Noise...\n" )
	planeTensorLargeNoise = generateSurfaceTensor(indexX,indexY,indexZ,25)
	stratificationTest(planeTensorLargeNoise,70)
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
    print( "\n\nCurve Small Noise...\n" )
	curveTensor = generateCurveTensor(indexX,indexY2,indexZ3,2)
	curvificationTest(curveTensor,90)

	# diagonal curbe tensor big noise
    print( "\n\nCurve Medium Noise...\n" )
	curveTensorNoise = generateCurveTensor(indexX,indexY2,indexZ3, 3)
	curvificationTest(curveTensorNoise,90)

	# diagonal curbe tensor big noise
    print( "\n\nCurve Large Noise...\n" )
	curveTensorLargeNoise = generateCurveTensor(indexX,indexY2,indexZ3, 4)
	curvificationTest(curveTensorLargeNoise,90)

	# surface tensor small noise
    print( "\n\nSurface Small Noise...\n" )
	surfaceTensor = generateSurfaceTensor(indexX,indexY2,indexZ3,3)
	stratificationTest(surfaceTensor,90)

	# surface tensor big noise
    print( "\n\nSurface Medium Noise...\n" )
	surfaceTensorNoise = generateSurfaceTensor(indexX,indexY2,indexZ3,10)
	stratificationTest(surfaceTensorNoise,90)

	# surface tensor big noise
    print( "\n\nSurface Large Noise...\n" )
	surfaceTensorLargeNoise = generateSurfaceTensor(indexX,indexY2,indexZ3,20)
	stratificationTest(surfaceTensorLargeNoise,90)
end

function runTests()
	runBlockTests()
	runSmoothTests()
	runCurvyTests()
end