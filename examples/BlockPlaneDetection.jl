include("../Dleto.jl") 

# Detecting surface can be used for detecting block plane

# function for remving small entries
dropSmall = x -> abs(x)< 1e-10 ? 0 : x

# Generate Face Block Tensor

#[(-4:4)...] is a shorthand for [-4,-3,-2,-1,0,1,2,3,4] -- the arguments needs to be a vector and -4:4 does not work
Xes = [1,1,1,1,2,2,2,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5]
Yes = [1,1,2,2,2,2,2,3,3,3,4,4,4,4,5,5,5,5,5,5,5,5]
Zes = [-8,-8,-8,-8,-7,-7,-7,-6,-6,-6,-6,-6,-5,-5,-5,-5,-5,-4,-4,-4,-4,-4,-4]

tensorBlockPlane = randomSurfaceTensor( Xes, Yes, Zes, 0.1)
# measure
testSurfaceTensor(tensorBlockPlane, Xes, Yes, Zes)


# run alg on the orig
# @time is a Julia macro which prints info about how long the computation took
@time resOrigBlockPlane=toSurfaceTensor(tensorBlockPlane)


# show result
resOrigBlockPlane.tensor .|> dropSmall

# show matrices
resOrigBlockPlane.Xchange .|> dropSmall
resOrigBlockPlane.Ychange .|> dropSmall
resOrigBlockPlane.Zchange .|> dropSmall
#measure
testSurfaceTensor(resOrigBlockPlane.tensor, resOrigBlockPlane.Xes, resOrigBlockPlane.Yes, resOrigBlockPlane.Zes)



#randomize
rTensorBlockPlane = randomizeTensor(tensorBlockPlane)
# run alg on the randomized tensor
@time resRandomBlockPlane=toSurfaceTensor(rTensorBlockPlane.tensor)
# show result
resRandomBlockPlane.tensor .|> dropSmall
# show matrices
resRandomBlockPlane.Xchange
resRandomBlockPlane.Ychange
resRandomBlockPlane.Zchange
# show product of transofromations
rTensorBlockPlane.Xchange * resRandomBlockPlane.Xchange .|> dropSmall
rTensorBlockPlane.Ychange * resRandomBlockPlane.Ychange .|> dropSmall
rTensorBlockPlane.Zchange * resRandomBlockPlane.Zchange .|> dropSmall

#measure
testSurfaceTensor(resRandomBlockPlane.tensor, resRandomBlockPlane.Xes, resRandomBlockPlane.Yes, resRandomBlockPlane.Zes)
