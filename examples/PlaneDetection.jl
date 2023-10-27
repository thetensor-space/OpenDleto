include("../Dleto.jl") 


# function for remving small entries
dropSmall = x -> abs(x)< 1e-10 ? 0 : x

# Generate Face Block Tensor

#[(-4:4)...] is a shorthand for [-4,-3,-2,-1,0,1,2,3,4] -- the arguments needs to be a vector and -4:4 does not work
Xes = [(-2:0.2:2)...]
Yes = [(-2:0.25:2)...]
Zes = [(-2:0.125:2)...]

tensorPlane = randomSurfaceTensor( Xes, Yes, Zes, 0.1)
# measure
testSurfaceTensor(tensorPlane, Xes, Yes, Zes)


# run alg on the orig
# @time is a Julia macro which prints info about how long the computation took
@time resOrigPlane=toSurfaceTensor(tensorPlane)


# show result
resOrigPlane.tensor .|> dropSmall

# show matrices
resOrigPlane.Xchange .|> dropSmall
resOrigPlane.Ychange .|> dropSmall
resOrigPlane.Zchange .|> dropSmall
#measure
testSurfaceTensor(resOrigPlane.tensor, resOrigPlane.Xes, resOrigPlane.Yes, resOrigPlane.Zes)



#randomize
rTensorPlane = randomizeTensor(tensorPlane)
# run alg on the randomized tensor
@time resRandomPlane=toSurfaceTensor(rTensorPlane.tensor)
# show result
resRandomPlane.tensor .|> dropSmall
# show matrices
resRandomPlane.Xchange
resRandomPlane.Ychange
resRandomPlane.Zchange
# show product of transofromations
rTensorPlane.Xchange * resRandomPlane.Xchange .|> dropSmall
rTensorPlane.Ychange * resRandomPlane.Ychange .|> dropSmall
rTensorPlane.Zchange * resRandomPlane.Zchange .|> dropSmall

#measure
testSurfaceTensor(resRandomPlane.tensor, resRandomPlane.Xes, resRandomPlane.Yes, resRandomPlane.Zes)



###########################
# thicker plane

tensorPlaneThick = randomSurfaceTensor( Xes, Yes, Zes, 0.75)
# measure
testSurfaceTensor(tensorPlaneThick, Xes, Yes, Zes)


# run alg on the orig
# @time is a Julia macro which prints info about how long the computation took
@time resOrigPlaneThick=toSurfaceTensor(tensorPlaneThick)


# show result
resOrigPlaneThick.tensor .|> dropSmall

# show matrices
resOrigPlaneThick.Xchange .|> dropSmall
resOrigPlaneThick.Ychange .|> dropSmall
resOrigPlaneThick.Zchange .|> dropSmall
#measure
testSurfaceTensor(resOrigPlaneThick.tensor, resOrigPlaneThick.Xes, resOrigPlaneThick.Yes, resOrigPlaneThick.Zes)



#randomize
rTensorPlaneThick = randomizeTensor(tensorPlaneThick)
# run alg on the randomized tensor
@time resRandomPlaneThick=toSurfaceTensor(rTensorPlaneThick.tensor)
# show result
resRandomPlaneThick.tensor .|> dropSmall
resRandomPlaneThick.tensor .|> x -> abs(x)< 1e-3 ? 0 : x
# show matrices
resRandomPlaneThick.Xchange
resRandomPlaneThick.Ychange
resRandomPlaneThick.Zchange
# show product of transofromations
rTensorPlaneThick.Xchange * resRandomPlaneThick.Xchange .|> dropSmall
rTensorPlaneThick.Ychange * resRandomPlaneThick.Ychange .|> dropSmall
rTensorPlaneThick.Zchange * resRandomPlaneThick.Zchange .|> dropSmall

#measure
testSurfaceTensor(resRandomPlaneThick.tensor, resRandomPlaneThick.Xes, resRandomPlaneThick.Yes, resRandomPlaneThick.Zes)

