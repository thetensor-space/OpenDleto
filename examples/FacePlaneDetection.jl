include("../Dleto.jl") 


# function for remving small entries
dropSmall = x -> abs(x)< 1e-10 ? 0 : x

# Generate Face Block Tensor

#[(-4:4)...] is a shorthand for [-4,-3,-2,-1,0,1,2,3,4] -- the arguments needs to be a vector and -4:4 does not work
Xes = [(-2:0.2:2)...]
Yes = [(-2:0.25:2)...]
Zes = [(-2:0.125:2)...]

tensorFacePlane = randomFaceCurveTensor( Xes, Yes, Zes, 0.3)
# measure
testFaceCurveTensor(tensorFacePlane, Xes, Yes, Zes)


# run alg on the orig
# @time is a Julia macro which prints info about how long the computation took
@time resOrigFacePlane=toFaceCurveTensor(tensorFacePlane)


# show result
resOrigFacePlane.tensor .|> dropSmall

# show matrices
resOrigFacePlane.Xchange .|> dropSmall
resOrigFacePlane.Ychange .|> dropSmall
resOrigFacePlane.Zchange .|> dropSmall
#measure
testFaceCurveTensor(resOrigFacePlane.tensor, resOrigFacePlane.Xes, resOrigFacePlane.Yes, resOrigFacePlane.Zes)



#randomize
rTensorFacePlane = randomizeTensor(tensorFacePlane)
# run alg on the randomized tensor
@time resRandomFacePlane=toFaceCurveTensor(rTensorFacePlane.tensor)
# show result
resRandomFacePlane.tensor .|> dropSmall
# show matrices
resRandomFacePlane.Xchange
resRandomFacePlane.Ychange
resRandomFacePlane.Zchange
# show product of transofromations
rTensorFacePlane.Xchange * resRandomFacePlane.Xchange .|> dropSmall
rTensorFacePlane.Ychange * resRandomFacePlane.Ychange .|> dropSmall
rTensorFacePlane.Zchange * resRandomFacePlane.Zchange .|> dropSmall

#measure
testFaceCurveTensor(resRandomFacePlane.tensor, resRandomFacePlane.Xes, resRandomFacePlane.Yes, resRandomFacePlane.Zes)



###########################
# thicker FacePlane

tensorFacePlaneThick = randomFaceCurveTensor( Xes, Yes, Zes, 1)
# measure
testFaceCurveTensor(tensorFacePlaneThick, Xes, Yes, Zes)


# run alg on the orig
# @time is a Julia macro which prints info about how long the computation took
@time resOrigFacePlaneThick=toFaceCurveTensor(tensorFacePlaneThick)


# show result
resOrigFacePlaneThick.tensor .|> dropSmall

# show matrices
resOrigFacePlaneThick.Xchange .|> dropSmall
resOrigFacePlaneThick.Ychange .|> dropSmall
resOrigFacePlaneThick.Zchange .|> dropSmall
#measure
testFaceCurveTensor(resOrigFacePlaneThick.tensor, resOrigFacePlaneThick.Xes, resOrigFacePlaneThick.Yes, resOrigFacePlaneThick.Zes)



#randomize
rTensorFacePlaneThick = randomizeTensor(tensorFacePlaneThick)
# run alg on the randomized tensor
@time resRandomFacePlaneThick=toFaceCurveTensor(rTensorFacePlaneThick.tensor)
# show result
resRandomFacePlaneThick.tensor .|> dropSmall
resRandomFacePlaneThick.tensor .|> x -> abs(x)< 1e-3 ? 0 : x
# show matrices
resRandomFacePlaneThick.Xchange
resRandomFacePlaneThick.Ychange
resRandomFacePlaneThick.Zchange
# show product of transofromations
rTensorFacePlaneThick.Xchange * resRandomFacePlaneThick.Xchange .|> dropSmall
rTensorFacePlaneThick.Ychange * resRandomFacePlaneThick.Ychange .|> dropSmall
rTensorFacePlaneThick.Zchange * resRandomFacePlaneThick.Zchange .|> dropSmall

#measure
testFaceCurveTensor(resRandomFacePlaneThick.tensor, resRandomFacePlaneThick.Xes, resRandomFacePlaneThick.Yes, resRandomFacePlaneThick.Zes)

