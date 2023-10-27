include("../Dleto.jl") 


# function for remving small entries
dropSmall = x -> abs(x)< 1e-10 ? 0 : x

# Generate Face Block Tensor

#[(-4:4)...] is a shorthand for [-4,-3,-2,-1,0,1,2,3,4] -- the arguments needs to be a vector and -4:4 does not work
Xes = [(-2:0.2:2)...] .|> x-> (x + 6)^2 - 40
Yes = [(-2:0.25:2)...] .|> x-> 40 - (x - 6)^2 
Zes = [(-2:0.1:2)...] .|> x-> 3* (x^3) 




tensorFaceCurve = randomFaceCurveTensor( Xes, Yes, Zes, 3)
# measure
testFaceCurveTensor(tensorFaceCurve, Xes, Yes, Zes)


# run alg on the orig
# @time is a Julia macro which prints info about how long the computation took
@time resOrigFaceCurve=toFaceCurveTensor(tensorFaceCurve)


# show result
resOrigFaceCurve.tensor .|> dropSmall
    
# show matrices
resOrigFaceCurve.Xchange .|> dropSmall
resOrigFaceCurve.Ychange .|> dropSmall
resOrigFaceCurve.Zchange .|> dropSmall

#measure
testFaceCurveTensor(resOrigFaceCurve.tensor, resOrigFaceCurve.Xes, resOrigFaceCurve.Yes, resOrigFaceCurve.Zes)
    

#randomize
rTensorFaceCurve = randomizeTensor(tensorFaceCurve)
# run alg on the randomized tensor
@time resRandomFaceCurve=toFaceCurveTensor(rTensorFaceCurve.tensor)
# show result
resRandomFaceCurve.tensor .|> dropSmall
# show matrices
resRandomFaceCurve.Xchange
resRandomFaceCurve.Ychange
resRandomFaceCurve.Zchange
# show product of transofromations
rTensorFaceCurve.Xchange * resRandomFaceCurve.Xchange .|> dropSmall
rTensorFaceCurve.Ychange * resRandomFaceCurve.Ychange .|> dropSmall
rTensorFaceCurve.Zchange * resRandomFaceCurve.Zchange .|> dropSmall

#measure
testFaceCurveTensor(resRandomFaceCurve.tensor, resRandomFaceCurve.Xes, resRandomFaceCurve.Yes, resRandomFaceCurve.Zes)



###########################
# thicker FaceCurve

tensorFaceCurveThick = randomFaceCurveTensor( Xes, Yes, Zes, 15)
# measure
testFaceCurveTensor(tensorFaceCurveThick, Xes, Yes, Zes)


# run alg on the orig
# @time is a Julia macro which prints info about how long the computation took
@time resOrigFaceCurveThick=toFaceCurveTensor(tensorFaceCurveThick)


# show result
resOrigFaceCurveThick.tensor .|> dropSmall

# show matrices
resOrigFaceCurveThick.Xchange .|> dropSmall
resOrigFaceCurveThick.Ychange .|> dropSmall
resOrigFaceCurveThick.Zchange .|> dropSmall
#measure
testFaceCurveTensor(resOrigFaceCurveThick.tensor, resOrigFaceCurveThick.Xes, resOrigFaceCurveThick.Yes, resOrigFaceCurveThick.Zes)



#randomize
rTensorFaceCurveThick = randomizeTensor(tensorFaceCurveThick)
# run alg on the randomized tensor
@time resRandomFaceCurveThick=toFaceCurveTensor(rTensorFaceCurveThick.tensor)
# show result
resRandomFaceCurveThick.tensor .|> dropSmall
resRandomFaceCurveThick.tensor .|> x -> abs(x)< 1e-1 ? 0 : x
# show matrices
resRandomFaceCurveThick.Xchange
resRandomFaceCurveThick.Ychange
resRandomFaceCurveThick.Zchange
# show product of transofromations
rTensorFaceCurveThick.Xchange * resRandomFaceCurveThick.Xchange .|> dropSmall
rTensorFaceCurveThick.Ychange * resRandomFaceCurveThick.Ychange .|> dropSmall
rTensorFaceCurveThick.Zchange * resRandomFaceCurveThick.Zchange .|> dropSmall

#measure
testFaceCurveTensor(resRandomFaceCurveThick.tensor, resRandomFaceCurveThick.Xes, resRandomFaceCurveThick.Yes, resRandomFaceCurveThick.Zes)

