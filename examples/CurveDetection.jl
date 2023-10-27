include("../Dleto.jl") 


# function for remving small entries
dropSmall = x -> abs(x)< 1e-10 ? 0 : x

# Generate Face Block Tensor

#[(-4:4)...] is a shorthand for [-4,-3,-2,-1,0,1,2,3,4] -- the arguments needs to be a vector and -4:4 does not work
Xes = [(-2:0.166666:2)...] .|> x-> (x + 6)^2 - 40
Yes = [(-2:0.125:2)...] .|> x-> 40 - (x - 6)^2 
Zes = [(-2:0.1:2)...] .|> x-> 3* (x^3) 


tensorCurve = randomCurveTensor( Xes, Yes, Zes, 7)
# measure
testCurveTensor(tensorCurve, Xes, Yes, Zes)


# run alg on the orig
# @time is a Julia macro which prints info about how long the computation took
@time resOrigCurve=toCurveTensor(tensorCurve)


# show result
resOrigCurve.tensor .|> dropSmall

# show matrices
resOrigCurve.Xchange .|> dropSmall
resOrigCurve.Ychange .|> dropSmall
resOrigCurve.Zchange .|> dropSmall
#measure
testCurveTensor(resOrigCurve.tensor, resOrigCurve.Xes, resOrigCurve.Yes, resOrigCurve.Zes)



#randomize
rTensorCurve = randomizeTensor(tensorCurve)
# run alg on the randomized tensor
@time resRandomCurve=toCurveTensor(rTensorCurve.tensor)
# show result
resRandomCurve.tensor .|> dropSmall
# show matrices
resRandomCurve.Xchange
resRandomCurve.Ychange
resRandomCurve.Zchange
# show product of transofromations
rTensorCurve.Xchange * resRandomCurve.Xchange .|> dropSmall
rTensorCurve.Ychange * resRandomCurve.Ychange .|> dropSmall
rTensorCurve.Zchange * resRandomCurve.Zchange .|> dropSmall

#measure
testCurveTensor(resRandomCurve.tensor, resRandomCurve.Xes, resRandomCurve.Yes, resRandomCurve.Zes)



###########################
# thicker Curve

tensorCurveThick = randomCurveTensor( Xes, Yes, Zes, 20)
# measure
testCurveTensor(tensorCurveThick, Xes, Yes, Zes)


# run alg on the orig
# @time is a Julia macro which prints info about how long the computation took
@time resOrigCurveThick=toCurveTensor(tensorCurveThick)


# show result
resOrigCurveThick.tensor .|> dropSmall

# show matrices
resOrigCurveThick.Xchange .|> dropSmall
resOrigCurveThick.Ychange .|> dropSmall
resOrigCurveThick.Zchange .|> dropSmall
#measure
testCurveTensor(resOrigCurveThick.tensor, resOrigCurveThick.Xes, resOrigCurveThick.Yes, resOrigCurveThick.Zes)



#randomize
rTensorCurveThick = randomizeTensor(tensorCurveThick)
# run alg on the randomized tensor
@time resRandomCurveThick=toCurveTensor(rTensorCurveThick.tensor)
# show result
resRandomCurveThick.tensor .|> dropSmall
resRandomCurveThick.tensor .|> x -> abs(x)< 1e-3 ? 0 : x
# show matrices
resRandomCurveThick.Xchange
resRandomCurveThick.Ychange
resRandomCurveThick.Zchange
# show product of transofromations
rTensorCurveThick.Xchange * resRandomCurveThick.Xchange .|> dropSmall
rTensorCurveThick.Ychange * resRandomCurveThick.Ychange .|> dropSmall
rTensorCurveThick.Zchange * resRandomCurveThick.Zchange .|> dropSmall

#measure
testCurveTensor(resRandomCurveThick.tensor, resRandomCurveThick.Xes, resRandomCurveThick.Yes, resRandomCurveThick.Zes)

