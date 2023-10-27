include("../Dleto.jl") 


# function for remving small entries
dropSmall = x -> abs(x)< 1e-10 ? 0 : x

# Generate Face Block Tensor

#[(-4:4)...] is a shorthand for [-4,-3,-2,-1,0,1,2,3,4] -- the arguments needs to be a vector and -4:4 does not work
Xes = [(-2:0.2:2)...]
Yes = [(-2:0.25:2)...]
Zes = [(-2:0.125:2)...]

tensorDiagonal = randomCurveTensor( Xes, Yes, Zes, 0.5)
# measure
testCurveTensor(tensorDiagonal, Xes, Yes, Zes)


# run alg on the orig
# @time is a Julia macro which prints info about how long the computation took
@time resOrigDiagonal=toCurveTensor(tensorDiagonal)


# show result
resOrigDiagonal.tensor .|> dropSmall

# show matrices
resOrigDiagonal.Xchange .|> dropSmall
resOrigDiagonal.Ychange .|> dropSmall
resOrigDiagonal.Zchange .|> dropSmall
#measure
testCurveTensor(resOrigDiagonal.tensor, resOrigDiagonal.Xes, resOrigDiagonal.Yes, resOrigDiagonal.Zes)



#randomize
rTensorDiagonal = randomizeTensor(tensorDiagonal)
# run alg on the randomized tensor
@time resRandomDiagonal=toCurveTensor(rTensorDiagonal.tensor)
# show result
resRandomDiagonal.tensor .|> dropSmall
# show matrices
resRandomDiagonal.Xchange
resRandomDiagonal.Ychange
resRandomDiagonal.Zchange
# show product of transofromations
rTensorDiagonal.Xchange * resRandomDiagonal.Xchange .|> dropSmall
rTensorDiagonal.Ychange * resRandomDiagonal.Ychange .|> dropSmall
rTensorDiagonal.Zchange * resRandomDiagonal.Zchange .|> dropSmall

#measure
testCurveTensor(resRandomDiagonal.tensor, resRandomDiagonal.Xes, resRandomDiagonal.Yes, resRandomDiagonal.Zes)



###########################
# thicker Diagonal

tensorDiagonalThick = randomCurveTensor( Xes, Yes, Zes, 2.5)
# measure
testCurveTensor(tensorDiagonalThick, Xes, Yes, Zes)


# run alg on the orig
# @time is a Julia macro which prints info about how long the computation took
@time resOrigDiagonalThick=toCurveTensor(tensorDiagonalThick)


# show result
resOrigDiagonalThick.tensor .|> dropSmall

# show matrices
resOrigDiagonalThick.Xchange .|> dropSmall
resOrigDiagonalThick.Ychange .|> dropSmall
resOrigDiagonalThick.Zchange .|> dropSmall
#measure
testCurveTensor(resOrigDiagonalThick.tensor, resOrigDiagonalThick.Xes, resOrigDiagonalThick.Yes, resOrigDiagonalThick.Zes)



#randomize
rTensorDiagonalThick = randomizeTensor(tensorDiagonalThick)
# run alg on the randomized tensor
@time resRandomDiagonalThick=toCurveTensor(rTensorDiagonalThick.tensor)
# show result
resRandomDiagonalThick.tensor .|> dropSmall
resRandomDiagonalThick.tensor .|> x -> abs(x)< 1e-3 ? 0 : x
# show matrices
resRandomDiagonalThick.Xchange
resRandomDiagonalThick.Ychange
resRandomDiagonalThick.Zchange
# show product of transofromations
rTensorDiagonalThick.Xchange * resRandomDiagonalThick.Xchange .|> dropSmall
rTensorDiagonalThick.Ychange * resRandomDiagonalThick.Ychange .|> dropSmall
rTensorDiagonalThick.Zchange * resRandomDiagonalThick.Zchange .|> dropSmall

#measure
testCurveTensor(resRandomDiagonalThick.tensor, resRandomDiagonalThick.Xes, resRandomDiagonalThick.Yes, resRandomDiagonalThick.Zes)

