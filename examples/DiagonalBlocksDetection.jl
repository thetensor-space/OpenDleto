include("../Dleto.jl") 

# Detecting curve can be used for detecting diagonal blocks

# function for remving small entries
dropSmall = x -> abs(x)< 1e-10 ? 0 : x

# Generate Face Block Tensor

#[(-4:4)...] is a shorthand for [-4,-3,-2,-1,0,1,2,3,4] -- the arguments needs to be a vector and -4:4 does not work
Xes = [1,1,1,1,2,2,2,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5]
Yes = [1,1,2,2,2,2,2,3,3,3,4,4,4,4,5,5,5,5,5,5,5,5]
Zes = [1,1,1,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5]

tensorDiagonalBlock = randomCurveTensor( Xes, Yes, Zes, 0.1)
# measure
testCurveTensor(tensorDiagonalBlock, Xes, Yes, Zes)


# run alg on the orig
# @time is a Julia macro which prints info about how long the computation took
@time resOrigDiagonalBlock=toCurveTensor(tensorDiagonalBlock)


# show result
resOrigDiagonalBlock.tensor .|> dropSmall

# show matrices
resOrigDiagonalBlock.Xchange .|> dropSmall
resOrigDiagonalBlock.Ychange .|> dropSmall
resOrigDiagonalBlock.Zchange .|> dropSmall
#measure
testCurveTensor(resOrigDiagonalBlock.tensor, resOrigDiagonalBlock.Xes, resOrigDiagonalBlock.Yes, resOrigDiagonalBlock.Zes)



#randomize
rTensorDiagonalBlock = randomizeTensor(tensorDiagonalBlock)
# run alg on the randomized tensor
@time resRandomDiagonalBlock=toCurveTensor(rTensorDiagonalBlock.tensor)
# show result
resRandomDiagonalBlock.tensor .|> dropSmall
# show matrices
resRandomDiagonalBlock.Xchange
resRandomDiagonalBlock.Ychange
resRandomDiagonalBlock.Zchange
# show product of transofromations
rTensorDiagonalBlock.Xchange * resRandomDiagonalBlock.Xchange .|> dropSmall
rTensorDiagonalBlock.Ychange * resRandomDiagonalBlock.Ychange .|> dropSmall
rTensorDiagonalBlock.Zchange * resRandomDiagonalBlock.Zchange .|> dropSmall

#measure
testCurveTensor(resRandomDiagonalBlock.tensor, resRandomDiagonalBlock.Xes, resRandomDiagonalBlock.Yes, resRandomDiagonalBlock.Zes)
