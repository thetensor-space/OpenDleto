include("../Dleto.jl") 

# Detecting Face curve can be used for detecting face blocks

# function for remving small entries
dropSmall = x -> abs(x)< 1e-10 ? 0 : x

# Generate Face Block Tensor

#[(-4:4)...] is a shorthand for [-4,-3,-2,-1,0,1,2,3,4] -- the arguments needs to be a vector and -4:4 does not work
Xes = [1,1,1,1,2,2,2,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5]
Yes = [1,1,2,2,2,2,2,3,3,3,4,4,4,4,5,5,5,5,5,5,5,5]
Zes = [(1:10)...]
tensorFaceBlock = randomFaceCurveTensor( Xes, Yes, Zes, 0.1)
# measure
testFaceCurveTensor(tensorFaceBlock, Xes, Yes, Zes)


# run alg on the orig
# @time is a Julia macro which prints info about how long the computation took
@time resOrigFaceBlock=toFaceCurveTensor(tensorFaceBlock)


# show result
resOrigFaceBlock.tensor .|> dropSmall

# show matrices
resOrigFaceBlock.Xchange .|> dropSmall
resOrigFaceBlock.Ychange .|> dropSmall
resOrigFaceBlock.Zchange .|> dropSmall
#measure
testFaceCurveTensor(resOrigFaceBlock.tensor, resOrigFaceBlock.Xes, resOrigFaceBlock.Yes, resOrigFaceBlock.Zes)



#randomize
rTensorFaceBlock = randomizeTensor(tensorFaceBlock)
# run alg on the randomized tensor
@time resRandomFaceBlock=toFaceCurveTensor(rTensorFaceBlock.tensor)
# show result
resRandomFaceBlock.tensor .|> dropSmall
# show matrices
resRandomFaceBlock.Xchange
resRandomFaceBlock.Ychange
resRandomFaceBlock.Zchange
# show product of transofromations
rTensorFaceBlock.Xchange * resRandomFaceBlock.Xchange .|> dropSmall
rTensorFaceBlock.Ychange * resRandomFaceBlock.Ychange .|> dropSmall
rTensorFaceBlock.Zchange * resRandomFaceBlock.Zchange .|> dropSmall
# this is a mess since we are not recovering the change in the Zes

#measure
testFaceCurveTensor(resRandomFaceBlock.tensor, resRandomFaceBlock.Xes, resRandomFaceBlock.Yes, resRandomFaceBlock.Zes)
