
# function for remving small entries
dropSmall = x -> abs(x)< 1e-10 ? 0 : x

# Generate Face Block Tensor

#[(-4:4)...] is a shorthand for [-4,-3,-2,-1,0,1,2,3,4] -- the arguments needs to be a vector and -4:4 does not work
Xes = [(0:0.1:10)...] .|> x-> (x*x- 9.0)
Yes = [(0:0.1:10)...] .|> x-> (x*x- 9.0)
Zes = [(0:0.1:10)...] .|> x-> (x*x- 9.0)

tensorSphere = randomSurfaceTensor( Xes, Yes, Zes, 1.5)
# measure
testSurfaceTensor(tensorSphere, Xes, Yes, Zes)
saveTensorToFile(tensorSphere, "sphere-51-41-81-orig.txt")