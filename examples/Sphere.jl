include("../Dleto.jl") 


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

# run alg on the orig
# @time is a Julia macro which prints info about how long the computation took
@time resOrigSphere=toSurfaceTensor(tensorSphere)


# show result
resOrigSphere.tensor .|> dropSmall

# show matrices
resOrigSphere.Xchange .|> dropSmall
resOrigSphere.Ychange .|> dropSmall
resOrigSphere.Zchange .|> dropSmall
#measure
testSurfaceTensor(resOrigSphere.tensor, resOrigSphere.Xes, resOrigSphere.Yes, resOrigSphere.Zes)



#randomize
rTensorSphere = randomizeTensor(tensorSphere)
saveTensorToFile(rTensorSphere.tensor, "sphere-51-41-81-rand.txt")

# run alg on the randomized tensor
@time resRandomSphere=toSurfaceTensor(rTensorSphere.tensor)
# show result
resRandomSphere.tensor .|> dropSmall
# show matrices
resRandomSphere.Xchange
resRandomSphere.Ychange
resRandomSphere.Zchange
# show product of transofromations
rTensorSphere.Xchange * resRandomSphere.Xchange .|> dropSmall
rTensorSphere.Ychange * resRandomSphere.Ychange .|> dropSmall
rTensorSphere.Zchange * resRandomSphere.Zchange .|> dropSmall
saveTensorToFile(resRandomSphere.tensor, "sphere-51-41-81-recov.txt")

#measure
testSurfaceTensor(resRandomSphere.tensor, resRandomSphere.Xes, resRandomSphere.Yes, resRandomSphere.Zes)



###########################
# thicker Sphere

tensorSphereThick = randomSurfaceTensor( Xes, Yes, Zes, 4)
saveTensorToFile(tensorSphereThick, "thicker-sphere-51-41-81-orig.txt")

# measure
testSurfaceTensor(tensorSphereThick, Xes, Yes, Zes)


# run alg on the orig
# @time is a Julia macro which prints info about how long the computation took
@time resOrigSphereThick=toSurfaceTensor(tensorSphereThick)


# show result
resOrigSphereThick.tensor .|> dropSmall

# show matrices
resOrigSphereThick.Xchange .|> dropSmall
resOrigSphereThick.Ychange .|> dropSmall
resOrigSphereThick.Zchange .|> dropSmall
#measure
testSurfaceTensor(resOrigSphereThick.tensor, resOrigSphereThick.Xes, resOrigSphereThick.Yes, resOrigSphereThick.Zes)



#randomize
rTensorSphereThick = randomizeTensor(tensorSphereThick)
saveTensorToFile(rTensorSphereThick.tensor, "thicker-sphere-51-41-81-rand.txt")

# run alg on the randomized tensor
@time resRandomSphereThick=toSurfaceTensor(rTensorSphereThick.tensor)
# show result
resRandomSphereThick.tensor .|> dropSmall
resRandomSphereThick.tensor .|> x -> abs(x)< 1e-3 ? 0 : x
# show matrices
resRandomSphereThick.Xchange
resRandomSphereThick.Ychange
resRandomSphereThick.Zchange
# show product of transofromations
rTensorSphereThick.Xchange * resRandomSphereThick.Xchange .|> dropSmall
rTensorSphereThick.Ychange * resRandomSphereThick.Ychange .|> dropSmall
rTensorSphereThick.Zchange * resRandomSphereThick.Zchange .|> dropSmall

saveTensorToFile(rTensorSphereThick.tensor, "thicker-sphere-51-41-81-recov.txt")

#measure
testSurfaceTensor(resRandomSphereThick.tensor, resRandomSphereThick.Xes, resRandomSphereThick.Yes, resRandomSphereThick.Zes)

