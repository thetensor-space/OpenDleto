include("../Dleto.jl") 



# generate tensor supported on a diagonal plane
T = randomSurfaceTensor([(-5:5)...],[(-7:7)...],[(-4:4)...],0.1)

#[(-4:4)...] is a shorthand for [-4,-3,-2,-1,0,1,2,3,4]

# changing the last papramter make the support a tichker plane 
#T = randomSurfaceTensor([(-5:5)...],[(-7:7)...],[(-4:4)...],1.5)


#verify that the tensor in supported on the plane  -- the number nneds to be very small
testSurfaceTensor(T, [(-5:5)...],[(-7:7)...],[(-4:4)...] )

#attempt to staratify it 
resorig  = toSurfaceTensor(T);

# show the resulting tensor
resorig.tensor

testSurfaceTensor(resorig.tensor, resorig.Xes, resorig.Yes, resorig.Zes)


# generate random othoergonal matrices to randomize the tensor
XC = randomOthogonalMatrix(11)
YC = randomOthogonalMatrix(15)
ZC = randomOthogonalMatrix(9)

# randomized tensor
Tnew = actAllDirections(T,[XC,YC,ZC])

#attempt to staratify it 
resnew  = toSurfaceTensor(Tnew);

# show the resulting tensor
resnew.tensor

testSurfaceTensor(resnew.tensor, resnew.Xes, resnew.Yes, resnew.Zes)

# check that the composition of the transformations is almost the identity
XC * resnew.Xchange
YC * resnew.Ychange
ZC * resnew.Zchange