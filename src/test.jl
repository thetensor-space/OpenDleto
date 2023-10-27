include("Dleto.jl") 

# Tensor generators

## Tensor supported on a diagonal plane 

#[(-4:4)...] is a shorthand for [-4,-3,-2,-1,0,1,2,3,4] -- the arguments needs to be a vector and -4:4 does not work
tensorPlaneThin = randomSurfaceTensor([(-5:5)...], [(-7:7)...], [(-4:4)...], 0.1)
tensorPlaneThick = randomSurfaceTensor([(-5:5)...], [(-7:7)...], [(-4:4)...], 2)
tensorPlaneVeryThick = randomSurfaceTensor([(-5:5)...], [(-7:7)...], [(-4:4)...], 5)

## Tensor supported on a face line 

tensorFaceLineThin = randomFaceCurveTensor([(-4:4)...], [(-4:4)...], [(-2:2)...], 0.1)
tensorFaceLineThick = randomFaceCurveTensor([(-4:4)...], [(-4:4)...], [(-2:2)...], 2)
tensorFaceLineVeryThick = randomFaceCurveTensor([(-4:4)...], [(-4:4)...], [(-2:2)...], 5)


## Tensor supported on a diagonal

tensorDiagonalLineThin = randomCurveTensor([(-4:4)...], [(-4:4)...], [(-4:4)...], 0.1)
tensorDiagonalLineThick = randomCurveTensor([(-4:4)...], [(-4:4)...], [(-4:4)...], 2)
tensorDiagonalLineVeryThick = randomCurveTensor([(-4:4)...], [(-4:4)...], [(-4:4)...], 5)




# Test functions 

## Test that tensors are supported on plane 

testSurfaceTensor(tensorPlaneThin, [(-5:5)...],[(-7:7)...],[(-4:4)...] )
testSurfaceTensor(tensorPlaneThick, [(-5:5)...],[(-7:7)...],[(-4:4)...] )
testSurfaceTensor(tensorPlaneVeryThick, [(-5:5)...],[(-7:7)...],[(-4:4)...] )

## compare to random tensor
testSurfaceTensor(randn(11,15,9), [(-5:5)...],[(-7:7)...],[(-4:4)...] )


## Test that tensors are supported on face line 

testFaceCurveTensor(tensorFaceLineThin,[(-4:4)...], [(-4:4)...], [(-2:2)...] )
testFaceCurveTensor(tensorFaceLineThick,[(-4:4)...], [(-4:4)...], [(-2:2)...] )
testFaceCurveTensor(tensorFaceLineVeryThick,[(-4:4)...], [(-4:4)...], [(-2:2)...] )

## compare to random tensor
testFaceCurveTensor(randn(9,9,5),[(-4:4)...], [(-4:4)...], [(-2:2)...] )


## Test that tensors are supported on diagonal line 
testCurveTensor(tensorDiagonalLineThin,[(-4:4)...], [(-4:4)...], [(-4:4)...] )
testCurveTensor(tensorDiagonalLineThick,[(-4:4)...], [(-4:4)...], [(-4:4)...] )
testCurveTensor(tensorDiagonalLineVeryThick,[(-4:4)...], [(-4:4)...], [(-4:4)...] )
## compare to random tensor
testCurveTensor(randn(9,9,9),[(-4:4)...], [(-4:4)...], [(-4:4)...] )


# Stratification Tests

## Original

### Surface
resorigPlaneThin  = toSurfaceTensor(tensorPlaneThin)
resorigPlaneThin.tensor
testSurfaceTensor(resorigPlaneThin.tensor, resorigPlaneThin.Xes, resorigPlaneThin.Yes, resorigPlaneThin.Zes)


resorigPlaneThick  = toSurfaceTensor(tensorPlaneThick)
resorigPlaneThick.tensor
testSurfaceTensor(resorigPlaneThick.tensor, resorigPlaneThick.Xes, resorigPlaneThick.Yes, resorigPlaneThick.Zes)


resorigPlaneVeryThick  = toSurfaceTensor(tensorPlaneVeryThick)
resorigPlaneVeryThick.tensor
testSurfaceTensor(resorigPlaneVeryThick.tensor, resorigPlaneVeryThick.Xes, resorigPlaneVeryThick.Yes, resorigPlaneVeryThick.Zes)


### FaceCurve

resorigFaceLineThin=toFaceCurveTensor(tensorFaceLineThin)
resorigFaceLineThin.tensor
testFaceCurveTensor(resorigFaceLineThin.tensor, resorigFaceLineThin.Xes, resorigFaceLineThin.Yes, resorigFaceLineThin.Zes)

resorigFaceLineThick=toFaceCurveTensor(tensorFaceLineThick)
resorigFaceLineThick.tensor
testFaceCurveTensor(resorigFaceLineThick.tensor, resorigFaceLineThick.Xes, resorigFaceLineThick.Yes, resorigFaceLineThick.Zes)

resorigFaceLineVeryThick=toFaceCurveTensor(tensorFaceLineVeryThick)
resorigFaceLineVeryThick.tensor
testFaceCurveTensor(resorigFaceLineVeryThick.tensor, resorigFaceLineVeryThick.Xes, resorigFaceLineVeryThick.Yes, resorigFaceLineVeryThick.Zes)


### Curve 
resorigDiagonalLineThin=toCurveTensor(tensorDiagonalLineThin)
resorigDiagonalLineThin.tensor
testCurveTensor(resorigDiagonalLineThin.tensor, resorigDiagonalLineThin.Xes, resorigDiagonalLineThin.Yes, resorigDiagonalLineThin.Zes)


resorigDiagonalLineThick=toCurveTensor(tensorDiagonalLineThick)
resorigDiagonalLineThick.tensor
testCurveTensor(resorigDiagonalLineThick.tensor, resorigDiagonalLineThick.Xes, resorigDiagonalLineThick.Yes, resorigDiagonalLineThick.Zes)


resorigDiagonalLineVeryThick=toCurveTensor(tensorDiagonalLineVeryThick)
resorigDiagonalLineVeryThick.tensor
testCurveTensor(resorigDiagonalLineVeryThick.tensor, resorigDiagonalLineVeryThick.Xes, resorigDiagonalLineVeryThick.Yes, resorigDiagonalLineVeryThick.Zes)


## Randomized

### Surface
RtensorPlaneThin = randomizeTensor(tensorPlaneThin)
resnewPlaneThin  = toSurfaceTensor(RtensorPlaneThin.tensor)
resnewPlaneThin.tensor
testSurfaceTensor(resnewPlaneThin.tensor, resnewPlaneThin.Xes, resnewPlaneThin.Yes, resnewPlaneThin.Zes)


RtensorPlaneThick = randomizeTensor(tensorPlaneThick)
resnewPlaneThick  = toSurfaceTensor(RtensorPlaneThick.tensor)
resnewPlaneThick.tensor
testSurfaceTensor(resnewPlaneThick.tensor, resnewPlaneThick.Xes, resnewPlaneThick.Yes, resnewPlaneThick.Zes)


RtensorPlaneVeryThick = randomizeTensor(tensorPlaneVeryThick)
resnewPlaneVeryThick  = toSurfaceTensor(RtensorPlaneVeryThick.tensor)
resnewPlaneVeryThick.tensor
testSurfaceTensor(resnewPlaneVeryThick.tensor, resnewPlaneVeryThick.Xes, resnewPlaneVeryThick.Yes, resnewPlaneVeryThick.Zes)


### FaceCurve
# not tested yet

resorigFaceLineThin=toFaceCurveTensor(tensorFaceLineThin)
resorigFaceLineThin.tensor
testFaceCurveTensor(resorigFaceLineThin.tensor, resorigFaceLineThin.Xes, resorigFaceLineThin.Yes, resorigFaceLineThin.Zes)

resorigFaceLineThick=toFaceCurveTensor(tensorFaceLineThick)
resorigFaceLineThick.tensor
testFaceCurveTensor(resorigFaceLineThick.tensor, resorigFaceLineThick.Xes, resorigFaceLineThick.Yes, resorigFaceLineThick.Zes)

resorigFaceLineVeryThick=toFaceCurveTensor(tensorFaceLineVeryThick)
resorigFaceLineVeryThick.tensor
testFaceCurveTensor(resorigFaceLineVeryThick.tensor, resorigFaceLineVeryThick.Xes, resorigFaceLineVeryThick.Yes, resorigFaceLineVeryThick.Zes)


### Curve 
resorigDiagonalLineThin=toCurveTensor(tensorDiagonalLineThin)
resorigDiagonalLineThin.tensor
testCurveTensor(resorigDiagonalLineThin.tensor, resorigDiagonalLineThin.Xes, resorigDiagonalLineThin.Yes, resorigDiagonalLineThin.Zes)


resorigDiagonalLineThick=toCurveTensor(tensorDiagonalLineThick)
resorigDiagonalLineThick.tensor
testCurveTensor(resorigDiagonalLineThick.tensor, resorigDiagonalLineThick.Xes, resorigDiagonalLineThick.Yes, resorigDiagonalLineThick.Zes)


resorigDiagonalLineVeryThick=toCurveTensor(tensorDiagonalLineVeryThick)
resorigDiagonalLineVeryThick.tensor
testCurveTensor(resorigDiagonalLineVeryThick.tensor, resorigDiagonalLineVeryThick.Xes, resorigDiagonalLineVeryThick.Yes, resorigDiagonalLineVeryThick.Zes)

