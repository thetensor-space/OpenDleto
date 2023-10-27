include("stratification.jl") 

Xes = [1,1,2,2,2,3,3,3]
Yes = [1,1,1,2,2,3,3,3]
Zes = [1,1,1,2,2,2,3,3]

tensorDiagonalBlock = randomCurveTensor( Xes, Yes, Zes, 0.1)

rTensorDiagonalBlock = randomizeTensor(tensorDiagonalBlock)

rTensorDiagonalBlock.tensor

resRandomDiagonalBlock=toCurveTensor(rTensorDiagonalBlock.tensor)

resRandomDiagonalBlock.tensor

resRandomDiagonalBlock.tensor .|> x -> (abs(x) < 1e-10 ? 0 : x )


resRandomDiagonalBlock.Xchange
resRandomDiagonalBlock.Ychange
resRandomDiagonalBlock.Zchange


rTensorDiagonalBlock.Xchange * resRandomDiagonalBlock.Xchange .|> x -> (abs(x) < 1e-10 ? 0 : x )
rTensorDiagonalBlock.Ychange * resRandomDiagonalBlock.Ychange .|> x -> (abs(x) < 1e-10 ? 0 : x )
rTensorDiagonalBlock.Zchange * resRandomDiagonalBlock.Zchange .|> x -> (abs(x) < 1e-10 ? 0 : x )

