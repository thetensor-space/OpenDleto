########################################################################
#  CC-BY 2021 
#  Peter A Brooksbank
#  Martin Kassabov
#  James B. Wilson
#
#    Distributed under MIT License.
########################################################################


include("../src/TensorGenerator.jl")
include("../src/TensorStratifyTests.jl")


#seed random number generator
Random.seed!(1234567890)

# construct a surface tensor
# first define the eigenvalues of the diagonalizable derivations (X,Y,Z) -- these will be scaled later on

const eigenX = [1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4, 5,5,5,5,5, 6,6,6,6,6, 7,7,7,7,7]
const eigenY = [1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4, 5,5,5,5,5, 6,6,6,6,6, 7,7,7,7,7]
const eigenZ = [1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4, 5,5,5,5,5, 6,6,6,6,6, 7,7,7,7,7]

# construct a tensor supported on a  curve
# the support is restrcted to the "curve" x[i] = y[j] = z[k], however we normalize first to ensure that the surface cuts through the middle of the cube
# width measure the thickness of the surface
# type is how to generate random numbers options are "uniform" and "normal"

const T = generateCurveTensor(eigenX, eigenY, eigenZ, 0.5 ;type="uniform")      



###############################################################
# test for stratification

# 1st argument is the tensor
# 2nd argument is the name of the files to generate

# optional arguments -- these have default values and does not need to be set
# control - apply stratification to the original tensor
# rounds - the number of random transvections to apply to each axis when scrambelling the tensor 
# noise  - number of added places with noise after scrableiing
# relsize - the size of noise to add relative to the maximal entry in the tensor
# type   - type of random number generator "uniform"/"normal"
# ratio - the size of the smallest boxes to print


curvificationTest(T,"block_diagonal_tensor_35_35_35"; ratio=1000,rounds=100,control=true,noise=0,relsize=0.00001,toprint=10,type="normal")
