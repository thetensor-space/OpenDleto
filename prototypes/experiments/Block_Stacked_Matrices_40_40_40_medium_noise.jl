########################################################################
#  CC-BY 2021 
#  Peter A Brooksbank
#  Martin Kassabov
#  James B. Wilson
#
#    Distributed under MIT License.
########################################################################


include("..\\tensorGenerator.jl")
include("..\\tensorStratifyTests.jl")


#seed random number generator
Random.seed!(1234567890)

# construct a surface tensor
# first define the eigenvalues of the diagonalizable derivations (X,Y,Z) -- these will be scaled later on

eigenX = [1,1,1, 2,2,2,2,2, 3,3,3,3,3,3,3, 4,4,4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
eigenY = [1,1,1, 2,2,2,2,2, 3,3,3,3,3,3,3, 4,4,4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
#eigenZ = [1,1,1, 2,2,2,2,2, 3,3,3,3,3,3,3, 4,4,4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
eigenZ = 1:40
#only used to get the size of the tensor

# construct a tensor supported on a  curve
# the support is restrcted to the "curve" x[i] = y[j] = z[k], however we normalize first to ensure that the surface cuts through the middle of the cube
# width measure the thickness of the surface
# type is how to generate random numbers options are "uniform" and "normal"

T = generateSurface12Tensor(eigenX, eigenY, eigenZ, 0.5 ;type="uniform")      



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
# ration - the size of the smallest boxes to print


stratification12FaceTest(T,"block_stacked_matrices_40_40_40"; ratio=250,rounds=100,control=true,noise=650,relsize=0.0003,toprint=20,type="normal")
