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

eigenX = 1:30
eigenY = 1:35
eigenZ = 1:40

# construct a tensor supported on a surface
# the support is restrcted to the "surface" x[i] + y[j] + z[k] =0, however we normalize first to ensure that the surface cuts through the middle of the cube
# width measure the thickness of the surface
# type is how to generate random numbers options are "uniform" and "normal"

T = generateSurfaceTensor(eigenX, eigenY, eigenZ, 15 ;type="uniform")      



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


stratificationTest(T,"planar_tensor_30_35_40"; ratio=100,rounds=70,control=true,noise=40,relsize=0.00001,toprint=10,type="normal")
