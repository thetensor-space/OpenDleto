
# Cylinder

include("../src/TensorStratify.jl")
include("../src/tensorRandomize.jl")
include("../src/Tensor3D.jl")

## Make a cylinder 
d = 25; e = d-9
t = zeros(Float64,d,d,e)
for k = 1:e
                                               t[k,k+4,k]=rand(); t[k,k+5,k]=rand(); 
            t[k+1,k+2,k]=rand(); t[k+1,k+3,k]=rand();                      t[k+1,k+6,k]=rand(); t[k+1,k+7,k]=rand(); 
    t[k+2,k+1,k]=rand();                                                                                   t[k+2,k+8,k]=rand(); 
    t[k+3,k+1,k]=rand();                                                                                   t[k+3,k+8,k]=rand(); 
t[k+4,k,k]=rand();                                                                                                     t[k+4,k+9,k] = rand(); 
t[k+5,k,k]=rand();                                                                                                     t[k+5,k+9,k] = rand(); 
    t[k+6,k+1,k]=rand();                                                                                   t[k+6,k+8,k]=rand(); 
    t[k+7,k+1,k]=rand();                                                                                   t[k+7,k+8,k]=rand(); 
            t[k+8,k+2,k]=rand(); t[k+8,k+3,k]=rand();                      t[k+8,k+6,k]=rand(); t[k+8,k+7,k]=rand(); 
                                                t[k+9,k+4,k]=rand(); t[k+9,k+5,k]=rand(); 
end

mkdir( "cylinder")
mkdir( "cylinder/images")
save3D(t,"cylinder/images/cylinder-orig.ply", "cylinder-orig.raw", 1000)

s, d = stratify(t)
save3D(s,"cylinder/images/cylinder-orig-strat.ply", "cylinder-orig.raw", 1000)

rt, rm = tensorRandomize(t, [10,10,10])
save3D(rt,"cylinder/images/cylinder-rand.ply", "cylinder-rand.raw", 1000)

rrt, rd = stratify(rt)
save3D(rrt,"cylinder/images/cylinder-rand-strat.ply", "cylinder-rand.raw", 1000)
