
using ITensors
x = Index(2, "x"); y = Index(3, "y"); z = Index(4, "z");
Γ = random_itensor( x, y, z);   

import Pkg; Pkg.activate("."); Pkg.instantiate();
using Dleto

ders = der(Γ);
res = stratify(Γ, ders[1])

# s51 = TensorIO.loadTensor( "labs/geometry/sphere-51x51x51-rand.txt" )
# ddLM, denLM = derden( [ 1 1 1 ], s51 )
# ddSMO, derSMO, denSMO = derdenSMO( [ 1 1 1 ], s51 )
# inv_dd51 = invderdenSMO( [ 1 1 1 ], s51 )

# vals, vecs = Arpack.eigs(inv_dd51, nev=10, which=:LM)

# # @time res = NullSolvers.solve(L51, :SVDSolver); # VERY SLOW
# @time res = NullSolvers.solve(S51, :ArpackSolver);

# @time res = NullSolvers.solve(S51, :LUSolver; tol=1e-6);
# # TensorIO.sidebyside( S51, ITensors.ITensor( reshape( res.vecs[:,1], size(s51)... ), inds(s51)... ); left_title="Original", right_title