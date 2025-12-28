
import Pkg;
Pkg.add( "ITensors" );
Pkg.add( "PlotlyJS" );
Pkg.add( "LinearMaps" );
Pkg.add( "LinearAlgebra" );
Pkg.add( "IterativeSolvers" );
Pkg.add( "Arpack" );
Pkg.add( "TensorOperations" );

using ITensors
using LinearMaps
using LinearAlgebra
using IterativeSolvers
using Arpack
using TensorOperations
import SciMLOperators as SMO
import LinearSolve as LS

include( "src/SylverLining.jl" )
include( "src/NullSolvers.jl" )
include( "src/TensorIO.jl" )

s51 = TensorIO.loadTensor( "labs/geometry/sphere-51x51x51-rand.txt" )
ddLM, denLM = derden( [ 1 1 1 ], s51 )
ddSMO, derSMO, denSMO = derdenSMO( [ 1 1 1 ], s51 )
inv_dd51 = invderdenSMO( [ 1 1 1 ], s51 )

vals, vecs = Arpack.eigs(inv_dd51, nev=10, which=:LM)

# @time res = NullSolvers.solve(L51, :SVDSolver); # VERY SLOW
@time res = NullSolvers.solve(S51, :ArpackSolver);

@time res = NullSolvers.solve(S51, :LUSolver; tol=1e-6);
# TensorIO.sidebyside( S51, ITensors.ITensor( reshape( res.vecs[:,1], size(s51)... ), inds(s51)... ); left_title="Original", right_title