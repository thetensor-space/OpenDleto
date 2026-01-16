
#
# Strata Dleto: Derivation Problem
#   Algorithms for solving Sylvester equations arising in chiseling.
# -----------------------------------------------------------------------------
# Copyright 2022-2026 Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the “Software”), 
# to deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.
# -----------------------------------------------------------------------------

# """
#     Derivations Methods

#     Interface to methods for solving Sylvester equations arising in chiseling.

# """
"""
    DerivationProblem

    A combination of a linear global operator chisel and 
    framings which are compatible
"""
struct DerivationProblem 
    ch          ::Chisel
    TOp         ::TransverseOps
    frames      ::Framed
    framesTemp  ::Framed
    DerivationProblem(TOp:: TransverseOps, ch::Chisel) =(
        @assert isLinear(TOp.LOps) "Only linear constraints";
        # needs to throw warning if some of the columns of the chise get removed
        return new(extend_chisel(ch, TOp.frames), TOp, TOp.frames, TOp.framesTemp )
    )
end;
## inner construct ensures that the framing of the chisel and the TOps are the same


function reduce(DP::DerivationProblem)::NamedTuple{(:DP,:reduce_map),Tuple{DerivationProblem, LinearMaps.LinearMap}}
    eng = engaged_axis(DP.ch)
    rch = reduceBy(DP.ch,eng)
    rTOp=reduceBy(DP.TOp,eng)
    rDP = DerivationProblem(rTOp.rTOp,rch)
    return (DP = rDP, reduce_map=rTOp.reduce_map)
end; 

"""
    Computes the dimension the space of trivial derivations 
"""
function dimTrivialDerivations(DP::DerivationProblem)::Integer
### TODO impement this function 
    A = scalarsMatrix(DP.TOp.LOps) * LinearAlgebra.transpose(DP.ch.ch_m)
    M = A * LinearAlgebra.transpose(A)
    eigens = LinearAlgebra.eigen( LinearAlgebra.Symmetric(M) )
    small = [ eigens.values[i] < 1e-4 for i = 1:size(M,1)]
    return sum(small)
end;