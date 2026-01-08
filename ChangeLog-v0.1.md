# Change Log

## File structure.

Grouped things in the base if they are general top-level structures that a user will need to interact with in some loose way to use Dleto.  This includes Tensors (both AbstractArrays and ITensors), Chisels, Operators, then Derivations and Densors.  Then for anything that needs multiple implementations, special algorithms, these will be folded into subfolders.

1. Base
    - Dleto.jl the module definition.
    - DletoExports.jl one place to collect all the public "exported" functions, this is how it is done in other Julia packages.  One thing I like about it is that it makes a single role, you aren't distracted by includes/imports/usings, just lists of what you export.    
    - DletoBase.jl general convenience extensions of ITensors and AbstractArray operations.
    - Chisels.jl general chisel interface.
    - Operators.jl general operator interface.
    - TransverseOperators.jl general transverse operator interface.
    - Derivations.jl general derivation interface.
    - Densors.jl general densor interface (this renames `stratify`)
2. ops
3. solvers
4. SylverLingin
5. util


## Base

 1. added `matched_idx[!]` to make local label change for ITensors.  Use this when we need `A*E` with `a in inds(A)` and `e in inds(E)`.  This does not perform the contraction but makes a new label `ae` shared by both `A` and `E`.  A default tag of "matched_idx" is added to record the operation.

 * engaged ---> support??
 * chisel matrix ----> space of linear maps
 * implement $\Gamma*p(Xs)$
 * implement $\Gamma*P(Xs)$

 ## Ops
 added dualize/unsafe-dualize for operators
 fixed TransverseOpSymetries