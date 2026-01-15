"""
    Simplify a tensor using solution of a derivation problem
    input 
        T   ::ITensor
        DP  ::DerivationProblem
        ders::Matrix            -- solution of the derivation problem
    output named tuple
        T       :: ITensor -- transformed tensor 
        Xs      :: Vector{ITensor} -- transformation change of basis 
        Ds      :: Vector{ITensor} -- almost diagonal approximate derivation as tensors in the new basis 
        D_TOp   :: TransverseOps   -- new TransverseOps for Ds, 
        d_data  :: Vector{<:Number}-- data for Ds 
        T_TOp   :: TransverseOps   -- new TransverseOps for Xs, 
        t_data  :: Vector{<:Number}-- data for Xs 

"""
function simplifyUsingDerivation(T::ITensors.ITensor, DP::DerivationProblem, ders::Matrix) #::NamedTuple(...)
#TODO add out put in the signature
    @assert true compatability
    @assert true size of the ders

    #generate random vector in der
    n_ders = size(ders,2)
    coefs = randn(n_ders)
    der = ders*coefs

    #construct simplifed transverse ops 
    # res_LOP=simplifyTo(DP.TOp.LOps)
    # dLOps = res_LOP.D
    # tLOps = res_LOP.T
    # DTOp = TransverseOps(DTOps, DP.TOp.frames, DP.TOp.framesTemp)
    # TTOp = TransverseOps(TTOps, DP.TOp.frames, DP.TOp.framesTemp)
    res_TOP = simplifyTo(DP.TOp)
    DTOp = res_TOP.D
    TTOp = res_TOP.T
  
    #construct simplifed solution
    res_der = simplify(DP.TOp.LOps,der) 
    dder = res_der.d
    tder = res_der.t

    #transofrom tensor
    chTensor = changeBasis(T, TTOp, tder; keep=true)
    Ds = embedITensors(DTOp,dder)
    DMats = embedMatrices(DTOp.LOps,dder)
    # deltas = DMats .|> (M -> [M[i,i] for i=1:size(M,1)])
    deltas = diag.(DMats)
    # return output
    return (
            T= chTensor.T, 
            Xs = chTensor.Xs, 
            Ds = Ds, 
            D_TOp  = DTOp, 
            d_data = dder, 
            T_TOp  = TTOp, 
            t_data = tder,
            deltas = deltas
        )
end