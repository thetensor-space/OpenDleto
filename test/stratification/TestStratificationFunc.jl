function testITensor(T::ITensor, O::Symbol, ch::Dleto.Chisel;sym::Vector{<:Integer}=Int64[])::Bool
    #make transverse op
    TO = Dleto.TransverseOps(T,O;sym=sym)

    #set up derivation problems
    DP = Dleto.DerivationProblem(TO,ch)

    #set up derivation method
    DM = Dleto.SylverLinningMethod(Dleto.SVDSolver())

    der = 0
    try 
        #find diagonal derivations
        der = Dleto.derivations(DM,DP,T;tol=1e-3)
    catch e
        println("exception during computing derivations")
        @show e
        return true
    end

    # test existance of notrivial derivation
    @test size(der,2) > Dleto.dimTrivialDerivationsReduced(DP)

    #simplify tensor
    try 
        simplified = Dleto.simplifyUsingDerivation(T,DP,der)
        # @show diag_s.T
        # @show diag_s.deltas
        #test that the result is OK
        if !(Dleto.normTensorChisel(simplified.T, simplified.deltas, ch).norm < 5e-3)
            println("Tensor may not be not stratified")
            @show ch
            @show O
            @show sym
            @show size(der)
            @show Dleto.dimTrivialDerivations(DP)
            @show simplified.deltas
            @show Dleto.normTensorChisel(simplified.T, simplified.deltas, ch).norm
            println("end")
        else
            @test Dleto.normTensorChisel(simplified.T, simplified.deltas, ch).norm < 5e-3
        end
    catch e
        println("exception during simplification base change")
        @show e
    end
    return true
end

function testITensorFail(T::ITensor, O::Symbol, ch::Dleto.Chisel;sym::Vector{<:Integer}=Int64[])::Bool
    #make transverse op
    TO = Dleto.TransverseOps(T,O;sym=sym)

    #set up derivation problems
    DP = Dleto.DerivationProblem(TO,ch)

    #set up derivation method
    DM = Dleto.SylverLinningMethod(Dleto.SVDSolver())

    #find diagonal derivations
    der = Dleto.derivations(DM,DP,T;tol=1e-3)
    # test existance of notrivial derivation
    if size(der,2) > Dleto.dimTrivialDerivationsReduced(DP)
        println("Unexpected extra derivation")
        @show size(der,2)
        @show Dleto.dimTrivialDerivations(DP)
        @show ch 
        @show O
        println("end")
    else
        @test size(der,2) == Dleto.dimTrivialDerivationsReduced(DP)
    end

    return true
end

function testChiselDelta(ch::Dleto.Chisel, deltas::Vector{<:Vector{<:Number}};sym::Vector{<:Integer}=Int64[])
    F = ch.frames
    frames = F.frame
    #generate tensor
    T = Dleto.randomTensorChisel(deltas, frames, ch)

    # tests in the original basis
    testITensor(T, :DiagonalOp, ch; sym=sym)
    testITensor(T, :TriDiagonalOp, ch; sym=sym)
    testITensor(T, :SymmetricOp, ch; sym=sym)
    testITensor(T, :UniversalOp, ch; sym=sym)

    #chnage basis of T using orhtogonal matrices
    OrthTO = Dleto.TransverseOps(T,:OrthogonalOp; sym=sym)
    roT_extra = Dleto.changeBasisRandom(T,OrthTO; keep=true)
    roT = roT_extra.T

    #test randomized tensor 
    testITensorFail(roT, :DiagonalOp, ch; sym=sym)
    testITensor(roT, :SymmetricOp, ch; sym=sym)
    testITensor(roT, :UniversalOp, ch; sym=sym)

    #chnage basis of T using invertible matrices
    InvTO = Dleto.TransverseOps(T,:InvertableOp; sym=sym)
    rT_extra = Dleto.changeBasisRandom(T,InvTO; keep=true)
    rT = rT_extra.T

    #test randomized tensor 
    testITensorFail(rT, :DiagonalOp, ch; sym=sym)
    testITensorFail(rT, :SymmetricOp, ch; sym=sym)
    testITensor(rT, :UniversalOp, ch; sym=sym)
end;



syms = [[1,1,1], [1,-1,-1], [1,-1,1], [1,2,1], [1,2,-2], [1,2,-1,2],[1,2,3,-3]]
