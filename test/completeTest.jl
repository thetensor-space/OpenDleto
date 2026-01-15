function dimenstions(T::ITensor)
    return ITensors.dim.(vcat(inds(T)...))
end;
@testset "NonDegenerate" begin
    axisdim = [5,6,7]
    axisdim2 = [5,3,2]
    frames = ITensors.Index.(axisdim)
    frames2 = ITensors.Index.(axisdim2)
    T = random_itensor(frames)
    T2 = ITensor(frames2)
    Tsum = ITensors.directsum( T => inds(T), T2=>inds(T2))[1]
    OrthTO = Dleto.TransverseOps(Tsum,:OrthogonalOp)
    rT_f = Dleto.changeBasisRandom(Tsum,OrthTO; keep=true)
    rT = rT_f.T 
    @test dimenstions(rT) == axisdim + axisdim2
    reduced_f = Dleto.nondeg(rT, vcat(inds(rT)...))
    reducedT = reduced_f.T  
    @test dimenstions(reducedT) == axisdim 
end

@testset "Simple Example of stratificiation" begin
    #set axises
    # axisdim = [12,13,14]
    # frames = (axisdim .|> x -> ITensors.Index(2*x+1) ) 
    # F = Dleto.Framing{Index}(frames)
    
    # #generate tensor on a plane
    # deltas = axisdim .|> (n -> [ i for i= (-n):n] ) 
    # # @show deltas
    # ch = Dleto.UniversalChisel(frames)

    axisdim = [12,12,14]
    frames = (axisdim .|> x -> ITensors.Index(2*x+1) ) 
    F = Dleto.Framing{Index}(frames)
    
    #generate tensor on a plane
    deltas = axisdim .|> (n -> [ i for i= (-n):n] ) 
    # @show deltas
    ch = Dleto.Chisel([1.0 1.0 0.0],frames)


    T = Dleto.randomTensorChisel(deltas, frames, ch)

    #chnage basis of T using orhtogonal matrices
    OrthTO = Dleto.TransverseOps(T,:OrthogonalOp)
    rT = Dleto.changeBasisRandom(T,OrthTO; keep=true)

    # @show rT.T

    # sertup Transverse Operators
    DiagTO = Dleto.TransverseOps(T,:DiagonalOp)
    SymTO = Dleto.TransverseOps(T,:SymmetricOp)
    UniTO = Dleto.TransverseOps(T,:UniversalOp)

    #set up derivation problems
    DiagDP = Dleto.DerivationProblem(DiagTO,ch)
    SymDP = Dleto.DerivationProblem(SymTO,ch)
    UniDP = Dleto.DerivationProblem(UniTO,ch)

    #verify the expected number of solutions
    # @test Dleto.dimTrivialDerivations(DiagDP)==2
    # @test Dleto.dimTrivialDerivations(SymDP)==2
    # @test Dleto.dimTrivialDerivations(UniDP)==2

    #set up derivation method
    DM = Dleto.SylverLinningMethod(Dleto.SVDSolver())

    #find diagonal derivations
    diagder = Dleto.derivations(DM,DiagDP,T)
    @test size(diagder,2) > Dleto.dimTrivialDerivations(DiagDP)
    #simplify tensor
    diag_s = Dleto.simplifyUsingDerivation(T,DiagDP,diagder)
    # @show diag_s.T
    # @show diag_s.deltas
    @test Dleto.normTensorChisel(diag_s.T, diag_s.deltas, ch).norm < 1e-5

    #find symmetric derivations
    symder = Dleto.derivations(DM,SymDP,T)
    @test size(symder,2) > Dleto.dimTrivialDerivations(SymDP)
    sym_s = Dleto.simplifyUsingDerivation(T,SymDP,symder)
    # @show sym_s.T
    @test Dleto.normTensorChisel(sym_s.T, sym_s.deltas, ch).norm< 1e-5

    #find all derivations
    unider = Dleto.derivations(DM,UniDP,T)
    @test size(unider,2) > Dleto.dimTrivialDerivations(UniDP)
    uni_s = Dleto.simplifyUsingDerivation(T,UniDP,unider)
    # @show uni_s.T
    @test Dleto.normTensorChisel(uni_s.T, uni_s.deltas, ch).norm < 1e-5


    #find symmetric derivations for randomized tensor
    rsymder = Dleto.derivations(DM,SymDP,rT.T)
    @test size(rsymder,2) > Dleto.dimTrivialDerivations(SymDP)
    rsym_s = Dleto.simplifyUsingDerivation(rT.T,SymDP,rsymder)
    # @show rsym_s.T
    @test Dleto.normTensorChisel(rsym_s.T, rsym_s.deltas, ch).norm < 1e-5

    #find all derivations for randomized tensor
    runider = Dleto.derivations(DM,UniDP,rT.T)
    @test size(runider,2) > Dleto.dimTrivialDerivations(UniDP)
    runi_s = Dleto.simplifyUsingDerivation(rT.T,UniDP,runider)
    # @show runi_s.T
    @test Dleto.normTensorChisel(runi_s.T, runi_s.deltas, ch).norm < 1e-5
end