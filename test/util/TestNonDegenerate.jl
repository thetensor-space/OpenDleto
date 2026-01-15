function dimenstions(T::ITensor)
    return ITensors.dim.(vcat(inds(T)...))
end;

function testNondegenerate(block::Vector{<:Integer},zblock::Vector{<:Integer})::Bool
    frames = ITensors.Index.(block)
    frames2 = ITensors.Index.(zblock)
    T = random_itensor(frames)
    T2 = ITensor(frames2)
    Tsum = ITensors.directsum( T => inds(T), T2=>inds(T2))[1]

    #random orthogonal change
    OrthTO = Dleto.TransverseOps(Tsum,:OrthogonalOp)
    roT_f = Dleto.changeBasisRandom(Tsum,OrthTO; keep=true)
    roT = roT_f.T 
    @test dimenstions(roT) == block + zblock
    reducedo_f = Dleto.nondeg(roT, vcat(inds(roT)...))
    reducedoT = reducedo_f.T  
    @test dimenstions(reducedoT) == block 

    #random invertible change
    InvTO = Dleto.TransverseOps(Tsum,:InvertableOp)
    rT_f = Dleto.changeBasisRandom(Tsum,InvTO; keep=true)
    rT = rT_f.T 
    @test dimenstions(rT) == block + zblock
    reduced_f = Dleto.nondeg(rT, vcat(inds(rT)...))
    reducedT = reduced_f.T  
    @test dimenstions(reducedT) == block
    return true 
end

@testset "NonDegenerate" begin
    for val=2:5
        for _=1:5
            blocksize = [rand(5:10) for _=1:val]
            if val==2
                blocksize[2] =blocksize[1] 
            end
            zeroblocksize = [rand(0:5) for _=1:val]
            testNondegenerate(blocksize, zeroblocksize)
        end
    end
end
