
function testBlockDiagonalTensors(numblock::Integer, val::Integer; sym::Vector{<:Integer}=Int64[])
    # println("Stratification of Block Diagonal Tensors with $numblock blocks valency $val and $sym")
    @testset "Stratification of Block Diagonal Tensors with $numblock blocks valency $val and $sym" begin
        #setup deltaes
        deltas = [vcat([[ i for _=1:rand(2:5)] for i=1:numblock]...) for _=1:val]
        if sym!=[]
            for i=1:length(sym)
                if sym[i]!=i
                    deltas[i] = deltas[abs(sym[i])]
                end
            end
        end
        # @show deltas .|> length
        # set up frames
        frames = deltas .|>  x-> ITensors.Index(length(x))
        #define chisel 
        ch = Dleto.CentroidChisel(frames)
        testChiselDelta(ch,deltas;sym=sym)
    end
end



@testset "Stratification of Block Diagonal Tensors" begin
    for val=3:4
        for _ = 1:5
            numblock = rand(3:5)
            testBlockDiagonalTensors(numblock,val)
        end        
    end
end


@testset "Stratification of Block Diagonal Tensors With Symmetries" begin
    for sym in syms
        for _ = 1:5
            numblock = rand(3:5)
            testBlockDiagonalTensors(numblock, length(sym); sym=sym)
        end        
    end
end
