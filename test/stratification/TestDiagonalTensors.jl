
function testDiagonalTensors(dim::Integer, val::Integer; sym::Vector{<:Integer}=Int64[])
    dims = [dim for i=1:val]
    # println("Stratification of Diagonal Tensors of size $dims and $sym")
    @testset "Stratification of Diagonal Tensors of size $dims and $sym" begin
        # set up frames
        frames = dims .|> ITensors.Index
        #setup deltaes
        deltas = dims .|> (n -> [ i for i= 1:n] ) 
        #define chisel 
        ch = Dleto.CentroidChisel(frames)
        testChiselDelta(ch,deltas;sym=sym)
    end
end



@testset "Stratification of Diagonal Tensors" begin
    for val=3:4
        for _ = 1:5
            dim = rand(7:10)
            sizes = [ dim for i=1:val ]
            testDiagonalTensors(dim,val)
        end        
    end
end


@testset "Stratification of Diagonal Tensors With Symmetries" begin
    for sym in syms
        for _ = 1:5
            dim = rand(7:10)
            testDiagonalTensors(dim, length(sym); sym=sym)
        end        
    end
end
