
function testFaceDiagonalTensors(val::Integer; sym::Vector{<:Integer}=Int64[])
    dims = [rand(5:10) for i=1:val]
    dims[2]=dims[1]

    # println("Stratification of Diagonal Tensors of size $dims and $sym")
    @testset "Stratification of Face Diagonal Tensors of size $dims and $sym" begin
        @show dims
        # set up frames
        frames = dims .|> ITensors.Index
        #setup deltaes
        deltas = dims .|> (n -> [ i for i= 1:n] ) 
        #define chisel 
        pch= Dleto.AdjointChisel(frames[1],frames[2])
        ch = Dleto.extend_chisel(pch,frames)
        @show ch

        testChiselDelta(ch,deltas;sym=sym)
    end
end



@testset "Stratification of Face Diagonal Tensors" begin
    for val=3:4
        for _ = 1:10
            testFaceDiagonalTensors(val)
        end        
    end
end

