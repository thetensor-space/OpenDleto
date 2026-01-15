function testPlaneTensors(axisdim::Vector{<:Number}; sym::Vector{<:Integer}=Int64[])
    dims = axisdim .|> x -> (2*x+1)
    # println("Stratification of Planar Tensors of size $dims and $sym")
    @testset "Stratification of Planar Tensors of size $dims and $sym" begin
        # set up frames
        frames = dims .|> ITensors.Index 
        #setup deltaes
        deltas = axisdim .|> (n -> [ i for i= (-n):n] ) 
        #define chisel 
        ch = Dleto.UniversalChisel(frames)
        testChiselDelta(ch,deltas;sym=sym)
    end
end



@testset "Stratification of Planar Tensors" begin
    for val=3:4
        for _ = 1:5
            sizes = [rand(4:7) for _=1:val]
            testPlaneTensors(sizes)
        end        
    end
end

@testset "Stratification of Planar Tensors With Symmetries" begin
    for sym in syms
        for _ = 1:5
            randomsizes = [rand(4:7) for _=1:10]
            sizes = [ randomsizes[abs(sym[i])] for i=1:length(sym) ]
            testPlaneTensors(sizes;sym=sym)
        end        
    end
end