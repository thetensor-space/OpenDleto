
function testFaceDiagonalTensors(val::Integer; sym::Vector{<:Integer}=Int64[])
    dims = [rand(5:10) for i=1:val]
    dims[2]=dims[1]
    if sym!=[]
        for i=1:length(sym)
            if sym[i]!=i
                dims[i] = dims[abs(sym[i])]
            end
        end
    end

    # println("Stratification of Diagonal Tensors of size $dims and $sym")
    @testset "Stratification of Face Diagonal Tensors of size $dims and $sym" begin
        # @show dims
        # set up frames
        frames = dims .|> ITensors.Index
        #setup deltaes
        deltas = dims .|> (n -> [ i for i= 1:n] ) 
        #define chisel 
        pch= Dleto.AdjointChisel(frames[1],frames[2])
        ch = Dleto.extend_chisel(pch,frames)
        # @show ch

        testChiselDelta(ch,deltas;sym=sym)
    end
end

function testFaceBlockDiagonalTensors(numblock::Integer, val::Integer; sym::Vector{<:Integer}=Int64[])
    deltas = [vcat([[ i for _=1:rand(2:5)] for i=1:numblock]...) for _=1:val]
    if sym!=[]
        for i=1:length(sym)
            if sym[i]!=i
                deltas[i] = deltas[abs(sym[i])]
            end
        end
    end
    frames = deltas .|>  x-> ITensors.Index(length(x))


    # println("Stratification of Diagonal Tensors of size $dims and $sym")
    @testset "Stratification of Face Block Diagonal Tensors" begin
        frames = deltas .|>  x-> ITensors.Index(length(x))

        #define chisel 
        pch= Dleto.AdjointChisel(frames[1],frames[2])
        ch = Dleto.extend_chisel(pch,frames)
        # @show ch

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

@testset "Stratification of Face Diagonal Tensors With Symmetries" begin
    for sym in syms
        for _ = 1:5
            testFaceDiagonalTensors(length(sym); sym=sym)
        end        
    end
end


@testset "Stratification of Face Block Diagonal Tensors" begin
    for val=3:4
        for _ = 1:10
            numblock = rand(3:5)
            testFaceBlockDiagonalTensors(numblock,val)
        end        
    end
end

@testset "Stratification of Face Block Diagonal Tensors With Symmetries" begin
    for sym in syms
        for _ = 1:5
            numblock = rand(3:5)
            testFaceBlockDiagonalTensors(numblock,length(sym); sym=sym)
        end        
    end
end
