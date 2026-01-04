#
# Test SylverLining
#
using Dleto
using ITensors
using LinearAlgebra
using LinearMaps

LΩs=[ LocalUniversalOps(), LocalDiagonalOps(), LocalSymmetricOps(), LocalAntiSymmetricOps(), LocalScalarOps(), LocalEmptyOps() ];
LΩs=[ LocalUniversalOps(), LocalDiagonalOps(), LocalSymmetricOps(), LocalAntiSymmetricOps() ];
syms = [ [1,1,1,1], [1,2,1,2,2,1], [1,2,3,-1,2,3], [1,2,-2,-2,5,-5,-1,-1], [1,-1,3,3,-3,1,7,8] ]

#only tests the maps are transpose to each other and that the composition is OK
#the test are as function since I am lazy to figure our how linear maps work

function testTranspose(f::LinearMap, num::Integer)
    n = size(f)[2]
    m = size(f)[1]
    # n < 3 && return true
    for i = 1:num
        a=rand(size(f)[2])
        B=rand(size(f)[1])
        A=f(a)
        b=f'(B)
        @assert isapprox(LinearAlgebra.dot(a,b),LinearAlgebra.dot(A,B)) "Failed Transpose dims $n $m" 
    end
    return true; 
end;

function testComposition(f::LinearMap, g::LinearMap, num::Integer)
    n = size(f)[2]
    m = size(f)[1]
    # n < 3 && return true
    for i = 1:num
        a=rand(size(f)[2])
        A=f(a)
        AA=f'(A)
        C=g(a)
        @assert isapprox(C,AA) "Failed Composition dims $n $m" 
    end
    return true; 
end;

function testGlobalOpChisel(Ω ::AbstractGlobalOps, ch::Matrix, num::Integer)
    # @show ch
    @testset "Testing with $ch" begin
        for i = 1:num
            Γ = random_itensor(frames(Ω))
            # ester, sylve, sylvester, opdim, densor_dim, derdensor_map, densor_map = sylvesterLM(Ω, ch, Γ)
            derdensor_map, densor_map = sylvesterLM(Ω, ch, Γ)
            n = size(densor_map)[2]
            m = size(densor_map)[1]
            if (n < 3) && isa(Ω, GlobalOpsSymmetries)
                # @show ch
                @show engaged(ch)
                @show Ω.syms
                @show Ω.duals
                rΩ = reduceByEngaged(Ω, engaged(ch))
                # @show rΩ.syms
                # @show rΩ.duals
                @show rΩ.axisDims
                @show rΩ.localOps
                @show globalDim(rΩ)
            end 
            @test testTranspose(densor_map, 100)
            @test testComposition(densor_map, derdensor_map, 100)
        end
    end
end

function testGlobalOp(Ω ::AbstractGlobalOps)
    # @show Ω
    d= globalDim(Ω)
    ntimes=5
    @testset "Testing Ω with dim $d" begin
        #Random Tucher  
        for i = 1:ntimes
            eng =  [ rand(1:10) % 2 == 1 for x in 1:valency(Ω) ]
            if sum(eng) != 0
                ch = TuckerChisel(eng)
                testGlobalOpChisel(Ω,ch,ntimes) 
            end
        end
        #Random Derivation  
        for i = 1:ntimes
            eng =  [ rand(1:10) % 2 == 1 for x in 1:valency(Ω) ]
            if sum(eng) != 0
                ch = UniversalChisel(eng)
                testGlobalOpChisel(Ω,ch,ntimes) 
            end
        end
        #Random Centroid  
        for i = 1:ntimes
            eng =  [ rand(1:10) % 2 == 1 for x in 1:valency(Ω) ]
            if sum(eng) > 1
                ch = CentroidChisel(eng)
                testGlobalOpChisel(Ω,ch,ntimes) 
            end
        end
    end
end

@testset "SylverLining Independent Tests" begin
    for val = 3:5
        @testset "Valency $val Tests" begin
            for i = 1:5
                axisdim = rand(2:10, val)
                frames = axisdim .|>  (i-> Index(i,"dim $i"))
                localops = rand(1:length(LΩs), val) .|> (i-> LΩs[i] )
                Ω = GlobalOpsIndependant(frames,localops)
                testGlobalOp(Ω)
            end 
        end
    end
end

@testset "SylverLining Trivial Symmetry Tests" begin
    for val = 3:5
        @testset "Valency $val Tests" begin
            for i = 1:5
                axisdim = rand(2:10, val)
                frames = axisdim .|>  (i-> Index(i,"dim $i"))
                localops = rand(1:length(LΩs), val) .|> (i-> LΩs[i] )
                Ω = GlobalOpsSymmetries(frames,localops,[i for i=1:val] )
                testGlobalOp(Ω)
            end 
        end
    end
end

val = 10
for sym in syms 
    @testset "SylverLining Symmetry $sym Tests" begin
        for i = 1:2
            # axisdim = rand(2:15, val)
            axisdim = rand(3:5, val)
            frames = axisdim .|>  (i-> Index(i,"dim $i"))
            localops = rand(1:length(LΩs), val) .|> (i-> LΩs[i] )
            Ω = GlobalOpsSymmetries(
                [ prime(frames[abs(sym[i])],i) for i=1:length(sym)],
                [ localops[abs(sym[i])] for i=1:length(sym)],
                sym)
            @show sym
            @show Ω.axisDims
            @show Ω.frames
            @show Ω.localOps
            @show Ω.globalDim
            testGlobalOp(Ω)
        end
    end 
end