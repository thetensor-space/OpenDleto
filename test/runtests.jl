#
# test/runtests.jl
# Run as:
#   Standalone:   julia --project test/runtests.jl
#   With Test:    JULIA_TEST_MODE=test julia --project test/runtests.jl
#   Pkg.test:     ] test  (Pkg sets up project; you can also export ENV variable if you prefer)


using Test
using Dleto
using ITensors

# --- Select mode via environment variable ---
const TEST_MODE = get(ENV, "JULIA_TEST_MODE", "assert") == "test"


include( "TestDletoBase.jl" )
include( "TestChisels.jl" )
include( "TestOperators.jl" )
include( "TestTransverseOps.jl" )
include( "TestTensorSynthesis.jl" )
include( "TestTensorDensity.jl" )
include( "TestFastDer3Valent.jl" )
include( "TestQuickDerN.jl" )
include( "TestAutoDer.jl" )
include( "TestDerivationLaws.jl" )
include( "TestNullVerdict.jl" )


include( "TestSylverLining.jl" )

# TestDerivations.jl lives in test/old-tests/ and is stale against the current
# API; it is not wired in.  See docs/review/Refactor-Plan.md Phase 0.


# if TEST_MODE
#     # Run all tests in test mode
#     @testset "Dleto.jl Tests" begin
#         # @testset "Testing Dleto Multiplication" begin
#         #     @test testMultiplication()
#         # end
#         @testset "Testing Chisels" begin
#             include( "TestChisels.jl" )
#             # @test testAllChisels()
#         end
#     end
# else
#     # Run all tests in assert mode
#     # @assert testMultiplication() "Failed Dleto multiplication test."
#     @assert testAllChisels() "Failed Chisel tests."
# end

# println("✓ All Dleto tests passed!")
