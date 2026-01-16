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

const test_stratification = true

const only_stratification = true

if test_stratification
    include("stratification/runtests.jl")
end

if !only_stratification
    include("solvers/runtests.jl") 
    include("chisels/runtests.jl") 
    include("util/runtests.jl")
    include("localops/runtests.jl") 
    include("ops/runtests.jl") 
end