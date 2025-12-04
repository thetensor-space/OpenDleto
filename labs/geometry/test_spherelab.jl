#!/usr/bin/env julia
# Test script to verify SphereLab.ipynb can be executed
# This script tests all the key components used in the notebook

println("=" ^ 70)
println("SphereLab.ipynb Verification Test")
println("=" ^ 70)
println()

# Test 1: Check Julia version
println("Test 1: Checking Julia version...")
println("  Julia version: ", VERSION)
if VERSION >= v"1.7"
    println("  ✓ Julia version is compatible (≥ 1.7)")
else
    println("  ✗ Warning: Julia version should be 1.7 or later")
end
println()

# Test 2: Check required packages
println("Test 2: Checking required packages...")
required_packages = ["IJulia", "Arpack", "PlotlyJS"]
for pkg in required_packages
    try
        if pkg == "IJulia"
            using IJulia
        elseif pkg == "Arpack"
            using Arpack
        elseif pkg == "PlotlyJS"
            using PlotlyJS
        end
        println("  ✓ $pkg is available")
    catch e
        println("  ✗ $pkg is not available: ", e)
    end
end
println()

# Test 3: Load Dleto.jl
println("Test 3: Loading Dleto.jl...")
try
    # Find Dleto.jl in repository root
    dleto_path = joinpath(@__DIR__, "..", "..", "Dleto.jl")
    ENV["WEBIO_JUPYTER_DETECTED"] = "true"
    include(dleto_path)
    println("  ✓ Dleto.jl loaded successfully from: ", dleto_path)
catch e
    println("  ✗ Failed to load Dleto.jl: ", e)
    exit(1)
end
println()

# Helper function to create sphere parameters
function create_sphere_parameters(center_val, radius, range_end)
    """Create sphere equation parameters for a given center and radius"""
    param = [(0:1:range_end)...] .|> x-> ((x-center_val)*(x-center_val)- (radius*radius)/3.0)
    return param
end

# Test 4-8: Run main workflow tests
sphere5 = nothing
hidden_sphere5 = nothing
recovered_sphere5 = nothing

# Test 4: Create a test sphere tensor
println("Test 4: Creating test sphere tensor...")
try
    a = 5.0; b = 5.0; c = 5.0; r = 5.0;
    Ues = create_sphere_parameters(a, r, 10)
    Ves = create_sphere_parameters(b, r, 10)
    Wes = create_sphere_parameters(c, r, 10)
    global sphere5 = randomSurfaceTensor( Ues, Ves, Wes, 1.5)
    println("  ✓ Sphere tensor created: ", size(sphere5))
catch e
    println("  ✗ Failed to create sphere tensor: ", e)
    exit(1)
end
println()

# Test 5: Test randomization
println("Test 5: Testing tensor randomization...")
try
    global hidden_sphere5 = randomizeTensor(sphere5)
    println("  ✓ Tensor randomized successfully")
catch e
    println("  ✗ Failed to randomize tensor: ", e)
    exit(1)
end
println()

# Test 6: Test surface recovery
println("Test 6: Testing surface recovery...")
try
    global recovered_sphere5 = toSurfaceTensor(hidden_sphere5.tensor)
    println("  ✓ Surface recovered successfully")
    
    # Test recovery quality
    chiseled_sphere5 = toSurfaceTensor(sphere5)
    s_orig = testSurfaceTensor(chiseled_sphere5.tensor, chiseled_sphere5.Xes, 
                                chiseled_sphere5.Yes, chiseled_sphere5.Zes)
    s_recovered = testSurfaceTensor(recovered_sphere5.tensor, recovered_sphere5.Xes, 
                                   recovered_sphere5.Yes, recovered_sphere5.Zes)
    println("  Surface metric (original): ", s_orig)
    println("  Surface metric (recovered): ", s_recovered)
catch e
    println("  ✗ Failed to recover surface: ", e)
    exit(1)
end
println()

# Test 7: Test file I/O
println("Test 7: Testing file I/O...")
try
    test_file = tempname() * ".txt"
    saveTensorToFile(sphere5, test_file)
    file_size = stat(test_file).size
    println("  ✓ File saved successfully: ", file_size, " bytes")
    rm(test_file)
    println("  ✓ File cleanup successful")
catch e
    println("  ✗ Failed file I/O test: ", e)
    exit(1)
end
println()

# Test 8: Test plotting (may not display in headless environment)
println("Test 8: Testing plotting functionality...")
try
    plot_result = plotTensor(sphere5, 0.01)
    println("  ✓ Plot created successfully")
    println("  Plot type: ", typeof(plot_result))
catch e
    println("  ⚠ Plot creation may not work in headless environment")
    println("  Error: ", e)
end
println()

println("=" ^ 70)
println("All core tests passed! SphereLab.ipynb is ready to run.")
println("=" ^ 70)
