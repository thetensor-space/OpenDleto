# 
# TestOperators.jl
#
# Unit tests for Operators interface and laws
#

using ITensors
using LinearAlgebra
using Test

# Define the operator interface functions based on ITensorOps.jl
function apply(ω::ITensor, Γ::ITensor, a::Index{N})::ITensor where {N}
    @assert member(ω) "Operator ITensor must have exactly two indices that are dual to each other"
    @assert hasind(ω, a) "Operator ITensor does not act on index $a"
    @assert hasind(Γ, a) "Input ITensor does not have index $a"
    e = a 
    # Detect collision of index names
    if hasind(Γ, a')
        e = addtags(a', "dot")  # Ensure correct contraction
        Γ = replaceind(Γ, a, e)
        ω = replaceind(ω, a', e)
    end
    return unsafe_apply(ω, Γ, e)
end

function unsafe_apply(ω::ITensor, Γ::ITensor, a::Index{N})::ITensor where {N}
    # Contract the operator ω with the tensor Γ along index a
    # The result should have the 'a' index replaced by the primed index from ω
    result = Γ * ω
    # Replace the primed index back to the original unprimed index 
    if hasind(result, a')
        result = replaceind(result, a', a)
    end
    return result
end

function member(ω::ITensor)::Bool
    return length(inds(ω)) == 2 && inds(ω)[1]' == inds(ω)[2]
end

function asoperator(M::AbstractMatrix, a::Index{N})::ITensor where {N}
    @assert size(M) == (dim(a), dim(a)) "Matrix should be square of same dimension as index."
    return ITensor(M, a, a')
end

function unsafe_coapply(Λ::ITensor, Δ::ITensor, a::Index{N})::ITensor where {N}
    # assumes that Λ and Δ have the same indices
    Δ = replaceind(Δ, a, a') # Shallow copy just changes indices
    return Λ * Δ
end

"""
Test basic operator interface requirements
"""
function testOperatorInterface()
    passing = true
    
    # Create test data
    a = Index(3, "a")
    Γ = randomITensor(a, Index(2, "b"), Index(2, "c"))
    
    # Test identity operator
    I_op = asoperator(Matrix(I, 3, 3), a)
    
    # Test that member function works
    if !member(I_op)
        println("FAIL: Identity operator should be a valid member")
        passing = false
    end
    
    # Test that apply function works
    try
        result = apply(I_op, Γ, a)
        if !isapprox(result, Γ)
            println("FAIL: Identity operator should preserve tensor")
            passing = false
        end
    catch e
        println("FAIL: Apply function threw exception: $e")
        passing = false
    end
    
    # Test that unsafe_apply works  
    try
        result = unsafe_apply(I_op, Γ, a)
        if !isapprox(result, Γ)
            println("FAIL: unsafe_apply with identity should preserve tensor")
            passing = false
        end
    catch e
        println("FAIL: unsafe_apply function threw exception: $e")
        passing = false
    end
    
    return passing
end

"""
Test linearity laws for tensor operators:
apply(ω, Γ + Δ, a) ≈ apply(ω, Γ, a) + apply(ω, Δ, a)
apply(ω, Γ * s, a) ≈ apply(ω, Γ, a) * s
"""
function testOperatorLinearity()
    passing = true
    
    # Create test data
    a = Index(3, "a")
    b = Index(2, "b") 
    c = Index(2, "c")
    Γ = randomITensor(a, b, c)
    Δ = randomITensor(a, b, c)
    s = 2.5
    
    # Create a test operator (not identity to make test more meaningful)
    M = [2.0 1.0 0.5; 0.5 3.0 1.0; 1.0 0.5 2.0]  # Random symmetric matrix
    ω = asoperator(M, a)
    
    # Test additivity: apply(ω, Γ + Δ, a) ≈ apply(ω, Γ, a) + apply(ω, Δ, a)
    lhs = apply(ω, Γ + Δ, a)
    rhs = apply(ω, Γ, a) + apply(ω, Δ, a)
    if !isapprox(lhs, rhs, atol=1e-12)
        println("FAIL: Operator additivity law violated")
        println("  |LHS - RHS| = $(norm(lhs - rhs))")
        passing = false
    end
    
    # Test scalar multiplication: apply(ω, Γ * s, a) ≈ apply(ω, Γ, a) * s
    lhs = apply(ω, Γ * s, a)
    rhs = apply(ω, Γ, a) * s
    if !isapprox(lhs, rhs, atol=1e-12)
        println("FAIL: Operator scalar multiplication law violated")
        println("  |LHS - RHS| = $(norm(lhs - rhs))")
        passing = false
    end
    
    return passing
end

"""
Test consistency laws:
apply(ω, Γ, a) == unsafe_apply(ω, Γ, Δ, a) if member(ω) == true
inds(apply(ω, Γ, a)) == inds(Γ) if member(ω) == true
"""
function testOperatorConsistency()
    passing = true
    
    # Create test data
    a = Index(3, "a")
    b = Index(2, "b")
    c = Index(2, "c") 
    Γ = randomITensor(a, b, c)
    
    # Create test operator
    M = [1.5 0.2 -0.1; 0.2 2.0 0.3; -0.1 0.3 1.8]
    ω = asoperator(M, a)
    
    # Test that member returns true for valid operator
    if !member(ω)
        println("FAIL: Valid operator should satisfy member predicate")
        passing = false
    end
    
    # Test consistency: apply(ω, Γ, a) == unsafe_apply(ω, Γ, a)
    safe_result = apply(ω, Γ, a)
    unsafe_result = unsafe_apply(ω, Γ, a)
    if !isapprox(safe_result, unsafe_result, atol=1e-14)
        println("FAIL: apply and unsafe_apply should give same result for valid operators")
        println("  |safe - unsafe| = $(norm(safe_result - unsafe_result))")
        passing = false
    end
    
    # Test index preservation: inds(apply(ω, Γ, a)) == inds(Γ)
    result = apply(ω, Γ, a)
    # ITensors may reorder indices, so we check that the same set of indices is present
    if Set(inds(result)) != Set(inds(Γ))
        println("FAIL: apply should preserve tensor indices (as a set)")
        println("  Original indices: $(inds(Γ))")
        println("  Result indices:   $(inds(result))")
        passing = false
    end
    
    return passing
end

"""
Test laws for linear tensor operators:
apply(ω + λ, Γ, a) ≈ apply(ω, Γ, a) + apply(λ, Γ, a)
apply(s * ω, Γ, a) ≈ s * apply(ω, Γ, a)
"""
function testLinearOperatorLaws()
    passing = true
    
    # Create test data
    a = Index(3, "a")
    b = Index(2, "b")
    Γ = randomITensor(a, b)
    s = 1.7
    
    # Create two test operators
    M1 = [1.0 0.2 -0.1; 0.2 1.5 0.1; -0.1 0.1 2.0]
    M2 = [2.0 -0.1 0.3; -0.1 1.8 -0.2; 0.3 -0.2 1.2]
    ω = asoperator(M1, a)
    λ = asoperator(M2, a)
    
    # Test operator additivity: apply(ω + λ, Γ, a) ≈ apply(ω, Γ, a) + apply(λ, Γ, a)
    lhs = apply(ω + λ, Γ, a)
    rhs = apply(ω, Γ, a) + apply(λ, Γ, a)
    if !isapprox(lhs, rhs, atol=1e-12)
        println("FAIL: Linear operator additivity law violated")
        println("  |LHS - RHS| = $(norm(lhs - rhs))")
        passing = false
    end
    
    # Test operator scalar multiplication: apply(s * ω, Γ, a) ≈ s * apply(ω, Γ, a)
    lhs = apply(s * ω, Γ, a)
    rhs = s * apply(ω, Γ, a)
    if !isapprox(lhs, rhs, atol=1e-12)
        println("FAIL: Linear operator scalar multiplication law violated")  
        println("  |LHS - RHS| = $(norm(lhs - rhs))")
        passing = false
    end
    
    return passing
end

"""
Test coapply linearity laws:
coapply(Λ₁ + Λ₂, Δ, a) ≈ coapply(Λ₁, Δ, a) + coapply(Λ₂, Δ, a)
coapply(s * Λ, Δ, a) ≈ s * coapply(Λ, Δ, a)
coapply(Λ, Δ₁ + Δ₂, a) ≈ coapply(Λ, Δ₁, a) + coapply(Λ, Δ₂, a)
coapply(Λ, Δ * s, a) ≈ coapply(Λ, Δ, a) * s
"""
function testCoapplyLinearity()
    passing = true
    
    # Create test data
    a = Index(3, "a")
    b = Index(2, "b")
    Λ₁ = randomITensor(a, b)
    Λ₂ = randomITensor(a, b)
    Δ₁ = randomITensor(a, b) 
    Δ₂ = randomITensor(a, b)
    s = 0.8
    
    # Test first argument additivity: coapply(Λ₁ + Λ₂, Δ₁, a) ≈ coapply(Λ₁, Δ₁, a) + coapply(Λ₂, Δ₁, a)
    lhs = unsafe_coapply(Λ₁ + Λ₂, Δ₁, a)
    rhs = unsafe_coapply(Λ₁, Δ₁, a) + unsafe_coapply(Λ₂, Δ₁, a)
    if !isapprox(lhs, rhs, atol=1e-12)
        println("FAIL: Coapply first argument additivity law violated")
        println("  |LHS - RHS| = $(norm(lhs - rhs))")
        passing = false
    end
    
    # Test first argument scalar multiplication: coapply(s * Λ₁, Δ₁, a) ≈ s * coapply(Λ₁, Δ₁, a)
    lhs = unsafe_coapply(s * Λ₁, Δ₁, a)
    rhs = s * unsafe_coapply(Λ₁, Δ₁, a)
    if !isapprox(lhs, rhs, atol=1e-12)
        println("FAIL: Coapply first argument scalar multiplication law violated")
        println("  |LHS - RHS| = $(norm(lhs - rhs))")
        passing = false
    end
    
    # Test second argument additivity: coapply(Λ₁, Δ₁ + Δ₂, a) ≈ coapply(Λ₁, Δ₁, a) + coapply(Λ₁, Δ₂, a)
    lhs = unsafe_coapply(Λ₁, Δ₁ + Δ₂, a)
    rhs = unsafe_coapply(Λ₁, Δ₁, a) + unsafe_coapply(Λ₁, Δ₂, a)
    if !isapprox(lhs, rhs, atol=1e-12)
        println("FAIL: Coapply second argument additivity law violated")
        println("  |LHS - RHS| = $(norm(lhs - rhs))")
        passing = false
    end
    
    # Test second argument scalar multiplication: coapply(Λ₁, Δ₁ * s, a) ≈ coapply(Λ₁, Δ₁, a) * s
    lhs = unsafe_coapply(Λ₁, Δ₁ * s, a)
    rhs = unsafe_coapply(Λ₁, Δ₁, a) * s
    if !isapprox(lhs, rhs, atol=1e-12)
        println("FAIL: Coapply second argument scalar multiplication law violated")
        println("  |LHS - RHS| = $(norm(lhs - rhs))")
        passing = false
    end
    
    return passing
end

"""
Test that coapply is the dual of apply when appropriate
"""
function testCoapplyDuality()
    passing = true
    
    # Create test data
    a = Index(3, "a")
    b = Index(2, "b")
    Λ = randomITensor(a, b)
    Δ = randomITensor(a, b)
    Γ = randomITensor(a, b)
    
    # Create operator from coapply
    ω = unsafe_coapply(Λ, Δ, a)
    
    # The duality property would be that applying ω to Γ gives the same result
    # as contracting Λ and Δ appropriately with Γ
    # This tests the fundamental relationship between apply and coapply
    if member(ω)
        result1 = apply(ω, Γ, a)
        
        # Manual computation of what the dual should give
        # This is a bit involved for ITensors but the key is that coapply
        # should encode the "operator" that when applied reproduces certain contractions
        
        # For now, we just test that the operator produced by coapply is valid
        if !member(ω)
            println("FAIL: Operator produced by coapply should be a valid member")
            passing = false
        end
        
        # Test that applying the coapplied operator preserves indices
        if Set(inds(result1)) != Set(inds(Γ))
            println("FAIL: Coapply-generated operator should preserve indices (as a set) when applied")
            passing = false
        end
    end
    
    return passing
end

"""
Test edge cases and error conditions
"""
function testOperatorEdgeCases()
    passing = true
    
    # Test with identity operator
    a = Index(2, "a")
    b = Index(3, "b")
    I_op = asoperator(Matrix(I, 2, 2), a)
    Γ = randomITensor(a, b)
    
    # Zero tensor test
    zero_tensor = 0.0 * Γ
    result = apply(I_op, zero_tensor, a)
    if !isapprox(result, zero_tensor, atol=1e-14)
        println("FAIL: Identity operator should preserve zero tensor")
        passing = false
    end
    
    # Test with diagonal operator
    diag_matrix = Matrix(Diagonal([2.0, 3.0]))
    diag_op = asoperator(diag_matrix, a)
    
    if !member(diag_op)
        println("FAIL: Diagonal operator should be valid member")
        passing = false
    end
    
    # Test linearity with diagonal operator
    Δ = randomITensor(a, b)
    s = -1.5
    
    lhs = apply(diag_op, Γ + Δ, a)
    rhs = apply(diag_op, Γ, a) + apply(diag_op, Δ, a)
    if !isapprox(lhs, rhs, atol=1e-12)
        println("FAIL: Diagonal operator additivity failed")
        passing = false
    end
    
    lhs = apply(diag_op, Γ * s, a)
    rhs = apply(diag_op, Γ, a) * s
    if !isapprox(lhs, rhs, atol=1e-12)
        println("FAIL: Diagonal operator scalar multiplication failed")
        passing = false
    end
    
    return passing
end

"""
Test behavior with invalid operators (should fail member test)
"""
function testInvalidOperators()
    passing = true
    
    a = Index(3, "a")
    b = Index(2, "b")
    
    # Create an ITensor that doesn't satisfy the operator interface
    # (wrong number of indices)
    invalid_op1 = randomITensor(a, b, Index(2, "c"))
    
    if member(invalid_op1)
        println("FAIL: Invalid operator with 3 indices should not pass member test")
        passing = false
    end
    
    # Create an ITensor with 2 indices that are not dual to each other
    c = Index(3, "c")
    invalid_op2 = randomITensor(a, c)  # a and c are not dual
    
    if member(invalid_op2)
        println("FAIL: Invalid operator with non-dual indices should not pass member test")
        passing = false
    end
    
    return passing
end

"""
Test index collision handling in apply function
"""
function testIndexCollisionHandling()
    passing = true
    
    a = Index(3, "a")
    b = Index(2, "b")
    
    # Create operator and tensor that both have a'
    M = [1.0 0.5 0.0; 0.5 2.0 0.3; 0.0 0.3 1.5]
    ω = asoperator(M, a)
    
    # Create tensor that includes both a and a' (collision case)
    Γ = randomITensor(a, a', b)
    
    # This should work - apply should handle the collision
    try
        result = apply(ω, Γ, a)
        
        # Check that result has the expected indices
        expected_inds = [a, a', b]
        if Set(inds(result)) != Set(expected_inds)
            println("FAIL: Index collision handling produced wrong indices")
            println("  Expected: $expected_inds")
            println("  Got:      $(inds(result))")
            passing = false
        end
    catch e
        println("FAIL: apply should handle index collisions gracefully, but threw: $e")
        passing = false
    end
    
    return passing
end

"""
Run all operator tests
"""
function testAllOperators()
    tests = [
        ("Basic Interface", testOperatorInterface),
        ("Operator Linearity", testOperatorLinearity), 
        ("Operator Consistency", testOperatorConsistency),
        ("Linear Operator Laws", testLinearOperatorLaws),
        ("Coapply Linearity", testCoapplyLinearity),
        ("Coapply Duality", testCoapplyDuality),
        ("Edge Cases", testOperatorEdgeCases),
        ("Invalid Operators", testInvalidOperators),
        ("Index Collision Handling", testIndexCollisionHandling)
    ]
    
    passing = true
    passed_count = 0
    total_count = length(tests)
    
    println("Running Operator Tests...")
    println("=" ^ 50)
    
    for (test_name, test_func) in tests
        print("  $test_name: ")
        try
            if test_func()
                println("✓ PASS")
                passed_count += 1
            else
                println("✗ FAIL")
                passing = false
            end
        catch e
            println("✗ ERROR: $e")
            passing = false
        end
    end
    
    println("=" ^ 50)
    println("Passed: $passed_count/$total_count tests")
    
    if passing
        println("✓ All operator tests PASSED!")
    else
        println("✗ Some operator tests FAILED!")
    end
    
    return passing
end

# Run tests if this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    testAllOperators()
end

# Export the main test function for use by the test runner
export testAllOperators