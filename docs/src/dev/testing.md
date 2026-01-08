# Testing

Dleto.jl includes comprehensive tests to ensure correctness and mathematical rigor.

## Test Structure

- `test/TestOperators.jl` - Complete operator interface tests
- `test/TestChisels.jl` - Chiseling functionality tests  
- `test/TestDleto.jl` - Core functionality tests
- `test/TestTransverseOpsIndependent.jl` - Transverse operator tests

## Operator Testing

The operator tests verify all documented mathematical laws:

```julia
include("test/TestOperators.jl")
testAllOperators()  # Runs comprehensive operator law verification
```

### Test Coverage

- ✅ Basic interface requirements (`member`, `apply`, `unsafe_apply`)
- ✅ Linearity laws for tensor operations
- ✅ Consistency between safe and unsafe operations  
- ✅ Linear operator laws (additivity, scalar multiplication)
- ✅ Coapplication laws and duality
- ✅ Edge cases (identity, diagonal, zero tensors)
- ✅ Invalid operator detection
- ✅ Index collision handling

## Running Specific Tests

```julia
# Test just the basic interface
testOperatorInterface()

# Test mathematical laws
testOperatorLinearity()
testLinearOperatorLaws()

# Test coapplication
testCoapplyLinearity()
testCoapplyDuality()
```

## Writing New Tests

When adding new functionality, follow the established patterns:

1. Create descriptive test function names
2. Use informative assertion messages
3. Test both success and failure cases
4. Verify mathematical properties where applicable