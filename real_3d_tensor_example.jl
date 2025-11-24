"""
Real 3D Tensor Example from OpenDleto Workspace

This loads and examines actual 3D tensors from the OpenDleto hypergraph analysis,
showing how they could represent neural network layers.
"""

include("Dleto.jl")

# Load a real 3D tensor from the workspace
println("Loading 3D tensor from OpenDleto workspace...")

# Load hypergraph tensor
hypergraph_tensor = loadTensorFromFile("examples/Hypergraph-Tensor.txt")
println("Hypergraph tensor shape: ", size(hypergraph_tensor))
println("This is a real 3D tensor: (", size(hypergraph_tensor)[1], " × ", 
        size(hypergraph_tensor)[2], " × ", size(hypergraph_tensor)[3], ")")

# Show some sample values
println("\nSample slice hypergraph_tensor[:, :, 1]:")
display(hypergraph_tensor[:, :, 1])

println("\nSample slice hypergraph_tensor[:, :, 2]:")
display(hypergraph_tensor[:, :, 2])

# Transform it into a stratified tensor (another 3D tensor)
println("\n" * "="^50)
println("Creating stratified tensor (surface detection)...")
strata_result = toSurfaceTensor(hypergraph_tensor)
strata_tensor = strata_result.tensor

println("Stratified tensor shape: ", size(strata_tensor))
println("Sample slice strata_tensor[:, :, 1]:")
display(strata_tensor[:, :, 1])

# Create a synthetic neural network-style 3D tensor based on the dimensions
println("\n" * "="^50)
println("Neural Network Interpretation of 3D Tensor:")
println("="^50)

dims = size(hypergraph_tensor)
println("If this were a neural network layer:")
println("• Dimension 1 (", dims[1], "): Could be input features")
println("• Dimension 2 (", dims[2], "): Could be hidden units")  
println("• Dimension 3 (", dims[3], "): Could be output classes or time steps")
println()
println("This 3D tensor could represent:")
println("1. Weight tensor for a multi-task network (features × hidden × tasks)")
println("2. RNN layer weights (input × hidden × time_steps)")  
println("3. Multi-head attention (queries × keys × heads)")
println("4. Graph convolution (nodes × features × layers)")

# Show sparsity pattern
nonzero_count = count(x -> abs(x) > 1e-10, hypergraph_tensor)
total_elements = length(hypergraph_tensor)
sparsity = 1.0 - (nonzero_count / total_elements)

println("\nTensor statistics:")
println("• Total elements: ", total_elements)
println("• Non-zero elements: ", nonzero_count) 
println("• Sparsity: ", round(sparsity * 100, digits=2), "%")
println("• Max value: ", maximum(hypergraph_tensor))
println("• Min value: ", minimum(hypergraph_tensor))
using Statistics
println("• Mean: ", round(mean(hypergraph_tensor), digits=6))

# Show how this could be used in a neural network forward pass
println("\n" * "="^50)
println("Example: Using 3D tensor in neural computation")
println("="^50)

# Simulate input
input_vec = randn(dims[1])  
println("Input vector shape: ", size(input_vec))

# Contract with tensor along first dimension (like matrix multiplication)
output = zeros(dims[2], dims[3])
for j in 1:dims[2], k in 1:dims[3]
    output[j, k] = sum(input_vec[i] * hypergraph_tensor[i, j, k] for i in 1:dims[1])
end

println("Output tensor shape: ", size(output))
println("Sample output values:")
display(output[1:min(5, dims[2]), 1:min(5, dims[3])])

println("\nThis demonstrates how a 3D tensor serves as learnable parameters")
println("that transform input data through tensor contractions!")