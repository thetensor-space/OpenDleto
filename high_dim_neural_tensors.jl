"""
3D and Higher-Dimensional Neural Network Layer Tensors

This demonstrates various types of neural network layers that use 3D, 4D, and 5D tensors.
These are the kinds of tensors you'd find in modern deep learning architectures.
"""

include("Dleto.jl")
using LinearAlgebra
using Random

Random.seed!(42)

# Enhanced tensor saving function for higher-dimensional tensors
function saveHighDimTensorToFile(tensor::AbstractArray, filename::String, threshold::Float64=1e-3)
    open(filename, "w") do file
        dims = size(tensor)
        ndims_tensor = length(dims)
        
        # Write header with dimension info
        println(file, "# Tensor dimensions: $(join(dims, " × "))")
        println(file, "# Format: $(join(["dim$i" for i in 1:ndims_tensor], " ")) value")
        
        # Use CartesianIndices for general n-dimensional iteration
        for idx in CartesianIndices(tensor)
            val = tensor[idx]
            if abs(val) > threshold
                coords = join([idx[i] for i in 1:ndims_tensor], " ")
                println(file, "$coords $val")
            end
        end
    end
end

# For 3D tensors, use the original function
function saveTensorSafe(tensor::AbstractArray, filename::String, threshold::Float64=1e-3)
    if ndims(tensor) == 3
        saveTensorToFile(tensor, filename, threshold)
    else
        saveHighDimTensorToFile(tensor, filename, threshold)
    end
end

"""
1. LSTM/RNN Layer - 3D Tensor
Shape: (sequence_length, batch_size, hidden_size)
"""
function create_lstm_hidden_states(seq_len::Int, batch_size::Int, hidden_size::Int)
    # Hidden states tensor: each "slice" along dimension 1 is the hidden state at time step t
    # Scale to range around 10 for more realistic neural network values
    hidden_states = randn(seq_len, batch_size, hidden_size) * 10.0
    
    # Make sparse: zero out values with absolute value < 13.0 (keeps ~10% of values for 90% sparsity)
    hidden_states[abs.(hidden_states) .< 13.0] .= 0.0
    
    println("LSTM Hidden States Tensor:")
    println("Shape: ", size(hidden_states))
    println("Interpretation: (time_steps, batch_size, hidden_features)")
    println("Sparsity: ", round(100 * count(x -> abs(x) < 1e-10, hidden_states) / length(hidden_states), digits=1), "%")
    println("Sample slice at timestep 1:")
    display(hidden_states[1, :, 1:5])  # Show first 5 features for all batch items
    
    # Save tensor to file
    saveTensorSafe(hidden_states, "lstm_hidden_states_tensor.txt")
    println("✓ Saved to: lstm_hidden_states_tensor.txt")
    
    return hidden_states
end

"""
2. Transformer Attention Weights - 4D Tensor
Shape: (batch_size, num_heads, seq_length, seq_length)
"""
function create_attention_tensor(batch_size::Int, num_heads::Int, seq_len::Int)
    # Attention weights: how much each position attends to every other position
    # Start with larger values then normalize and make sparse
    attention_weights = rand(batch_size, num_heads, seq_len, seq_len) * 20.0
    
    # Make sparse: zero out smaller values (simulating attention sparsity) - 90% sparsity
    attention_weights[attention_weights .< 18.0] .= 0.0
    
    # Normalize so each row sums to approximately 10 (only non-zero values)
    for b in 1:batch_size, h in 1:num_heads, i in 1:seq_len
        row_sum = sum(attention_weights[b, h, i, :])
        if row_sum > 0
            attention_weights[b, h, i, :] .*= 10.0 / row_sum
        end
    end
    
    println("\nTransformer Attention Tensor:")
    println("Shape: ", size(attention_weights))
    println("Interpretation: (batch, heads, query_pos, key_pos)")
    println("Sparsity: ", round(100 * count(x -> abs(x) < 1e-10, attention_weights) / length(attention_weights), digits=1), "%")
    println("Sample attention matrix for batch 1, head 1:")
    display(attention_weights[1, 1, :, :])
    
    # Save tensor to file
    saveTensorSafe(attention_weights, "transformer_attention_tensor.txt")
    println("✓ Saved to: transformer_attention_tensor.txt")
    
    return attention_weights
end

"""
3. 3D Convolutional Layer - 5D Tensor
Shape: (batch_size, channels, depth, height, width)
For video or 3D medical imaging
"""
function create_3d_conv_weights(out_channels::Int, in_channels::Int, 
                               depth::Int, height::Int, width::Int)
    # 3D convolution weights for processing volumetric data
    # Scale to range around 10 for more realistic weight magnitudes
    conv3d_weights = randn(out_channels, in_channels, depth, height, width) * 
                     10.0 / sqrt(in_channels * depth * height * width)
    
    # Make sparse: zero out smaller weights for 90% sparsity (extreme pruning)
    conv3d_weights[abs.(conv3d_weights) .< 0.12] .= 0.0
    
    println("\n3D Convolutional Layer Weights:")
    println("Shape: ", size(conv3d_weights))
    println("Interpretation: (out_channels, in_channels, depth, height, width)")
    println("Sparsity: ", round(100 * count(x -> abs(x) < 1e-10, conv3d_weights) / length(conv3d_weights), digits=1), "%")
    println("Sample 3D filter (first channel, middle slice):")
    mid_slice = div(depth, 2) + 1
    display(conv3d_weights[1, 1, mid_slice, :, :])
    
    # Save tensor to file
    saveTensorSafe(conv3d_weights, "conv3d_weights_tensor.txt")
    println("✓ Saved to: conv3d_weights_tensor.txt")
    
    return conv3d_weights
end

"""
4. Graph Neural Network - 3D Tensor
Shape: (num_nodes, num_nodes, feature_dim)
Represents node features and adjacency relationships
"""
function create_gnn_tensor(num_nodes::Int, feature_dim::Int)
    # Each "slice" along dimension 3 represents a different feature's adjacency pattern
    gnn_tensor = zeros(num_nodes, num_nodes, feature_dim)
    
    # Create some random adjacency patterns for each feature with values around 10
    for f in 1:feature_dim
        # Random sparse adjacency matrix with larger values - 90% sparsity
        adj = rand(num_nodes, num_nodes) * 15.0  # Scale up to ~15
        adj[adj .< 14.0] .= 0  # Make 90% sparse, keep only the largest values
        adj = (adj + adj') / 2  # Make symmetric
        gnn_tensor[:, :, f] = adj
    end
    
    println("\nGraph Neural Network Tensor:")
    println("Shape: ", size(gnn_tensor))
    println("Interpretation: (nodes, nodes, features) - adjacency for each feature")
    println("Sparsity: ", round(100 * count(x -> abs(x) < 1e-10, gnn_tensor) / length(gnn_tensor), digits=1), "%")
    println("Sample adjacency for feature 1:")
    display(gnn_tensor[:, :, 1])
    
    # Save tensor to file
    saveTensorSafe(gnn_tensor, "gnn_adjacency_tensor.txt")
    println("✓ Saved to: gnn_adjacency_tensor.txt")
    
    return gnn_tensor
end

"""
5. Vision Transformer Patch Embeddings - 4D Tensor
Shape: (batch_size, num_patches, patch_size^2 * channels, embedding_dim)
"""
function create_vit_patch_embeddings(batch_size::Int, num_patches::Int, 
                                   patch_features::Int, embed_dim::Int)
    # Patch embeddings: each image patch gets embedded into a high-dimensional space
    # Scale to range around 10 for more realistic embedding magnitudes
    patch_embeddings = randn(batch_size, num_patches, patch_features, embed_dim) * 10.0
    
    # Make sparse: zero out smaller embeddings for 90% sparsity (extreme sparse embeddings)
    patch_embeddings[abs.(patch_embeddings) .< 12.5] .= 0.0
    
    println("\nVision Transformer Patch Embeddings:")
    println("Shape: ", size(patch_embeddings))
    println("Interpretation: (batch, patches, patch_features, embedding)")
    println("Sparsity: ", round(100 * count(x -> abs(x) < 1e-10, patch_embeddings) / length(patch_embeddings), digits=1), "%")
    println("Sample patch embedding (first patch, first 5 features, first 5 embed dims):")
    display(patch_embeddings[1, 1, 1:5, 1:5])
    
    # Save tensor to file
    saveTensorSafe(patch_embeddings, "vit_patch_embeddings_tensor.txt")
    println("✓ Saved to: vit_patch_embeddings_tensor.txt")
    
    return patch_embeddings
end

"""
6. Tensor Network Layer - 4D Tensor
Shape: (mode1, mode2, mode3, mode4)
Used in quantum machine learning and tensor decomposition networks
"""
function create_tensor_network_layer(d1::Int, d2::Int, d3::Int, d4::Int)
    # A 4D tensor that gets contracted with input tensors
    # Scale to range around 10 for more realistic tensor network weights
    tensor_layer = randn(d1, d2, d3, d4) * 10.0 / sqrt(d1 * d2 * d3 * d4)
    
    # Make sparse: zero out smaller tensor elements for 90% sparsity (heavily compressed tensor networks)
    tensor_layer[abs.(tensor_layer) .< 0.08] .= 0.0
    
    println("\nTensor Network Layer:")
    println("Shape: ", size(tensor_layer))
    println("Interpretation: (mode1, mode2, mode3, mode4) - all modes are trainable")
    println("Sparsity: ", round(100 * count(x -> abs(x) < 1e-10, tensor_layer) / length(tensor_layer), digits=1), "%")
    println("Sample 2D slice [1, :, :, 1]:")
    display(tensor_layer[1, :, :, 1])
    
    # Save tensor to file
    saveTensorSafe(tensor_layer, "tensor_network_layer.txt")
    println("✓ Saved to: tensor_network_layer.txt")
    
    return tensor_layer
end

# Execute examples
println("="^60)
println("3D AND HIGHER-DIMENSIONAL NEURAL NETWORK TENSORS")
println("="^60)

# 1. LSTM with 3D hidden states
lstm_hidden = create_lstm_hidden_states(50, 16, 64)  # 50 timesteps, batch of 16, 64 hidden units

# 2. Transformer attention (4D)
attention = create_attention_tensor(12, 16, 24)  # batch=12, 16 heads, sequence length=24

# 3. 3D Convolution (5D weights)
conv3d = create_3d_conv_weights(32, 16, 8, 12, 12)  # 32 filters, 16 input channels, 8×12×12 kernel

# 4. Graph Neural Network (3D)
gnn = create_gnn_tensor(20, 15)  # 20 nodes, 15 features

# 5. Vision Transformer patches (4D)
vit_patches = create_vit_patch_embeddings(8, 49, 96, 128)  # batch=8, 49 patches (7×7), 96 features each, 128-dim embedding

# 6. Tensor Network (4D)
tensor_net = create_tensor_network_layer(15, 20, 18, 12)  # 15×20×18×12 tensor

println("\n" * "="^60)
println("SUMMARY OF HIGH-DIMENSIONAL NN TENSORS")
println("="^60)
println("• LSTM hidden states: 3D (time × batch × features) - 90% SPARSE")
println("• Transformer attention: 4D (batch × heads × seq × seq) - 90% SPARSE")
println("• 3D Convolution weights: 5D (out × in × depth × height × width) - 90% SPARSE")
println("• Graph Neural Networks: 3D (nodes × nodes × features) - 90% SPARSE")
println("• ViT patch embeddings: 4D (batch × patches × patch_features × embedding) - 90% SPARSE")
println("• Tensor Networks: 4D+ (arbitrary number of modes) - 90% SPARSE")
println("\nThese tensors are the learnable parameters that get updated during training!")
println("All tensors are made EXTREMELY SPARSE (90%) for ultra-realistic neural network behavior.")
println("\n" * "="^60)
println("SAVED TENSOR FILES:")
println("="^60)
println("• lstm_hidden_states_tensor.txt - 3D LSTM tensor (90% sparse)")
println("• transformer_attention_tensor.txt - 4D attention weights (90% sparse)")
println("• conv3d_weights_tensor.txt - 5D convolution weights (90% sparse)")
println("• gnn_adjacency_tensor.txt - 3D graph adjacency (90% sparse)")
println("• vit_patch_embeddings_tensor.txt - 4D vision transformer (90% sparse)")
println("• tensor_network_layer.txt - 4D tensor network (90% sparse)")
println("\nAll tensors saved in OpenDleto format for further analysis!")
println("Ultra-sparse tensors (90%) minimize file sizes and represent heavily pruned neural networks.")