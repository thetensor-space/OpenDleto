"""
Neural Network Layer Tensor Example

This file demonstrates what a tensor representing a layer in a neural network 
would look like. While the main OpenDleto codebase focuses on stratified tensor 
decomposition, here we show how tensors can represent neural network parameters.
"""

using LinearAlgebra
using Random

# Set seed for reproducible results
Random.seed!(42)

"""
Simple Dense Layer Representation as a 2D Tensor (Weight Matrix)
"""
function create_dense_layer(input_size::Int, output_size::Int)
    # Weight matrix: each row represents connections from all inputs to one output neuron
    weights = randn(output_size, input_size) * sqrt(2.0 / input_size)  # Xavier initialization
    bias = zeros(output_size)
    
    return (weights=weights, bias=bias)
end

"""
Convolutional Layer as a 4D Tensor
For a convolutional layer: (output_channels, input_channels, kernel_height, kernel_width)
"""
function create_conv_layer(input_channels::Int, output_channels::Int, 
                          kernel_height::Int, kernel_width::Int)
    # 4D tensor representing convolutional weights
    conv_weights = randn(output_channels, input_channels, kernel_height, kernel_width) * 
                   sqrt(2.0 / (input_channels * kernel_height * kernel_width))
    bias = zeros(output_channels)
    
    return (weights=conv_weights, bias=bias)
end

"""
Example: Multi-Layer Perceptron represented as a collection of 2D tensors
"""
function create_mlp_layers()
    # Input layer to first hidden layer (784 -> 128)
    layer1 = create_dense_layer(784, 128)  # For MNIST-like input
    
    # First hidden to second hidden layer (128 -> 64)
    layer2 = create_dense_layer(128, 64)
    
    # Second hidden to output layer (64 -> 10)
    layer3 = create_dense_layer(64, 10)    # For 10-class classification
    
    return [layer1, layer2, layer3]
end

"""
Example: Simple forward pass using the layer tensors
"""
function forward_pass(x, layers)
    activation = x
    
    for layer in layers
        # Linear transformation: y = W * x + b
        activation = layer.weights * activation + layer.bias
        
        # Apply ReLU activation (except for last layer)
        if layer != layers[end]
            activation = max.(0, activation)  # ReLU
        end
    end
    
    return activation
end

# Example usage:
println("Creating neural network layer tensors...")

# Create a simple MLP
mlp_layers = create_mlp_layers()

println("\nLayer 1 (Input → Hidden1):")
println("Weight tensor shape: ", size(mlp_layers[1].weights))
println("Bias vector shape: ", size(mlp_layers[1].bias))
println("Sample weights (first 3×3):")
display(mlp_layers[1].weights[1:3, 1:3])

println("\nLayer 2 (Hidden1 → Hidden2):")
println("Weight tensor shape: ", size(mlp_layers[2].weights))
println("Sample weights (first 3×3):")
display(mlp_layers[2].weights[1:3, 1:3])

println("\nLayer 3 (Hidden2 → Output):")
println("Weight tensor shape: ", size(mlp_layers[3].weights))
println("Sample weights (first 3×3):")
display(mlp_layers[3].weights[1:3, 1:3])

# Create a convolutional layer example
println("\n" * "="^50)
println("Convolutional Layer Example")
println("="^50)

conv_layer = create_conv_layer(3, 32, 3, 3)  # RGB input, 32 filters, 3×3 kernels
println("Conv layer weight tensor shape: ", size(conv_layer.weights))
println("This represents: (32 output channels, 3 input channels, 3×3 kernel)")
println("Sample filter (first channel of first filter):")
display(conv_layer.weights[1, 1, :, :])

# Demonstrate forward pass with dummy data
println("\n" * "="^50)
println("Forward Pass Example")
println("="^50)

# Create dummy input (flattened 28×28 image)
dummy_input = randn(784)
output = forward_pass(dummy_input, mlp_layers)
println("Input shape: ", size(dummy_input))
println("Output shape: ", size(output))
println("Output values: ", output)

println("\n" * "="^50)
println("Key Points about Neural Network Tensors:")
println("="^50)
println("1. Dense layers use 2D tensors (matrices) for weights")
println("2. Convolutional layers use 4D tensors")
println("3. Each tensor represents learnable parameters")
println("4. The tensor values are what get updated during training")
println("5. These tensors define the network's computational graph")