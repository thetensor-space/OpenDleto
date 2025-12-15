
"""
    change_point_detection(values::AbstractVector{T}; threshold::Real=0.1, min_segment_length::Integer=3) :: Vector{Integer} where T <: Real

    Detects change points in an increasing sequence of floating point values.
    
    # Arguments
    - `values`: Vector of increasing floating point values
    - `threshold`: Minimum relative change in slope to consider a change point (default: 0.1 = 10%)
    - `min_segment_length`: Minimum number of points between change points (default: 3)
    
    # Returns
    - Vector of indices where change points occur
    
    # Example
    ```julia
    vals = [0.1, 0.2, 0.3, 0.31, 0.32, 0.33, 0.8, 0.9, 1.0]
    change_points = change_point_detection(vals, threshold=0.5)
    ```
"""
function change_point_detection(values::AbstractVector{T}; 
                               threshold::Real=0.1, 
                               min_segment_length::Integer=3) :: Vector{Integer} where T <: Real
    
    if length(values) < 2*min_segment_length + 1
        return Integer[]
    end
    
    # Compute first differences (slopes)
    slopes = diff(values)
    
    # Smooth slopes with a simple moving average to reduce noise
    window_size = max(2, min_segment_length ÷ 2)
    smoothed_slopes = similar(slopes)
    
    for i in 1:length(slopes)
        start_idx = max(1, i - window_size + 1)
        end_idx = min(length(slopes), i + window_size - 1)
        smoothed_slopes[i] = mean(slopes[start_idx:end_idx])
    end
    
    change_points = Integer[]
    last_change_point = 1
    
    for i in (min_segment_length+1):(length(slopes)-min_segment_length)
        # Skip if too close to last change point
        if i - last_change_point < min_segment_length
            continue
        end
        
        # Calculate average slope before and after potential change point
        left_window = max(1, i - min_segment_length)
        right_window = min(length(smoothed_slopes), i + min_segment_length - 1)
        
        left_slope = mean(smoothed_slopes[left_window:i-1])
        right_slope = mean(smoothed_slopes[i:right_window])
        
        # Avoid division by zero
        if left_slope ≈ 0.0
            left_slope = eps(T)
        end
        
        # Calculate relative change in slope
        relative_change = abs(right_slope - left_slope) / abs(left_slope)
        
        # Detect significant change
        if relative_change > threshold
            push!(change_points, i)
            last_change_point = i
        end
    end
    
    return change_points
end

"""
    plot_change_points(values::AbstractVector{T}, change_points::Vector{Integer}; 
                      title::String="Change Point Detection") where T <: Real

    Helper function to visualize change points in a sequence.
    Note: Requires Plots.jl to be loaded for visualization.
"""
function plot_change_points(values::AbstractVector{T}, change_points::Vector{Integer}; 
                           title::String="Change Point Detection") where T <: Real
    try
        p = plot(values, label="Values", linewidth=2, title=title)
        
        for cp in change_points
            vline!([cp], label="Change Point $cp", linestyle=:dash, alpha=0.7)
        end
        
        return p
    catch
        println("Plots.jl not available. Change points detected at indices: $change_points")
        return nothing
    end
end
# ...existing code...

"""
    elbow_detection(values::AbstractVector{T}; method::Symbol=:curvature) :: Integer where T <: Real

    Detects the elbow point in a decreasing sequence using various methods.
    
    # Arguments
    - `values`: Vector of decreasing values (e.g., singular values)
    - `method`: Detection method (:curvature, :variance, :gradient)
    
    # Methods
    - `:curvature`: Finds point of maximum curvature
    - `:variance`: Uses variance-based approach 
    - `:gradient`: Finds largest change in gradient
    
    # Returns
    - Index of the elbow point
    
    # Example
    ```julia
    vals = [10.0, 5.0, 2.0, 1.8, 1.7, 1.65, 1.6, 1.58, 1.57]
    elbow_idx = elbow_detection(vals)
    ```
"""
function elbow_detection(values::AbstractVector{T}; method::Symbol=:curvature) :: Integer where T <: Real
    
    if length(values) < 3
        return 1
    end
    
    if method == :curvature
        return elbow_curvature(values)
    elseif method == :variance
        return elbow_variance(values)
    elseif method == :gradient
        return elbow_gradient(values)
    else
        error("Unknown method: $method. Use :curvature, :variance, or :gradient")
    end
end

"""
    elbow_curvature(values::AbstractVector{T}) :: Integer where T <: Real
    
    Finds elbow using maximum curvature method.
    Computes second derivative and finds point of maximum absolute curvature.
"""
function elbow_curvature(values::AbstractVector{T}) :: Integer where T <: Real
    n = length(values)
    if n < 3
        return 1
    end
    
    # Normalize values to [0,1] to make curvature calculation scale-independent
    normalized = (values .- minimum(values)) ./ (maximum(values) - minimum(values))
    
    # Create x coordinates
    x = collect(1:n)
    
    # Compute first and second derivatives using finite differences
    first_deriv = diff(normalized)
    second_deriv = diff(first_deriv)
    
    # Compute curvature: |y''| / (1 + y'^2)^(3/2)
    curvatures = zeros(T, n-2)
    for i in 1:(n-2)
        y_prime = first_deriv[i]
        y_double_prime = second_deriv[i]
        curvatures[i] = abs(y_double_prime) / (1 + y_prime^2)^(3/2)
    end
    
    # Find point of maximum curvature (add 1 to account for indexing)
    return argmax(curvatures) + 1
end

"""
    elbow_variance(values::AbstractVector{T}) :: Integer where T <: Real
    
    Finds elbow using variance-based method.
    Splits data at each point and minimizes within-group variance.
"""
function elbow_variance(values::AbstractVector{T}) :: Integer where T <: Real
    n = length(values)
    if n < 3
        return 1
    end
    
    min_variance = Inf
    elbow_point = 1
    
    # Try each possible split point
    for i in 2:(n-1)
        left_group = values[1:i]
        right_group = values[(i+1):end]
        
        # Calculate weighted within-group variance
        left_var = length(left_group) > 1 ? var(left_group) : 0.0
        right_var = length(right_group) > 1 ? var(right_group) : 0.0
        
        weighted_var = (length(left_group) * left_var + length(right_group) * right_var) / n
        
        if weighted_var < min_variance
            min_variance = weighted_var
            elbow_point = i
        end
    end
    
    return elbow_point
end

"""
    elbow_gradient(values::AbstractVector{T}) :: Integer where T <: Real
    
    Finds elbow using gradient change method.
    Finds point where the rate of change (gradient) changes most dramatically.
"""
function elbow_gradient(values::AbstractVector{T}) :: Integer where T <: Real
    n = length(values)
    if n < 3
        return 1
    end
    
    # Compute first differences (gradients)
    gradients = diff(values)
    
    # Compute second differences (change in gradients)
    gradient_changes = abs.(diff(gradients))
    
    # Find point of maximum gradient change (add 1 to account for indexing)
    return argmax(gradient_changes) + 1
end

"""
    plot_elbow(values::AbstractVector{T}; method::Symbol=:curvature, title::String="Elbow Detection") where T <: Real
    
    Visualizes the elbow detection result.
"""
function plot_elbow(values::AbstractVector{T}; 
    method::Symbol=:curvature, 
    title::String="Elbow Detection") where T <: Real
    elbow_idx = elbow_detection(values, method=method)
    
    try
        p = plot(values, marker=:circle, linewidth=2, label="Values", title=title)
        vline!([elbow_idx], label="Elbow at $elbow_idx", linestyle=:dash, linewidth=2, color=:red)
        scatter!([elbow_idx], [values[elbow_idx]], color=:red, markersize=8, label="")
        xlabel!("Index")
        ylabel!("Value")
        return p
    catch
        println("Plots.jl not available. Elbow detected at index: $elbow_idx")
        println("Value at elbow: $(values[elbow_idx])")
        return nothing
    end
end

# ...existing code...
function critique(spall::Spall)

    # # extract the correct vector (use last available column if fewer than 3)
    # col_idx = min(3, size(lastsvds, 2))

    
    # eigenvector = lastsvds[:, col_idx]

    # # expand to matrices
    # offset = 0
    # mats = Vector{AbstractMatrix}(undef, valence)
    # for a in 1:valence
    #     eng = ⚒.category[a]
    #     if eng == Primal || eng == Ambidextrous
    #         mats[a] = Ω.toMatrix(maineigenvector, dims[a], offset )
    #         offset += Ω.dimFormula(a)
    #     elseif eng == Dual
    #         mats[a] = Ω.toMatrix(maineigenvector, dims[a], offset )'
    #         offset += Ω.dimFormula(a)
    #     end # Skip Disengaged
    # end

    # # Diagonalize
    # trans = [ LinearAlgebra.eigen( mat ) for mat in mats ]
    # vals = [t.values for t in trans]
    # vecs = [real.(t.vectors) for t in trans]  # Take real part to avoid complex number issues
    # tensor=act(t_float, vecs )
    # return (;tensor, matrices=vecs, eigenvalues=vals)
end;
