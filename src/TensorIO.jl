#
# Strata Dleto: Tensor I/O
#   Methods for loading, saving, visualizing, and 3D printing tensors.
#
# Copyright 2022-2025 Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the “Software”), 
# to deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.
# 

# module TensorIO

using ITensors
# using LinearAlgebra
# using SparseArrays
using PlotlyJS

"""
    normalize_tensor(t::ITensor)

    Normalize the tensor `t` so that its Frobenius norm is 1. If the norm is zero, 
    returns the tensor unchanged.
"""
function normalize_tensor(t::ITensor)
    norm_factor = norm(store(t))
    if norm_factor == 0
        return t
    else
        return t / norm_factor
    end
end

## DONT USE, use Array(Γ, inds(Γ)) instead!!
# function asarray(Γ::ITensor)
#     dims = size(Γ)
#     arr = zeros(Float64, dims...)
#     for idx in CartesianIndices(arr)
#         arr[idx] = Γ[Tuple(idx)...]
#     end
#     return arr
# end

"""
        side_by_side(left, right; left_title="Left", right_title="Right")

Render two array-like values side-by-side in notebook output using HTML.
Works in Jupyter/VS Code notebooks. Titles are optional.
"""
function side_by_side(left, right;
    left_title::AbstractString="Left", 
    right_title::AbstractString="Right")

    the_left = (typeof(left) <: ITensor) ? Array(left, inds(left)) : left
    left_txt = repr("text/plain", the_left)

    the_right = (typeof(right) <: ITensor) ? Array(right,inds(right)) : right
    right_txt = repr("text/plain", the_right)
    
    html = """
        <div style=\"display:flex; gap:16px; align-items:flex-start\">
            <div style=\"flex:1\">
                <h4>$(left_title)</h4>
                <pre style=\"font-family: SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;\">$(left_txt)</pre>
            </div>
            <div style=\"flex:1\">
                <h4>$(right_title)</h4>
                <pre style=\"font-family: SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;\">$(right_txt)</pre>
            </div>
        </div>
    """
    display(MIME("text/html"), html)
    return nothing
end
 
"""
    load_tensor(filename::String) -> ITensor

    [TBD: This should work with any valence!]
    Load a tensor from a file in sparse format. The file should contain lines of the form:
    i j k value
    where i, j, k are the indices and value is the tensor entry at that position.
"""
function load_tensor(filename::String)::ITensor
    # Read the file and parse entries
    entries = []
    max_dims = []
    
    valence = -1
    open(filename, "r") do file
        for line in eachline(file)
            # Skip comments and empty lines
            if startswith(line, "#") || isempty(strip(line))
                continue
            end
            
            # Parse the line into values
            parts = split(strip(line))
            if valence == -1
                valence = length(parts) - 1
                max_dims = zeros(Int, valence)
            end

            if length(parts) != valence + 1
                error("Inconsistent valence in tensor file: expected $valence indices, got $(length(parts)-1)")
            end
            is = [ parse(Int, parts[i]) for i in 1:(valence) ]
            val = parse(Float64, parts[valence+1])
                
            push!(entries, (is, val))
            for d in 1:valence
                max_dims[d] = max(max_dims[d], is[d])
            end
        end
    end
    # create a sparse array to hold the tensor data
    axes = [Index(max_dims[a], "x_$a") for a in 1:valence]
    # Create ITensor from array
    Γ = ITensor(axes...)
    for (is, val) in entries
        Γ[(axes[a] => is[a] for a in 1:valence)...] = val
    end
    return Γ
end



"""
    save(tensor::ITensor, filename::String, threshold::Float64=1e-3)

    Save the tensor to a file in sparse format, writing only entries 
    whose absolute value exceeds the given threshold.

    - `tensor`: The input tensor to save
    - `filename`: The name of the output file
    - `threshold`: Minimum absolute value for entries to be saved (default: 1e-3)
"""
function save(tensor::ITensor, filename::String, threshold::Float64=1e-3)
    open(filename, "w") do file
        dims = size(tensor)
        println(file, "# i j k value")
        for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]
            val = tensor[i, j, k]
            if abs(val) > threshold
                println(file, "$i $j $k $val")
            end
        end
    end
end

"""
    plot_tensor(tensor::ITensor, threshold::Float64=1e-2; 
                   xlabel::String="X", ylabel::String="Y", zlabel::String="Z",
                   title::String="3D Tensor Visualization", color::Symbol=:blue,
                   mode::Symbol=:image)   
                   
    Visualize a 3D tensor, plotting only entries whose absolute value exceeds 
    the given threshold.
    - `tensor`: The input tensor to visualize
    - `threshold`: Minimum absolute value for entries to be plotted (default: 1e
    # Arguments
    - `mode::Symbol=:image`: Use `:static` for static plot (Plots.jl) or 
      `:interactive` for interactive 3D viewer (PlotlyJS).
"""
function plot_tensor(tensor, threshold::Float64=1e-6; 
                   xlabel::String="X", ylabel::String="Y", zlabel::String="Z",
                   title::String="3D Tensor Visualization", color::Symbol=:blue,
                   viewer::Symbol=:static)

    # Convert ITensor to array for processing
    arr = (typeof(tensor) <: ITensor) ? Array(tensor, inds(tensor)...) : tensor
    
    # function for rounding to threshold decimal places
    roundToThreshold = x -> round(x, digits=Int(-log10(threshold)))
    arr = arr .|> roundToThreshold
    
    # Get indices of non-zero values in the tensor
    indices = findall(x -> x != 0, arr)
    dims = size(arr)

    # Extract x, y, z coordinates and values
    x_coords = [idx[1] for idx in indices]
    y_coords = [idx[2] for idx in indices]
    z_coords = [idx[3] for idx in indices]
    values = [abs(arr[idx]) for idx in indices]
    
    # Scale marker sizes proportional to values
    # Normalize values to a reasonable marker size range (2-10)
    if length(values) > 0
        min_val, max_val = extrema(values)
        if min_val ≈ max_val
            marker_sizes = fill(5.0, length(values))  # All same size if all values equal
        else
            # Scale values to range [2, 10] for marker sizes
            marker_sizes = 2.0 .+ 8.0 .* (values .- min_val) ./ (max_val - min_val)
        end
    else
        marker_sizes = Float64[]
    end

    @info "Plotting $(length(indices)) points with value-proportional sizes..."
    
    if viewer == :interactive
        # Create interactive 3D scatter plot using PlotlyJS
        # Convert color symbol to a PlotlyJS-compatible color string
        color_str = string(color)
        
        trace = PlotlyJS.scatter3d(;
            x=x_coords, 
            y=y_coords, 
            z=z_coords,
            mode="markers",
            marker=attr(
                size=marker_sizes,
                opacity=0.6,
                color=color_str
            ),
            type="scatter3d"
        )
        
        layout = PlotlyJS.Layout(;
            title=title,
            scene=attr(
                xaxis=attr(title=xlabel, range=[1, dims[1]+1]),
                yaxis=attr(title=ylabel, range=[1, dims[2]+1]),
                zaxis=attr(title=zlabel, range=[1, dims[3]+1])
            ),
            showlegend=false
        )
        
        p = PlotlyJS.plot([trace], layout)
        return p
    else
        # Create static 3D scatter plot using Plots.jl (default :image mode)
        p = Plots.scatter3d(x_coords, y_coords, z_coords;
            markersize=marker_sizes,
            markeralpha=0.6,
            color=color,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            title=title,
            xlims=(1, dims[1]+1),
            ylims=(1, dims[2]+1),
            zlims=(1, dims[3]+1),
            legend=false
        )
        
        # Return the plot object
        return p
    end
end


# end # module TensorIO