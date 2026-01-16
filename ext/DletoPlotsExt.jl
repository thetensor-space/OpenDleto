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

module DletoPlotsExt

using Dleto
import PlotlyBase
import PlotlyKaleido
import PlotlyJS

using ITensors
# using LinearAlgebra
# using SparseArrays
# using PlotlyJS
using Plots

const COMPARE_LAYOUT = Ref{Symbol}(:widescreen)

"""
    set_compare_layout(layout::Symbol)

Set the default layout for the `compare` function.

- `:widescreen` - Display side-by-side (default)
- `:vertical` - Display stacked vertically
"""
function Dleto.set_compare_layout(layout::Symbol)
    if layout ∉ (:widescreen, :vertical)
        error("Layout must be either :widescreen or :vertical")
    end
    COMPARE_LAYOUT[] = layout
    return nothing
end

"""
    get_compare_layout()

Get the current default layout for the `compare` function.
"""
Dleto.get_compare_layout() = COMPARE_LAYOUT[]

function Dleto.plot_layout(type::Symbol)
    if type == :vertical
        Plots.default(size=(400,900))
    else # default :widescreen
        Plots.default(size=(900,400))
        println("Using widescreen layout for plots.")
    end
end

"""
        compare(left, right; left_title="Left", right_title="Right", layout=nothing)

Render two array-like values in notebook output using HTML.
Works in Jupyter/VS Code notebooks. Titles are optional.

- `layout=nothing`: Use global default (set via `set_compare_layout`)
- `layout=:widescreen`: Display side-by-side
- `layout=:vertical`: Display stacked vertically
"""
function Dleto.compare(left, right;
    left_title::AbstractString="Left", 
    right_title::AbstractString="Right",
    layout::Union{Symbol,Nothing}=nothing)

    # Use global default if layout not specified
    actual_layout = isnothing(layout) ? get_compare_layout() : layout

    the_left = (typeof(left) <: ITensor) ? Array(left, inds(left)) : left
    left_txt = repr("text/plain", the_left)

    the_right = (typeof(right) <: ITensor) ? Array(right,inds(right)) : right
    right_txt = repr("text/plain", the_right)
    
    if actual_layout == :vertical
        html = """
            <div style=\"display:flex; flex-direction:column; gap:16px;\">
                <div>
                    <h4>$(left_title)</h4>
                    <pre style=\"font-family: SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;\">$(left_txt)</pre>
                </div>
                <div>
                    <h4>$(right_title)</h4>
                    <pre style=\"font-family: SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;\">$(right_txt)</pre>
                </div>
            </div>
        """
    else  # :widescreen
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
    end
    display(MIME("text/html"), html)
    return nothing
end
 

"""
    plot_tensor(tensor::ITensor, threshold::Float64=1e-2;
                   xlabel::String="X", ylabel::String="Y", zlabel::String="Z",
                   title::String="3D Tensor Visualization", color::Symbol=:blue
        )

    Visualize a 3D tensor, plotting only entries whose absolute value exceeds
    the given threshold.
    - `tensor`: The input tensor to visualize
    - `threshold`: Minimum absolute value for entries to be plotted (default: 1e
    # Arguments
"""
function Dleto.plot_tensor(tensor, threshold::Float64=1e-4;
                   xlabel::String="X", ylabel::String="Y", zlabel::String="Z",
                   title::String="3D Tensor Visualization", color::Symbol=:blue
    )

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

    if length(indices) == 0
        @warn "No tensor entries exceed the threshold of $threshold. Nothing to plot."
        return nothing
    elseif length(indices) > 1000
        @warn "More than 1,000 viewable entries, plotting only the largest points."
        # Keep only the top 1,000 entries by value
        sorted_indices = sortperm(values, rev=true)[1:1000]
        x_coords = x_coords[sorted_indices]
        y_coords = y_coords[sorted_indices]
        z_coords = z_coords[sorted_indices]
        values = values[sorted_indices]
    end
    # @info "Plotting $(length(indices)) points with value-proportional sizes..."

    # Scale marker sizes proportional to values
    # Normalize values to a reasonable marker size range (1-5)
    if length(values) > 0
        min_val, max_val = extrema(values)
        if min_val ≈ max_val
            marker_sizes = fill(5.0, length(values))  # All same size if all values equal
        else
            # Scale values to range [1, 5] for marker sizes
            marker_sizes = 1.0 .+ 4.0 .* (values .- min_val) ./ (max_val - min_val)
        end
    else
        marker_sizes = Float64[]
    end

        corner_x = [1, dims[1]+1]
        corner_y = [1, dims[2]+1]
        corner_z = [1, dims[3]+1]
        
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
        
        # Add invisible corner points to enforce limits
        Plots.scatter3d!(p, corner_x, corner_y, corner_z;
            markersize=0.1,
            markeralpha=0.0,
            color=:white,
            legend=false
        )

        # Return the plot object
        return p
    # end
end


function __init__()
    # Suppress WebIO warnings
    ENV["WEBIO_WARN"] = "false"
    println("Loading Dleto Plots Extension")
    # Only set backend if we're not precompiling
    if ccall(:jl_generating_output, Cint, ()) != 1
        try
            Plots.plotlyjs()
        catch e
            @warn "Failed to set Plotly backend" exception=e
        end
    end
end

end