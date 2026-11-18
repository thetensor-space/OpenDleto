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
using LinearAlgebra: svd, norm
import Random

using ITensors
# using LinearAlgebra
# using SparseArrays
# using PlotlyJS
using Plots

# Global variable to store the default compare layout
const COMPARE_LAYOUT = Ref{Symbol}(:widescreen)

_unwrap_plot_input(x) = x
_unwrap_plot_input(x::Dleto.TensorSpace.TensorElement) = Dleto.TensorSpace.unwrap(x)

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
    normalize_tensor(t::ITensor)

    Normalize the tensor `t` so that its Frobenius norm is 1. If the norm is zero, 
    returns the tensor unchanged.
"""
function Dleto.normalize_tensor(t::ITensor)
    norm_factor = norm(store(t))
    if norm_factor == 0
        return t
    else
        return t / norm_factor
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

    left_unwrapped = _unwrap_plot_input(left)
    the_left = (typeof(left_unwrapped) <: ITensor) ? Array(left_unwrapped, inds(left_unwrapped)) : left_unwrapped
    left_txt = repr("text/plain", the_left)

    right_unwrapped = _unwrap_plot_input(right)
    the_right = (typeof(right_unwrapped) <: ITensor) ? Array(right_unwrapped, inds(right_unwrapped)) : right_unwrapped
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
    load_tensor(filename::String) -> ITensor

    [TBD: This should work with any valence!]
    Load a tensor from a file in sparse format. The file should contain lines of the form:
    i j k value
    where i, j, k are the indices and value is the tensor entry at that position.
"""
function Dleto.load_tensor(filename::String)::ITensor
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
function Dleto.save(tensor::ITensor, filename::String, threshold::Float64=1e-3)
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

function _tensor_marker_sizes(values; equal::Bool=false)
    isempty(values) && return Float64[]
    equal && return fill(5.0, length(values))
    peak = maximum(values)
    iszero(peak) && return fill(5.0, length(values))
    return 5.0 .* values ./ peak
end

"""
    plot_tensor(tensor, threshold=1e-4; controls=false,
        size_mode=:proportional, color_mode=:monochrome, ...)

Plot entries whose absolute value exceeds `threshold`. By default, return a
`Plots.Plot`, as used in notebook layouts. Set `controls=true` for a native
`PlotlyJS.Plot` with menus for proportional or equal marker sizes and
monochrome or signed-value colors. The static plot accepts the same
`size_mode` and `color_mode` keywords. At most 1,000 entries are shown.
"""
function Dleto.plot_tensor(tensor, threshold::Float64=1e-4;
                   xlabel::String="X", ylabel::String="Y", zlabel::String="Z",
                   title::String="3D Tensor Visualization", color::Symbol=:blue,
                   controls::Bool=false, size_mode::Symbol=:proportional,
                   color_mode::Symbol=:monochrome
    )

    isfinite(threshold) && threshold >= 0 ||
        throw(ArgumentError("threshold must be finite and nonnegative"))
    size_mode in (:proportional, :equal) ||
        throw(ArgumentError("size_mode must be :proportional or :equal"))
    color_mode in (:monochrome, :value) ||
        throw(ArgumentError("color_mode must be :monochrome or :value"))
    tensor = _unwrap_plot_input(tensor)

    # Convert ITensor to array for processing
    arr = (typeof(tensor) <: ITensor) ? Array(tensor, inds(tensor)...) : tensor
    (controls || color_mode == :value) && !(eltype(arr) <: Real) &&
        throw(ArgumentError("value colors require real tensor entries"))

    # Keep coefficients by magnitude; thresholds need not be powers of ten.
    indices = findall(x -> abs(x) > threshold, arr)
    dims = size(arr)

    # Extract x, y, z coordinates and values
    x_coords = [idx[1] for idx in indices]
    y_coords = [idx[2] for idx in indices]
    z_coords = [idx[3] for idx in indices]
    values = [abs(arr[idx]) for idx in indices]
    signed_values = [arr[idx] for idx in indices]

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
        signed_values = signed_values[sorted_indices]
    end
    # @info "Plotting $(length(indices)) points with value-proportional sizes..."

    proportional_sizes = _tensor_marker_sizes(values)
    equal_sizes = _tensor_marker_sizes(values; equal=true)
    marker_sizes = size_mode == :equal ? equal_sizes : proportional_sizes

    if controls
        scale = maximum(abs, signed_values)
        scale = iszero(scale) ? one(scale) : scale
        marker_color = color_mode == :value ? signed_values : string(color)
        trace = PlotlyJS.scatter3d(; x=x_coords, y=y_coords, z=z_coords,
            mode="markers", showlegend=false, customdata=signed_values,
            hovertemplate="(%{x}, %{y}, %{z})<br>coefficient=%{customdata:.5g}<extra></extra>",
            marker=PlotlyJS.attr(size=marker_sizes, color=marker_color,
                colorscale="RdBu", cmin=-scale, cmax=scale, opacity=0.6))
        size_buttons = [
            PlotlyJS.attr(label="Proportional", method="restyle",
                args=[Dict("marker.size" => [proportional_sizes]), [0]]),
            PlotlyJS.attr(label="Equal", method="restyle",
                args=[Dict("marker.size" => [equal_sizes]), [0]]),
        ]
        color_buttons = [
            PlotlyJS.attr(label="Monochrome", method="restyle",
                args=[Dict("marker.color" => [string(color)]), [0]]),
            PlotlyJS.attr(label="By value", method="restyle",
                args=[Dict("marker.color" => [signed_values]), [0]]),
        ]
        layout = PlotlyJS.Layout(; title, width=700, height=570, showlegend=false,
            scene=PlotlyJS.attr(
                xaxis=PlotlyJS.attr(title=xlabel, range=[0, dims[1]+1]),
                yaxis=PlotlyJS.attr(title=ylabel, range=[0, dims[2]+1]),
                zaxis=PlotlyJS.attr(title=zlabel, range=[0, dims[3]+1])),
            margin=PlotlyJS.attr(l=20, r=20, t=115, b=20),
            updatemenus=[
                PlotlyJS.attr(type="dropdown", x=0.05, y=1.15, xanchor="left",
                    active=size_mode == :equal ? 1 : 0, buttons=size_buttons),
                PlotlyJS.attr(type="dropdown", x=0.52, y=1.15, xanchor="left",
                    active=color_mode == :value ? 1 : 0, buttons=color_buttons),
            ])
        return PlotlyJS.Plot([trace], layout)
    end

    # if viewer == :interactive
    #     # Create interactive 3D scatter plot using PlotlyJS
    #     # Convert color symbol to a PlotlyJS-compatible color string
    #     color_str = string(color)

    #     trace = PlotlyJS.scatter3d(;
    #         x=x_coords,
    #         y=y_coords,
    #         z=z_coords,
    #         mode="markers",
    #         marker=attr(
    #             size=marker_sizes,
    #             opacity=0.6,
    #             color=color_str
    #         ),
    #         type="scatter3d"
    #     )

    #     layout = PlotlyJS.Layout(;
    #         title=title,
    #         scene=attr(
    #             xaxis=attr(title=xlabel, range=[1, dims[1]+1]),
    #             yaxis=attr(title=ylabel, range=[1, dims[2]+1]),
    #             zaxis=attr(title=zlabel, range=[1, dims[3]+1])
    #         ),
    #         showlegend=false
    #     )

    #     p = PlotlyJS.plot([trace], layout)
    #     return p
    # else
        # Create static 3D scatter plot using Plots.jl (default :image mode)
        # Add invisible corner points to enforce axis limits with PlotlyJS backend
        corner_x = [1, dims[1]+1]
        corner_y = [1, dims[2]+1]
        corner_z = [1, dims[3]+1]
        
        color_kwargs = color_mode == :monochrome ? (; color) :
            (; color=:RdBu, marker_z=signed_values)
        p = Plots.scatter3d(x_coords, y_coords, z_coords;
            markersize=marker_sizes,
            markeralpha=0.6,
            color_kwargs...,
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

# """
#     plot_tensor(tensor::ITensor, threshold::Float64=1e-2; 
#                    xlabel::String="X", ylabel::String="Y", zlabel::String="Z",
#                    title::String="3D Tensor Visualization", color::Symbol=:blue,
#                    mode::Symbol=:image)   
                   
#     Visualize a 3D tensor, plotting only entries whose absolute value exceeds 
#     the given threshold.
#     - `tensor`: The input tensor to visualize
#     - `threshold`: Minimum absolute value for entries to be plotted (default: 1e
#     # Arguments
#     - `mode::Symbol=:image`: Use `:static` for static plot (Plots.jl) or 
#       `:interactive` for interactive 3D viewer (PlotlyJS).
# """
# function plot_tensor(tensor, threshold::Float64=1e-6; 
#                    xlabel::String="X", ylabel::String="Y", zlabel::String="Z",
#                    title::String="3D Tensor Visualization", color::Symbol=:blue,
#                    viewer::Symbol=:static)

#     # Convert ITensor to array for processing
#     arr = (typeof(tensor) <: ITensor) ? Array(tensor, inds(tensor)...) : tensor
    
#     # function for rounding to threshold decimal places
#     roundToThreshold = x -> round(x, digits=round(Int,-log10(threshold)))
#     arr = arr .|> roundToThreshold
    
#     # Get indices of non-zero values in the tensor
#     indices = findall(x -> x != 0, arr)
#     dims = size(arr)

#     # Extract x, y, z coordinates and values
#     x_coords = [idx[1] for idx in indices]
#     y_coords = [idx[2] for idx in indices]
#     z_coords = [idx[3] for idx in indices]
#     values = [abs(arr[idx]) for idx in indices]
    
#     # Scale marker sizes proportional to values
#     # Normalize values to a reasonable marker size range (2-10)
#     if length(values) > 0
#         min_val, max_val = extrema(values)
#         if min_val ≈ max_val
#             marker_sizes = fill(5.0, length(values))  # All same size if all values equal
#         else
#             # Scale values to range [2, 10] for marker sizes
#             marker_sizes = 2.0 .+ 8.0 .* (values .- min_val) ./ (max_val - min_val)
#         end
#     else
#         marker_sizes = Float64[]
#     end

#     @info "Plotting $(length(indices)) points with value-proportional sizes..."
    
#     if viewer == :interactive
#         # Create interactive 3D scatter plot using PlotlyJS
#         # Convert color symbol to a PlotlyJS-compatible color string
#         color_str = string(color)
        
#         trace = PlotlyJS.scatter3d(;
#             x=x_coords, 
#             y=y_coords, 
#             z=z_coords,
#             mode="markers",
#             marker=attr(
#                 size=marker_sizes,
#                 opacity=0.6,
#                 color=color_str
#             ),
#             type="scatter3d"
#         )
        
#         layout = PlotlyJS.Layout(;
#             title=title,
#             scene=attr(
#                 xaxis=attr(title=xlabel, range=[1, dims[1]+1]),
#                 yaxis=attr(title=ylabel, range=[1, dims[2]+1]),
#                 zaxis=attr(title=zlabel, range=[1, dims[3]+1])
#             ),
#             showlegend=false
#         )
        
#         p = PlotlyJS.plot([trace], layout)
#         return p
#     else
#         # Create static 3D scatter plot using Plots.jl (default :image mode)
#         p = Plots.scatter3d(x_coords, y_coords, z_coords;
#             markersize=marker_sizes,
#             markeralpha=0.6,
#             color=color,
#             xlabel=xlabel,
#             ylabel=ylabel,
#             zlabel=zlabel,
#             title=title,
#             xlims=(1, dims[1]+1),
#             ylims=(1, dims[2]+1),
#             zlims=(1, dims[3]+1),
#             legend=false
#         )
        
#         # Return the plot object
#         return p
#     end
# end


"""
    derivation_spectrum_widget(Γ::Dleto.TensorSpace.TensorElement; factorization=nothing,
        descending_indices=nothing, initial_descending=nothing,
        mode=:auto, window=16, max_points=500, max_frames=500,
        max_dense_bytes=256*1024^2,
        size_mode=:proportional, color_mode=:monochrome,
        title="Derivation-map singular directions",
        axis_labels=("Axis 1", "Axis 2", "Axis 3"))

Return a native `PlotlyJS.Plot` with precomputed slider frames.
`Γ` is a `TensorElement` wrapping a real third-order `ITensor`.
For a small map, `mode=:auto` plots the full singular spectrum of the
rectangular universal-chisel derivation map, largest to smallest.
When the map fits `max_dense_bytes` but has more than `max_frames` directions,
`mode=:auto` computes the full SVD and plots every singular value, while the
slider offers 3D frames only for the last `window` directions. `mode=:window`
requests this explicitly. A full SVD is used because a values-only SVD plus
matrix-free candidate vectors would not pair the plotted values with their
3D views. For a map beyond the dense-memory budget, `mode=:auto` uses Dleto's
`AutoSolver` on the matrix-free square map and plots only candidate equation
residuals relative to an estimated largest singular value; this is not a full
SVD. `mode=:near_null` requests that matrix-free route explicitly.
Set `mode=:full` to require the full spectrum and, unless
`descending_indices` is supplied, all slider frames. `window` is a direction
count, or a fraction between 0 and 1 of the map's columns. Increase it to
save more 3D frames, up to `max_frames`; in matrix-free mode this also requests
more candidate directions. The dot marks the selected singular value or
candidate residual. The 3D panel is an adapted coordinate view from
`Dleto.stratify(Γ, D)` for the selected SVD vector or matrix-free candidate
embedded as transverse operators. No Tucker reduction, thresholding of `Γ`, or law
certification is performed.

`factorization` may be a previously computed thin `svd(Matrix(N))` of this
same tensor's derivation map. Its dimensions are checked, but the caller must
ensure it came from the same tensor. Without it, the rectangular map is
converted to a dense matrix and a thin SVD is computed, which can require
substantial time and memory.
`descending_indices` selects frames only in full mode; the full spectrum
remains visible. Subsetting frames does not reduce the full SVD cost.
Without a supplied factorization, `max_dense_bytes` guards the rectangular
matrix allocation for both SVD routes; they need additional workspace.
`descending_indices` requires full mode, while `factorization` can also be
used with `mode=:window`.

Each precomputed 3D frame displays at most `max_points` largest-magnitude
coefficients. The 3D display menu offers proportional or equal marker sizes,
each in monochrome or signed-value colors, as in `plot_tensor`. Set
`size_mode` and `color_mode` to choose the initial display. Sizes and color
scales are computed within each frame, so the viewer does
not certify zeros or compare coefficient magnitudes across frames. Singular
vectors outside a certified null space are approximate coordinate probes,
not certified derivations; a separate solver must certify the law. This
function does not inspect derivation eigenvalues.
"""
function Dleto.derivation_spectrum_widget(
    Γ::Dleto.TensorSpace.TensorElement;
    factorization=nothing,
    descending_indices=nothing,
    initial_descending=nothing,
    mode::Symbol=:auto,
    window::Real=16,
    max_points::Integer=500,
    max_frames::Integer=500,
    max_dense_bytes::Integer=256*1024^2,
    size_mode::Symbol=:proportional,
    color_mode::Symbol=:monochrome,
    title::AbstractString="Derivation-map singular directions",
    axis_labels::NTuple{3,<:AbstractString}=("Axis 1", "Axis 2", "Axis 3"),
)
    raw = Dleto.TensorSpace.unwrap(Γ)
    raw isa ITensor || throw(ArgumentError("3D viewer requires a TensorElement wrapping an ITensor"))
    ndims(raw) == 3 || throw(ArgumentError("3D viewer requires a third-order tensor"))
    eltype(raw) <: Real || throw(ArgumentError("3D viewer currently requires real tensor entries"))
    max_points > 0 || throw(ArgumentError("max_points must be positive"))
    max_frames > 0 || throw(ArgumentError("max_frames must be positive"))
    max_dense_bytes > 0 || throw(ArgumentError("max_dense_bytes must be positive"))
    size_mode in (:proportional, :equal) ||
        throw(ArgumentError("size_mode must be :proportional or :equal"))
    color_mode in (:monochrome, :value) ||
        throw(ArgumentError("color_mode must be :monochrome or :value"))
    isfinite(window) && window > 0 && (window < 1 || isinteger(window)) ||
        throw(ArgumentError("window must be a positive integer count or a fraction between 0 and 1"))
    mode in (:auto, :full, :window, :near_null) ||
        throw(ArgumentError("mode must be :auto, :full, :window, or :near_null"))

    tensor_frame = inds(raw)
    ops = Dleto.UniversalOps(raw)
    square, N = Dleto.sylvesterLM(ops, Dleto.UniversalChisel(3), raw)
    rows, columns = size(N)
    dense_bytes = Int128(rows) * columns * sizeof(eltype(N))
    dense_gib = round(dense_bytes / 2.0^30; digits=2)
    full_n = min(rows, columns)
    forced_full = !isnothing(descending_indices)
    actual_mode = mode == :auto ?
        (forced_full ? :full :
            (!isnothing(factorization) || dense_bytes <= max_dense_bytes) ?
                (full_n <= max_frames ? :full : :window) : :near_null) : mode
    actual_mode == :near_null && (!isnothing(factorization) || forced_full) && throw(ArgumentError(
        "factorization and descending_indices require a full-SVD mode"))
    actual_mode == :window && !isnothing(descending_indices) && throw(ArgumentError(
        "descending_indices requires mode=:full"))

    if actual_mode in (:full, :window)
        if actual_mode == :window
            requested = window < 1 ? max(1, ceil(Int, window * full_n)) : Int(window)
            requested = min(requested, full_n)
            requested <= max_frames || throw(ArgumentError(
                "window requests $requested frames, exceeding max_frames=$max_frames"))
            indices = collect(full_n-requested+1:full_n)
        else
            indices = isnothing(descending_indices) ? collect(1:full_n) :
                collect(Int, descending_indices)
        end
        isempty(indices) && throw(ArgumentError("descending_indices cannot be empty"))
        all(k -> 1 <= k <= full_n, indices) || throw(BoundsError(1:full_n, indices))
        length(unique(indices)) == length(indices) ||
            throw(ArgumentError("descending_indices must not contain duplicates"))
        length(indices) <= max_frames || throw(ArgumentError(
            "$(length(indices)) frames exceeds max_frames=$max_frames for a $rows×$columns map " *
            "(dense matrix about $dense_gib GiB). Pass descending_indices to reduce " *
            "frames, or use mode=:window; the full SVD cost remains."))
        if isnothing(factorization) && dense_bytes > max_dense_bytes
            throw(ArgumentError("full SVD of the $rows×$columns map needs a dense " *
                "matrix of about $dense_gib GiB plus workspace, exceeding " *
                "max_dense_bytes=$max_dense_bytes. Use mode=:near_null or an " *
                "independently computed factorization; descending_indices only " *
                "reduces saved frames."))
        end
        F = isnothing(factorization) ? svd(Matrix(N); full=false) : factorization
        singular = F.S
        length(singular) == full_n || throw(DimensionMismatch(
            "factorization must have $full_n singular values for a $rows×$columns map"))
        size(F.V) == (columns, full_n) || throw(DimensionMismatch(
            "factorization must have $columns rows and $full_n right singular vectors"))
        full_n > 0 && isfinite(first(singular)) && first(singular) > 0 ||
            throw(ArgumentError("derivation map needs a positive finite largest singular value"))
        all(isfinite, singular) || throw(ArgumentError("nonfinite singular value"))
        issorted(singular; rev=true) || throw(ArgumentError("singular values must be descending"))
        ratios = singular ./ first(singular)
        vectors = F.V
        n = full_n
        x_title = "Descending SVD index"
        y_title = "σ / σ₁"
        slider_prefix = "SVD index: "
        hover_label = "descending SVD index"
        display_title = actual_mode == :window ?
            "$title (full singular values; 3D window: $(length(indices)) of $n directions)" : title
    else
        requested = window < 1 ? max(1, ceil(Int, window * columns)) : Int(window)
        requested = min(requested, columns)
        requested <= max_frames || throw(ArgumentError(
            "window requests $requested frames, exceeding max_frames=$max_frames"))
        result = Dleto.solve(square, :AutoSolver; nv=requested, seed=20261118)
        hasproperty(result, :converged) && !result.converged &&
            throw(ArgumentError("matrix-free near-null solve did not converge"))
        available = min(requested, length(result.vals), size(result.vecs, 2))
        available > 0 || throw(ArgumentError("near-null solver returned no directions"))
        keep = sortperm(abs.(result.vals))[1:available]
        candidate_vectors = result.vecs[:, keep]
        residuals = [norm(N * candidate_vectors[:, j]) / norm(candidate_vectors[:, j])
                     for j in 1:available]
        order = sortperm(residuals; rev=true)
        vectors = candidate_vectors[:, order]
        scale = sqrt(Dleto.opnorm_estimate(square;
            iters=10, rng=Random.MersenneTwister(20261118)))
        scale > 0 || throw(ArgumentError("derivation map needs a positive norm estimate"))
        ratios = residuals[order] ./ scale
        n = available
        indices = collect(1:n)
        x_title = "Near-null candidate index"
        y_title = "‖N v‖ / estimated σ₁"
        slider_prefix = "Candidate: "
        hover_label = "near-null candidate"
        display_title = "$title (near-null window: $n of $columns directions)"
    end
    sort!(indices)  # slider and spectrum dot both move left to right
    first_index = isnothing(initial_descending) ? last(indices) : Int(initial_descending)
    first_index in indices || throw(ArgumentError("initial_descending must be in the slider window"))
    start_position = findfirst(==(first_index), indices)

    positive_ratios = filter(>(0), Float64.(ratios))
    graph_floor = isempty(positive_ratios) ? eps(Float64) :
        max(nextfloat(0.0), minimum(positive_ratios) / 10)
    graph_ratios = max.(Float64.(ratios), graph_floor)
    any(iszero, ratios) && (y_title *= " (zeros shown at floor)")

    function marker(k)
        PlotlyJS.scatter(; x=[k], y=[graph_ratios[k]], customdata=[ratios[k]],
            mode="markers", xaxis="x", yaxis="y",
            showlegend=false, marker=PlotlyJS.attr(size=13, color="crimson"),
            hovertemplate="$hover_label %{x}<br>relative value=%{customdata:.5g}<extra></extra>")
    end

    display_modes = ((:proportional, :monochrome), (:equal, :monochrome),
        (:proportional, :value), (:equal, :value))
    active_display = findfirst(==((size_mode, color_mode)), display_modes)

    function views(k; initial=false)
        operators = Dleto.embedITensors(ops, vectors[:, k])
        adapted = Array(Dleto.stratify(raw, operators).Σ, tensor_frame...)
        flat = vec(adapted)
        chosen = partialsortperm(abs.(flat), 1:min(max_points, length(flat)); rev=true)
        positions = CartesianIndices(adapted)
        xyz = [Tuple(positions[j]) for j in chosen]
        values = flat[chosen]
        scale = maximum(abs, values)
        iszero(scale) && (scale = one(scale))
        sizes = (_tensor_marker_sizes(abs.(values)),
            _tensor_marker_sizes(abs.(values); equal=true))
        [PlotlyJS.scatter3d(; x=first.(xyz), y=getindex.(xyz, 2), z=getindex.(xyz, 3),
            mode="markers", scene="scene", showlegend=false, customdata=values,
            (initial ? (; visible=i == active_display) : NamedTuple())...,
            hovertemplate="(%{x}, %{y}, %{z})<br>coefficient=%{customdata:.5g}<extra></extra>",
            marker=PlotlyJS.attr(size=sizes[mode[1] == :equal ? 2 : 1],
                color=mode[2] == :value ? values : "royalblue",
                colorscale="RdBu", cmin=-scale, cmax=scale, opacity=0.6))
            for (i, mode) in enumerate(display_modes)]
    end

    frames = PlotlyJS.PlotlyFrame[]
    steps = Any[]
    for k in indices
        name = string(k)
        push!(frames, PlotlyJS.frame(; name, traces=[1, 2, 3, 4, 5],
            data=[marker(k); views(k)]))
        push!(steps, PlotlyJS.attr(label=name, method="animate",
            args=[[name], PlotlyJS.attr(mode="immediate",
                frame=PlotlyJS.attr(duration=0, redraw=true),
                transition=PlotlyJS.attr(duration=0))]))
    end

    line = PlotlyJS.scatter(; x=1:n, y=graph_ratios, customdata=ratios,
        mode="lines", xaxis="x", yaxis="y", showlegend=false,
        hovertemplate="$hover_label %{x}<br>relative value=%{customdata:.5g}<extra></extra>",
        line=PlotlyJS.attr(color="seagreen", width=2))
    display_buttons = [PlotlyJS.attr(label=label, method="restyle",
        args=[Dict("visible" => [j == i for j in 1:4]), [2, 3, 4, 5]])
        for (i, label) in enumerate(("Proportional · monochrome", "Equal · monochrome",
            "Proportional · by value", "Equal · by value"))]
    window_shapes = n > length(indices) ? [PlotlyJS.attr(type="line", xref="x",
        yref="paper", x0=first(indices)-0.5, x1=first(indices)-0.5,
        y0=0, y1=1, line=PlotlyJS.attr(color="crimson", dash="dot"))] : []
    window_labels = n > length(indices) ? [PlotlyJS.attr(xref="x", yref="paper",
        x=first(indices)-0.5, y=1, text="3D window →", showarrow=false,
        xanchor="right", font=PlotlyJS.attr(color="crimson"))] : []
    dims = size(raw)
    layout = PlotlyJS.Layout(; title=display_title, width=1100, height=650, showlegend=false,
        xaxis=PlotlyJS.attr(domain=[0.0, 0.27], title=x_title, range=[1, n]),
        yaxis=PlotlyJS.attr(type="log", title=y_title),
        scene=PlotlyJS.attr(domain=PlotlyJS.attr(x=[0.34, 1.0], y=[0.12, 1.0]),
            xaxis=PlotlyJS.attr(title=axis_labels[1], range=[0, dims[1]+1]),
            yaxis=PlotlyJS.attr(title=axis_labels[2], range=[0, dims[2]+1]),
            zaxis=PlotlyJS.attr(title=axis_labels[3], range=[0, dims[3]+1])),
        margin=PlotlyJS.attr(l=50, r=20, t=90, b=110),
        shapes=window_shapes, annotations=window_labels,
        updatemenus=[PlotlyJS.attr(type="dropdown", x=0.34, y=1.08,
            xanchor="left", active=active_display-1, buttons=display_buttons)],
        sliders=[PlotlyJS.attr(active=start_position-1,
            currentvalue=PlotlyJS.attr(prefix=slider_prefix),
            pad=PlotlyJS.attr(t=40), len=0.27, x=0.0, steps=steps)])
    PlotlyJS.Plot(vcat([line, marker(first_index)],
        views(first_index; initial=true)), layout, frames)
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
