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

using PlotlyJS


function loadTensorFromFile(filename::String)
    # Read the file and parse entries
    entries = []
    max_i, max_j, max_k = 0, 0, 0
    
    open(filename, "r") do file
        for line in eachline(file)
            # Skip comments and empty lines
            if startswith(line, "#") || isempty(strip(line))
                continue
            end
            
            # Parse the line: i j k value
            parts = split(strip(line))
            if length(parts) == 4
                i = parse(Int, parts[1])
                j = parse(Int, parts[2])
                k = parse(Int, parts[3])
                val = parse(Float64, parts[4])
                
                push!(entries, (i, j, k, val))
                max_i = max(max_i, i)
                max_j = max(max_j, j)
                max_k = max(max_k, k)
            end
        end
    end
    
    # Create tensor array with appropriate dimensions
    tensor = zeros(Float64, max_i, max_j, max_k)
    
    # Populate the tensor with values
    for (i, j, k, val) in entries
        tensor[i, j, k] = val
    end
    
    return tensor
end


function saveTensorToFile(tensor::AbstractArray, filename::String, threshold::Float64=1e-3)
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

function plotTensor(tensor::AbstractArray, threshold::Float64=1e-2; 
                   xlabel::String="X", ylabel::String="Y", zlabel::String="Z",
                   title::String="3D Tensor Visualization", color::String="blue")

                # function for rounding to threshold decimal places
                roundToThreshold = x -> round(x, digits=Int(-log10(threshold)))
                tensor = tensor .|> roundToThreshold
    # # function for removing small entries
    # dropSmall = x -> abs(x)< threshold ? 0 : x
    # tensor = tensor .|> dropSmall
    
    # Get indices of non-zero values in the tensor
    indices = findall(x -> x != 0, tensor)
    dims = size(tensor)

    # Extract x, y, z coordinates and values
    x_coords = [idx[1] for idx in indices]
    y_coords = [idx[2] for idx in indices]
    z_coords = [idx[3] for idx in indices]
    values = [abs(tensor[idx]) for idx in indices]
    
    # Scale marker sizes proportional to values
    # Normalize values to a reasonable marker size range (2-20)
    if length(values) > 0
        min_val, max_val = extrema(values)
        if min_val ≈ max_val
            marker_sizes = fill(5.0, length(values))  # All same size if all values equal
        else
            # Scale values to range [2, 20] for marker sizes
            marker_sizes = 2.0 .+ 18.0 .* (values .- min_val) ./ (max_val - min_val)
        end
    else
        marker_sizes = Float64[]
    end

    println("Plotting $(length(indices)) points with value-proportional sizes...")
    
    # Create 3D scatter plot with bounding box based on tensor dimensions
    p = PlotlyJS.Plot(scatter3d(
        x=x_coords, 
        y=y_coords, 
        z=z_coords,
        mode="markers",
        marker=attr(size=marker_sizes, opacity=0.6, color=color)
    ), Layout(
        scene=attr(
            xaxis=attr(range=[1, dims[1]+1], title=xlabel),
            yaxis=attr(range=[1, dims[2]+1], title=ylabel),
            zaxis=attr(range=[1, dims[3]+1], title=zlabel),
            aspectmode="cube"
        ),
        title=title
    ))
    
    # Return the plot object to let notebook handle rendering
    return p
end
