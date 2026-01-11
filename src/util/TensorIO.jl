#
# Strata Dleto: Tensor I/O Placeholder
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

const COMPARE_LAYOUT = Ref{Symbol}(:widescreen)

function set_compare_layout end

function get_compare_layout end 

function plot_layout end 

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



function compare end

side_by_side = compare
 
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

function plot_tensor end

