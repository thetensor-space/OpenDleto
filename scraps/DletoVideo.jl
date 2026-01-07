#
# Strata Dleto: Video
#   Special tools for processing video tensors.
#
# Copyright 2022-2026 Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson
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


include("Dleto.jl") 


## Uncomment if not already installed.
# using Pkg
# Pkg.add("VideoIO")
# Pkg.add("ColorTypes")
# Pkg.add("FixedPointNumbers")
# Pkg.add("ProgressMeter")
using VideoIO
using ColorTypes  # Import RGB and Gray types
using FixedPointNumbers  # Import N0f8
using Serialization
using ProgressMeter  # For progress bars

function videoTensor(video_path::String, max_frames::Int=60) :: Array{RGB{N0f8}, 3}
    # Load the video file
    video = VideoIO.openvideo(video_path)

    frames = []
    while !eof(video) && length(frames) < max_frames
        frame = RGB{N0f8}.(read(video))
        push!(frames, frame)
    end

    close(video)

    # Create 3D tensor: (height, width, frames)
    return cat(frames..., dims=3)
end


# Pad dimensions to make them even if needed
function padToEven(tensor::Array{T, 3}) where T
    h, w, d = size(tensor)
    new_h = iseven(h) ? h : h + 1
    new_w = iseven(w) ? w : w + 1
    new_d = iseven(d) ? d : d + 1
    
    if new_h != h || new_w != w || new_d != d
        # Create padded tensor filled with zeros
        padded = fill(T(0), new_h, new_w, new_d)
        # Copy original data
        padded[1:h, 1:w, 1:d] = tensor
        return padded
    else
        return tensor
    end
end

# function saveAsVideo(video_tensor::Array{RGB{N0f8}, 3}, output_path::String, fps::Int=30)
#     num_frames = size(video_tensor, 3)
    
#     frames = [video_tensor[:, :, i] for i in 1:num_frames]
    
#     try
#         # Use simple VideoIO.save approach
#         ## keep it somewhat faithful, these are ffmpeg options
#         encoder_options = (crf=18, preset="veryslow") # Optional: adjust quality and encoding speed
#         VideoIO.save(output_path, frames, framerate=fps, # Set an appropriate framerate
#              encoder_options=encoder_options, 
#              target_pix_fmt=VideoIO.AV_PIX_FMT_YUV420P)
#         println("✓ Video saved to: ", output_path)
#     catch e
#         println("Video save failed: ", e)
#         println("Attempting alternative approach...")
        
#         try
#             # Fallback using video writer
#             first_frame = frames[1]
#             writer = VideoIO.open_video_out(output_path, first_frame)
            
#             for frame in frames
#                 write(writer, frame)
#             end
            
#             finalize(writer)
#             println("✓ Video saved to: ", output_path)
#         catch e2
#             println("Alternative video save also failed: ", e2)
#             error("Could not save video to $output_path")
#         end
#     end
# end

function saveVideo(tensor::Array{T,3}, output_path::String; fps::Int=30) where T
    # Many video codecs require even dimensions
    t_padded = padToEven(tensor)

    num_frames = size(t_padded, 3)

    # If already color selected, keep and save, else use grayscale
    frames = []; 
    codec_opt = (crf=18, preset="veryslow") # Optional: adjust quality and encoding speed
    fmt = VideoIO.AV_PIX_FMT_YUV420P
    if T <: RGB
        frames = [t_padded[:, :, i] for i in 1:num_frames]
        
    elseif T <: Gray
        frames = [t_padded[:, :, i] for i in 1:num_frames]
        fmt=VideoIO.AV_PIX_FMT_GRAY8
    else
        # For Float types, normalize and convert to grayscale
        normalized = abs.(t_padded) ./ maximum(abs.(t_padded))
        gray_tensor = Gray{N0f8}.(normalized)
        frames = [gray_tensor[:, :, i] for i in 1:num_frames]
        fmt=VideoIO.AV_PIX_FMT_GRAY8
    end
    
    try
        # Approach 1: Simple VideoIO.save with minimal options
        VideoIO.save(output_path, 
                    frames, 
                    framerate=fps,
                    encoder_options=codec_opt,
                    target_pix_fmt=fmt 
                )
        # println("✓ Grayscale video saved to: ", output_path, " (simple method)")
        return
    catch e1
        println("Simple grayscale save failed: ", e1)
        
        try
            # Approach 2: Use basic encoder options without problematic presets
            encoder_options = (crf=23,)  # Good quality, no preset
            VideoIO.save(output_path, frames, framerate=fps, 
                        encoder_options=encoder_options)
            println("✓ Grayscale video saved to: ", output_path, " (with CRF)")
            return
        catch e2
            println("CRF grayscale save failed: ", e2)
            
            try
                # Approach 3: Use open_video_out method
                println("Attempting grayscale video writer approach...")
                first_frame = frames[1]
                writer = VideoIO.open_video_out(output_path, first_frame, framerate=fps)
                
                for (i, frame) in enumerate(frames)
                    write(writer, frame)
                    if i % 10 == 0
                        println("Written grayscale frame $i/$num_frames")
                    end
                end
                
                close_video_out!(writer)
                println("✓ Grayscale video saved to: ", output_path, " (video writer)")
                return
            catch e3
                println("Grayscale video writer failed: ", e3)
                
                try
                    # Approach 4: Force specific pixel format
                    encoder_options = (pix_fmt="yuv420p",)
                    VideoIO.save(output_path, frames, framerate=fps, 
                                encoder_options=encoder_options)
                    println("✓ Grayscale video saved to: ", output_path, " (forced yuv420p)")
                    return
                catch e4
                    println("Forced pixel format grayscale save failed: ", e4)
                    error("All grayscale video save methods failed. Last error: $e4")
                end
            end
        end
    end
end

function boxVideo( video::Array{RGB{N0f8}, 3}, box::Tuple{Int, Int, Int})
    # split tensors into chunks and extract color channels
    return [ 
        (;
            red = map(x -> Float32(x.r), video[(1+(i-1)*box[1]):(i*box[1]), 
                                              (1+(j-1)*box[2]):(j*box[2]), 
                                              (1+(k-1)*box[3]):(k*box[3])]),
            green = map(x -> Float32(x.g), video[(1+(i-1)*box[1]):(i*box[1]), 
                                              (1+(j-1)*box[2]):(j*box[2]), 
                                              (1+(k-1)*box[3]):(k*box[3])]),
            blue = map(x -> Float32(x.b), video[(1+(i-1)*box[1]):(i*box[1]), 
                                              (1+(j-1)*box[2]):(j*box[2]), 
                                              (1+(k-1)*box[3]):(k*box[3])]),
            index = (i, j, k)
        )
                    for i in 1:div(size(video, 1), box[1]), 
                        j in 1:div(size(video, 2), box[2]), 
                        k in 1:div(size(video, 3), box[3]) 
        ]
end

struct RGB2
    red::Float16
    green::Float16
    blue::Float16
end
function assembleVideo( boxes,
            box_size::Tuple{Int, Int, Int}, 
            original_size::Tuple{Int, Int, Int} )

    # Reassemble the tensors in the same striding pattern used to create chunks
    video = Array{Tuple{Float16, Float16, Float16}}(undef, original_size[1], original_size[2], original_size[3])

    # green = zeros(Float16, original_size[1], original_size[2], original_size[3])
    # blue = zeros(Float16, original_size[1], original_size[2], original_size[3])

    chunk_idx = 1
    for i in 1:div(original_size[1], box_size[1])
        for j in 1:div(original_size[2], box_size[2])
            for k in 1:div(original_size[3], box_size[3])
                # zip together boxes
                video[(1+(i-1)*box_size[1]):(i*box_size[1]), 
                      (1+(j-1)*box_size[2]):(j*box_size[2]), 
                      (1+(k-1)*box_size[3]):(k*box_size[3])] .= ( boxes[chunk_idx].red, boxes[chunk_idx].green, boxes[chunk_idx].blue ) 
                chunk_idx += 1
            end
        end
    end
    
    # red_norm = normalize_channel(red)
    # green_norm = normalize_channel(green)
    # blue_norm = normalize_channel(blue)
    # red_norm = clamp.(red, 0.0, 1.0)
    # green_norm = clamp.(green, 0.0, 1.0)
    # blue_norm = clamp.(blue, 0.0, 1.0)
    
    # Create RGB tensor with properly normalized values
    # strata = RGB{N0f8}.(red_norm, green_norm, blue_norm)
    return video .|> x -> RGB2(x[1], x[2], x[3])
end

# Normalize each channel to 0-1 range by shifting and scaling
function normalize_channel(channel)
    min_val = minimum(channel)
    max_val = maximum(channel)
    println("Normalizing channel...", " min: ", min_val, " max: ", max_val)
    if max_val <= min_val
        return zeros(size(channel))  # Handle constant arrays
    end
    normalized = (channel .- min_val) ./ (max_val - min_val)
    return clamp.(normalized, 0.0, 1.0)  # Ensure values are in [0,1]
end

struct Sculpture{T}
    tensor::Array{T, 3}
    Xchange::Matrix{T}
    Ychange::Matrix{T}
    Zchange::Matrix{T}
    # chisel?
end

struct VideoStrata{T}
    red_tensor::Array{Array{Float16, 3}, 3}
    green_tensor::Array{Array{Float16, 3}, 3}
    blue_tensor::Array{Array{Float16, 3}, 3}
    Xchanges::Vector{Matrix}
    Ychanges::Vector{Matrix}
    Zchanges::Vector{Matrix}
    box_size::Tuple{Int, Int, Int}
end

"""
Stratify a video tensor into smaller boxes and process each color channel separately.
- `video::Array{RGB{N0f8}, 3}`: The input video tensor
- `box::Tuple{Int, Int, Int}`: The size of the boxes to stratify the video into

Returns
- `VideoStrata`: A struct containing the stratified video and associated changes
"""
function stratifyByBoxes(
        video::Array{RGB{N0f8}, 3}, 
        box::Tuple{Int, Int, Int}
    ) :: VideoStrata
    boxes = boxVideo(video, box)
    # Extract color channels from boxes
    red_boxes = [box.red for box in boxes]
    green_boxes = [box.green for box in boxes]
    blue_boxes = [box.blue for box in boxes]

    println("Stratifying video into boxes of size ", box, 
            " resulting in (", length(boxes), ") total boxes.")
    # stratify each chunk
    
    # println("Processing red channel...")
    red_strata = @showprogress color=:red "Red channel: " map(toSurfaceTensor, red_boxes)
    # println("Processing green channel...")
    green_strata = @showprogress color=:green "Green channel: " map(toSurfaceTensor, green_boxes)
    # println("Processing blue channel...")
    blue_strata = @showprogress color=:blue "Blue channel: " map(toSurfaceTensor, blue_boxes)
    # recombine the chunks into a single stratified tensor
    rd = [Float16.(s.tensor) for s in red_strata]
    gd = [Float16.(s.tensor) for s in green_strata]
    bd = [Float16.(s.tensor) for s in blue_strata]
    
    # strata = assembleVideo(rd, gd, bd, box, size(video))
    
    r_trans = Vector.([(Float16.(s.Xchange),Float16.(s.Ychange),Float16.(s.Zchange)) for s in red_strata])
    g_trans = Vector.([(Float16.(s.Xchange),Float16.(s.Ychange),Float16.(s.Zchange)) for s in green_strata])
    b_trans = Vector.([(Float16.(s.Xchange),Float16.(s.Ychange),Float16.(s.Zchange)) for s in blue_strata])
    return VideoStrata(rd, gd, bd, r_trans, g_trans, b_trans, box)
end

function decode(strata::VideoStrata)
    red_boxes, green_boxes, blue_boxes = boxVideo(strata, strata.box_size)

    # zip together.
    reds = zip(red_boxes, strata.Xchanges)
    greens = zip(green_boxes, strata.Ychanges)
    blues = zip(blue_boxes, strata.Zchanges)
    # assuming orthogonal matrices the inverse is a transpose.
    red_decoded = @showprogress color=:red "Red channel: " [ 
        actAllDirections(box, [trans.Xchange', trans.Ychange', trans.Zchange'] ) for (box, trans) in reds]
    green_decoded = @showprogress color=:green "Green channel: " [ 
        actAllDirections(box, [trans.Xchange', trans.Ychange', trans.Zchange'] ) for (box, trans) in greens]
    blue_decoded = @showprogress color=:blue "Blue channel: " [ 
        actAllDirections(box, [trans.Xchange', trans.Ychange', trans.Zchange'] ) for (box, trans) in blues]

    video = assembleVideo( red_decoded,
                            green_decoded,
                            blue_decoded,
                            bounding_box, size(strata) )
    return video
end

#####################

function plotTensor(tensor::Array{RGB{N0f8}, 3}, threshold::Float64=1e-2; 
                   xlabel::String="X", ylabel::String="Y", zlabel::String="Z",
                   title::String="3D Tensor Visualization")

    # function for rounding to threshold decimal places
    roundToThreshold = x -> round(x, digits=round(Int, -log10(threshold)))
    
    # Get indices of non-zero values in the tensor (check if any RGB component is non-zero)
    indices = findall(x -> x.r != 0 || x.g != 0 || x.b != 0, tensor)
    dims = size(tensor)

    # Extract x, y, z coordinates and RGB values
    x_coords = [idx[1] for idx in indices]
    y_coords = [idx[2] for idx in indices]
    z_coords = [idx[3] for idx in indices]
    
    # Extract RGB values and convert to hex strings for PlotlyJS
    rgb_values = [tensor[idx] for idx in indices]
    colors = ["rgb($(Int(round(rgb.r*255))), $(Int(round(rgb.g*255))), $(Int(round(rgb.b*255))))" for rgb in rgb_values]
    
    # Calculate intensity for marker sizes (using luminance formula)
    intensities = [0.299*rgb.r + 0.587*rgb.g + 0.114*rgb.b for rgb in rgb_values]
    intensities = intensities .|> roundToThreshold
    
    # Scale marker sizes proportional to intensity
    if length(intensities) > 0
        min_val, max_val = extrema(intensities)
        if min_val ≈ max_val
            marker_sizes = fill(5.0, length(intensities))
        else
            marker_sizes = 2.0 .+ 18.0 .* (intensities .- min_val) ./ (max_val - min_val)
        end
    else
        marker_sizes = Float64[]
    end

    println("Plotting $(length(indices)) RGB points...")
    
    # Create 3D scatter plot with RGB colors
    p = PlotlyJS.Plot(scatter3d(
        x=x_coords, 
        y=y_coords, 
        z=z_coords,
        mode="markers",
        marker=attr(size=marker_sizes, opacity=0.7, color=colors)
    ), Layout(
        scene=attr(
            xaxis=attr(range=[1, dims[1]+1], title=xlabel),
            yaxis=attr(range=[1, dims[2]+1], title=ylabel),
            zaxis=attr(range=[1, dims[3]+1], title=zlabel),
            aspectmode="cube"
        ),
        title=title
    ))
    
    return p
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

println("✓ DletoVideo.jl loaded.")