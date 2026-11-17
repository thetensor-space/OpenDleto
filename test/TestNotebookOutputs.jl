using JSON
using Test

@testset "Notebooks have no saved computation" begin
    labs = joinpath(@__DIR__, "..", "labs")
    for (root, _, files) in walkdir(labs), file in files
        endswith(file, ".ipynb") || continue
        path = joinpath(root, file)
        notebook = JSON.parsefile(path)
        @test !haskey(get(notebook, "metadata", Dict()), "widgets")
        for cell in notebook["cells"]
            get(cell, "cell_type", "") == "code" || continue
            @test get(cell, "execution_count", nothing) === nothing
            @test isempty(get(cell, "outputs", []))
        end
    end
end
