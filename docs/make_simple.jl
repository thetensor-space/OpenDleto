using Documenter, DocStringExtensions, ITensors

# Create a minimal version of Dleto module just for documentation
module Dleto

using ITensors
using DocStringExtensions

# Include only the core files that don't have heavy dependencies
include("../src/Operators.jl")
using .Operators

# Placeholder for other modules that would be included
export apply, unsafe_apply, member, coapply, unsafe_coapply, asoperator

end # module Dleto

# Define custom CSS for better formatting
css_dir = joinpath(@__DIR__, "src", "assets")
mkpath(css_dir)

# Create custom CSS file
custom_css = """
.docstring .docstring-category {
    margin-top: 1rem;
}
.documenter-sidebar .docs-logo > img {
    max-height: 6rem;
}
"""

open(joinpath(css_dir, "custom.css"), "w") do io
    write(io, custom_css)
end

makedocs(;
    modules=[Dleto, Dleto.Operators],
    authors="Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson",
    sitename="Dleto.jl",
    format=Documenter.HTML(;
        canonical="https://your-username.github.io/OpenDleto",
        assets=["assets/custom.css"],
        edit_link="main"
    ),
    pages=[
        "Home" => "index.md",
        "Installation" => "installation.md",
        "API Reference" => [
            "Core" => "api/core.md",
            "Operators" => "api/operators.md",
            "Chisels" => "api/chisels.md",
            "Derivations" => "api/derivations.md",
            "Transverse Operators" => "api/transverse.md",
            "Utilities" => "api/utils.md"
        ],
        "Examples" => [
            "Basic Usage" => "examples/basic.md",
            "Tensor Operations" => "examples/tensors.md",
            "Detection Algorithms" => "examples/detection.md"
        ],
        "Developer Guide" => [
            "Contributing" => "dev/contributing.md",
            "Testing" => "dev/testing.md"
        ]
    ],
    warnonly=[:missing_docs, :cross_references],
    checkdocs=:none  # Don't check docs for now since we're only including minimal modules
)

# Uncomment the following lines when ready to deploy to GitHub Pages
# deploydocs(;
#     repo="github.com/your-username/OpenDleto.git",
#     devbranch="main"
# )