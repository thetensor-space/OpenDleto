using Documenter

makedocs(;
    authors="Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson",
    sitename="Dleto.jl",
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Installation" => "installation.md", 
        "Examples" => [
            "Basic Usage" => "examples/basic.md",
            "Tensor Operations" => "examples/tensors.md",
        ],
        "Developer Guide" => [
            "Contributing" => "dev/contributing.md",
            "Testing" => "dev/testing.md"
        ]
    ],
    warnonly=true
)

# Success message
println("✓ Documentation built successfully!")
println("📖 Open docs/build/index.html to view the documentation")