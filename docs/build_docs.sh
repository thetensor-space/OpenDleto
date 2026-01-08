#!/bin/bash

# Build Documentation for Dleto.jl
# This script sets up the environment and builds the documentation

echo "🏗️  Setting up documentation environment..."

# Navigate to docs directory
cd "$(dirname "$0")"

# Install dependencies
echo "📦 Installing documentation dependencies..."
julia --project -e "using Pkg; Pkg.instantiate()"

# Build documentation
echo "📖 Building documentation..."
julia --project make_basic.jl

# Check if build was successful
if [ -f "build/index.html" ]; then
    echo "✅ Documentation built successfully!"
    echo ""
    echo "📂 Documentation location: docs/build/"
    echo "🌐 Open docs/build/index.html in your browser to view"
    echo ""
    echo "💡 To view locally, you can run:"
    echo "   open docs/build/index.html  # macOS"
    echo "   xdg-open docs/build/index.html  # Linux"
else
    echo "❌ Documentation build failed!"
    exit 1
fi