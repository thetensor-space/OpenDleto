# Dleto.jl Documentation

This directory contains the documentation setup for Dleto.jl using Documenter.jl.

## Quick Start

### Build Documentation

```bash
cd docs/
./build_docs.sh
```

Or manually:

```bash
cd docs/
julia --project -e "using Pkg; Pkg.instantiate()"
julia --project make_basic.jl
```

### View Documentation

After building, open `docs/build/index.html` in your web browser.

## File Structure

- `make_basic.jl` - Basic documentation build script
- `make.jl` - Full documentation build script (for when package is complete)
- `build_docs.sh` - Convenient build script 
- `Project.toml` - Documentation dependencies
- `src/` - Documentation source files
  - `index.md` - Homepage
  - `installation.md` - Installation guide
  - `api/` - API reference documentation
  - `examples/` - Usage examples
  - `dev/` - Developer documentation

## Current Status

✅ **Basic documentation generation working**
- Static documentation pages build successfully
- Manual content renders properly  
- Site navigation and styling functional

🔧 **In Progress**
- Automatic API documentation from docstrings
- Integration with full Dleto module (pending dependency resolution)

## Adding Content

### Adding New Pages

1. Create `.md` files in appropriate `src/` subdirectories
2. Add to the `pages` array in `make_basic.jl`
3. Rebuild documentation

### API Documentation

When the full package is buildable, uncomment the relevant sections in `make.jl` and switch to using that instead of `make_basic.jl`.

## Deployment

For GitHub Pages deployment:

1. Update the `repo` field in `make.jl`
2. Uncomment the `deploydocs()` call
3. Set up GitHub Actions workflow

## Dependencies

- **Documenter.jl** - Core documentation generator
- **DocStringExtensions.jl** - Enhanced docstring support  
- **ITensors.jl** - For package compatibility