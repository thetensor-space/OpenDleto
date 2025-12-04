# Installing Dleto

Dleto is family of tools written in Julia for finding sparsity patterns in tensors. While in principle this is compatible with Julia 1.7, we have successfully tested it with later versions as well.  To test your own installation you can run the following.

> **Notice.** At present we have make Dleto a group of files to include but not a formal Julia package to be installed by the Julia package manager.  This is a point of embarrassement and frustration as we have not successfully wrangle Julia's package management creation.  Since `OpenDleto` is an open software project, we welcome and invite support on this desired future goal.

## Table of Contents
- [Installing Dleto](#installing-dleto)
- [Accessing through Notebooks](#accessing-through-notebooks)
    - [Testing notebook Julia access](#testing-notebook-julia-access)
- [Command line Julia](#command-line-getting-julia)

## Accessing through Notebooks
Most notebook envorinments that support Julia can run Dleto directly.  You may need to install [Julia](https://julialang.org) and [IJulia](https://julialang.github.io/IJulia.jl/stable/).  We have found general purpose generative AI models a good source for trouble shooting installation of these systems.

### Testing notebook Julia access

If using a Jupyter notebook you can run 
```julia
using IJulia
println("Julia version: ", VERSION) 
```
A successful output would read something like this. 
```notebook
Julia kernel is active!
Julia version: 1.10.3
```
When you first create or run your notebook you may be ask you to select a kernel.  Select the Julia kernel of your choice.  If you do not see a Julia kernel then your Julia installation is not configured for use that notebook.  Follow online instruction to install Julia for Jupyter notebooks.

Once you succeed in selecting Julia for your notebook, if the above test fails try adding this command to install IJulia.  Note you only need to install IJulia once.
```julia
using Pkg
Pkg.add("IJulia")
```
**Note.** After installing IJulia, some notebooks need to be restarted to work properly.

## Command line Julia

For the command line version you need only a suitable installtion of Julia.
```bash
$ julia --version
Julia version: 1.10.3
```
If you do not have version of Julia installed visit [julialang.org](https://julialang.org).

---

<sub>CC-BY 2025 Brooksbank, Kassabov, Wilson</sub>
