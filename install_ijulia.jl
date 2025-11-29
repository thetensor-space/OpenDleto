using Pkg
Pkg.add("IJulia")
using IJulia
try
  installkernel()
catch e
  println("installkernel error:", e)
end
