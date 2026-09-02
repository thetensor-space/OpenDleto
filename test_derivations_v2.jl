using Pkg
Pkg.activate(".")
using Random, ITensors, Dleto

function make_sphere_tensor(a, b, c, r; shell_thickness=2.0)
    f(x, y, z) = (x - a)^2 + (y - b)^2 + (z - c)^2
    ds = (ceil(Int, 2r + a), ceil(Int, 2r + b), ceil(Int, 2r + c))
    S = zeros(Float64, ds...)
    for x in 1:ds[1], y in 1:ds[2], z in 1:ds[3]
        if abs(f(x - a, y - b, z - c) - r^2) < shell_thickness
            S[x, y, z] = randn()
        end
    end
    return S
end

Random.seed!(42)
println("Building sphere...")
S = make_sphere_tensor(12, 11, 18, 25)
println("\nRandomizing...")
rn = randomize_tensor(S)
nd = nondeg(rn.Δ)

for t in [1e-6, 1e-10]
    println("\n--- Testing tol=$t ---")
    print("SylverLining: ")
    stratify(nd.Δ; method=:SylverLining, tol=t)
    print("FastDer3Valent: ")
    stratify(nd.Δ; method=:FastDer3Valent, tol=t)
end
