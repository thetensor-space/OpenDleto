using Dleto, ITensors, LinearAlgebra, Statistics, Metal
BLAS.set_num_threads(1)
folder=ARGS[1]
function med(f)
    for _ in 1:5; Dleto.gpu_sync(f); end
    xs=Float64[]
    for _ in 1:11
        t=time_ns();Dleto.gpu_sync(f);push!(xs,(time_ns()-t)/1e6)
    end
    median(xs)
end
for n in (64,96,128)
    T=Float32.(reshape(collect(reinterpret(Float64, read(joinpath(folder,"tensor_$n.bin")))),n,n,n))
    x=Float32.(collect(reinterpret(Float64, read(joinpath(folder,"x_$n.bin")))))
    fr=[Index(n,"axis$a") for a in 1:3]
    Γ=ITensor(T,fr...); Ω=IndTransverseOps(fr,UniversalOp());P=UniversalChisel(3)
    for backend in (:array,:metal)
        A,_=sylvesterLM(Ω,P,Γ;backend)
        println("NORMAL n=$n backend=$backend ms=$(med(() -> A*x))")
        flush(stdout)
    end
end
