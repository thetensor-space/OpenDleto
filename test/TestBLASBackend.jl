using Test
using LinearAlgebra

@testset "BLAS backend" begin
    @test string(Dleto.blas_backend()) == string(BLAS.get_config())
    if Sys.isapple() && get(ENV, "DLETO_BLAS", "auto") == "auto"
        version = try
            VersionNumber(strip(read(`/usr/bin/sw_vers -productVersion`, String)))
        catch
            v"0"
        end
        handle = Dleto.Libdl.dlopen_e(Dleto._accelerate_library)
        available = handle != C_NULL && Dleto.Libdl.dlsym_e(handle, "dgemm\$NEWLAPACK\$ILP64") != C_NULL
        handle != C_NULL && Dleto.Libdl.dlclose(handle)
        if version >= v"13.4" && available
            @test occursin("Accelerate", string(Dleto.blas_backend()))
        end
    end
    A = [1.0 2.0; 3.0 5.0]
    @test A * A ≈ [7.0 12.0; 18.0 31.0]
    @test A * (A \ [1.0, 2.0]) ≈ [1.0, 2.0]
    # A fresh process checks the opt-out before Dleto initializes.
    script = "using LinearAlgebra; before = string(BLAS.get_config()); using Dleto; @assert string(Dleto.blas_backend()) == before"
    cmd = `$(Base.julia_cmd()) --project=$(dirname(Base.active_project())) -e $script`
    @test success(addenv(cmd, "DLETO_BLAS" => "openblas"))
end
