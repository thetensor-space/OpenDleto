using Test, Dleto, ITensors, LinearAlgebra, Random, Plots, PlotlyJS

@testset "Derivation spectrum widget" begin
    rng = MersenneTwister(20261118)
    dims = (3, 4, 5)
    # SphereLab encodes each coordinate as i² - r²/3 and samples the
    # surface where the sum is near zero. Seed a task-local RNG so the
    # synthesis does not reset the test runner's random stream.
    radius = 3.0
    axes = [Float64[i^2 - radius^2 / 3 for i in 0:(d - 1)] for d in dims]
    wrapped = fetch(@async begin
        Random.seed!(20261118)
        Dleto.randSurfaceTensor(axes..., 1.5)
    end)
    Γ = Dleto.TensorSpace.unwrap(wrapped)
    _, N = Dleto.sylvesterLM(Dleto.UniversalOps(Γ), Dleto.UniversalChisel(3), Γ)
    F = svd(Matrix(N); full=false)
    n = length(F.S)
    # Supply unsorted indices to check the slider follows the graph left to right.
    chosen = [1, n, 2]
    fig = Dleto.derivation_spectrum_widget(wrapped; factorization=F,
        descending_indices=chosen, initial_descending=2, max_points=7, max_frames=3)

    @test fig isa PlotlyJS.Plot
    @test length(fig.data) == 6
    @test length(fig.frames) == 3
    @test [f[:data][1][:x][1] for f in fig.frames] == [1, 2, n]
    @test all(f -> f[:traces] == [1, 2, 3, 4, 5], fig.frames)
    @test [step[:label] for step in fig.layout[:sliders][1][:steps]] == ["1", "2", string(n)]
    @test fig.layout[:xaxis][:domain][2] <= 0.3
    @test fig.layout[:sliders][1][:active] == 1
    @test fig.data[2][:x] == [2]
    @test fig.data[2][:y][1] ≈ F.S[2] / F.S[1]
    @test length(fig.layout[:updatemenus]) == 1
    @test [b[:label] for b in fig.layout[:updatemenus][1][:buttons]] ==
        ["Proportional · monochrome", "Equal · monochrome",
         "Proportional · by value", "Equal · by value"]
    @test [fig.data[i][:visible] for i in 3:6] == [true, false, false, false]
    for frame in fig.frames
        points = frame[:data][2]
        @test points[:marker][:color] == "royalblue"
        @test length(points[:x]) == length(points[:y]) == length(points[:z]) == 7
        @test frame[:data][3][:marker][:size] == fill(5.0, 7)
        @test frame[:data][4][:marker][:color] == points[:customdata]
        @test frame[:data][5][:marker][:color] == points[:customdata]
        @test all(v -> !haskey(v, :visible), frame[:data][2:5])
        for (coord, dim) in zip((points[:x], points[:y], points[:z]), dims)
            @test all(x -> 1 <= x <= dim, coord)
        end
    end
    alternative = Dleto.derivation_spectrum_widget(wrapped; factorization=F,
        descending_indices=[1], size_mode=:equal, color_mode=:value)
    @test alternative.layout[:updatemenus][1][:active] == 3
    @test [alternative.data[i][:visible] for i in 3:6] ==
        [false, false, false, true]
    @test_throws ArgumentError Dleto.derivation_spectrum_widget(wrapped;
        size_mode=:unknown)

    rank2 = Dleto.TensorSpace.tensor(ITensor(randn(rng, 3, 4), Index(3), Index(4)))
    @test_throws ArgumentError Dleto.derivation_spectrum_widget(rank2)
    @test_throws ArgumentError Dleto.derivation_spectrum_widget(wrapped; factorization=F,
        descending_indices=[2, 2])
    @test_throws ArgumentError Dleto.derivation_spectrum_widget(wrapped; factorization=F,
        descending_indices=chosen, max_frames=2)
    @test_throws ArgumentError Dleto.derivation_spectrum_widget(wrapped;
        mode=:full, max_frames=2)
    @test_throws ArgumentError Dleto.derivation_spectrum_widget(wrapped;
        mode=:full, descending_indices=chosen, max_dense_bytes=1)
    partial = Dleto.derivation_spectrum_widget(wrapped;
        mode=:near_null, window=3, max_points=7)
    @test length(partial.frames) == 3
    @test occursin("near-null window: 3", partial.layout[:title])
    @test partial.layout[:xaxis][:title] == "Near-null candidate index"
    @test issorted(partial.data[1][:y]; rev=true)
    windowed = Dleto.derivation_spectrum_widget(wrapped;
        mode=:window, window=3, max_points=7)
    @test length(windowed.frames) == 3
    @test length(windowed.data[1][:x]) == n
    @test occursin("full singular values; 3D window: 3", windowed.layout[:title])
    @test windowed.layout[:xaxis][:title] == "Descending SVD index"
    @test [f[:data][1][:x][1] for f in windowed.frames] == collect(n-2:n)
    @test [step[:label] for step in windowed.layout[:sliders][1][:steps]] ==
        string.(n-2:n)
    @test length(windowed.layout[:shapes]) == 1
    @test windowed.layout[:shapes][1][:x0] == n-2.5
    @test issorted(windowed.data[1][:y]; rev=true)
    @test windowed.data[2][:customdata][1] ≈ F.S[end] / F.S[1]
    auto_window = Dleto.derivation_spectrum_widget(wrapped;
        max_frames=3, window=3, max_points=7)
    @test occursin("full singular values; 3D window: 3", auto_window.layout[:title])
    reused_window = Dleto.derivation_spectrum_widget(wrapped;
        factorization=F, max_frames=3, window=3, max_points=7)
    @test [f[:data][1][:x][1] for f in reused_window.frames] == collect(n-2:n)
    automatic = Dleto.derivation_spectrum_widget(wrapped;
        max_dense_bytes=1, window=2, max_points=7)
    @test length(automatic.frames) == 2
    @test occursin("near-null window: 2", automatic.layout[:title])
    fractional = Dleto.derivation_spectrum_widget(wrapped;
        mode=:near_null, window=0.04, max_points=7)
    @test length(fractional.frames) == 2
    @test_throws ArgumentError Dleto.derivation_spectrum_widget(wrapped;
        mode=:near_null, window=1.5)
    @test_throws BoundsError Dleto.derivation_spectrum_widget(wrapped; factorization=F,
        descending_indices=[n + 1])
    @test_throws DimensionMismatch Dleto.derivation_spectrum_widget(wrapped;
        factorization=(; S=F.S, V=zeros(size(F.V, 1) + 1, size(F.V, 2))),
        descending_indices=chosen)

    sample = zeros(3, 3, 3)
    sample[1, 1, 1] = 1
    sample[2, 2, 2] = -2
    interactive = Dleto.plot_tensor(Dleto.TensorSpace.tensor(sample);
        controls=true)
    @test interactive isa PlotlyJS.Plot
    @test length(interactive.layout[:updatemenus]) == 2
    @test interactive.data[1][:marker][:size] == [2.5, 5.0]
    @test interactive.layout[:updatemenus][1][:buttons][2][:label] == "Equal"
    @test interactive.layout[:updatemenus][2][:buttons][2][:label] == "By value"
    equal_color = Dleto.plot_tensor(Dleto.TensorSpace.tensor(sample);
        controls=true, size_mode=:equal, color_mode=:value)
    @test equal_color.data[1][:marker][:size] == [5.0, 5.0]
    @test equal_color.data[1][:marker][:color] == [1.0, -2.0]
    @test equal_color.layout[:updatemenus][1][:active] == 1
    @test equal_color.layout[:updatemenus][2][:active] == 1
    @test_throws ArgumentError Dleto.plot_tensor(Dleto.TensorSpace.tensor(sample);
        controls=true, size_mode=:unknown)
end
