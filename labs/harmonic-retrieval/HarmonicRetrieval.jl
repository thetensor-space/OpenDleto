# Helpers for the Clustered Frequency Lab: reading a Bruker 2D HSQC, band
# selection, the Hankel third mode, and reading frequencies out of Dleto's
# stratification.  Pure Julia (no NMR packages).
module HarmonicRetrieval

using Dleto, ITensors, LinearAlgebra, Statistics, FFTW, Downloads, Random

export read_bruker_params, read_bruker_hsqc, remove_digital_filter, echo_antiecho, mix_decimate,
       hankel_tensor, reference_peaks, fit_exponential, download_bmrb_experiment,
       synthetic_record, frequency, fft_peaks_1d

human_chisel_name(P::AbstractMatrix; atol::Real = 1e-8) = begin
    Pm = Matrix{Float64}(real.(P))
    val = size(Pm, 2)
    eng = engaged(Pm, atol)
    if size(Pm, 1) == 1 && isapprox(Pm, UniversalChisel(eng); atol = atol)
        return "universal"
    elseif size(Pm, 1) == count(eng) && isapprox(Pm, TuckerChisel(eng); atol = atol)
        return "tucker"
    elseif size(Pm, 1) == count(eng) * (count(eng) - 1) ÷ 2 && isapprox(Pm, CentroidChisel(eng); atol = atol)
        return "centroid"
    elseif size(Pm, 1) == 1
        nz = findall(a -> any(abs.(Pm[:, a]) .> atol), 1:val)
        if length(nz) == 2
            left, right = nz
            scale = Pm[1, left]
            if abs(scale) > atol && isapprox(Pm, scale .* AdjointChisel(val, left, right); atol = atol)
                return "adjoint"
            end
        end
    end
    return "custom"
end

human_operator_name(op) =
    op isa UniversalOp ? "full" :
    op isa SymmetricOp ? "symmetric" :
    op isa DiagonalOp ? "diagonal" :
    op isa AntiSymmetricOp ? "antisymmetric" :
    op isa ScalarOp ? "scalar" :
    op isa EmptyOp ? "empty" : string(nameof(typeof(op)))

function print_stratification_setting(P::AbstractMatrix, Ω)
    println("Chisel matrix: ", human_chisel_name(P), "  ", size(P, 1), "×", size(P, 2))
    println("Operator family by axis: ", join([human_operator_name(op) for op in Ω.localOps], ", "))
    if any(op -> op isa UniversalOp, Ω.localOps)
        println("Note: in this codebase, \"full\" corresponds to `UniversalOp()`.")
    end
end

# ---------------------------------------------------------------- Bruker files
"Parse a Bruker JCAMP parameter file (acqus/acqu2s) into a Dict of scalars/strings."
function read_bruker_params(path::AbstractString)
    d = Dict{String,Any}()
    for line in eachline(path)
        m = match(r"^##\$(\w+)=\s*(.*)$", line); m === nothing && continue
        key, val = m.captures; val = strip(val)
        if startswith(val, "<"); d[key] = strip(val, ['<', '>'])
        elseif startswith(val, "("); continue                      # arrays: skipped
        else; v = tryparse(Float64, val); d[key] = v === nothing ? val : v; end
    end
    return d
end

"""
    read_bruker_hsqc(dir) -> (fid, p)

Read `ser` of a 2D Bruker experiment as complex FIDs (rows = increments in
acquisition order, columns = complex t2 points) with the parameters `p`:
spectral widths (Hz), spectrometer frequencies (MHz), carrier offsets (Hz),
group delay, FnMODE.  Assumes 32-bit integer little-endian data (DTYPA 0,
BYTORDA 0), the common case; errors otherwise.
"""
function read_bruker_hsqc(dir::AbstractString)
    a = read_bruker_params(joinpath(dir, "acqus")); a2 = read_bruker_params(joinpath(dir, "acqu2s"))
    get(a, "DTYPA", 0.0) == 0.0 || error("ser is not int32 (DTYPA = $(a["DTYPA"]))")
    get(a, "BYTORDA", 0.0) == 0.0 || error("ser is big-endian; not handled")
    td2 = Int(a["TD"]); td1 = Int(a2["TD"])
    raw = reinterpret(Int32, read(joinpath(dir, "ser")))
    blk = max(td2, 256)                                 # each FID is padded to a 1 kB multiple (256 int32)
    blk = cld(td2, 256) * 256
    rows = length(raw) ÷ blk
    rows >= td1 || error("ser holds $rows FIDs, acqu2s says $td1")
    fid = Matrix{ComplexF64}(undef, td1, td2 ÷ 2)
    for r in 1:td1
        x = @view raw[(r-1)*blk+1 : (r-1)*blk+td2]
        fid[r, :] = complex.(Float64.(x[1:2:end]), Float64.(x[2:2:end]))
    end
    p = (; sw2 = a["SW_h"], sfo2 = a["SFO1"], o1 = a["O1"], sw1 = a2["SW"] * a2["SFO1"], sfo1 = a2["SFO1"], o2 = a2["O1"],
           grpdly = get(a, "GRPDLY", 0.0), fnmode = Int(get(a2, "FnMODE", 0.0)), ns = Int(get(a, "NS", 0.0)),
           pulprog = get(a, "PULPROG", ""), nuc2 = get(a, "NUC1", ""), nuc1 = get(a2, "NUC1", ""))
    return fid, p
end

"Remove the Bruker digital-filter group delay: shift each FID back by `grpdly` samples (a linear phase in the spectrum) and drop the wrapped-around transient."
function remove_digital_filter(fid::AbstractMatrix, grpdly::Real)
    n = size(fid, 2); k = 0:n-1
    phase = exp.(2π * im * grpdly .* k ./ n)
    out = ifft(fft(fid, 2) .* transpose(phase), 2)
    skip = Int(floor(grpdly + 2))
    return out[:, 1:n-skip]
end

"Combine echo/antiecho pairs (FnMODE 6) into a complex t1: rows 2i-1 and 2i."
function echo_antiecho(fid::AbstractMatrix)
    e = fid[1:2:end, :]; a = fid[2:2:end, :]
    return (e .+ a) .+ im .* (im .* (e .- a))
end

# ---------------------------------------------------------------- band selection
"Mix axis `dim` of `x` to `center` Hz (sampling rate `sw`), low-pass with a Hann-windowed sinc, keep every `D`-th sample."
function mix_decimate(x::AbstractMatrix, dim::Int, sw::Real, center::Real, D::Int)
    n = size(x, dim); t = (0:n-1) ./ sw
    carrier = exp.(-2π * im * center .* t)
    y = dim == 2 ? x .* transpose(carrier) : x .* carrier
    L = 8D; h = [k == 0 ? 1 / D : sin(π * k / D) / (π * k) for k in -L:L] .* [0.5 * (1 + cos(π * k / (L + 1))) for k in -L:L]
    h ./= sum(h)
    m = cld(n, D); out = zeros(ComplexF64, dim == 2 ? (size(x, 1), m) : (m, size(x, 2)))
    for s in 1:m
        c = (s - 1) * D + 1                                   # output sample s sits at input sample c
        acc = zeros(ComplexF64, dim == 2 ? size(x, 1) : size(x, 2))
        for (i, k) in enumerate(-L:L)
            j = c - k; 1 <= j <= n || continue
            acc .+= h[i] .* (dim == 2 ? y[:, j] : y[j, :])
        end
        dim == 2 ? (out[:, s] = acc) : (out[s, :] = acc)
    end
    return out
end

# ---------------------------------------------------------------- tensors and readout
"Hankel third mode on the columns: T[t1, j, k] = Y[t1, j + k - 1], j = 1..J, k = 1..K with J + K - 1 = size(Y, 2)."
function hankel_tensor(Y::AbstractMatrix, K::Int)
    n2 = size(Y, 2); J = n2 - K + 1
    return [Y[t, j + k - 1] for t in 1:size(Y, 1), j in 1:J, k in 1:K]
end

"Damping and frequency (cycles/sample) of a sampled exponential by an amplitude-weighted log-ratio fit."
function fit_exponential(v::AbstractVector)
    r = [v[i+1] / v[i] for i in 1:length(v)-1]; w = abs.(v[1:end-1]).^2
    z = sum(w .* log.(r)) / sum(w)
    return (-real(z), imag(z) / 2π)
end

"Local maxima of a 2D magnitude spectrum above `thr` times its maximum, as (i, j, height) sorted by height."
function reference_peaks(mag::AbstractMatrix; thr = 0.05, radius = 3)
    pk = Tuple{Int,Int,Float64}[]
    m = maximum(mag)
    for i in 1+radius:size(mag,1)-radius, j in 1+radius:size(mag,2)-radius
        v = mag[i, j]; v > thr * m || continue
        v == maximum(@view mag[i-radius:i+radius, j-radius:j+radius]) && push!(pk, (i, j, v))
    end
    return sort(pk; by = x -> -x[3])
end

"Download ser, acqus, acqu2s (and the title) of a BMRB metabolomics experiment into `dir`."
function download_bmrb_experiment(entry::AbstractString, set::AbstractString, exp::Integer, dir::AbstractString)
    base = "https://bmrb.io/ftp/pub/bmrb/metabolomics/entry_directories/$entry/nmr/$set/$exp"
    mkpath(joinpath(dir, "pdata", "1"))
    for f in ("ser", "acqus", "acqu2s", "pulseprogram"); isfile(joinpath(dir, f)) || Downloads.download("$base/$f", joinpath(dir, f)); end
    isfile(joinpath(dir, "pdata", "1", "title")) || Downloads.download("$base/pdata/1/title", joinpath(dir, "pdata", "1", "title"))
    return dir
end

# ---------------------------------------------------------------- notebook conveniences
"Sum of M separable damped exponentials on an n³ grid: T = Σ c_m a_m ⊗ b_m ⊗ c_m, frequencies in cycles/sample."
function synthetic_record(n::Int, fx, fy, fz; damping = 0.02, amps = ones(length(fx)), snr_db = Inf, rng = Random.default_rng())
    e(f) = [exp((-damping + 2π * im * f) * t) for t in 0:n-1]
    T = zeros(ComplexF64, n, n, n)
    for m in eachindex(fx); T .+= amps[m] .* reshape(e(fx[m]), n, 1, 1) .* reshape(e(fy[m]), 1, n, 1) .* reshape(e(fz[m]), 1, 1, n); end
    snr_db == Inf && return T
    return T .+ (norm(T) / sqrt(length(T)) * 10^(-snr_db / 20)) .* (randn(rng, ComplexF64, n, n, n) ./ sqrt(2))
end

"Frequency (cycles/sample, wrapped to (-0.5, 0.5]) of a recovered factor vector."
frequency(v::AbstractVector) = (f = fit_exponential(v)[2]; f - round(f))

"Frequencies the FFT resolves along one axis of T: local maxima of the dominant mode's n-point spectrum above 10% of its peak."
function fft_peaks_1d(T::AbstractArray{<:Number,3}, axis::Int)
    n = size(T, axis)
    U = reshape(permutedims(T, (axis, setdiff(1:3, axis)...)), n, :)
    u = svd(U).U[:, 1]
    S = abs.(fft(u))
    return sort([(i - 1) / n for i in 1:n if S[i] > S[mod1(i - 1, n)] && S[i] >= S[mod1(i + 1, n)] && S[i] > 0.1 * maximum(S)])
end

end # module
