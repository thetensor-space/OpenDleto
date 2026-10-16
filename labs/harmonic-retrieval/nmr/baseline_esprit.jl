# ESPRIT (multiple-snapshot, Hankel-stacked) baseline on the same decimated sub-band,
# per axis: are the clustered 13C shifts separable from this record at all?
using HDF5, LinearAlgebra, Statistics, Printf
path = ARGS[1]; N1s = parse.(Int, split(ARGS[2], ",")); Ms = parse.(Int, split(ARGS[3], ","))
fid, ref, sw1, sw2, f1c, f2c = h5open(path, "r") do h
    read(h["fid"]), read(h["ref_peaks"]), read_attribute(h, "sw1"), read_attribute(h, "sw2"), read_attribute(h, "f1_center"), read_attribute(h, "f2_center")
end
fid = permutedims(fid); ref = permutedims(ref)
tohz(f, sw, c) = ((f - round(f)) * sw + c)
"ESPRIT along axis 1 of Y (columns are snapshots): Hankel window L, model order M."
function esprit(Y::AbstractMatrix, L::Int, M::Int)
    N = size(Y, 1); H = hcat([hcat([Y[i:i+L-1, c] for i in 1:N-L+1]...) for c in 1:size(Y, 2)]...)
    U = svd(H).U[:, 1:M]
    Φ = pinv(U[1:end-1, :]) * U[2:end, :]
    z = eigvals(Φ)
    return [(-real(log(zi)), imag(log(zi)) / 2π) for zi in z], svdvals(H)
end
println("reference (F1, F2 Hz): ", [(round(ref[r,1]), round(ref[r,2])) for r in 1:size(ref,1)])
for N1 in N1s
    Y1 = fid[1:N1, :]                       # t1 along rows, t2 columns as snapshots
    L1 = max(2, min(N1 ÷ 2, 24))
    for M in Ms
        est, sv = esprit(Y1, L1, M)
        f1 = sort([tohz(e[2], sw1, f1c) for e in est])
        @printf("N1=%3d M=%d  ESPRIT 13C (Hz): %s   | singular values of the stacked Hankel: %s\n", N1, M, join([@sprintf("%8.1f", x) for x in f1], " "), join([@sprintf("%.2f", s / sv[1]) for s in sv[1:min(6, end)]], " "))
    end
end
# t2 axis (1H), snapshots = t1 rows
Y2 = permutedims(fid); est, sv = esprit(Y2, max(2, size(Y2, 1) ÷ 2), 3)
println("ESPRIT 1H (Hz), M=3, full t1 as snapshots: ", join([@sprintf("%7.1f", tohz(e[2], sw2, f2c)) for e in est], " "), "  (reference 326, 346, 373)")
