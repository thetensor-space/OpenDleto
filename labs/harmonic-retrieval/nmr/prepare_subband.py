"""Band-select and decimate the nmrglue 2D test FID (15N-13C correlation) to a
sub-band holding a handful of peaks.  Declared preprocessing: mix each time axis
to the band centre and low-pass decimate (scipy.signal.resample_poly, zero-phase
FIR).  Output: HDF5 with the decimated complex FID, the new spectral widths,
carrier offsets, and the reference peak list from the full record."""
import sys, numpy as np, h5py
from scipy.signal import resample_poly
from scipy.ndimage import maximum_filter
src, out = sys.argv[1], sys.argv[2]
f2_center, f2_half, D2 = float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])     # F2 band (Hz offsets from carrier), t2 decimation
f1_center, f1_half, D1 = float(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8])     # F1 band, t1 decimation (1 = none)
with h5py.File(src) as h:
    fid = h['fid'][()]
    sw1 = float(h.attrs['sw1']) if 'sw1' in h.attrs else float(h.attrs['FDF1SW']); sw2 = float(h.attrs['sw2']) if 'sw2' in h.attrs else float(h.attrs['FDF2SW'])
n1, n2 = fid.shape
# reference peaks from the full record (cosine apodized FT)
w1 = np.cos(np.pi * np.arange(n1) / (2 * n1)); w2 = np.cos(np.pi * np.arange(n2) / (2 * n2))
mag = np.abs(np.fft.fftshift(np.fft.fft2(fid * w1[:, None] * w2[None, :], s=(2 * n1, 2 * n2))))
floor = np.median(mag); mad = np.median(np.abs(mag - floor))
pk = (mag == maximum_filter(mag, size=5)) & (mag > floor + 10 * mad)
ii, jj = np.nonzero(pk); F1 = (ii - n1) * sw1 / (2 * n1); F2 = (jj - n2) * sw2 / (2 * n2); H = mag[ii, jj] / floor
inband = (np.abs(F2 - f2_center) < f2_half * 0.85) & (np.abs(F1 - f1_center) < f1_half * 0.85)
ref = np.c_[F1[inband], F2[inband], H[inband]]; ref = ref[np.argsort(-ref[:, 2])]
print("reference peaks in the sub-band (F1 Hz, F2 Hz, height/floor):"); print(np.round(ref, 1))
# mix to band centre and decimate.  Sample k of axis a is at time k/sw_a.
t1 = np.arange(n1) / sw1; t2 = np.arange(n2) / sw2
mixed = fid * np.exp(-2j * np.pi * f1_center * t1)[:, None] * np.exp(-2j * np.pi * f2_center * t2)[None, :]
dec = resample_poly(mixed, 1, D2, axis=1)          # zero-phase FIR low-pass + decimate along t2
dec = resample_poly(dec, 1, D1, axis=0) if D1 > 1 else dec
sw1n, sw2n = sw1 / D1, sw2 / D2
print("decimated FID shape", dec.shape, " new spectral widths (Hz): F1", sw1n, " F2", sw2n)
# noise estimate: the tail of the decimated t2 axis carries little signal (T2 decay); compare energies
e = np.mean(np.abs(dec) ** 2); e_tail = np.mean(np.abs(dec[:, -3:]) ** 2)
print("decimated record: mean energy", round(float(e), 2), " last-3-samples energy", round(float(e_tail), 2))
with h5py.File(out, "w") as h:
    h["fid"] = dec.astype(np.complex128); h["ref_peaks"] = ref
    h.attrs.update(dict(sw1=sw1n, sw2=sw2n, f1_center=f1_center, f2_center=f2_center, D1=D1, D2=D2, sw1_orig=sw1, sw2_orig=sw2))
# what the FFT of a SHORT record of this sub-band shows: first 16 t1 points x 12 t2 points, zero-filled x8
n2d = dec.shape[1]
for N1 in (dec.shape[0], dec.shape[0] // 2, dec.shape[0] // 4):
    blk = dec[:N1, :]; spec = np.abs(np.fft.fftshift(np.fft.fft2(blk, s=(8 * N1, 8 * n2d))))
    thr = 0.15 * spec.max(); pks = (spec == maximum_filter(spec, size=9)) & (spec > thr)
    print(f"FFT of the short record ({N1} x {n2d}, zero-filled x8): {pks.sum()} peaks above 15% of max; resolution F1 {sw1n / N1:.0f} Hz, F2 {sw2n / n2d:.0f} Hz")
