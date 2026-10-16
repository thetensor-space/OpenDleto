"""BMRB 15063 3D HNCO (Varian, 600 MHz): combine the hypercomplex increments into a
complex (15N, 13C', 1H) array, save it with acquisition parameters, and list the
reference peaks from the full record."""
import sys, numpy as np, h5py, nmrglue as ng
from scipy.ndimage import maximum_filter
src, out = sys.argv[1], sys.argv[2]
dic, data = ng.varian.read(src, torder="regular")   # (2*ni2, 2*ni, np/2) complex, phase2 / phase interleaved
pp = dic["procpar"]; g = lambda k: float(pp[k]["values"][0])
sw3, sfrq, tof = g("sw"), g("sfrq"), g("tof"); sw1, dfrq, dof = g("sw1"), g("dfrq"), g("dof"); sw2, dfrq2, dof2 = g("sw2"), g("dfrq2"), g("dof2")
# 13C' (axis 1): States, f1coef empty -> complex = A + iB
A, B = data[:, 0::2, :], data[:, 1::2, :]; d1 = A + 1j * B
# 15N (axis 0): echo/antiecho pairs.  The echo trace alone is a single-quadrant signal
# (the antiecho would add sqrt(2) sensitivity when combined with the right phases; the
# combinations tried produced quadrature images, so only the echo is used).  Conjugated
# so that the amide 1H offsets are positive.
d2 = np.conj(d1[0::2])
fid = d2.astype(np.complex128)                   # (ni2=40 15N, ni=32 13C', 1024 1H)
print("complex 3D FID", fid.shape, "| 1H sw", sw3, "Hz at", sfrq, "MHz | 13C' sw", sw1, "Hz at", dfrq, "MHz | 15N sw", sw2, "Hz at", dfrq2, "MHz")
# reference spectrum: cosine apodization, zero-fill x2 on indirect axes
n0, n1, n2 = fid.shape
w = lambda n: np.cos(np.pi * np.arange(n) / (2 * n))
sp = np.fft.fftshift(np.fft.fftn(fid * w(n0)[:, None, None] * w(n1)[None, :, None] * w(n2)[None, None, :], s=(2 * n0, 2 * n1, n2)))
mag = np.abs(sp); floor = np.median(mag); mad = np.median(np.abs(mag - floor))
pk = (mag == maximum_filter(mag, size=(5, 5, 5))) & (mag > floor + 12 * mad)
ii, jj, kk = np.nonzero(pk); H = mag[ii, jj, kk] / floor
# frequency axes (Hz offsets from carriers; Varian: carrier at tof/dof/dof2 relative to sfrq*... we report ppm = (carrier_hz + offset)/MHz)
F2 = (ii - n0) * sw2 / (2 * n0); F1 = (jj - n1) * sw1 / (2 * n1); F3 = (kk - n2 / 2) * sw3 / n2
# Varian transmitter offsets tof/dof/dof2 are relative to the base frequencies; absolute ppm needs the reference; report offsets in ppm from the carrier
ppmN, ppmC, ppmH = F2 / dfrq2, F1 / dfrq, F3 / sfrq
order = np.argsort(-H)
print(f"3D peaks above floor+12MAD: {len(ii)}; bins: 15N {sw2/n0:.0f} Hz ({sw2/n0/dfrq2:.2f} ppm), 13C' {sw1/n1:.0f} Hz ({sw1/n1/dfrq:.2f} ppm), 1H {sw3/n2:.1f} Hz")
print(" rank  15N(ppm from carrier)  13C'(ppm)  1H(ppm)  height/floor")
for r, k in enumerate(order[:30]): print(f"{r+1:5d} {ppmN[k]:10.2f} {ppmC[k]:10.2f} {ppmH[k]:9.3f} {H[k]:8.1f}")
with h5py.File(out, "w") as h:
    h["fid"] = fid; h["ref_peaks"] = np.c_[F2, F1, F3, H][order]          # Hz offsets: 15N, 13C', 1H
    h.attrs.update(dict(sw1=sw2, sw2=sw1, sw3=sw3, obsN=dfrq2, obsC=dfrq, obsH=sfrq))
# how crowded are 1H bands?
for width in (0.1, 0.15, 0.2, 0.3):
    counts = [(np.abs(ppmH - c) < width / 2).sum() for c in np.arange(ppmH.min(), ppmH.max(), width / 2)]
    print(f"1H band width {width} ppm: peaks per band median {np.median(counts):.0f}, max {max(counts)}")
# candidate 1H bands: strong peaks (> 150 x floor) per 0.12 ppm band and their 13C' spread
strong = H > 150
for c in np.arange(2.8, 3.8, 0.04):
    sel = strong & (np.abs(ppmH - c) < 0.06)
    if sel.sum() < 4: continue
    cs = np.sort(ppmC[sel]); gaps = np.diff(cs)
    print(f"1H band {c-0.06:.2f}-{c+0.06:.2f} ppm: {sel.sum():2d} strong peaks, 13C' spread {cs.min():.2f}..{cs.max():.2f} ppm, min 13C' gap {gaps.min() if len(gaps) else 0:.2f} ppm, 15N spread {np.ptp(ppmN[sel]):.2f} ppm")
e = np.mean(np.abs(fid) ** 2); e_tail = np.mean(np.abs(fid[:, :, -n2 // 8:]) ** 2); print("time-domain energy mean", round(float(e), 1), " 1H tail", round(float(e_tail), 1))
