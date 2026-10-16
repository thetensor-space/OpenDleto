"""Select a 1H band of the 3D HNCO: mix the direct axis to the band centre and
polyphase-decimate it; keep the indirect axes untouched.  Saves the (15N, 13C', 1H)
complex sub-band record with the in-band reference peaks (Hz offsets from carriers)."""
import sys, numpy as np, h5py
from scipy.signal import resample_poly
src, out, c_ppm, half_ppm, D = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
with h5py.File(src) as h:
    fid = h["fid"][()]; ref = h["ref_peaks"][()]; swN, swC, swH = (float(h.attrs[k]) for k in ("sw1", "sw2", "sw3")); obsN, obsC, obsH = (float(h.attrs[k]) for k in ("obsN", "obsC", "obsH"))
n0, n1, n2 = fid.shape
fc = c_ppm * obsH; half = half_ppm * obsH
t3 = np.arange(n2) / swH
dec = resample_poly(fid * np.exp(-2j * np.pi * fc * t3)[None, None, :], 1, D, axis=2)
inband = (np.abs(ref[:, 2] - fc) < 0.85 * half)
r = ref[inband]
print(f"band {c_ppm - half_ppm:.2f}..{c_ppm + half_ppm:.2f} ppm 1H -> decimated record {dec.shape}, new 1H sw {swH / D:.1f} Hz; {inband.sum()} reference peaks in band (height > 12 MAD)")
print("  strong ones (height/floor > 100):  15N ppm   13C' ppm   1H ppm   height")
for row in r[r[:, 3] > 100]: print(f"      {row[0] / obsN:8.2f} {row[1] / obsC:9.2f} {row[2] / obsH:8.3f} {row[3]:8.0f}")
with h5py.File(out, "w") as h:
    h["fid"] = dec.astype(np.complex128); h["ref_peaks"] = r
    h.attrs.update(dict(swN=swN, swC=swC, swH=swH / D, obsN=obsN, obsC=obsC, obsH=obsH, fcH=fc))
