"""Read the cyclosporin edited HSQC (Zenodo 13307304, experiment 12, Bruker AV500,
CC BY 4.0): remove the digital filter, combine echo/antiecho pairs into a complex
t1, save the complex (t1, t2) FID with parameters, and list reference peaks from
the full record."""
import sys, numpy as np, h5py, nmrglue as ng
from scipy.ndimage import maximum_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src, out, fig = sys.argv[1], sys.argv[2], sys.argv[3]
dic, data = ng.bruker.read(src)
data = ng.bruker.remove_digital_filter(dic, data)
print("raw ser shape after filter removal", data.shape, data.dtype)
acq, acq2 = dic["acqus"], dic["acqu2s"]
sw2 = acq["SW_h"]; sfo2 = acq["SFO1"]; sfo1 = acq2["SFO1"]; sw1 = acq2["SW"] * sfo1; o1 = acq["O1"]; o2 = acq2["O1"]
print("F2 (1H): SW", sw2, "Hz, SFO1", sfo2, "MHz, offset", o1, " | F1 (13C): SW", sw1, "Hz, SFO1", sfo1, "MHz, offset", o2, " FnMODE", acq2["FnMODE"])
# echo/antiecho -> States-type complex t1 (constant factors and the sign convention only mirror F1)
echo, anti = data[0::2], data[1::2]
cos_part, sin_part = echo + anti, 1j * (echo - anti)
fid = cos_part + 1j * sin_part                     # complex along t1 (hypercomplex collapsed)
n1, n2 = fid.shape; print("complex (t1, t2) shape", fid.shape)
# reference spectrum from the full record (cosine apodization, zero fill x2)
w1 = np.cos(np.pi * np.arange(n1) / (2 * n1)); w2 = np.cos(np.pi * np.arange(n2) / (2 * n2))
sp = np.fft.fftshift(np.fft.fft2(fid * w1[:, None] * w2[None, :], s=(2 * n1, 2 * n2))); mag = np.abs(sp)
floor = np.median(mag); mad = np.median(np.abs(mag - floor))
pk = (mag == maximum_filter(mag, size=7)) & (mag > floor + 15 * mad)
ii, jj = np.nonzero(pk); H = mag[ii, jj] / floor
F1 = (ii - n1) * sw1 / (2 * n1); F2 = (jj - n2) * sw2 / (2 * n2)      # Hz offsets from the carriers
ppm1 = o2 / sfo1 + F1 / sfo1; ppm2 = o1 / sfo2 + F2 / sfo2
order = np.argsort(-H)
print(f"peaks above floor+15MAD: {len(ii)}; floor {floor:.0f}; bin widths F1 {sw1/n1:.0f} Hz ({sw1/n1/sfo1:.2f} ppm), F2 {sw2/n2:.1f} Hz")
print(" rank   F1(Hz)  13C ppm   F2(Hz)  1H ppm  height/floor")
for r, k in enumerate(order[:60]):
    print(f"{r+1:5d} {F1[k]:8.1f} {ppm1[k]:8.2f} {F2[k]:8.1f} {ppm2[k]:7.2f} {H[k]:8.1f}")
with h5py.File(out, "w") as h:
    h["fid"] = fid.astype(np.complex128); h["ref_peaks"] = np.c_[F1, F2, H, ppm1, ppm2][order]
    h.attrs.update(dict(sw1=sw1, sw2=sw2, sfo1=sfo1, sfo2=sfo2, o1=o1, o2=o2))
e = np.mean(np.abs(fid) ** 2); e_tail = np.mean(np.abs(fid[:, -n2 // 10:]) ** 2); e_tail1 = np.mean(np.abs(fid[-n1 // 8:, :]) ** 2)
print("time-domain energy: mean", round(float(e), 1), " t2 tail", round(float(e_tail), 1), " t1 tail", round(float(e_tail1), 1))
plt.figure(figsize=(11, 7)); plt.imshow(mag ** 0.5, aspect="auto", origin="lower", cmap="magma", extent=[ppm2.min() if False else (o1 - sw2/2)/sfo2, (o1 + sw2/2)/sfo2, (o2 - sw1/2)/sfo1, (o2 + sw1/2)/sfo1])
plt.gca().invert_xaxis(); plt.gca().invert_yaxis(); plt.xlabel("1H ppm"); plt.ylabel("13C ppm"); plt.title("cyclosporin edited HSQC, full record (sqrt magnitude)"); plt.savefig(fig, dpi=110)
