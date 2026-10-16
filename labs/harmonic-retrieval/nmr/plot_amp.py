"""Figure: zero-filled 2D FFT of the short AMP ribose record against Dleto's
components (from the log) and the full-record reference peaks."""
import sys, re, numpy as np, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
h5path, log, out = sys.argv[1], sys.argv[2], sys.argv[3]
with h5py.File(h5path) as h:
    fid = h["fid"][()]; ref = h["ref_peaks"][()]; sw1 = float(h.attrs["sw1"]); sw2 = float(h.attrs["sw2"]); f1c = float(h.attrs["f1_center"]); f2c = float(h.attrs["f2_center"])
sfo1, sfo2, o1, o2 = 125.69555121378, 499.842349248, 2349.248, 10683.21378     # AMP entry (bmse000005 exp 6)
ppm1 = lambda f1: (o2 + f1) / sfo1; ppm2 = lambda f2: (o1 + f2) / sfo2
blocks = re.split(r"\nN1=", open(log).read())[1:]
panels = []
for b in blocks:
    N1 = int(b.split()[0]); comps = [(float(m.group(1)), float(m.group(2))) for m in re.finditer(r"F1 +(-?[\d.]+) Hz, F2 +(-?[\d.]+) Hz", b)]
    panels.append((N1, comps))
fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5), sharey=True)
strong = ref[ref[:, 2] > 100]
for ax, (N1, comps) in zip(axes, panels):
    blk = fid[:N1, :]; n2 = blk.shape[1]
    w1 = np.cos(np.pi * np.arange(N1) / (2 * N1)); w2 = np.cos(np.pi * np.arange(n2) / (2 * n2))
    sp = np.abs(np.fft.fftshift(np.fft.fft2(blk * w1[:, None] * w2[None, :], s=(16 * N1, 8 * n2))))
    F1 = (np.arange(16 * N1) - 8 * N1) * sw1 / (16 * N1) + f1c; F2 = (np.arange(8 * n2) - 4 * n2) * sw2 / (8 * n2) + f2c
    ax.contour(ppm2(F2), ppm1(F1), sp / sp.max(), levels=np.geomspace(0.05, 1, 10), cmap="Greys", linewidths=0.7)
    ax.scatter(ppm2(strong[:, 1]), ppm1(strong[:, 0]), s=90, facecolors="none", edgecolors="tab:blue", linewidths=1.5, label="reference (full record)")
    ax.scatter([ppm2(c[1]) for c in comps], [ppm1(c[0]) for c in comps], s=40, marker="x", c="tab:red", linewidths=2, label="Dleto components")
    ax.set_title(f"{N1} 13C increments: FFT bin {sw1 / N1 / sfo1:.1f} ppm"); ax.set_xlabel("1H (ppm)"); ax.invert_xaxis()
    ax.set_ylim(ppm1(f1c - 3200), ppm1(f1c + 600)); ax.invert_yaxis()
axes[0].set_ylabel("13C (ppm)"); axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle("AMP ribose carbons from short HSQC records: zero-filled FFT (contours) vs Dleto (crosses)")
plt.tight_layout(); plt.savefig(out, dpi=120); print("saved", out)
