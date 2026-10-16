# Prepared sub-band records (small, versioned)

Both files are complex time-domain sub-bands cut from public BMRB raw data by the
scripts in `../nmr/`; the raw files are not versioned (they live in `labs/data/`).

| file | source | preparation |
|---|---|---|
| `bmr15063_hnco_band_3.56-3.68ppm.h5` | BMRB entry 15063, `timedomain_data/hnco.fid` (Varian, 600 MHz, 3D HNCO of mouse ThTPase) | `read_hnco.py` (States on 13C', echo trace on 15N, trace order "regular"), `prepare_subband3d.py 3.62 0.06 150` (1H band 3.56–3.68 ppm from carrier, decimated ×150). Dataset `fid` is (1H, 13C', 15N) in HDF5 order, i.e. (15N, 13C', 1H) = 40 × 32 × 7 in Julia after `permutedims(…, (3,2,1))`; `ref_peaks` rows: 15N Hz, 13C' Hz, 1H Hz offsets, height/floor; attributes swN, swC, swH (Hz), obsN, obsC, obsH (MHz), fcH (band centre, Hz). |
| `bmse000005_hsqc_ribose_band.h5` | BMRB metabolomics bmse000005 (AMP 100 mM), `nmr/set01/6` (Bruker 500 MHz edited HSQC) | `read_hsqc.py`, `prepare_subband.py -220 225 14 0 30000 1` (1H 3.8–4.7 ppm, decimated ×14). `fid` is (t2, t1) in HDF5 order = 128 × 32 in Julia after `permutedims`; attributes sw1, sw2 (Hz), f1_center, f2_center. |

Chemical shifts are reported relative to the carriers (ppm from carrier); absolute
referencing was not applied.
