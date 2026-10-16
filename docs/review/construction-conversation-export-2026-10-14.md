# Construction Discussion Export (2026-10-14)

This file summarizes the conversation about the synthetic tensor construction in `labs/HNCOFocusedLab.ipynb`, including terminology and modeling choices.

## 1) Notebook construction updates made during the session

- Reworked the warm-up to explicitly construct a dense 6x6x6 tensor in code instead of hiding it behind a helper call.
- Renamed the toy tensor variable from `Ttoy` to `Gamma_synth`.
- Renamed frequency vectors to subscript-style names: `f_x`, `f_y`, `f_z`.
- Updated the oscillator demo cells to show:
  - continuous curve plus sampled points,
  - stronger visible damping,
  - a window that can show growth then decay,
  - multi-frequency carriers from user-specified frequency/amplitude lists.
- Fixed execution bugs from cell refactors (undefined names and a naming collision).
- Added dotted/light plotting style on request.

## 2) Key conceptual Q/A from the discussion

### Q: What is a damped oscillatory component?
A: A wave-like term that oscillates and decays in magnitude. Typical form:

s(t) = A * exp(-alpha * t) * exp(i * (2*pi*f*t + phi))

In the tensor example, each component contributes a separable 3D factor over x/y/z sample indices.

### Q: Why did the first demo not look wave-like?
A: With very short records and low frequency, the real part can look almost monotone. We switched to a higher visible frequency and overlaid envelopes/samples.

### Q: Can we show full continuous wave with discrete points overlaid?
A: Yes. The demo was updated to plot dense continuous time and sample markers on top.

### Q: Can we make damping more obvious?
A: Yes. A larger demo decay rate was used for visualization.

### Q: Can the window show growth then decay?
A: Yes. The envelope was centered at an interior peak and plotted over a finite window.

### Q: Can the wave be a sum of multiple frequencies?
A: Yes. The carrier was changed to a linear combination from lists of frequencies, amplitudes, and optional phases.

### Q: What does "Hz" mean in this context?
A: Hertz means cycles per second. If the x-axis is sample index (not seconds), then frequencies are cycles per sample-step unless sampling rate is specified.

### Q: Canonical symbol for frequency?
A: Usually f. Also nu (nu) in some fields; angular frequency is omega with omega = 2*pi*f.

## 3) Clarification about physical intuition

User intuition discussed:
- "three orthogonal wires plucked independently, center motion in 3D, maybe projection onto (1,1,1)."

Clarification provided:
- The notebook construction is not a vector field projection model.
- It builds a scalar complex-valued field sampled on a 3D grid.
- Tensor entry is a sum across components/axes-of-superposition:

Gamma[i,j,k] = sum_m a_m * exp(i*phi_m) * sx_m(i) * sy_m(j) * sz_m(k)

where each sx/sy/sz is a damped complex exponential along that coordinate.

So the "adding" is superposition/interference of separable scalar components, not dot-product projection of a 3-vector onto (1,1,1).

## 4) Terminology alignment requested by user

- User preference: use "axes" as the name for what was previously referred to as "modes/components" in explanation.
- Agreement: subsequent explanations used that wording preference.

## 5) 2D slow-walk model that user proposed

User proposed 2D construction:
- Two damped oscillators f_x and f_y sampled on x_1..x_6 and y_1..y_4.
- Build vector field V[j,i] = (f_x(x_i), f_y(y_j)) on a 4x6 grid (an order-3 tensor 4x6x2).
- Project with fixed vector (1,1) to get scalar field S[j,i] = f_x(x_i) + f_y(y_j), i.e., a 4x6 matrix.

Assessment given:
- This is mathematically correct and meaningful.
- It is a different model from separable superposition used by notebook synthetic tensor.
- Projection model represents "multi-channel vector measurement + readout direction".
- Notebook model represents "sum of separable damped oscillatory scalar components".

## 6) Latest request before export

User asked if construction explanation could be exported to a file in order to switch LLMs.
This file is that export.
