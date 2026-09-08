# docs/beta — the `beta` branch review set

Written 2026-09-08 against `beta` at tag `v1.5-beta-2026-09-04` (113 commits ahead of `main`).

| Document | Read it if you want to know |
|---|---|
| [NEW-FEATURES.md](NEW-FEATURES.md) | what beta gives a *user* that `main` did not, in a page |
| [CHANGELOG.md](CHANGELOG.md) | every change, grouped by beta tag, with commit hashes; API and dependency deltas |
| [REVIEW.md](REVIEW.md) | the code review: verdict, test run, bugs, correctness risks, API and maintainability findings, coverage gaps, hygiene, ordered work list |
| [DESIGN-CHOICES.md](DESIGN-CHOICES.md) | the fifteen design decisions on beta and whether each holds up (keep / keep but finish / revisit) |
| [COMING-FEATURES.md](COMING-FEATURES.md) | what is committed, planned or gated for the next beta releases |
| [review/](review/) | the five detailed slice reviews the consolidated review was built from (QuickDer; null solvers and precision; SylverLining, FastDer and QuickSylver; core types and package spine; repository hygiene) |

Status of the test suite on the reviewed commit: 51 testsets, 14,279 passes, 0 failures,
Julia 1.12, run under `bench/jl` with the beta worktree selected explicitly.
