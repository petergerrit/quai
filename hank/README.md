# SWIFTbot

Subgroup Workflow for Identifying Fault-tolerant T-extensions. A five-stage
pipeline for numerically surveying finite-subgroup + single-gate extensions of
`SU(d)` by the quantum circuit overhead `Q_T` of Słowik, Dulian, and
Sawicki (arXiv:2505.00683). Companion code for Fleming, Lamm,
Tame-Narvaez, and Vander Griend, *Efficient gate-set extensions for fault-tolerant
qudit quantum computation* (FERMILAB-PUB-26-TBA-T, in preparation).

Practical qudit fault-tolerant computation needs a finite gate set that is
both universal and efficient: universal so that short words are dense in
`SU(d)`, and efficient so that the approximation exponent `Q_T` is small.
SWIFTbot scores candidate `(base group, extension)` pairs against `Q_T`
and---separately---against a mean-depth figure of merit on structured
target families. A curated QEC-code and distillation catalogue flags
which passing pairs currently admit an end-to-end fault-tolerant stack.

## Installation

Python 3.12 is the supported interpreter (3.11 works but is not tested on
CI). All compute paths run locally with `numpy` + `scipy`; the optional LLM
supervisor calls the Anthropic API.

```bash
git clone https://github.com/petergerrit/quai.git
cd quai/hank
python -m venv swiftbot/.venv
source swiftbot/.venv/bin/activate
pip install -r swiftbot/requirements.txt
```

Set `ANTHROPIC_API_KEY` only if you want to use the `explore` / `sweep`
commands, which delegate group ranking and extension proposals to an LLM
supervisor. Every paper figure is produced by the deterministic panel
commands (`q-panel`, `cover-panel`, `codes`), which do not require an API
key.

## Quickstart (30 seconds)

Browse the curated QEC-code and distillation catalogue (no compute):

```bash
python -m swiftbot.cli codes --dim 2 --distillation
```

Runs in well under a second; prints one line per code with its
`[[n,k,d]]_q` parameters, transversal groups, and reference.

Run a minimal Q_T panel on one `d=3` group, three extensions (two fixed,
one Haar-random batch of three), parallelised across two threads:

```bash
python -m swiftbot.cli q-panel \
    --panel d3_survey --groups S216 --t 3 \
    --rnd-samples 3 --workers 2 \
    --run-id demo --db /tmp/swiftbot_demo.db
```

Expected output: a progress line per panel entry on stderr, then a ranked
table with columns `group | extension | kind | n | best_δ | mean_δ | std_δ |
Q_T(b) | Q_T(μ) | Q_opt | sec`, followed by the per-row JSON dump on
stdout. The first call on a group warms the Π(g) cache under
`swiftbot/.pi_cache/`; subsequent panels on `S216` skip that cost.
Wall time for this example is well under a minute on a laptop.

## Reproducing the paper

- **Table 1 (d=2 validation, `tab:d2_validation`)**: run
  ```bash
  python sweep_runs/manual_d2_paper_validation.py
  ```
  Reproduces the Clifford + P(π/4) `Q_T ≈ 52` row and the
  Haar-random `Q_T ≈ 3.79` / `4.34` rows against the Słowik *et al.*
  reference values. Takes ~10 minutes on a single workstation.
- **Table 2 (d=3 Tier 1 `Q_T` survey, `tab:d3_tier1`)**:
  ```bash
  OMP_NUM_THREADS=1 python -m swiftbot.cli q-panel \
      --panel d3_survey --t 5 --rnd-samples 10 --workers 8 \
      --run-id d3_tier1 --db sweep_runs/d3_tier1.db
  ```
  3 base groups (`S216`, `S648`, `S1080`) × 8 extensions. On a 32-core
  box with BLAS pinned to one thread, completes in ~17 minutes.
- **Table 3 (d=3 apples-to-apples Haar panel, `tab:d3_apples`)**:
  ```bash
  OMP_NUM_THREADS=1 python -m swiftbot.cli q-panel \
      --panel d3_rnd_apples --t 5 --workers 8 \
      --run-id d3_apples --db sweep_runs/d3_apples.db
  ```
  Same 3 groups × 10 fixed Haar SU(3) matrices (seed 42). Completes in
  ~2 minutes once the Π(g) cache built by the Tier 1 run is warm.
- **Table 4 (DSA Σ(36×3) target-family coverage, `tab:lamm_sigma36`)**:
  ```bash
  python -m swiftbot.cli cover-panel \
      --family lamm_sigma36 --base clifford \
      --panel lamm_d2 --workers 7 \
      --db sweep_runs/cover_panel.db
  ```
  BFS word-tree coverage of 7 candidate extensions against the DSA
  (discrete subgroup approximation) primitive-gate target family `T_36`
  of Lamm *et al.* (arXiv:1903.08807). Completes in ~5 minutes. The
  `lamm_sigma36` / `lamm_d2` identifiers are retained as internal symbol
  names for backward compatibility with pre-`v0.2` sweep artefacts; the
  paper refers to this family as the DSA primitive-gate target family.

All four runs write a single-file SQLite cache under `sweep_runs/`; rerun
the same command and only the missing rows are recomputed (content-
addressable keys). The `q-panel-summary` subcommand aggregates a stored
run into a ranked best + mean ± std table.

### Π(g) cache

Stage 3 builds the per-irrep `π_λ(c)` matrices for each base group once
per `(group, t-design weight)` pair and persists them under
`swiftbot/.pi_cache/` as pickled ndarrays. The default location can be
overridden per run:

```bash
export QCO_PI_CACHE_DIR=/scratch/swiftbot_pi_cache
```

The first `q-panel` invocation on `S648` at `t=5` takes ~2 minutes to
build and write the cache; subsequent panels against the same group at
the same `t` load it in a few seconds. The cache is keyed by a
fingerprint of the `C` array and the highest weight `λ`, so
editing the group registry invalidates stale entries automatically.

## Pipeline stages

The five-stage architecture described in §4 of the paper maps to the
repository as follows.

- **Stage 1: group discovery and registration** --- `swiftbot/tools/groups.py`.
  A pre-seeded registry of 16 groups (`d ∈ {2,3,4}`) loaded byte-for-byte
  from the canonical `qco-main_opt/*.txt` gate lists, plus
  `clifford_t/genGROUP.py` for on-demand projective closure of arbitrary
  generator sets. `register_custom()` adds ad-hoc groups at runtime.
- **Stage 2: universality and reducibility screen** ---
  `swiftbot/tools/sawicki.py`, `swiftbot/stages/check_extension.py`.
  Schur-character computation of `dim Comm(Ad_C)` at O(d²) memory,
  plus a bounded-closure check against the `{120, 1080, 5040}`
  classification caps at `d ∈ {2,3,4}`.
- **Stage 3: QCO numerics** --- `swiftbot/stages/s3_efficiency.py`.
  Thin wrapper around the QCO kernel in `qco-main_opt/`; supports both
  subprocess mode (crash isolation) and in-process mode (faster for
  small panels). Applies the `g c g†` conjugation-cache and the on-disk
  `π_λ(c)` cache for the 3--6× speedup described in §4.3 of the paper.
- **Stage 4: QEC-code catalogue** --- `swiftbot/tools/codes.py`.
  Twelve curated entries: Steane [[7,1,3]], 5-qubit perfect [[5,1,3]],
  Reed--Muller [[15,1,3]], Bravyi--Haah [[49,1,5]] triorthogonal, Kubischta
  2I ((7,2,3)) and 2O families, qutrit triorthogonal [[20,7,2]]_3, ternary
  Golay [[11,1,5]]_3, Denys--Leverrier 2T-qutrit bosonic code, the
  Kubischta--Teixeira twisted-unitary-t-group framework entry, and two
  permutation-invariant (PI) entries used by the Ouyang--Jing--Brennen
  code-switching route (Pollatsek--Ruskai [[7,1,3]]_PI and the
  Kubischta--Teixeira (2b+3)-qubit PI family). Returns a "research
  needed" marker and an EC Zoo pointer when no curated match exists.
- **Stage 5: distillation protocols** --- `swiftbot/tools/distillation.py`.
  Nine curated protocols spanning four target-gate families:
  Bravyi--Kitaev 15-to-1 and Bravyi--Haah triorthogonal for `qubit T`;
  Bravyi--Haah CCZ for `qubit CCZ`; Duclos-Cianci--Poulin,
  Campbell--O'Gorman, and Campbell--Howard for `qubit Z-rotation
  (programmable)` (a Clifford+T substrate that bypasses the
  Anderson--Jochym-O'Connor no-go for non-Clifford-hierarchy angles);
  Ouyang--Jing--Brennen PI code-switching for `qubit rational-angle
  (non-stabilizer code-switching)` (a second AJOC bypass that does not
  route through Clifford+T); Campbell--Anwar--Browne qutrit triorthogonal
  [[9m-k,k,2]]_3 for `qutrit T`; and ternary Golay strange-state for
  `qutrit strange`. `family_for_extension(spec)` coarse-tags extensions
  into one of these families; `ajoc_excluded(spec, d)` reports whether
  the Anderson--Jochym-O'Connor classification forbids a direct
  stabilizer-code transversal implementation.
- **LLM supervisor** --- `swiftbot/llm.py`, `swiftbot/supervisor.py`.
  Anthropic SDK wrapper with Pydantic-enforced structured output via
  tool-use. A `ScriptedLLM` backend makes CI deterministic.

A separate target-family coverage module in `swiftbot/stages/target_coverage.py`
implements the bounded-depth BFS of §6 used for Table 4.

## Extending the catalogue

Pull requests adding new entries are welcome. Each entry type has a
cross-validation test that catches dangling references.

- **New base group**: append to `REGISTRY` in
  `swiftbot/tools/groups.py` using `_spec_txt(...)` if you have a
  canonical `qco-main_opt/NAME.txt` file, or `_spec_inline(...)` with a
  `@_inline_generators("key")` builder otherwise. `test_groups.py`
  checks that every registered group closes to its declared
  `expected_size`.
- **New QEC code**: append a `CodeRecord` to `_CODES` in
  `swiftbot/tools/codes.py`. The `transversal_groups` strings must be
  names registered in `tools/groups.py`;
  `test_codes_distillation.py::test_code_transversal_groups_registered`
  catches typos.
- **New distillation protocol**: append to `_PROTOCOLS` in
  `swiftbot/tools/distillation.py`. The `code_name` must match a
  registered code, and `target_gate_family` must be one of the
  recognised coarse tags; `test_codes_distillation.py` enforces both.
- **New panel**: append to `PANELS` in `swiftbot/cli.py` with an
  `extensions` builder, a `default_groups` tuple, and the target `dim`.
  Add a choice to the `q-panel` / `q-panel-summary` `--panel`
  argparse entries. No automated test, but
  `_d3_survey_panel()` / `_d3_rnd_apples_panel()` are clean templates.

`pytest swiftbot/tests/` runs in ~25 seconds locally (161 tests); all pass
deterministically without network or API access (the LLM tests use the
`ScriptedLLM` backend).

## Performance notes

- **Π(g) disk cache**: gives 3--6× speedup on multi-extension panels
  against the same base group. Persisted under
  `swiftbot/.pi_cache/` (override via `QCO_PI_CACHE_DIR`). The cache
  key includes the group content hash, so registry edits invalidate
  stale entries.
- **BLAS oversubscription**: always pin `OMP_NUM_THREADS=1` when using
  `--workers > 1`. The QCO kernel is already multi-threaded internally
  at the per-irrep level; letting BLAS pick its own thread count on top
  of `ThreadPoolExecutor` worker fan-out produces 10× slowdowns from
  core contention.
- **In-process mode**: `--in-process` on `q-panel` (or `in_process=True`
  on `evaluate_extension(...)`) skips the subprocess fork and saves
  ~5 s per call. Useful for interactive small-panel debugging; loses
  crash isolation, so do not combine with `--workers > 1`.

## License

```
MIT License (see LICENSE)
```

## Citation

Until the paper is posted to arXiv, please cite the in-preparation manuscript:

```bibtex
@misc{swiftbot_paper,
    author    = {Fleming, George T. and Lamm, Henry and
                 Tame-Narvaez, Karla and Vander Griend, Peter},
    title     = {Efficient gate-set extensions for fault-tolerant
                 qudit quantum computation},
    howpublished = {FERMILAB-PUB-26-TBA-T, in preparation},
    year      = {2026},
    note      = {Companion code: \url{https://github.com/petergerrit/quai}}
}

@misc{swiftbot_github,
    author    = {Fleming, George T. and Lamm, Henry and
                 Tame-Narvaez, Karla and Vander Griend, Peter},
    title     = {{SWIFTbot}: a pipeline for efficient qudit gate-set discovery},
    howpublished = {\url{https://github.com/petergerrit/quai}},
    year      = {2026},
    note      = {Software release accompanying the paper}
}
```
