# Weekday probes at the injection layer: ON vs OFF vs the injection directions

Ridge probes trained on each arm's residual stream right after block 3 (the
injection point, site output) against the SAME aligned+thresholded gemma
weekday z-scores training injected (held-out climbmix shard 100, prescored,
2,500 docs / 1.37M tokens / 161k active rows, 0 store misses). Conditions:
4 arms x gate {on, off} through the pinned harness path, plus bolt-on controls
(each arm's checkpoint D attached to the never-injected baseline at the trained
gate 0.0273). Probes: standardized ridge from exact fp64 moments, lambda by
held-out R2 (every 10th doc); per probe we save the DECODER readout V = W/sigma
(covariance-whitened) and the ENCODING estimate Cxy (stream-target
cross-covariance — where the signal lives, whitening-free). Ran on one H100
(4 arms in parallel, ~35 min, ~$2); pipeline validated by CPU tests incl. a
planted-direction recovery (enc cos 0.96) and an exact moment-vs-direct R2
crosscheck (0.0000 diff on the real data). All numbers: `probe_report.json`.

## Headline numbers
- Probe quality (active tokens, held-out R2 mean / day-argmax acc):
  OFF ~0.25 / 0.41 for ALL arms incl. baseline; ON: trainable 0.46/0.55,
  sphere 0.40/0.47, orthogonal 0.46/0.55; bolt-on 0.33-0.38/0.44-0.50.
- Encoding-vs-D matched-day cosine (all tokens): ON 0.29/0.38/0.31
  (trainable/sphere/orthogonal), OFF 0.11/0.16/0.14, bolt 0.30/0.22/0.17,
  baseline null +0.13 vs trainable D but **-0.01 / 0.00** vs sphere/orthogonal D.
- Decoder-vs-D matched-day cosine: ON 0.62/0.75/0.55, OFF 0.18/0.12/0.24,
  bolt 0.60/0.61/0.53.
- CE sanity (matches the eval suite's direction): ON < OFF for every injected
  arm (e.g. trainable 2.8389 vs 2.8407).

## Figures
- **figP1_probe_r2.png** — held-out R2 (active tokens) per condition with argmax
  accuracy annotated. OFF bars are flat at ~0.25 across all four arms; ON lifts
  each arm to 0.40-0.46; bolt-on reaches only 0.33-0.38. *Injection makes the
  weekday code substantially more decodable, and a TRAINED arm carries it better
  than the same packet bolted onto the baseline.*
- **figP2_encoding_vs_D.png** — per-day cos(encoding row, D row): ON vs OFF vs
  bolt, one panel per arm. Sphere/orthogonal: ON (0.38/0.31) far above bolt
  (0.22/0.17), OFF (0.16/0.14) above the ~0 baseline null. Trainable: ON 0.29
  ~= bolt 0.30. *Frozen-direction arms internalized D — their streams carry the
  injected directions beyond the raw packet imprint, even with injection off.*
- **figP3_enc_heatmaps.png** — 7x7 cos(ON encoding rows, D rows), calendar
  order. Trainable and orthogonal are cleanly diagonal (~0.3 vs ~0.08 off-diag);
  sphere is positive everywhere (0.16-0.41) with a weak diagonal. *Near-orthogonal
  D rows give day-separable codes; the sphere's correlated rows (shared weekday
  component) blur day identity in the stream.*
- **figP4_decoder_and_rotation.png** — left: matched-day mean vs D for ON
  encoder/decoder, OFF encoder, bolt encoder. The decoder (whitened readout)
  aligns with D much more than the raw encoding estimate (e.g. sphere 0.75 vs
  0.38). Right: cos(ON encoding row, OFF encoding row) ~= 0.97 for every day/arm.
  *The overall weekday signal location barely moves when injection turns on (the
  packet is a 2.7%-RMS perturbation); the D-aligned component rides on top of a
  dominant natural encoding.*

## One-line interpretation
Probes at the injection layer read the injected code well (R2 0.25 -> 0.46,
acc 0.41 -> 0.55); the frozen-direction arms show weight-level internalization
of D (OFF-stream alignment 0.14-0.16 vs a 0.00 baseline null; ON above the
bolt-on packet imprint), while the trainable arm shows the mirror image — its
LEARNED D moved toward the model-natural weekday geometry (baseline null +0.13
vs its D) rather than the model moving toward D.

## Reproduce
Pod: `run_probes_pod.sh` via `pod_bootstrap.sh` (H100, torch-v240 template;
stage the tokenizer first if the pod cache is empty: `python -m
nanochat.dataset -n 8 && python -m scripts.tok_train`). Local analysis:
`python3 runs/weekdays/probes/compare_probes.py`. CPU tests:
`python3 runs/weekdays/probes/test_probes.py`.
