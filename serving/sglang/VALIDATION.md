# Valen plugin validation

Validated on 22 October 2026, Asia/Singapore. This is an experimental serving-kit build, version `0.1.0.dev2`. The machine-readable receipt is `validation.json`.

## Tested combination

| Component | Tested value |
| --- | --- |
| Valen checkpoint | `Valen-Team/Valen-2B` |
| Checkpoint revision | `fb9205c2541abc97ce276670e66f79b78d9068fc` |
| GPU | One NVIDIA H200 |
| Engine | Unmodified upstream SGLang `c1c287a42928495166e14dbcef31b9e1f1dac199` |
| Valen wheel | `valen_sglang-0.1.0.dev2-py3-none-any.whl`, 24,787 bytes |
| Wheel SHA256 | `9d67db1d007ec7aa6edff46f6c969adc49507b0d4b6f03c23a7606f447a13ddc` |
| Runtime | Exact package versions in `validation.json` and the wheel's `runtime-lock.json` |

The cluster run installed a separate engine wheel built from unmodified upstream source using the upstream `SGLANG_BUILD_RUST_EXTS=none` option, then installed the Valen kit with `install.sh` in an existing CUDA 13 SGLang container. `PYTHONPATH` was unset. It did not load the Valen fork or a Valen source overlay. This establishes the wheel installation path; it is not a full pull/run verification of the public nightly base image.

## Results

- **CPU regression:** 62 passed, 1 skipped, across 63 tests. This includes Valen behavior and existing Clef, System One and decision API tests. The final wheel separately passed all 21 plugin/native tests on both the local CPU environment and the GPU host.
- **GPU comparison:** five cases covering text, image, choice, noul, score and shared-state questions. Input token counts and selected choices matched the checkpoint's Transformers implementation.
- **Maximum absolute probability error:** `0.007673213242758242`, below the harness tolerance of `0.02`.
- **Concurrency:** twenty requests with four client workers; maximum probability difference from sequential requests was `0.007422405802532395`.
- **Engine files:** all 4,527 installed SGLang Python source files had the same fingerprint before and after the run: `4a4ba3a9f8c3836cca43f759075e228a36a5784f90a1213a9a313e1505bd3018`.
- Default overlap scheduling, native System One warmup, image preprocessing, BF16 backbone and FP32 decision head completed successfully. The validation job exited with code 0.
- Python formatting/lint, shell syntax, wheel namespace and bundle checksums passed.

The probability tolerance is an integration-test threshold, not a claim of exact numerical equivalence or downstream task accuracy. Other checkpoint sizes, GPUs, quantization and distributed execution have not been validated by this kit release.

The harness records timing for diagnosis. Its cases are synthetic, and the Transformers environment lacked optional `causal_conv1d` and `flash-linear-attention` acceleration. These timings do not establish a general performance advantage.

## Reproduce

Inside the compatible environment, with the checkpoint already downloaded:

```bash
bash sglang/install.sh
python -m valen_sglang.compatibility --source-tree > engine-before.json
python sglang/validate.py --model-path /path/to/Valen-2B --output-dir results
python -m valen_sglang.compatibility --source-tree > engine-after.json
cmp engine-before.json engine-after.json
```

For a new checkpoint or engine version, retain a new receipt rather than replacing this historical validation result.
