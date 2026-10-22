# Valen vLLM plugin validation

Validated on 22 October 2026, Asia/Singapore. Experimental build `0.1.0.dev1`.
The machine-readable receipt is [validation.json](validation.json).

## Tested combination

| Component | Value |
| --- | --- |
| Checkpoint | `Valen-Team/Valen-2B` |
| Checkpoint revision | `fb9205c2541abc97ce276670e66f79b78d9068fc` |
| GPU | One NVIDIA H200, driver 595.71.05 |
| Python / glibc | 3.12.3 / 2.39 |
| Engine | Unmodified official vLLM `30aded62d46dfcf52c815925a8acec5a8bebddc6` |
| vLLM wheel | `0.32.1rc1.dev253+g30aded62d` |
| Valen wheel | `valen_vllm-0.1.0.dev1-py3-none-any.whl`, 24,554 bytes |
| Valen wheel SHA256 | `57940221ed33ae278e98a06073bd9b8e6af0d6c148e0b8dbc82c6399906e3ae3` |
| Runtime | Exact package pins in `runtime-requirements.txt`; PyTorch 2.14.0 (CUDA 13.0), Transformers 5.19.0 |

Validation used the official vLLM binary wheel in a fresh Python virtual
environment inside an existing CUDA 13 container. `PYTHONPATH` was unset. The
server installed the standalone Valen wheel; it did not use a Valen repository
import, a source overlay or modified engine files. This was wheel validation,
not a newly built vLLM container image.

## Results

- **21 tests passed, no skips.** Coverage includes native configuration loading
  without remote code, lazy model registration, real vLLM pooling metadata and
  batched role reads, missing/duplicate head weights, compatibility rejection,
  tokenizer spans, FP32 retention, and equality with the trained Mixer head.
- **`pip check` passed:** `No broken requirements found.`
- **Five GPU parity cases:** text, dialogue, eleven-level Score, long text and an
  image; together covering Choice, Noul, Score and shared-state questions.
  Input token counts and selected choices matched Transformers.
- **Maximum absolute probability error:** `0.004466097501141242`,
  below the `0.02` harness tolerance.
- **Twenty concurrent requests with four client workers:** maximum probability
  difference from sequential requests `0.005675112242458646`.
- Every native response was checked for finite probabilities, normalization,
  confidence bounds, Score/distribution agreement and expected question keys.
- **All 3,117 vLLM Python files were unchanged**, fingerprint:
  `7e3a221b079d3ee532f783ac8f1ea8fa52793aba5a23ecdd29b1caaff2fc1016`.
- Source/package formatting and lint, shell syntax, wheel namespace, entry points,
  engine/runtime locks and kit checksums passed.

The final job `valen-vllm-final-261022-093939` exited with code 0. The earlier
successful development run also passed parity. The initial online setup job
was stopped after a proxy timeout. All three jobs are inactive; no validation
GPU allocation remains running.

## Reproduce

On an otherwise idle GPU, prepare the exact environment in README.md, then:

```bash
bash vllm-kit/install.sh
python -m pip check
python -m valen_vllm.compatibility --source-tree > engine-before.json
python vllm-kit/validate.py --model-path /path/to/Valen-2B --output-dir results
python -m valen_vllm.compatibility --source-tree > engine-after.json
cmp engine-before.json engine-after.json
```

Run the source tests in the same environment with
`VALEN_TEST_VLLM=1 VLLM_PLUGINS=valen VLLM_USE_V2_MODEL_RUNNER=0`
and `VALEN_TEST_MODEL=/path/to/Valen-2B` using
`python -m unittest discover -s serving/vllm/tests -v` from the Valen repository.
The server itself does not require that repository.

## Limits

Only this Valen-2B checkpoint, H200 and pinned runtime were GPU-validated.
Other checkpoint sizes, GPUs, quantization, distributed execution and direct
video are not established by this result. Save a new receipt for each new
checkpoint, engine or runtime combination.

The tolerance is an integration-test threshold, not exact numerical equivalence
or a downstream accuracy guarantee. Cases are synthetic. Transformers lacked
optional `causal_conv1d` and `flash-linear-attention` acceleration, so recorded
latencies must not be used as a general speed comparison.
