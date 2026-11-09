# llmA10

Homework 10: Efficient Training and Inference (11-967).

## Contents

- `11_967_2026_homeworks-3.pdf`: assignment instructions.
- `hw_10.pdf`: supplied answer-format template.
- `Assignment.tex` and `Assignment.pdf`: HW10 answer document with Q1.1 completed in English using the full experiment results. Q1.2 and Q1.3 remain blank.
- `11967-hw10-dev-main/`: starter code and prepared Q1.1 configuration. Its README contains the original environment and model-download instructions.
- `q11-results/backups/`: original training script and configuration before Q1.1 preparation.
- `q11-results/run-q11.sh`: launcher for the previously configured GPU3 server; its activation path and GPU checks are server-specific and must be adapted before use on another cluster.

The HKU helper scripts install Q1.1 dependencies, perform a short eager-training memory check, and launch the full experiment while retaining the scheduler's allocated GPU visibility. FlashAttention installation is deferred until Q1.2.

## Current status

**Q1.1 is complete.** The full run succeeded on an RTX 4080 SUPER on 2026-11-09. Q1.2 code is implemented and CUDA-validated (36 passed, 0 skipped). Full Q1.2 training is running on the school RTX 4080 SUPER; Q1.3 remains pending.

| Metric | Measured result |
| --- | ---: |
| Validation perplexity, length 512 | 39.93 |
| Validation perplexity, length 2048 | 137.58 |
| Trainer runtime | 1594.7876 s (26.58 min) |
| Peak training memory, PyTorch allocated | 13124.14 MiB |
| Peak training memory, PyTorch reserved | 14132.00 MiB |
| Peak device memory sampled over the full run | 14456 MiB |

The Trainer completed all 3,609 configured optimizer steps. Its reported epoch is 0.999861 due to the incomplete final accumulation group. The input split contains 115,503 sequences (59,137,536 tokens). The completed model is preserved on the school server, and the GPU allocation has been released.

`q11-results/hku/` contains the full-run metrics, Trainer state, exact training configuration, GPU samples, environment/code hashes, initialization revision, and clearly named preflight records. `measurement-summary.json` defines the timing and memory measurements. The actual Trainer path passed at batch size 16 with accumulation 2, including both validation lengths; batch size 32 exceeded the available GPU memory in the actual Trainer path.

To regenerate only the English Q1.1 answer from the completed metrics:

```bash
python3 q11-results/write_q11_answer.py --metrics q11-results/hku/metrics.json --gpu-csv q11-results/hku/gpu-usage.csv
pdflatex Assignment.tex
```

The report helper rejects incomplete/smoke-test runs. Review and render the compiled PDF after regenerating it.

The Q1.1 configuration uses eager attention, sequence length 512, BF16, one epoch, per-device batch size 16 with 2 gradient accumulation steps (effective batch size 32), AdamW with learning rate 6e-4, weight decay 0.1, warmup ratio 0.01, cosine scheduling, and seed 42. The training script records validation perplexities at lengths 512 and 2048, Trainer runtime, and peak allocated/reserved CUDA memory in `q11_metrics.json` under the model output directory. Peak training memory is captured before the final evaluations; Trainer runtime includes epoch-end validation.

## Running on the HKU GPU Farm

1. Connect to HKUVPN and log in to your GPU Farm gateway.
2. Run `gpu-interactive` once to obtain a GPU compute session, then inspect the assigned GPU with `nvidia-smi`.
3. Create a dedicated Python 3.11 environment on the compute node and follow `11967-hw10-dev-main/README.md`. Choose a PyTorch/CUDA/FlashAttention combination compatible with the allocated GPU. The GPU3 environment used newer versions for its Blackwell card; that environment is not bundled in this repository.
4. Check the batch size against the allocated GPU's memory. The HKU configuration uses per-device batch size 16 and 2 accumulation steps for the 16GB RTX 4080 SUPER; run the smoke check before a full experiment. For Q1.1/Q1.2 comparisons, retain matching hyperparameters and hardware.
5. Download the initial model and run from the starter-code directory:

```bash
cd 11967-hw10-dev-main
bash scripts/setup_hku_packages.sh
bash get_initial_model.sh
$HOME/hw10-env/bin/python scripts/smoke_q11.py --batch-size 16 --output ../q11-results/smoke-batch16.json
bash scripts/launch_q11_hku.sh
```

The training command starts a full experiment; it is not part of environment setup. Observe the school's allocated GPU visibility instead of using the GPU3-specific activation script. Preserve trained models for subsequent questions.

Local credentials, SSH connection files, machine-specific notes, model weights, environment directories, logs, and temporary LaTeX outputs are excluded from Git.

## Q1.2 code and tests

`11967-hw10-dev-main/model_src/` versions the supplied custom model source, with only the FlashAttention forward implementation changed. Model weights stay outside Git. `scripts/install_custom_model.py` copies that implementation into the downloaded initialization directory and preserves the original Python source. It does not modify the Q1.1 trained checkpoint or model weights. The initialization downloader pins the same Hugging Face revision used in Q1.1 and applies it after downloading.

The implementation converts Q/K/V to the FlashAttention layout, forwards the padding mask, causal alignment and training dropout, and merges the attention heads before output projection. Dense attention probabilities are not returned by this backend; head masks raise an explicit error.

On an allocated school compute node, using the existing HW10 environment:

```bash
cd 11967-hw10-dev-main
$HOME/hw10-env/bin/python scripts/setup_hku_flash.py
$HOME/hw10-env/bin/python scripts/install_custom_model.py
$HOME/hw10-env/bin/python -m pytest tests/test_flash_attention.py -v
```

The installer selects the matching wheel from the official FlashAttention 2.6.3 GitHub release without replacing PyTorch. Tests compare eager and real FlashAttention outputs, losses and all parameter gradients in FP16/BF16 against an additional FP32 eager reference, including 512/2048-token contexts, left/right padding, causal non-leakage, cached single/multiple-token decoding, and dropout behavior. GPU tests are skipped if CUDA/FlashAttention is absent; a skipped suite is not successful CUDA validation.

`configs/512_flash_wikitext.json` starts from the same random initialization and keeps Q1.1 hyperparameters, changing only the attention backend and output/run names. The full training command is:

```bash
$HOME/hw10-env/bin/python lm_train.py configs/512_flash_wikitext.json
```

Full Q1.2 training was started after the code-only validation. Its launcher is `scripts/launch_q12_hku.sh`, with isolated results under `q12-results/hku/`. The completed-run helper `q12-results/write_q12_answer.py` fills Q1.2.A/B/C after successful training and computes comparisons from the saved Q1.1 baseline.

CUDA validation completed on the school RTX 4080 SUPER: **36 passed, 0 failed, 0 skipped**, including a real 160M-model forward/backward check. `q12-results/validation.json`, `junit.xml`, and `flash-install.json` record the tested source hashes and environment. Tolerances derive from the dtype epsilon; gradient comparisons use scale-invariant L2 and maximum-error criteria to avoid false failures at cancellation near zero. The causal-prefix test requires exact equality. Representative FP32 gradient diagnostics are also retained.
