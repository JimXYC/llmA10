# llmA10

Homework 10: Efficient Training and Inference (11-967).

## Contents

- `11_967_2026_homeworks-3.pdf`: assignment instructions.
- `hw_10.pdf`: supplied answer-format template.
- `Assignment.tex` and `Assignment.pdf`: recreated HW10 answer template, with the author's name and all question labels. Answers are pending measured experiment results.
- `11967-hw10-dev-main/`: starter code and prepared Q1.1 configuration. Its README contains the original environment and model-download instructions.
- `q11-results/backups/`: original training script and configuration before Q1.1 preparation.
- `q11-results/run-q11.sh`: launcher for the previously configured GPU3 server; its activation path and GPU checks are server-specific and must be adapted before use on another cluster.

## Current status

Q1.1 is prepared but **training has not started**. Q1.2 and Q1.3 remain unimplemented.

The Q1.1 configuration uses eager attention, sequence length 512, BF16, one epoch, batch size 32, AdamW with learning rate 6e-4, weight decay 0.1, warmup ratio 0.01, cosine scheduling, and seed 42. The training script records validation perplexities at lengths 512 and 2048, Trainer runtime, and peak allocated/reserved CUDA memory in `q11_metrics.json` under the model output directory. Peak training memory is captured before the final evaluations; Trainer runtime includes epoch-end validation.

## Running on the HKU GPU Farm

1. Connect to HKUVPN and log in to your GPU Farm gateway.
2. Run `gpu-interactive` once to obtain a GPU compute session, then inspect the assigned GPU with `nvidia-smi`.
3. Create a dedicated Python 3.11 environment on the compute node and follow `11967-hw10-dev-main/README.md`. Choose a PyTorch/CUDA/FlashAttention combination compatible with the allocated GPU. The GPU3 environment used newer versions for its Blackwell card; that environment is not bundled in this repository.
4. Check the batch size against the allocated GPU's memory. The existing batch size 32 was prepared for a 96GB GPU and has not been tested on the inspected 16GB RTX 4080 SUPER. For Q1.1/Q1.2 comparisons, retain matching hyperparameters and hardware.
5. Download the initial model and run from the starter-code directory:

```bash
cd 11967-hw10-dev-main
bash get_initial_model.sh
# Set WANDB_MODE=offline here if using local experiment logging.
python lm_train.py configs/512_eager_wikitext.json
```

The training command starts a full experiment; it is not part of environment setup. Observe the school's allocated GPU visibility instead of using the GPU3-specific activation script. Preserve trained models for subsequent questions.

Local credentials, SSH connection files, machine-specific notes, model weights, environment directories, logs, and temporary LaTeX outputs are excluded from Git.
