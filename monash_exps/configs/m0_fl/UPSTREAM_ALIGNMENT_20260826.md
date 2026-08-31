# M0 FL configuration alignment audit (2026-08-26)

This note compares the frozen, successfully executed M0 local-FL templates
with Slakshna revision `2b205b1` and its pinned Bhaskera revision `b3d8c8d`.
The existing AU and India templates remain unchanged because they are part of
the accepted run provenance. They must not be replaced until the upstream
configuration ambiguity below is resolved.

| Setting | Updated Slakshna template | Effective Bhaskera value | Accepted M0 local-FL value |
|---|---:|---:|---:|
| Model | OLMo-2-1124-7B | OLMo-2-1124-7B | OLMo-2-1124-7B-Instruct |
| Sequence length | 2048 | 2048 (current default) | 1024 |
| Per-GPU batch size | 4 | 4 | 2 |
| Gradient accumulation | 8 | 4 (field is ignored) | 4 |
| Effective batch per site (2 GPUs) | 64 intended | 32 actual | 16 |
| Learning rate | 3e-4 | 3e-4 (translated by Slakshna) | 1e-4 |
| Optimizer | Muon | Muon | AdamW |
| Distributed mode | FSDP | FSDP | DDP |
| LoRA rank / alpha / dropout | 16 / 64 / 0.03 | 16 / 64 / 0.03 | 16 / 64 / 0.03 |
| Local step budget | 50 | 50 | round-specific 116--117 AU, 191--192 India |
| Checkpoint retention | 1 intended | 2 (field is ignored) | 1 |

The updated Slakshna template also declares a 10,000-sample South Asia view,
whereas the accepted M0 run used deterministic full-pass AU and India shards.
Its model and token-cache paths are absolute paths from another installation.
The current local assets contain the Instruct checkpoint and sequence-1024
token caches, not the base checkpoint and sequence-2048 caches needed to enact
the new template literally.

Several template names are not part of the Bhaskera schema. Canonical names
include `data.name`, `data.seq_len`, `training.grad_accum`, `training.lr`,
`lora.r`, and `checkpoint.keep_last_n`. The stock template instead uses
`dataset_name`, `sequence_length`, `gradient_accumulation_steps`, `rank`, and
`retention`; Bhaskera silently falls back to defaults for these fields. The
canonical South Asia training file inside Bhaskera is not an alternative
source of truth: it specifies the Instruct model, LoRA rank 256, batch size 1,
and learning rate 2e-5, and Slakshna does not load it in the node-template
workflow.

Before executable M0 templates are aligned, the collaborating team must name
one authoritative configuration and settle four material choices: base versus
Instruct model, intended versus actually parsed accumulation, sequence length,
and whether the new 50-step subset schedule replaces the two-complete-pass M0
schedule. Once fixed, both site templates should use canonical schema names,
portable path placeholders, identical model/optimizer/LoRA settings, and only
site-specific data paths and deterministic step schedules. New sequence-2048
token caches and a matching common initial adapter are required if the updated
base-model setting is selected.
