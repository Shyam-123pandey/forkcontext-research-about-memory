# forkcontext-research-about-memory
# Cross-Turn State Binding & Workspace Readouts in LLM Agents

An experimental design and mechanistic interpretability specification testing how open-weight LLMs (`Qwen-2.5-14B-Instruct`, `Llama-3.1-8B-Instruct`) represent and act on persistent cross-turn memory versus surface-level context injection.

## Motivation (Connection to ForkContext)
While building **ForkContext**—a persistent context-continuity layer across AI interfaces backed by FastAPI, `pgvector`, and local IndexedDB—a fundamental systems question emerged: *When an agent operates over multi-turn persistent state, does the model form distinct internal self-state representations, or does it treat injected memory identically to third-party text?*

This repository contains the experimental protocol, subspace ablation specs, and synthetic evaluation harnesses designed to test that hypothesis.

## Core Methodology
1. **Subspace & Workspace Readouts (Jacobian Lens):**
   - Computes local Jacobian-space (`J-space`) projections across mid-to-late transformer layers to isolate verbalizable global broadcast representations.
2. **Causal Interventions:**
   - **Directional Subspace Ablation:** Suppresses target workspace directions during multi-turn decision steps to dissociate verbalized reporting from downstream action selection.
   - **Cross-Turn Activation Patching:** Patches residual stream states from Turn $t-1$ into Turn $t$ to test whether internal state persists beyond raw token re-reading.
3. **Construct Validity Controls:**
   - **Equal-Rank Orthogonal Control:** Ablates an orthogonal subspace of identical rank to rule out generic capability degradation.
   - **Ghost-Memory & Obfuscation Controls:** Injects third-party trajectory logs and keyword-obfuscated prompts to separate genuine state tracking from surface keyword triggers.

## Repository Structure
- `specs/` — Experimental protocol, pre-registered decision thresholds, and compute budget estimates (~40–120 GPU-hours on A100).
- `figures/` — Architecture pipeline diagrams and predicted dissociation plots.
- `scripts/` — Plotting and subspace projection prototypes.

## Compute & Ethics Guardrails
- Designed for reproducibility on open-weight 8B–14B models (`Modal` / `RunPod` single-node A100 setups).
- Capped episode budgets (300–500 pre-registered trials) using abstract resource-allocation tasks.
