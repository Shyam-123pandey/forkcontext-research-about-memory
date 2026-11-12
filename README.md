# forkcontext-research-about-memory
# Cross-Turn State Binding & Workspace Readouts in LLM Agents

An experimental design and mechanistic interpretability specification testing how open-weight LLMs (`Qwen-2.5-14B-Instruct`, `Llama-3.1-8B-Instruct`) represent and act on persistent cross-turn memory versus surface-level context injection.

## Motivation (Connection to ForkContext)
While building **ForkContext**—a persistent context-continuity layer across AI interfaces backed by FastAPI, `pgvector`, and local IndexedDB—a fundamental systems question emerged: *When an agent operates over multi-turn persistent state, does the model form distinct internal self-state representations, or does it treat injected memory identically to third-party text?*

This repository contains the experimental protocol, subspace ablation specs, and synthetic evaluation harnesses designed to test that hypothesis.

**why we** care about this while building ForkContext, you can explain it in two sentences:

The Engineering Insight: When ForkContext moves project state from ChatGPT to Claude or Gemini, you aren't just copying text—you are transplanting a partial reasoning trajectory into a fresh model instance.

The Practical Takeaway: Because LLMs treat first-person assistant turns differently from user/system reference blocks in their attention and residual-stream representations, how you structure injected memory (reconstructing assistant-turn history vs. dumping raw system-prompt bullets) directly affects whether the model treats that state as an active commitment or passive background text.
**Thinking**
While building ForkContext—a Chrome extension that syncs persistent project memory across ChatGPT, Claude, and Gemini using FastAPI, pgvector, and local IndexedDB—I ran into a deeper systems question beyond just vector retrieval: How does an LLM actually represent injected cross-turn state internally?

Since transformers are stateless between API calls, transplanting a reasoning trajectory from one session into another relies on how the residual stream binds that context. In mid-to-late layers, models separate 'my own prior assistant trajectory' from passive third-party reference text. If you dump retrieved memory as a generic system block, the model often treats it like external documentation rather than an active commitment—leading to plan drift.

Understanding those internal state-tracking representations directly shaped how we structure context injection in ForkContext, preserving turn-level continuity so the target model picks up the exact reasoning state seamlessly.

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
