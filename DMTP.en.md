# DMTP: A Better First Token for Block Drafting via MTP

**Yuhao Shen · October 2026 · Exploration Note**

Speculative decoding has become an important component of modern LLM inference infrastructure. Over the past year, models such as [DFlash](https://z-lab.ai/projects/dflash/) have shown considerable potential. DeepSeek has also deployed [DSpark](https://arxiv.org/abs/2607.05147) in production, reporting substantial gains over its previous MTP-1 baseline. However, in my experiments and recent discussions with Domino author Jianuo Huang about MTP and DFlash, we noticed that DFlash does not always offer a significant advantage over MTP, particularly on models with well-trained native MTP modules, such as GLM and Qwen, or on certain chat workloads.

Our tests revealed a useful distinction: native MTP has stronger first-token acceptance, particularly on OOD workloads, while DFlash generates a longer continuation in parallel. Can we combine MTP's pretrained first-token capability with DFlash's parallel drafting?

Several approaches address the decline in acceptance at later positions within a DFlash-style block. **We focus on a different question: can we improve the first draft token itself?** Under prefix verification, rejecting that first candidate prevents the rest of the block from being committed. DMTP starts with a pretrained MTP prediction, then passes its hidden representation to DFlash for the continuation.

We introduce **DMTP**: pretrained MTP proposes the first token, and DFlash generates the continuation in parallel. Our main method is **MTP-style token and hidden-state fusion**: independently normalize the T1 embedding and its predicting MTP hidden, concatenate them, and project the pair into the T1 input position.

> **The key idea:** let MTP start the block and DFlash continue it in parallel. The two models share both a concrete candidate and a predictive representation, while retaining their original backbones.

## The Key Observations

- **Reuse pretrained first-position capability.** Native MTP has higher first-position acceptance across the workloads in Figure 1. DMTP combines this capability with parallel DFlash continuation.
- **MTP starts; DFlash continues in parallel.** We reuse MTP's pretrained first-position capability while controlling serial draft latency. Multiple MTP tokens and adaptive step counts are natural extensions.
- **Combine predictions and representations.** MTP supplies the first candidate and passes its predictive state to the parallel drafter. We evaluate candidate conditioning and representation transfer separately.
- **Evaluate the interface during generation.** On eight Code inputs, MTP-style token and hidden-state fusion reaches a mean acceptance length of **3.963**, versus **3.770** when conditioning on the candidate tokens alone.

## 1. First-Token Acceptance and Parallel Drafting

Figure 1 compares **DFlash and native MTP** using the same target model. DFlash is trained on a mixture of tasks with 32K sequences; MTP uses the pretrained weights bundled with the same target.

Both arms use the same Qwen3.6-35B-A3B target and input set across Code, Math, MT-Bench, Agent, SWE and Long-horizon. **DMTP itself uses MTP for T1 and DFlash for six parallel continuation candidates.**

![DFlash and native MTP first-position acceptance across six workloads](assets/first-token.png)

*Figure 1. First-token acceptance across six workloads on Qwen3.6-35B-A3B at a generation temperature of 1.*

On Code, first-position acceptance is 71.00% for DFlash and 82.32% for MTP.


**The first-token advantage has a straightforward explanation: native MTP benefits from extensive pretraining.** Its first-step prediction is trained alongside the target on far more data than our separately trained DFlash draft, giving it a strong foundation for prediction and generalization.

We therefore view the observed gap as a reason to reuse that pretrained capability, rather than as evidence that MTP is inherently a better architecture. DMTP builds on this training investment and passes the MTP representation to the parallel continuation.

This leads to a natural question: **how can we combine MTP's pretrained first-token capability with DFlash's parallel block prediction?** Rather than choosing one over the other, we use MTP to start the block and DFlash to continue it.

## 2. DMTP: MTP Starts, DFlash Continues

The division of labor is straightforward:

```text
MTP:     propose T1
DFlash:  condition on T1 and propose T2–T7 in one forward pass
Target:  verify seven candidates and commit the accepted prefix plus correction/bonus
```

![MTP-style token and hidden-state fusion at the first candidate input position](assets/architecture.png)

*Figure 2. Original DFlash (top) and DMTP (bottom). Read left to right: MTP generates T1; its hidden state is fused with the T1 token embedding; DFlash predicts T2–T7 in parallel; the target verifies all seven candidates. The upper branch preserves the original target-context-to-KV path. The T1 embedding is a token lookup, not the target hidden state at T1.*

There are three reasons to explore this combination.

**Reuse the strong first-token predictor.** Native MTP stays frozen, preserving its first-step behavior while the new draft is trained. A reliable beginning creates an opportunity to realize the value of the remaining candidates.

**Give DFlash a closer prediction anchor.** Given T1, the original second future token becomes the first unknown position after the new anchor. DFlash generates the continuation conditioned on T1, and the target verifies the complete candidate block.

**Pass on MTP's pretrained representation.** T1 gives DFlash a discrete prediction. MTP hidden adds features learned during pretraining, which we hope will help the following DFlash positions generalize and pass verification. This is a separate hypothesis from the benefit of using MTP for T1.

### Single-Step MTP Drafting

Using MTP for just T1 preserves its strongest pretrained prediction while limiting serial draft latency. Generating more MTP candidates requires successive forwards and token selection; DFlash instead predicts the remaining six positions in one parallel pass. Additional MTP steps are worthwhile only when their acceptance gains offset their execution cost. Multiple MTP tokens and adaptive step counts remain useful extensions.

## 3. MTP Hidden-State Integration and Ablations

The next question is how to pass MTP's representation to DFlash. We use **MTP-style token and hidden-state fusion** and compare generation-time acceptance across candidate conditioning and representation interfaces.

Let \(A = x_a\) be the pending target-produced token. Target context ends at \(x_{a-1}\). Native MTP consumes \(A\) and the preceding target hidden, producing the representation that predicts \(T_1 = x_{a+1}\). No target hidden for \(A\) or \(T_1\) is required.

Fusion reads **the MTP hidden that predicted T1**, not one computed after consuming T1:

$$
\widetilde{\mathbf e}_{T_1}=W_f\left[\operatorname{RMS}_e(\mathbf e_{T_1});\operatorname{RMS}_m(\mathbf h_{\mathrm{MTP}})\right].
$$

Each 2048-dimensional input has its own learned RMSNorm, followed by a 4096→2048 linear projection. Only the T1 embedding is replaced. A, all six MASK embeddings, and the original target-context FC→KV path stay unchanged; no MTP KV slot is added.

### The MTP-Style Fusion Interface

**Reuse an established interface pattern.** Native MTP independently normalizes token embeddings and hidden states before concatenation and linear projection. The main method adopts that structure with separately trained fusion parameters; it does not copy the native MTP projection weights.

**Bind the representation to a concrete candidate.** E(T1) identifies the selected token; MTP hidden describes the prediction that produced it. Six MASK queries attend to the fused T1 instead of receiving the same additive residual at every position.

**Keep the original context pathway.** Token and hidden-state fusion preserves historical context, the DFlash backbone, and the proposal budget, making it straightforward to reuse existing models and inference implementations.

### Two-Anchor Conditioning

> **Two known inputs, seven proposals.** A has already been emitted by the target; T1 is the new MTP proposal and can still be rejected. Only T1–T7 enter the seven-candidate verification budget.

**A closes the context boundary.** Target context stops before A. Removing it removes the explicit identity of the last emitted token, leaving DFlash to recover the boundary through the MTP hidden.

Keeping E(A) supplies that identity directly, avoiding the need to reconstruct it from a compressed representation before predicting the continuation.

**T1 specifies the continuation branch.** MTP hidden describes the prediction, but is not interchangeable with the selected token. Explicit T1 lets all six MASK positions attend to the same concrete proposal, rather than infer which possible first token they should continue.

**Two anchors give parallel continuation a more explicit prefix.** A identifies the last emitted token; T1 identifies the proposed next step. All six MASK positions share those conditions instead of reconstructing the boundary and guessing T1.

This does not lower the sampling temperature or make T1 a verified token.

We compare the combined MTP-first, two-anchor design against original DFlash, then fix the input layout to evaluate the additional MTP representation.

![Two known positions A and T1, followed by six supervised MASK positions](assets/anchor-layout.png)

*Figure 3a. Eight input slots, seven proposals: A is already emitted; T1 comes from MTP; T2–T7 come from DFlash. Neither A nor the known T1 slot receives draft prediction loss.*

### Ablation Design

- **Original DFlash:** single-anchor external baseline with `[A,MASK×7]`; DFlash proposes all seven candidates.
- **Candidate-conditioned two-anchor drafting:** `[A,T1,MASK×6]`, without additional MTP state injection; original target context remains available.
- **Hidden-state residual at the first candidate:** project MTP hidden through a low-rank network and add the residual only to the T1 embedding; all six mask inputs stay unchanged.
- **MTP-style token and hidden-state fusion (main method):** independently normalize and concatenate MTP hidden and E(T1), then project to replace only the T1 embedding.

Supplementary controls are a hidden-state residual at six mask positions, target and MTP context fusion, and MTP-only context. Context fusion concatenates five target feature streams and aligned MTP features, then projects 6H→H; it is not the earlier five-projection layerwise pilot. Every DMTP group retains A and T1.

![The four main-table methods: original DFlash, candidate conditioning, first-candidate hidden-state residual, and token and hidden-state fusion](assets/injection-variants.png)

*Figure 3b. Only the four main-table methods: original DFlash, candidate-conditioned two-anchor drafting, a hidden-state residual at the first candidate, and MTP-style token and hidden-state fusion. All retain the original target-context pathway and the same seven-candidate budget. The final two compare additive residual injection with replacement-style fusion at the first candidate input.*

Conditioning on the candidate tokens alone provides a control for the additional MTP representation.

### Inference Ablation: Code Acceptance Length

We compare representation interfaces on the same eight code-task inputs, holding the models, candidate count, and generation settings fixed.

| Method | Representation interface | Code length | vs candidate conditioning |
| --- | --- | --- | --- |
| Original DFlash | Single anchor; seven proposals | 3.360 | -10.89% |
| Candidate-conditioned two-anchor drafting | Condition on the proposed token without extra state injection | 3.770 | +0.00% |
| Hidden-state residual at the first candidate | Add a residual only at the first candidate input | 3.864 | +2.48% |
| **MTP-style token and hidden-state fusion (main method)** | **Replace only the first candidate input representation** | 3.963 | +5.12% |

**MTP-style token and hidden-state fusion adds 5.12% in acceptance length.** MTP-first, two-anchor drafting raises the original DFlash result from 3.360 to 3.770 (+12.22%). Representation fusion then reaches 3.963. The hidden-state residual at the first candidate reaches 3.864, 2.48% above candidate conditioning.

*Values are mean acceptance lengths over eight Code inputs, including target correction/bonus advancement. The main table uses the unified adapter; the supplementary six-mask residual uses an independent serving implementation.*

<details>
<summary>Supplement: other uses of MTP hidden</summary>

| Method | Representation interface | Code length | vs candidate conditioning |
| --- | --- | --- | --- |
| Hidden-state residual at six mask positions | Add an MTP hidden-state residual to six mask inputs | 3.842 | +1.92% |
| Target and MTP context fusion | Concatenate and project target and MTP features | 3.714 | -1.49% |
| MTP-only context | Supply context from position-aligned MTP features | 3.565 | -5.43% |

</details>

### Prediction Reuse and Representation Transfer

**MTP's pretrained capability is already reused at the output level:** it generates T1. Holding the candidate inputs fixed lets us evaluate what its internal representation adds to parallel continuation.

The interface determines where and how DFlash reads MTP information. The representation and its injection point should be designed together.

MTP-style fusion achieves the highest acceptance length in this Code comparison and follows native MTP's normalization and projection structure, binding the candidate and its predictive state at one input position.

### Compatibility with Existing Drafting Methods

**MTP-style token and hidden-state fusion retains the original DFlash backbone and target-context interface.** A compatible backbone can be loaded and trained with the new T1 fusion module while Target and MTP remain frozen.

Fine-tuning adds the T1 fusion module, uses `[A,T1,MASK×6]` as input, and supervises the six continuation positions.

DMTP changes how MTP and DFlash work together. It retains both backbones, so improvements to MTP weights, DFlash objectives, path selectors and runtime kernels can be explored alongside it.

**DMTP retains both backbones and connects them through a lightweight representation interface.** Existing training recipes remain useful, and the design can incorporate stronger prediction heads and inference implementations.

### Early MTP-Layer Experiments

In July and August, we copied MTP layers for DFlash-style parallel prediction. Joint training affected first-position performance and yielded limited gains. DMTP instead freezes that pretrained first step and trains the parallel continuation around it.

## 4. Training DMTP

We use Qwen3.6-35B-A3B and a mixture of tasks, primarily software engineering and long-horizon tasks. Training uses 32K contexts and approximately 250K records; conversation and agent tasks are used to evaluate generalization.

Training follows three steps:

1. **Extract frozen features.** Target processes the training sequence. At each block boundary, MTP consumes A and the preceding target hidden to obtain the representation for predicting T1. DFlash context uses only target features before A.
2. **Build the draft input.** Use `[A,T1,MASK×6]` with the dataset's T1. Independently normalize E(T1) and its predicting MTP hidden, concatenate, and project to replace the T1 embedding. All six MASK embeddings remain unchanged.
3. **Optimize the continuation.** DFlash learns T2–T7 with the original position-decayed CE objective. Target and MTP stay frozen; only DFlash and the T1 fusion module are updated. A and the known T1 input slot receive no draft prediction loss.

Training uses <mark>GT T1</mark>; inference uses MTP-generated T1, leaving a train–inference gap. The shared-hidden pathway is the same in both cases: it uses the MTP representation that predicts T1, not a hidden obtained after consuming T1.

## 5. Inference: MTP First, DFlash Second

Each generation round follows the order in Figure 2:

1. **MTP proposes T1.** Read the available target hidden and consume pending token A. Keep both the selected T1 and the MTP hidden that predicted it.
2. **DFlash predicts T2–T7.** Form `[A,T1,MASK×6]`, fuse E(T1) with its predicting MTP hidden, replace only the T1 embedding, and run one parallel DFlash forward.
3. **Target verifies the block.** Verify T1–T7 together, commit the accepted prefix and correction/bonus token, then advance to the next round.

The runtime keeps MTP and DFlash state separate, commits only accepted history, and rolls back rejected candidates.

The additional computation consists of an MTP forward, its first-token LM head, two normalizations, and a T1 fusion projection. Target context and DFlash attention lengths remain unchanged.

Systems optimization focuses on state updates, execution scheduling, and CUDA Graph coverage to reduce the cost of connecting the two drafting stages.

## 6. Main-Method Evaluation Across Six Workloads

We evaluate DMTP against DFlash and native MTP on long-context benchmarks covering code, mathematics, conversation, agents, software engineering, and long-horizon tasks. Inputs are capped at **128K tokens** and outputs at **28K tokens**. All methods use the same inputs at a generation temperature of 1.

| Workload | DFlash length | Native MTP length* | DMTP length | DMTP vs DFlash |
| --- | --- | --- | --- | --- |
| Code | 3.239 | 3.496 | 3.847 | +18.78% |
| Math | 3.497 | 3.619 | 4.084 | +16.78% |
| MT-Bench | 2.849 | 3.153 | 3.406 | +19.56% |
| Agent | 4.742 | 4.201 | 5.369 | +13.22% |
| SWE | 3.355 | 3.402 | 3.935 | +17.29% |
| Long-horizon | 4.479 | 4.045 | 4.894 | +9.25% |

*Native MTP proposes four candidates; DFlash and DMTP each propose seven.*

*Mean acceptance length includes the target correction or bonus token.*

![Acceptance length of DMTP, DFlash and native MTP across six workloads](assets/main-acceptance.png)

*Figure 5. DMTP improves mean acceptance length by 9.25%–19.56% over DFlash across six workloads.*

### Position-Wise Acceptance

Figure 6 breaks down the same generation evaluation by candidate position.

![Position-wise cumulative acceptance for DFlash and main-method T1-fusion DMTP](assets/main-position-acceptance.png)

*Figure 6. Cumulative acceptance P(L≥k): the fraction of verification rounds accepting the first k draft candidates, excluding correction/bonus tokens. Evaluation settings are unchanged.*

DMTP achieves higher cumulative acceptance at every position from T1 to T7 across all six workloads, showing how a stronger first proposal and conditioned parallel prediction improve acceptance throughout the candidate block.

<details>
<summary>Conditional acceptance: accepting candidate k given that the first k−1 were accepted</summary>

Conditional acceptance measures the probability of accepting the next candidate after reaching a position.

![Position-wise conditional acceptance for DFlash and main-method T1-fusion DMTP](assets/main-position-conditional.png)

| Workload / model | Verify rounds | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Code / DFlash | 523201 | 71.00% | 70.38% | 71.64% | 70.79% | 70.43% | 70.91% | 70.97% |
| Code / DMTP (T1 fusion) | 434780 | 81.81% | 76.83% | 74.34% | 73.43% | 73.00% | 73.02% | 71.76% |
| Math / DFlash | 473538 | 73.73% | 72.93% | 73.82% | 73.87% | 73.36% | 71.90% | 70.59% |
| Math / DMTP (T1 fusion) | 404689 | 83.72% | 78.84% | 76.25% | 76.36% | 75.64% | 74.54% | 72.62% |
| MT-Bench / DFlash | 54479 | 61.36% | 62.04% | 68.34% | 73.36% | 76.38% | 76.11% | 74.92% |
| MT-Bench / DMTP (T1 fusion) | 41259 | 73.72% | 66.90% | 67.78% | 73.03% | 76.24% | 76.77% | 74.74% |
| Agent / DFlash | 22213 | 79.53% | 81.84% | 84.00% | 85.87% | 87.05% | 86.81% | 84.74% |
| Agent / DMTP (T1 fusion) | 17371 | 86.98% | 86.12% | 85.51% | 86.91% | 87.72% | 87.34% | 86.18% |
| SWE / DFlash | 4795 | 59.33% | 64.25% | 68.38% | 73.92% | 79.65% | 80.98% | 84.90% |
| SWE / DMTP (T1 fusion) | 3836 | 71.79% | 72.59% | 74.64% | 79.69% | 84.44% | 85.06% | 82.20% |
| Long-horizon / DFlash | 2919 | 75.13% | 76.84% | 80.18% | 80.01% | 82.89% | 83.71% | 84.40% |
| Long-horizon / DMTP (T1 fusion) | 2798 | 80.56% | 80.66% | 79.98% | 83.22% | 85.04% | 86.10% | 85.78% |

</details>

Overall, first-position enhancement and conditional parallel continuation improve acceptance length. The next systems challenge is reducing the execution cost of the extra MTP stage.

## 7. Future Directions

There are several directions we would like to explore:

- **A unified MTP–DFlash formulation.** Can we train both within one framework and better understand why their first-token acceptance and later-position decay differ?
- **Joint training.** MTP does not have to stay frozen. We can train it together with DFlash and study how to improve block prediction while preserving its first-token strength.
- **Multi-layer MTP for block prediction.** Can multiple MTP layers learn to predict a block, combining pretrained MTP capabilities with parallel drafting?
- **A more efficient inference pipeline.** Improve KV handling, CUDA Graph coverage, and execution scheduling to reduce the extra cost of combining the two models.
- **Larger models.** Extend DMTP to larger target models and explore different MTP–DFlash combinations to evaluate its generalization and acceleration potential at scale.

DMTP is an initial exploration. We will share results on larger models, and the paper and open-source code will be released soon.

## References

1. [DFlash: Block Diffusion for Flash Speculative Decoding](https://z-lab.ai/projects/dflash/)
2. [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147)
3. [SpecForge v0.3.0](https://www.lmsys.org/blog/2026-08-04-specforge-v0-3)

## Acknowledgements

We thank the **Qwen Team, the GLM/Domino Team, and the Qwen Application Team** for their support and discussions. This work was carried out during my **internship with the Qwen Infra team**.

**Qwen Team**: Yuhao Shen, Di Liu, Yucheng Li, Hao Yu, Yang Xu, Xi Lin, Yi Zhang.

**GLM/Domino Team**: Jianuo Huang.

**Qwen Application Team**: Tianyu Liu, Jun Zhang.

## Citation

If you find this work useful, please consider citing this technical blog:

```bibtex
@misc{shen2026dmtp,
  author = {Shen, Yuhao},
  title  = {{DMTP}: A Better First Token for Block Drafting via MTP},
  year   = {2026},
  note   = {Exploration note}
}
```
