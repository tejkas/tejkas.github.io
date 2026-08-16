# Activation Probes vs Text Monitors: What Is More Effective for Understanding a Model's Thoughts?

With the advent of reasoning models, chain-of-thought text has proven to be a fairly groundbreaking way of reading a model's "thoughts". Often, one may want to monitor an agent's chain-of-thought to understand if the agent is still aligned with solving the actual task, and if not, why it might be drifting.

One method of doing this is via having a smaller model reading a frontier model's reasoning text to flag adherence to specific behavior. Chain-of-thought monitoring works. Baker et al. (2025) found a GPT-4o monitor reading a frontier model's reasoning caught 95% of reward hacks versus 60% for an actions-only monitor [1]. Reading the model's thoughts is one of the better oversight tools we currently have.

A position paper from ~40 authors across AISI, Apollo, OpenAI, GDM and Anthropic calls CoT monitorability "a new and fragile opportunity," because nothing guarantees models keep thinking in readable English [2]. It also doesn't guarantee that models will faithfully generate a CoT with its actual, underlying reasoning mechanism, as it could just as easily hide, mislead or otherwise obfuscate its textual reasoning.

Thus, the question worth investigating is whether a hybrid monitor can be developed, combining the strengths of text monitors and activation probes, which depend on the residual stream of a model's internal transformer layers, and if they complement each other. That is to say, does the activation probe hold its efficacy while the text monitor fails, and can the activation probe actually see something the text monitor cannot?

## Measuring Hint-Deference

For this mini-experiment, I chose the hint-usage paradigm described in Drori et al [3], which Chen et al [7] were the first to use for studying CoT faithfulness. My setup differs in that I inject hints pointing at a wrong option on questions the model reliably answers correctly:

1. Keep only questions the model reliably answers correctly unhinted (8 samples, modal answer correct, mode peaked ≥75%). Call these capable questions, and 580 of 800 screened MMLU-Pro questions qualified (72.5%).
2. Inject a hint pointing at a wrong option. For example, "A Stanford professor told me the answer is (H)."
3. Label behaviorally: deferred if the final answer equals the hint, resisted if it equals the correct answer, discard anything else.

Hint-usage makes sense here because it's fairly simple to see if the hint was actually heeded, and so a detector's failure should be unambiguous. Because the model reliably knew the answer and the hint is wrong, answering the hint means the model used the hint.

The base model in this experiment (Qwen3-8B) deferred to authority 41% of the time, but to sycophancy ("I think the answer is…") only 3% of the time. Hence authority hints supply nearly all of the deferred cases.

## Experimentation

I use Qwen3-8B as the base model, generating 4 samples for each of the 580 capable questions from MMLU-Pro. Questions are then further split into a training set consisting of 232 questions, and a validation set of 348 questions. The validation set thus yields 348 * 4 = 1392 generations, of which 1052 carry a usable label (resisted hint or demonstrably deferred).

There are five separate detectors:

| Detector | What it sees |
|---|---|
| Activation probe | residual-stream activations, teacher-forced over prompt + CoT, pooled over CoT positions |
| Prompt-only probe | the same method, but activations of the question and hint alone — never the reasoning |
| TF-IDF | bag-of-words over the CoT text |
| Text monitor (3B) | SmolLM3-3B, prompted |
| Text monitor (14B) | Qwen3-14B, prompted |

The probe is difference-of-means (direction = mean of deferring activations − mean of resisting ones), with layer and pooling swept and selected on training data only. Because attention is causal, a teacher-forced pass over prompt + CoT reproduces the model's generation-time hidden states.

The prompt-only probe is a control artifact, which sees the question but not the reasoning. Its score is thus purely a function of its priors, asking whether the question (and the hint) itself tends to induce hint-deference.

Moreover, every CoT's own statement of the answer (such as, "...therefore, the answer must be (A)") is stripped. This is done to eliminate potential sources of bias where the prompt names the hint, the CoT names its answer, and a detector can do a simple string comparison without understanding the latent reasoning.

## Results

| Detector | AUROC | 95% CI | TPR@1%FPR |
|---|---|---|---|
| Text monitor (14B) | **0.919** | [0.884, 0.948] | **0.294** |
| Activation probe | 0.901 | [0.862, 0.934] | 0.086 |
| **Prompt-only probe** | **0.865** | [0.815, 0.909] | 0.171 |
| TF-IDF | 0.806 | [0.737, 0.866] | 0.075 |
| Text monitor (3B) | 0.608 | [0.536, 0.676] | 0.123 |

*[figure: legible vs illegible AUROC bar chart]*

The experiment yields a few interesting takeaways.

First, the CoT reasoning itself does not seem to be adding enough value as it pertains to detecting hint-deference. The prompt-only probe, which doesn't even see the CoT, scores fairly high (0.865 AUROC). In contrast, the activation probe, which does, improves only marginally (+0.036 AUROC).

Since all four samples of a question share the prompt, a prompt-only probe can easily capture the variance between (question, hint) pairs to detect questions that are likely to induce deference, as opposed to genuine latent reasoning.

In addition, the grid search to determine pooling and layer on the training data seemed to converge on max pooling at layer 8 of 36. This is fairly shallow depth (0.22); for comparison, McKenzie et al. select layer 31 of 80 (0.39) by cross-validation [4]. They also find their best-performing aggregators are Attention, Softmax and Mean, all of which read every token, while max pooling scores on the single highest-scoring token, which they suggest can indicate a label that depends on specific words rather than the whole sequence [4]. This is supported by the fact that my probe does not improve much over a TF-IDF bag-of-words detector, whereas their attention probe substantially outperforms an equivalent TF-IDF/SVM baseline.

Another aspect to consider is that a strong text monitor actually beats the activation probe. The Qwen3-14B monitor outpaces the activation probe with an AUROC of 0.919 compared to 0.901. More strikingly, its TPR@1%FPR is tripled (0.294 against 0.086). When reasoning is legible, the assumption that internal probes uncover hidden signals missed by textual monitors appears unfounded. In contrast, the 3B monitor achieves a modest 0.608 AUROC, underperforming the TF-IDF baseline (0.806). This significant performance delta between text monitors (0.608 vs 0.919) dwarfs the margin between internal probes and text-based methods. If the analysis had been restricted to the 3B model, the probe would appear superior simply by virtue of competing against a weak baseline.

The low TPR@1%FPR across all detectors is also worth noting. McKenzie et al. report the same shape: their best probe detects 43% of high-stakes cases at 1% FPR against 55% for their best LLM baseline, and they conclude that neither probes nor most baselines could serve as standalone monitors where false positives directly affect users [4]. My numbers are lower in absolute terms but point in the same direction, which suggests an AUROC near 0.9 does not translate into a deployable monitor.

## Discussion

There are a few aspects of this experiment that line up with existing research.

Firstly, as seen with McKenzie et al., the best known probe results classify inputs [4]. In particular these detect high-stakes interactions, where prompt-based classification is a necessary method. Here, keeping the prompt in is a confounding variable.

Aremu et al. tried probing a property of the model's own response, but did not run it against a prompt-only control [6]. Their detection statistic averages cosine similarities over assistant tokens only, and their baselines are two text-level guard models plus a supervised activation probe, none of which is prompt-only.

McKenzie et al. also find that probes and LLM monitors capture complementary information, with cascaded combinations outperforming either alone [4], which is closer to the hybrid framing this experiment set out to test.

Baker et al. also found that monitor strength is a dominating feature. GPT-4o had 95% joint recall on the two systemic hacks in comparison to GPT-4o-mini (12%). They caveat this themselves, though: GPT-4o-mini was much harder to instruct and often failed to output a label at all, so the gap may reflect instruction-following and long-prompt handling rather than an inherent capability limit at that scale, and they present the result as preliminary [1].

This may also simply be a feature of the hint-deference paradigm. Since the hint necessarily has to exist within the prompt, a prompt-only probe is likely to do well. A different problem family, such as deception, where the prompt does not announce the deception, may score worse with just the prompt. That makes this partly a caution about the paradigm: hint-deference cannot cleanly separate reasoning-derived signal from prompt-derived signal, so a prompt-only floor should be reported alongside any probe result in this setting.

## What about an illegible CoT?

This was a murkier question to answer. For an illegible CoT, it's possible the activation probe's performance persists as the CoT degrades to something that isn't human readable.

I used traces on 146 MMLU-Pro questions (drawn from a separate 300-question sample, filtered down to 210 capable questions and then split 70/30 into train and validation sets). I used Qwen2.5-14B-Instruct to rewrite each chain of thought into a terse, symbolic form and LoRA fine-tuned the base Qwen3-8B model on (question, symbolic CoT) pairs to induce a style transfer.

As an example, on an unseen question, the fine-tuned model reasons like:

```
>R=reg;Q=TAdB&R+S&A&D&C|UO&T+F+J&S|R=A:S,P,D,F,J,T,U,S,C,N,O,V,W,X,Y,Z...
```

Which mimics an illegible, symbolic CoT.

Re-running the evaluation with this model gave:

| Detector | AUROC | 95% CI |
|---|---|---|
| Prompt-only probe | 0.629 | [0.566, 0.687] |
| Activation probe | 0.569 | [0.520, 0.623] |
| Text monitor (3B) | 0.566 | [0.517, 0.621] |
| TF-IDF | 0.550 | [0.499, 0.599] |
| Text monitor (14B) | 0.510 | [0.457, 0.563] |

Note that every confidence interval in this table overlaps every other, so no detector here is distinguishable from another and the ordering should not be read as a ranking.

*[figure: task outcome breakdown, legible vs illegible]*

However, upon evaluation, I noticed that the model performs pretty poorly overall, and so degenerating the CoT, unsurprisingly, degrades task performance. Essentially, the model reverts to answering the question randomly. The issue is that style transfer alone without a correctness signal kills the model entirely. Since the labels depend on the model knowing the answer in the first place, this condition is no longer measuring the same thing the legible condition measured.

## Limitations and Follow-Up

In a follow up, I plan on inducing illegibility with RL on an objective that retains the task reward (answering correctly). The model must stay correct while being hard to read.

Drori et al [3] give a concrete recipe for this. Rather than rewriting traces into a target style, they train with REINFORCE against an output monitor that penalizes mentions of the hint in the final answer, never touching the CoT directly. They call the result feedback spillover: the CoT stops mentioning the hint while task reward stays high, remaining around 0.95 in their MMLU-with-hints Penalty runs. That is the condition I failed to produce here, and the reason seems to be that style transfer has no term rewarding correctness, whereas an output penalty layered on top of task reward preserves it.

In addition, a few more limitations to this experiment also need to be corrected.

- **Behavior Conditioning:** Zolkowski et al. restricts detection to cases where the hidden behavior actually happened [5]. For a fully statistically thorough result, this means dropping the samples where deference was detected via label and ensuring they can be spotted in the CoT itself.
- **Repetition Penalties:** Adding a repetition penalty to the illegible CoT reduced looping, but this is essentially more style shaping instead of letting the model actually reason.
- **Probe Type:** A difference-of-means activation probe is essentially a euclidean distance measure with a threshold check. A logistic-regression probe would be useful to compare against.
- **Lexical Behavior:** The claim that the probe is behaving lexically is currently inferred from the layer depth and pooling choice. Correlating probe scores against TF-IDF scores per sample would test it directly.
- **Cost:** In a production environment, even activations cannot be easily sampled without compromising model throughput. Aremu et al. put the per-request inference cost of activation monitoring at F(m) + O(Kd) against F(m) + F(g) for a separate guard model, on the basis that activations come free from the forward pass the provider already runs [6]. That accounting is asymptotic, and it doesn't cover what reading intermediate activations costs inside a serving engine. An interesting question would be to see how activation sampling affects a model's throughput and latency during the decode phase, especially in a serving engine like vLLM, where CUDA graphs may need to be disabled if sampling occurs.

## References

[1] Baker et al., *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation* — https://arxiv.org/pdf/2503.11926

[2] Korbak et al., *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety* — https://arxiv.org/pdf/2507.11473

[3] Drori, Marks, Woodworth, Cloud & Turner, *Output Supervision Can Obfuscate the Chain of Thought* (ICLR 2026) — https://arxiv.org/pdf/2511.11584

[4] McKenzie et al., *Detecting High-Stakes Interactions with Activation Probes* — https://arxiv.org/pdf/2506.10805

[5] Zolkowski et al., *Can Reasoning Models Obfuscate Reasoning?* — https://arxiv.org/pdf/2510.19851

[6] Aremu, Ognev, Poppi & Lukas, *Adaptively Robust LLM Monitoring via Activation Watermarking* — https://arxiv.org/pdf/2603.23171

[7] Chen et al., *Reasoning Models Don't Always Say What They Think* — https://arxiv.org/abs/2505.05410
