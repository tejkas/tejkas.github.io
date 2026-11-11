---
title: 'Extending KV Cache Management Techniques from MHA to DeepSeek MLA'
description: 'CacheBlend and RocketKV rank tokens by per-head keys that DeepSeek MLA never stores. Measured on real weights, the staleness ranking transfers exactly; the relevance ranking lives almost entirely in the 64-wide rotary key, not the 512-wide latent.'
pubDate: 2026-11-10
tags: ['llm-inference', 'kv-cache', 'deepseek']
---

*Two popular KV-cache techniques rank tokens using quantities defined over each attention head's key. DeepSeek's MLA never writes those keys down. I measured, on real weights, whether the rankings survive the move into MLA's compressed cache. One does. The other turns out to be reading the wrong 512 numbers.*

## Background: the KV cache and MLA

When a transformer generates text, it keeps a key and a value for every past token at every layer, referred to as the KV cache. For long contexts this cache is the thing that fills the GPU's HBM, and a small cornucopia of techniques exists to make it smaller or to avoid recomputing it. These include reusing cached chunks across requests, evicting tokens the model no longer attends to, and reading only the tokens that matter at each step.

Almost all of these techniques are written for the cache as Llama or Mistral store it, namely one key vector and one value vector per attention head (or per group of heads), per token. The methods compute something over those vectors (how far a key moved, how much attention it would receive) and rank tokens by it.

DeepSeek's Multi-head Latent Attention (MLA) stores something else. Per token and layer it keeps a 512-wide *latent* `c`, plus a 64-wide rotary key `k^R` that all heads share. The per-head keys and values are rebuilt from `c` by a projection whenever they are needed (and in the efficient implementation they are never rebuilt at all), because the query is folded through the projection instead ("absorption"), so the query scores the latent directly.

![What one token stores, per layer](./images/fig1-what-a-token-stores.svg)

*Figure 1. Per token and layer, Mistral-7B stores 2,048 numbers; DeepSeek-V2-Lite stores 576. On MLA the attention score has two terms: the absorbed query against the 512-wide latent, and the rotary query against the 64-wide shared key.*

So the question is concrete. Those per-head quantities the techniques depend on: can they be computed in MLA's latent space, and do the token *rankings* they produce survive? I picked one technique from each family — CacheBlend for cache reuse, RocketKV for token selection — and measured.

## Two methods, one assumption

**CacheBlend** reuses the KV cache of text chunks the model has seen before, even when they appear in a new order or alongside new chunks. A reused key is slightly wrong, because the token's attention over its new neighbours was never computed. CacheBlend recomputes only the tokens whose keys and values moved the most — the top 15% by deviation — and relies on an observation that the tokens most deviated at one layer are the most deviated at the next, so a check at an early layer picks the tokens for later ones. The quantity it ranks by is the per-head key's deviation.

**RocketKV** (NVIDIA, 2025) compresses the cache for decoding in two stages. Stage one permanently evicts tokens the prompt never attended to. Stage two, at every decode step, estimates which of the surviving tokens the current query will attend to and reads only those. The estimate is cheap by design: it keeps only the largest coordinates of the query, takes an elementwise max over small pages of keys, and averages the resulting attention across heads that share a key. The paper derives the whole thing from one budget `c`, the target compression ratio: with `S` tokens in context the method may touch `t = S/c` tokens' worth of memory per step, half for the estimate and half for attending to the tokens it picks. The quantity it ranks by is an estimated attention score, built from per-head keys.

## Setup

Model under test: `DeepSeek-V2-Lite` — 27 layers, 16 heads, a 512-wide latent and a 64-wide `k^R`. Control: `Mistral-7B-Instruct-v0.2` — 32 query heads sharing 8 key/value heads, the architecture RocketKV was built for, running RocketKV's own released selection code. Data: 60 multi-hop question-answering documents from MuSiQue, 1,379 to 4,457 tokens each (median 2,414). One rented GPU session captured the caches and queries in bfloat16; everything after that ran on a laptop against the captured tensors and the model's real projection weights.

The measurement for selection is **share of ideal**. At one layer, take the real attention weights of the last prompt token over the context. A selector picks `keep` tokens. Its score is the attention mass on the tokens it picked, divided by the mass on the `keep` tokens the model actually attends to most. Picking at random scores about `keep/S` — 0.004 at `c = 64`. One detail that matters later: the ideal is restricted to what the selector could have chosen. On MLA every head shares one token set, so the ideal is one set chosen on head-averaged attention; on Mistral it is one set per group of four heads.

Before trusting any of it I checked that scoring from the latent reproduces the model's own attention: on the real DeepSeek-V2-Lite, with `output_attentions=True`, the two agree to Spearman 0.99996 and a top-15% overlap of 0.989 at layers 0, 13 and 26. The remaining differences are bfloat16 rounding.

## Staleness: the half that works

For CacheBlend the news is good, and it comes from one line of algebra. Each head's key is a linear function of the latent, `k_h = W_h c`, so the squared deviation of all heads' keys together is a quadratic form in the latent's deviation:

```
Σ_h ‖Δk_h‖²  =  Δcᵀ (Σ_h W_hᵀ W_h) Δc
```

The matrix in the middle is 512 × 512, computed once per layer from the weights. So CacheBlend's exact ranking quantity *is* computable without materializing a single per-head key. Checked against keys rebuilt the slow way, the relative error is 1.0e-06 over 60 documents.

A close cousin does the same work and more. Build the matrix from the whole `kv_b_proj`, the key and value blocks together, instead of the key block alone. Same cost, and it scores the value's deviation too, which the key-only identity does not. Against the exact key ranking it keeps 0.953 of the top-15% set (rank correlation 0.998). That is the signal I would recommend. The matrix-free alternative, the plain squared deviation of the 576 stored numbers, keeps 0.847, just under the bar I had set; the latent alone, dropping `k^R`, keeps 0.752.

CacheBlend's cross-layer observation also transfers: the deviation ranking at one layer predicts the next layer's at Spearman 0.946.

One number from this half foreshadows the other. Early in the network, 95% of the staleness comes from the 64-wide `k^R`, not the 512-wide latent. The latent passes through a layer norm that fixes its scale; nothing constrains `k^R`, and its norm is about ten times larger.

## Relevance, part 1: the budget arithmetic

RocketKV's selection step scores a compressed summary of each page of keys. Its cost is the number of pages times the width of the summary. Substituting the paper's own formula for how many pages survive, the page size cancels and the cost becomes

```
selection traffic  =  t × (summary width) / (numbers stored per token)
```

On Mistral the summary is a key and a key is half of what a token stores (the value is the other half, and it is only fetched for tokens that survive selection). So the selection step costs exactly `t/2` — precisely the half the paper assigns it. The derivation is built around this identity, and it holds at every compression ratio.

On MLA the value is rebuilt from the same latent that is being scored. Nothing escapes being summarized, so the ratio is close to 1 and the selection step alone costs about `1.1t` — more than the whole budget.

![What the selection step reads](./images/fig2-selection-traffic.svg)

*Figure 2. Selection-step traffic in units of the budget `t`. Mistral lands on `t/2` exactly. MLA as published pays 1.06–1.12`t` for the selection step alone; scoring only the 64-wide `k^R` with pages of two brings it to 0.26–0.35`t`.*

Asked for 64× compression, RocketKV on MLA delivers 39.5×. This is arithmetic from the paper's own formulas and independent of every measured result below. It is also the one finding that is directly actionable for the authors: the fix is to send more of the compression to stage one, or to summarize far less than they do.

## Relevance, part 2: the latent barely ranks tokens

I expected the interesting failure to be in RocketKV's query-coordinate reduction — keeping only the largest 12.5% of the query's coordinates seemed unlikely to survive the query being folded through a 512-wide projection. That was wrong: at the kept fraction the paper's derivation actually produces (about half the coordinates), the reduction costs at most 0.006 of share-of-ideal on either architecture.

The interesting failure was somewhere else. On MLA the attention score has two terms (Figure 1): the absorbed query against the 512-wide latent, and the rotary query against the 64-wide `k^R`. I scored tokens on each term alone.

![Share of ideal by selector](./images/fig3-share-of-ideal.svg)

*Figure 3. Share of ideal attention mass for six selectors at three compression ratios. Scoring the 64-wide half alone captures 0.97–0.98. Scoring the 512-wide latent alone captures 0.19 at the loosest budget and 0.02 at the tightest, against a random floor of 0.016–0.001.*

The 64 numbers carry the ranking. The 512 do not. RocketKV's published pipeline scores all 576 and lands at 0.869 / 0.875 / 0.731 — which is almost exactly what `k^R` alone scores once the same page-max is applied to it (0.860 / 0.873 / 0.767). The latent contributes nothing to the ordering.

The reason is magnitude, and the same one as in the staleness half. Over 27 layers, 16 heads and 60 documents, the rotary term's spread across context tokens is 2.08× the latent term's, and the two terms are close to independent. Under independence that ratio alone predicts rank correlations of 0.90 and 0.43 between the full score and each term; measured, they are 0.904 and 0.523.

![Which half carries the ranking, per layer](./images/fig4-which-half-ranks.svg)

*Figure 4. Rank correlation between the full score and each term alone, by layer. The rotary half leads in 24 of 27 layers; the latent takes over at layers 12, 19 and 24.*

The obvious objection is that `k^R` is the *rotary* key — the half of the key that carries position — so maybe it is just picking recent tokens. It is not. `k^R` is a 64-wide projection of the token's content that is then rotated by position; it is content first. Three checks:

- Picking the `keep` most recent tokens captures 0.028 of ideal.
- Keeping every rotation at its own position but shuffling which token's content sits there — exact on the stored keys, because rotation is a rotation — collapses `k^R` selection from 0.898 to 0.011, two or three times the random floor and no more.
- The first context token (the "attention sink") does carry mass on its own, 0.605 of ideal at `c = 64`. Giving every selector that token for free and scoring only what they pick beyond it, `k^R` adds 0.29–0.36 above the floor and the latent adds 0.07.

## A better operating point for MLA

If the 64 numbers carry the ranking, the selection step should read those 64 instead of all 576. That drops its traffic by 9×, which turns the page size into a free parameter. Pages of two dominate the published setting on both axes at every budget:

| asked `c` | setting | selection traffic | delivered ratio | share of ideal |
|---|---|---|---|---|
| 16 | as published | 1.06`t` | 10.2× | 0.869 |
| 16 | `k^R`, pages of 2 | 0.26`t` | **21.0×** | **0.891** |
| 64 | as published | 1.12`t` | 39.5× | 0.875 |
| 64 | `k^R`, pages of 2 | 0.35`t` | **75.6×** | **0.894** |
| 256 | as published | 1.11`t` | 159× | 0.731 |
| 256 | `k^R`, pages of 2 | 0.33`t` | **309×** | **0.925** |

It also meets the budget the paper's own derivation asks for, which the published setting on MLA does not.

## Trying to break it

A result this clean deserves suspicion, so I spent more time attacking it than finding it.

- **Softmax temperature.** The method divides page scores by a factor that depends on the width of what is scored, so each variant ran at a different sharpness, and the head-averaging step is sharpness-dependent. Forcing every variant to the model's own sharpness moved no `k^R` variant by more than 0.014 and made the latent *worse* (0.070 → 0.049 at `c = 64`). The natural temperature had been flattering the latent.
- **Which query token.** Every headline number scores the last prompt token. Across 16 stored query positions the spread ratio stays within 1.82–2.08 and `k^R` selection within 0.89–0.92.
- **Query distance.** Re-rotating the query up to 1,024 tokens further from the context changes the ratio by less than 0.04. Beyond 4,096 the attention collapses onto a handful of tokens and every selector scores 1.0, so the measurement degenerates rather than improving.
- **Rotation versus content**, the shuffle above.
- **The sink**, the conditioned sweep above.
- **The instrument itself**, the Spearman 0.99996 agreement with the model's own attention, at the content-dominated end of the network as well as the rotary-dominated middle.

## What a reviewer caught

An outside review of the write-up found a dozen things, all but one of them prose — a range quoted from the wrong file, a "three orders of magnitude" that was a factor of 55, a layer called third-best that was first. The eleventh was a measurement defect.

Share of ideal is a ratio, and its denominator is the best selection the selector *could* have made. On MLA I had built that oracle as one set chosen on head-averaged attention, matching what the selector emits. On Mistral the code gave each of the 32 heads its own best set — a ceiling the per-group selector cannot reach by construction, because the four heads in a group disagree about which tokens matter. Mistral was being divided by too large a number.

The error ran against the MLA-is-worse conclusion, not for it. With a matched oracle, Mistral's scores rose from 0.907 / 0.897 / 0.844 to 0.932 / 0.921 / 0.865, and the gap to MLA widened from 0.038 / 0.022 / 0.113 to 0.063 / 0.046 / 0.134. The finding survived. But it was the second time an outside reading had caught a convention that differed between the two sides of a comparison — the budget split was the first — and both times the two code paths had been written weeks apart and never diffed against each other. If I did this again, that diff would be the first test.

## What this does not show

- **The MLA-versus-Mistral gap is weakly supported.** Per-layer spread on MLA runs 0.33 to 0.95, wider than every gap, and the trend across budgets is not monotone. The within-MLA comparison — latent versus `k^R` — is the strong result; the cross-architecture one is suggestive.
- **Attention mass is not output quality.** Capturing the right attention is necessary for an unchanged answer, not sufficient. Measuring end-task quality needs RocketKV implemented inside MLA attention and a benchmark harness.
- **Stage one is untested.** The permanent eviction does most of the compression and shares the same forced head grouping.
- **Sixteen heads is small.** DeepSeek-V3 has 128 heads sharing one coordinate set.
- **One thing I cannot explain.** Layer 5 is the best of 27 layers for latent-only selection and 26th of 27 for how much the latent term contributes to the score. Across layers the two rank-correlate at only 0.24. The per-layer story does not line up with the aggregate one.

## What I would do next

Implement the `k^R`-only selection inside MLA attention and measure end-task accuracy on a long-context benchmark; measure real bandwidth rather than counting bytes; test stage-one eviction; and try a 128-head model, where the shared coordinate set spans eight times as many per-head bases.

Code, captured results and the claims ledger with every number's provenance are at [github.com/tejkas/absorbkv](https://github.com/tejkas/absorbkv). Every number in the figures is in `results/*.json`; `blog/figures.py` draws them.
