# One LoRA for all verticals? (observation, 2026-09-27)

**中文摘要：** 用一个 LoRA 代替每个任务各自一个 LoRA：在 13 个 vertical 的训练数据（外加几个 Decision Index 弱项的训练数据）上混训一个 LoRA，
1 epoch，一张 RTX PRO 6000 约 2.5 GPU 小时。在各任务固定测试集上逐题配对：读过 Jev 的 12 个任务里 **7 个显著超过 Jev**；和每个任务自己的
LoRA 比，When2Call / Mind2Web / RewardBench 打平、SGD intent 只差 1.4、agent 轨迹护栏（hard）反而 +3.5；BANKING77 / AndroidControl /
MetaTool / 提示注入差 4–7 分；**ToxicChat（−11）和两个路由任务（F1 ≈ 2，"永远不交给强模型"）失效**。结论：一个共享适配器能覆盖大多数意图和
agent 决策任务，代价 1–5 分；正例稀少的护栏任务和按概率排序的路由任务还需要单独的适配器。

## 1. Question

Can one LoRA serve every vertical, instead of one adapter per task? A customer with a dozen decision points (intent routing,
tool choice, guardrails, agent next-step) would rather deploy one adapter than twelve.

## 2. Setup

- **Model:** one LoRA on an intermediate Wald-4B build between v0.9 and v1.0 (the v1.0 base before its final
  task-coverage LoRA was merged; see Versioning in the README). Recipe: LoRA r32 / α64 on every projection, lr 1e-4 cosine,
  5 % warm-up, **1 epoch**, letter readout, and 30 % replay rows trained toward the base's own answer distribution (a KL
  anchor). The LoRA is kept separate, never merged into the released weights.
- **Data** (commercial licences only): 53,243 rows = 37,270 labelled + 15,973 replay, about 37M prompt tokens.
  - 13 verticals, each capped at 2,500 rows after removing near-duplicates of **every** vertical's test / calibration /
    development items. Rows per task after that: When2Call, BANKING77, SGD intent, ToxicChat, MetaTool, AndroidControl,
    RewardBench, Mind2Web 2,500 each; model routing 1,174 (and 1,174 for a second framing of the same prompts); prompt
    injection 780; agent-trajectory safety 52; **agent-trajectory safety (hard) 0**.
  - Training splits of a few Decision Index benchmarks where the base trails (a home-appliance simulator, iSarcasmEval,
    API-Bank, ContractNLI), about 16.5k rows.
  - Uniform mixing: no per-task sampling weight, no loss re-weighting.
- **Cost:** about 2.5 GPU-hours of training on one RTX PRO 6000 (≈ $2.4 at the GPU-hour price), about $0.8 for the 13
  evaluation reads.
- **Evaluation:** each task's **fixed test items**, a temperature fitted on that task's calibration split, paired on the
  same items (1,000 item resamples; SGD intent 2,000) against:
  - **that task's own LoRA** (all labels). **Caveat: every own LoRA sits on Wald-4B v0.9**, a different build, so this
    comparison is paired on items, not on base;
  - **Jev** (same request bytes);
  - **the same base zero-shot**, where a hosted read exists (a different reader than ours).

## 3. Results

| Task | Metric | one LoRA | own LoRA (v0.9) | Jev | base zero-shot | one − own | one − Jev | one − zero-shot |
|---|---|---:|---:|---:|---:|---|---|---|
| When2Call | accuracy | 82.4 | 82.8 | 72.6 | 76.6 | −0.4 [−2.8, +2.2] | **+9.8** [+6.2, +13.2] | +5.8 [+2.2, +9.2] |
| BANKING77 | macro-F1 | 88.2 | 92.8 | 78.1 | — | −4.6 [−7.2, −2.6] | **+10.1** [+7.3, +14.1] | — |
| SGD intent | macro-F1 | 96.6 | 98.0 | 92.8 | — | −1.4 [−2.7, −0.2] | **+3.7** [+1.5, +6.6] | — |
| AndroidControl | accuracy | 81.8 | 85.8 | 73.4 | — | −4.0 [−6.8, −1.0] | **+8.4** [+4.4, +12.4] | — |
| MetaTool | accuracy | 90.2 | 97.0 | 83.0 | 81.4 | −6.8 [−9.2, −4.4] | **+7.2** [+4.4, +10.4] | +8.8 [+6.2, +11.6] |
| Agent-trajectory safety (hard) | unsafe F1 | 75.7 | 72.2 | 47.0 | 29.3 | **+3.5** [+1.8, +5.2] | **+28.7** [+21.6, +36.3] | +46.5 [+39.5, +53.7] |
| Agent-trajectory safety | unsafe F1 | 98.0 | — | 92.2 | 74.8 | — | **+5.8** [+3.2, +8.7] | +23.2 [+18.7, +28.1] |
| Mind2Web | accuracy | 51.2 | 51.9 | 48.5 | — | −0.6 [−5.0, +3.3] | +2.7 [−1.5, +6.9] | — |
| RewardBench | accuracy | 82.8 | 83.4 | 90.6 | 83.6 | −0.6 [−2.5, +1.1] | −7.8 [−9.9, −5.7] | −0.8 [−2.4, +0.8] |
| Prompt injection | injection F1 | 75.0 | 79.4 | 82.8 | 72.7 | −4.4 [−7.9, −1.4] | −7.8 [−14.1, −0.9] | +2.3 [−1.8, +6.4] |
| ToxicChat | toxic F1 | 73.8 | 84.9 | 80.4 | 71.8 | **−11.1** [−14.5, −7.9] | −6.6 [−10.0, −3.4] | +2.0 [−0.6, +4.6] |
| Model routing | needs-strong F1 | 1.9 | 26.8 | 0.0 | 10.4 | **−24.9** [−34.0, −14.7] | +1.9 [+0.0, +6.2] | −8.5 [−16.9, −0.6] |
| Model routing (difficulty framing) | hard F1 | 2.0 | — | — | 16.2 | — | — | −14.2 [−22.7, −6.0] |

## 4. Three groups

**One adapter is enough.**
- When2Call (−0.4), Mind2Web (−0.6) and RewardBench (−0.6) are level with their own LoRA; SGD intent is 1.4 behind.
- Agent-trajectory safety (hard) is **+3.5 ahead** of its own LoRA with **zero** rows of its own task in the mix: the gain is
  transfer from the other agent and guard tasks. Its own LoRA had only 250 training rows and was often confidently wrong,
  so the bar was low.

**A small cost (−4 to −7).**
- BANKING77 −4.6, AndroidControl −4.0, MetaTool −6.8, prompt injection −4.4. All but prompt injection still beat Jev.
- The first three had 2,500 rows in the mix vs 8,000–30,000 for their own LoRA, so part of the gap is data volume (the own
  LoRAs gain about one point per doubling of labels), not interference.

**It breaks.**
- **ToxicChat −11.1** vs its own LoRA, back to the zero-shot level. 2,500 rows at about 7 % toxic means about 175 positives
  among 37k labelled rows: the rare-positive signal is drowned.
- **Model routing collapses to "never route"** (F1 1.9 vs 26.8; 2.0 for the second framing). Routing is a probability-
  ranking task: its value is the ordering of P(needs a strong model) across prompts, and the mixed model puts nearly all mass
  on the majority option. It also had only 1,174 rows after near-duplicate removal (own LoRA: 8,000).

## 5. Why (hypotheses, not tested)

1. **Class imbalance under uniform mixing.** Rare-positive tasks (ToxicChat 7 %, routing 20 %) contribute few positive
   gradients in a 37k-row mix, so the model learns their majority option.
2. **Ranking / threshold tasks lose their calibrated margin.** The replay anchor and other tasks' confident targets sharpen
   the answer head; routing needs small, well-ordered probability differences that argmax F1 at the default threshold misses.
3. **Label-space interference.** The letter readout shares (A) / (B) across binary tasks with different meanings and base
   rates, which can pull against each other.
4. **Data volume and isolation losses.** The per-task cap (2,500) and the cross-task near-duplicate removal (routing
   8,000 → 1,174; hard agent-safety 250 → 0) mean the mix is not the data the dedicated LoRAs saw.
5. **Base difference.** The one LoRA and the dedicated LoRAs sit on different builds. On one browser task a base swap alone
   changed nothing (−0.4 [−2.9, +2.1]), so this likely explains little, but it is not ruled out per task.

## 6. What to try next

| Idea | What | Rough cost |
|---|---|---|
| Per-task sampling weights | sample tasks ∝ n^α (α ≈ 0.3–0.5), or up-weight rare-positive and ranking tasks | ≈ $2.5 + $0.8 reads |
| Task-balanced or focal loss | class-balanced weights within each binary task, or focal loss on rare-positive tasks | ≈ $2.5 + $0.8 |
| Hybrid adapters | the shared adapter plus small separate adapters for ToxicChat, prompt injection and routing | ≈ $0.2–1.5 per extra adapter; multi-LoRA serving |
| Per-task calibration | per-task temperature and decision threshold on calibration data (routing: rank-based, not argmax) | ≈ $0 (offline) |
| Same-base comparison | dedicated and shared LoRAs on the same build | ≈ $2.5 |
| Lighter isolation | remove near-duplicates only against each task's own evaluation items | $0 data + one re-train |

## 7. Product takeaway

One adapter covers most intent and agent-decision tasks at a 1–5 point cost against dedicated adapters, and still beats Jev
on 7 of 12 tasks. Imbalanced guard tasks (ToxicChat) and probability-ranking tasks (routing) still need their own adapter,
or a hybrid of one shared adapter plus small per-task ones.

## Caveats

- For 7 tasks the Jev numbers are Jev's reads from the dedicated-LoRA evaluations on the same item ids, not a fresh read.
- Mind2Web was read with plain knockout, because the adapter was trained on knockout-shaped questions; the knockout +
  top-10 re-read gives 49.0.
- The dedicated LoRAs are on Wald-4B v0.9: paired on items, not on base.
- The base zero-shot column comes from a hosted read, not our evaluation reader.
- One seed, one epoch, one mix; no sampling-weight search.
