# Contamination statement

This covers the checkpoint submitted as Wald-4B and **every stage of its lineage**, back to the base model. The Decision
Index rule is that any test item an entry was trained on counts as wrong. The list below is for the maintainers to apply
that rule. We also count these items wrong in our own blocklist-adjusted scores.

## 1. Decision Index items in our training data

**[contamination/trained-on-di-ids.json](contamination/trained-on-di-ids.json)** lists 363 item ids. For each id it
gives the benchmark, the lineage stage whose data contains it, and why it is there. Please count all 363 as trained on.

| benchmark | ids | scored in 0.2.1 | why they are in our data |
|---|---:|---|---|
| ToolRet (#2) | 67 | yes | **shared upstream source.** ToolRet aggregates the user queries of public tool-calling datasets. Our data contains the train splits of ToolACE and Glaive function-calling. |
| BANKING77 (#4) | 49 | yes | **public train split duplicates a test sentence.** The BANKING77 train split is in our data, and these test sentences also appear verbatim in train. |
| CLINC150+OOS (#5) | 44 | yes | **public train split duplicates a test sentence** (as above, for the CLINC-OOS train split). |
| MMLU-Pro (#57) | 6 | yes | **shared upstream source.** MMLU-Pro includes MATH / TheoremQA problems, and the same problems are in our math data (MATH train). |
| SATA-Bench (#33) | 4 | yes | **shared upstream source.** Reading passages from RACE, which makes up most of MMLU auxiliary_train. |
| ANLI (#12) | 3 | yes | **shared upstream text.** The premise text also appears in other public datasets in our data. |
| WinoGrande (#28) | 2 | yes | **public train split duplicates a test sentence.** |
| BRIGHT (#36) | 1 | yes | **shared upstream source.** A public programming problem statement. |
| RouterBench (#6) | 170 | no (removed in 0.2.1) | **shared upstream source.** RouterBench prompts embed GSM8K / MMLU questions, and those questions are in our math and QA training data (GSM8K train and others). |
| MMLU (#24), ARC-Easy (#26), ARC-Challenge (#27) | 6 / 7 / 4 | display only | **public train split duplicates** (ARC train, MMLU auxiliary_train). These are the same kind of duplicates the board already records for other entries. |

That is 176 ids on scored benchmarks. On our 6,948-request sample, counting them wrong moves the index by 0.3 points:
53.91 → 53.62 for v1.0 (effort medium, `repeat_state_plain`), and 49.62 → 49.34 for its parent at high-k4.

**How the items got in.** We never trained on a Decision Index test split. Every hit above reached us through a public
train split or an upstream dataset that the benchmark itself draws from. Stages 1–3 of the lineage were built before we
had the full blocklist (below), so those stages contain these items. Stage 4 (the task-coverage LoRA) was filtered with the full blocklist before training: 0 hits.

## 2. Training data, by source

This section names every public source in the lineage and the split we read. It gives no proportions: no ratios, token
or row counts per source, or mixing weights. The training data itself is not released.

### 2.1 The lineage

| stage | what it is | data |
|---|---|---|
| 1 | full-parameter decision training of Qwen/Qwen3.5-4B-Base | converted public datasets (§2.2), code-generated decision families, and synthetic tool, document-QA and style items |
| 2 | LoRA refinement with KL replay (merged) | a curated mix of human-labelled public sets and code-generated items, synthetic hard items, and a KL-only replay of stage-1 rows |
| 3 | short-thought distillation (LoRA, merged) | synthetic short thoughts on questions from the stage-2 pool, one-pass rows from the same pool, and replay rows whose target is the stage-2 model's own distribution |
| 4 | task-coverage LoRA (merged; the submitted checkpoint) | the train splits of iSarcasmEval, API-Bank, ContractNLI, VAST, NLI4CT, ACOS and RAGTruth, new Home-appliance households from the Decision Index kit's generator, and a KL-only replay of stage-2 rows |

Stage 1 drew on a public decision-training mixture. Its constituent datasets are listed one by one below, and its custom
and routing parts were left out. Stage 4 teaches the formats of eight Decision Index benchmarks from their source train
splits and the kit's own generator. No test item is included. **Home appliance scores 1.00 (skill) for this checkpoint
on our sample, so please weigh that benchmark with this training in mind.** The maintainers can regenerate the rows
with the kit's generator and any new seed, and compare them with their test rows.

### 2.2 Public sources

Every public dataset was read through its **train split only**, unless the split column says otherwise. No test split
of any source was used.

In the stage column, **(replay)** means that the stage re-used rows from that source only as KL replay. For those rows
the training target is the model's own earlier distribution, not the source label.

The licence column gives the licence **as the source states it**. "card" means the Hugging Face dataset card, and
"upstream" means the original release, used where the card is empty or says unknown. We have not reviewed these terms
here.

#### Constituents of the stage-1 public mixture

| dataset | split(s) used | stages | licence (as stated) | note |
|---|---|---|---|---|
| [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) (all) | auxiliary_train | 1, 2, 3, 4 (replay) | MIT (card) | dev / validation / test never used. auxiliary_train is mostly RACE passages (see §1, SATA-Bench) |
| [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) (ARC-Challenge, ARC-Easy) | train | 1, 2, 3, 4 (replay) | CC-BY-SA-4.0 (card) | also in the wide replay (ARC-Easy). Some test questions duplicate train questions (§1) |
| [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa) | train | 1, 2, 3, 4 (replay) | MIT (card) | also read directly as a human-labelled set |
| [google/boolq](https://huggingface.co/datasets/google/boolq) | train | 1, 2, 3, 4 (replay) | CC-BY-SA-3.0 (card) | also read directly as a human-labelled set |
| [nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli) | train | 1, 2, 3, 4 (replay) | mixed: CC-BY-3.0 / CC-BY-SA-3.0 / MIT / OANC (card) | validation_matched / mismatched never used. Also read directly |
| [stanfordnlp/snli](https://huggingface.co/datasets/stanfordnlp/snli) | train | 1, 2, 3, 4 (replay) | CC-BY-SA-4.0 (card) | |
| [mteb/banking77](https://huggingface.co/datasets/mteb/banking77) | train | 1, 2, 3, 4 (replay) | MIT (card); upstream PolyAI release CC-BY-4.0 | some test sentences also appear verbatim in train (§1) |
| [allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa) (main) | train | 1, 2, 3, 4 (replay) | Apache-2.0 (upstream allenai/OpenBookQA) | also read directly as a human-labelled set |
| [GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | train | 1, 2, 3, 4 (replay) | CC-BY-4.0 (card); upstream MIT | |
| [allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande) (winogrande_xl) | train | 1, 2, 3, 4 (replay) | CC-BY-4.0 (upstream allenai/winogrande) | also in the wide replay. Some test sentences duplicate train (§1) |
| [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) (plus) | train | 1, 2, 3, 4 (replay) | CC-BY-3.0 (card) | also in the wide replay. Some test sentences duplicate train (§1) |
| [SetFit/amazon_massive_intent_en-US](https://huggingface.co/datasets/SetFit/amazon_massive_intent_en-US) | train | 1, 2, 3, 4 (replay) | CC-BY-4.0 (upstream Amazon MASSIVE) | |
| [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | train (the only split) | 1, 2, 3, 4 (replay) | CDLA-Sharing-1.0 (card) | |
| [fancyzhx/dbpedia_14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | train | 1, 2, 3, 4 (replay) | CC-BY-SA-3.0 (card) | also read directly as a human-labelled set |
| [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) (simplified) | train | 1, 2, 3, 4 (replay) | Apache-2.0 (card) | |
| [SetFit/hate_speech_offensive](https://huggingface.co/datasets/SetFit/hate_speech_offensive) | train | 1, 2, 3, 4 (replay) | MIT (upstream t-davidson/hate-speech-and-offensive-language) | |
| [SetFit/amazon_counterfactual_en](https://huggingface.co/datasets/SetFit/amazon_counterfactual_en) | train | 1, 2, 3, 4 (replay) | CC-BY-SA-4.0 (upstream amazon-research/amazon-multilingual-counterfactual-dataset) | |
| [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) | train (a leading slice) | 1, 2, 3, 4 (replay) | CC0-1.0 (card) | also in the wide replay |
| [ucirvine/sms_spam](https://huggingface.co/datasets/ucirvine/sms_spam) | train (the only split) | 1, 2, 3, 4 (replay) | CC-BY-4.0 (upstream UCI SMS Spam Collection) | |
| [allenai/qasc](https://huggingface.co/datasets/allenai/qasc) | train | 1, 2, 3, 4 (replay) | CC-BY-4.0 (card) | also in the wide replay |
| [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) | train | 1, 2, 3, 4 (replay) | MIT (upstream rowanz/hellaswag) | validation / test never used |
| [ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa) | train | 1, 2, 3, 4 (replay) | unknown | also in the wide replay |
| [LabHC/bias_in_bios](https://huggingface.co/datasets/LabHC/bias_in_bios) | train | 1, 2, 3, 4 (replay) | MIT (card) | |
| [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | train | 2, 3 | CC-BY-4.0 (card) | |
| [nvidia/HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) (preference) | train | 1, 2, 3, 4 (replay) | CC-BY-4.0 (card) | also in the wide replay |
| [ucberkeley-dlab/measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) | train (the only split) | 1, 2, 3, 4 (replay) | CC-BY-4.0 (card) | |
| [chengxuphd/liar2](https://huggingface.co/datasets/chengxuphd/liar2) | train | 1, 2, 3, 4 (replay) | Apache-2.0 (card) | |
| [allenai/prosocial-dialog](https://huggingface.co/datasets/allenai/prosocial-dialog) | train | 1, 2, 3, 4 (replay) | CC-BY-4.0 (card) | also in the wide replay |
| [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | train_prefs | 1, 2, 3, 4 (replay) | MIT (card) | test_prefs never used |
| [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) | train | 1, 2, 3, 4 (replay) | MIT (card) | |
| [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | train (the only split) | 1, 2, 3, 4 (replay) | Apache-2.0 (card) | also a seed source for the synthetic tool items. Shares user queries with ToolRet (§1) |
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | train (the only split) | 1, 2, 3, 4 (replay) | Apache-2.0 (card) | also in the wide replay and the tool items. Shares user queries with ToolRet (§1) |
| [copenlu/fever_gold_evidence](https://huggingface.co/datasets/copenlu/fever_gold_evidence) | train | 1, 2, 3, 4 (replay) | unknown | |
| [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) | train | 1, 2, 3, 4 (replay) | Apache-2.0 (card); upstream MIT | also in the wide replay |
| [osunlp/Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) | train | 1, 2, 3, 4 (replay) | CC-BY-4.0 (card) | also a seed source for the synthetic tool items |

#### Wide replay of public train splits (sources not listed above)

| dataset | split(s) used | stages | licence (as stated) | note |
|---|---|---|---|---|
| [allenai/cosmos_qa](https://huggingface.co/datasets/allenai/cosmos_qa) | train | 1, 2 (replay), 3, 4 (replay) | CC-BY-4.0 (card) | |
| [ucinlp/drop](https://huggingface.co/datasets/ucinlp/drop) | train | 1, 2 (replay), 3, 4 (replay) | CC-BY-SA-4.0 (card) | |
| [allenai/quoref](https://huggingface.co/datasets/allenai/quoref) | train | 1, 2 (replay), 3, 4 (replay) | CC-BY-4.0 (card) | |
| [allenai/quartz](https://huggingface.co/datasets/allenai/quartz) | train | 1, 2 (replay), 3, 4 (replay) | CC-BY-4.0 (card) | |
| [allenai/ropes](https://huggingface.co/datasets/allenai/ropes) (plain_text) | train | 1, 2 (replay), 3, 4 (replay) | CC-BY-4.0 (card) | |
| [alisawuffles/WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) | train | 1, 2 (replay), 3, 4 (replay) | CC-BY-4.0 (card) | crowd labels that revise model-written candidates |
| [allenai/social_i_qa](https://huggingface.co/datasets/allenai/social_i_qa) | train | 1, 2 (replay), 3, 4 (replay) | CC-BY-4.0 (card text) | |
| [deepmind/aqua_rat](https://huggingface.co/datasets/deepmind/aqua_rat) (raw) | train | 1, 2 (replay), 3, 4 (replay) | Apache-2.0 (card) | |
| [ChilleD/SVAMP](https://huggingface.co/datasets/ChilleD/SVAMP) | train | 1, 2 (replay), 3, 4 (replay) | MIT (card) | |
| [math-eval/TAL-SCQ5K](https://huggingface.co/datasets/math-eval/TAL-SCQ5K) (EN) | train | 1, 2 (replay), 3, 4 (replay) | MIT (card) | |
| [tasksource/ruletaker](https://huggingface.co/datasets/tasksource/ruletaker) | train | 1, 2 (replay), 3, 4 (replay) | Apache-2.0 (card) | |
| [ImperialCollegeLondon/health_fact](https://huggingface.co/datasets/ImperialCollegeLondon/health_fact) (PUBHEALTH) | train | 1, 2 (replay), 3, 4 (replay) | MIT (card) | |
| [wenhu/tab_fact](https://huggingface.co/datasets/wenhu/tab_fact) | train | 1, 2 (replay), 3, 4 (replay) | CC-BY-4.0 (card) | |
| [lighteval/wikitablequestions](https://huggingface.co/datasets/lighteval/wikitablequestions) | train | 1, 2 (replay), 3, 4 (replay) | CC-BY-SA-4.0 (upstream ppasupat/WikiTableQuestions) | |
| [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive) (en-US) | train | 1, 2 (replay), 3, 4 (replay) | CC-BY-4.0 (card) | |
| [google-research-datasets/schema_guided_dstc8](https://huggingface.co/datasets/google-research-datasets/schema_guided_dstc8) (SGD) | train | 1, 2 (replay), 3, 4 (replay) | CC-BY-SA-4.0 (card) | |
| [tasksource/esci](https://huggingface.co/datasets/tasksource/esci) (us) | train | 1, 2 (replay), 3, 4 (replay) | Apache-2.0 (card) | |
| [google-research-datasets/poem_sentiment](https://huggingface.co/datasets/google-research-datasets/poem_sentiment) | train | 1, 2 (replay), 3, 4 (replay) | CC-BY-4.0 (card) | |
| [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | train | 1, 2 (replay), 3, 4 (replay) | MIT (card) | |
| [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) (glaive subset) | train | 1, 2 (replay), 3, 4 (replay) | Apache-2.0 (card) | |
| [openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) (evol_instruct, ultrachat) | train | 1, 2 (replay), 3, 4 (replay) | MIT (card) | ratings in the source are model-written |
| [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | train | 1, 2 (replay), 3, 4 (replay) | Apache-2.0 (card) | |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | train | 1, 2 (replay), 3, 4 (replay) | Apache-2.0 (card) | |
| [tonytan48/TempReason](https://huggingface.co/datasets/tonytan48/TempReason) | train_l1 | 1, 2 (replay), 3, 4 (replay) | CC-BY-SA-3.0 (card) | answers recomputed by code |

#### Other public decision sets

| dataset | split(s) used | stages | licence (as stated) | note |
|---|---|---|---|---|
| [jaredpalmer/kev-suites](https://huggingface.co/datasets/jaredpalmer/kev-suites) | v7/decision-v7/train.jsonl | 2, 3 | Apache-2.0 (card); rows derive from public datasets under their own terms | Kev's public training partition. Its rows come from the BANKING77, BoolQ, DBpedia-14 and MNLI train splits |
| [thu-coai/cold](https://huggingface.co/datasets/thu-coai/cold) (COLD, Chinese) | train | 2, 3 | Apache-2.0 (repo) | test never used for training |
| [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) | train | 1, 2, 3, 4 (replay) | CC0-1.0 (card); code MIT | labels by rules, solvers and generator latents. Its calibration / validation / test / OOD rows were never used. `customer-control-v1` excluded |
| [ZefanCai/Open-Jev-v1.1](https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1) | train (`wanli-decisions-v1`, `community-diversity-v2`) | 2, 3 | CC0-1.0 (community-diversity-v2); CC-BY-4.0 (wanli-decisions-v1, from WANLI) | |

#### Prompt sources for routing and verification items

The items themselves are ours. A routing item labels a public prompt by the kind of source it came from, and a
verification item plants an error in a human-written solution by code.

| dataset | split(s) used | stages | licence (as stated) | note |
|---|---|---|---|---|
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (main) | train | 1, 2, 3, 4 (replay) | MIT (card) | routing prompts and the worked solutions of the verification items. Test never used. RouterBench embeds GSM8K questions (§1) |
| [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | train | 1, 2, 3, 4 (replay) | MIT (card) | test never used. MMLU-Pro includes MATH problems (§1) |
| [deepmind/code_contests](https://huggingface.co/datasets/deepmind/code_contests) | train | 1, 2, 3, 4 (replay) | CC-BY-4.0 (card) | valid / test never used |
| [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) (full) | train, validation, prompt | 2, 3 | CC-BY-4.0 (card) | test never used |
| [princeton-nlp/SWE-bench](https://huggingface.co/datasets/princeton-nlp/SWE-bench) | train | 1, 2, 3, 4 (replay) | MIT (card) | test (the benchmark) never used |
| [mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa) (rc.nocontext) | train | 1, 2, 3, 4 (replay) | Apache-2.0 (card) | |
| [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) (3.0.0) | train | 1, 2, 3, 4 (replay) | Apache-2.0 (card) | |
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | train | 1, 2, 3, 4 (replay) | Apache-2.0 (card) | |
| [HuggingFaceM4/WebSight](https://huggingface.co/datasets/HuggingFaceM4/WebSight) (v0.2) | train (the only split) | 1, 2, 3 | CC-BY-4.0 (card) | |

#### Text embedded in long states

| dataset | split(s) used | stages | licence (as stated) | note |
|---|---|---|---|---|
| [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample-10BT) | train (the only split) | 1, 2, 3, 4 (replay) | ODC-BY-1.0 (card) | documents placed inside code-generated long-state items |
| [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (20231101.en) | train (the only split) | 1, 2, 3, 4 (replay) | CC-BY-SA-3.0 and GFDL (card) | as above |

#### Seed sources of the synthetic tool items

| dataset | split(s) used | stages | licence (as stated) | note |
|---|---|---|---|---|
| [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) | train_sft | 1 | CC-BY-4.0 (card) | test never used |
| [liminghao1630/API-Bank](https://huggingface.co/datasets/liminghao1630/API-Bank) | training-data: lv1 (stage 1); lv1 + lv2 (stage 4) | 1, 4 | MIT (card); the GitHub repo AlibabaResearch/DAMO-ConvAI (api-bank) states Apache-2.0 | test-data never used. In stage 4, 53-option tool catalogs are built the kit's way from the train tool pool |
| [AgentGym/AgentTraj-L](https://huggingface.co/datasets/AgentGym/AgentTraj-L) | train | 1 | unknown | |

#### Stage 4: Decision Index benchmark sources

| dataset | split(s) used | stages | licence (as stated) | note |
|---|---|---|---|---|
| [iSarcasmEval](https://github.com/iabufarha/iSarcasmEval) (SemEval-2022 Task 6) | train.En.csv (tasks A-En, B-En), train.Ar.csv (task A-Ar) | 4 | MIT (repo LICENSE); the README asks for the task citation | test never used |
| [ContractNLI](https://github.com/stanfordnlp/contract-nli) | train | 4 | CC BY 4.0 (README / project site) | train rows only. Rows that share clauses with test contracts were dropped |
| [VAST](https://github.com/emilyallaway/zero-shot-stance) @ e7c4775 | data/VAST/vast_train.csv | 4 | unknown | test never used |
| [NLI4CT](https://github.com/ai-systems/Task-2-SemEval-2024) (SemEval-2024 Task 2) @ 7f32fa6 | train.json, dev.json | 4 | unknown | test never used |
| [ACOS](https://github.com/NUSTM/ACOS) @ 45d179a | data/*/*_quad_train.tsv | 4 | unknown | test never used |
| [RAGTruth](https://github.com/ParticleMedia/RAGTruth) @ c103204 | response.jsonl split = train, source_info.jsonl | 4 | MIT (repo LICENSE); contexts: unknown | test never used |
| [Decision Index kit](https://github.com/apolinario/decision-index) @ 19ad28e, Home-appliance generator (`decision_index/suite/build/home_appliance.py`) | new households from our own seed | 4 | MIT (repo) | code, not a dataset. 0 states are shared with the benchmark's dev + test rows |

That is **84 public sources**: 83 datasets and one generator.

### 2.3 Synthetic data

Everything that is not a public source above falls into four categories.

- **Code-generated and code-labelled items.** These are decision families whose answer a program computes: rule
  application, dates and business days, numeric extraction, lookup, ordering, long policies with amendments and decoys,
  temporal traces, multi-hop and buried-evidence long states, and tool-use worlds. The judge items plant an error in a
  human-written GSM8K solution by code, so the label holds by construction. The routing items take their label from the
  prompt's source dataset. This category also includes the rule-based renderings (JSON, described, wide and
  single-question forms) and the rule-twin items that ship with the stage-1 mixture's public code, and the stage-4
  Home-appliance households.
- **Synthetic items and short thoughts.** Part of the data is synthetic: items and short thoughts written or labelled
  by much larger frontier LLMs — far above 120B parameters where the size is published. An item is kept only when
  checks by code or independent answers agree and it passes our schema, shortcut and overlap gates; a thought is kept
  only if its final answer matches the gold label.
- **Our own model's errors, used for judge data.** None of the four stages' training files contain this category. The
  judge items of v1.0 are the code-planted errors described above, plus synthetic judge items.
- **KL replay.** Stages 2, 3 and 4 re-show earlier rows, and the target is the model's own distribution before that
  stage (§2.2, "(replay)"). This keeps earlier abilities while the new rows are learned. No label is added.

The training data is not released.

### 2.4 How Decision Index items were screened

- **Stages 1–3** were built before the full blocklist existed. For stage 3, we removed the 5 rows that held the strictly
  verified hits known at the time. Every remaining hit in the stage-1, stage-2 and stage-3 training files is listed in
  §1: 363 ids in [contamination/trained-on-di-ids.json](contamination/trained-on-di-ids.json). Please count them as
  trained on.
- **The blocklist.** Since 2026-09-27, every Decision Index 0.2 request we rebuilt (all but HLE) is on a permanent
  blocklist in our training repository (`trainer/data/blocklists`). Items are stored as fingerprints, not text. The data acceptance gate fails any
  training row that matches it (§3).
- **Stage 4** was checked row by row before training against the full blocklist, our sample rows, the calibration
  holdouts and our vertical test sets. It passed the gate with 0 rows that contain a Decision Index item.

## 3. How we scanned

- **Suite.** The Decision Index 0.2 suite was rebuilt with the maintainers' public kit (github.com/apolinario/decision-index
  @ 19ad28e) from its pinned public sources. It has 154,877 requests; HLE (501 rows, gated) is not included.
- **Match rule** (applied to every string of every training record):
  - Words are lower-case `[a-z0-9]+`.
  - Each item's state is taken after stripping the benchmark's fixed instruction line. A shared prompt prefix is
    stripped too, so training text without the DI prompt is still recognized.
  - A state of 4–11 words must appear verbatim.
  - A state of 12 words or more needs ≥ 80 % of its 8-word shingles in one record. Shingles that occur in more than 20
    states are dropped as boilerplate.
  - Benchmarks whose options are the item's content (MMLU / ARC / MMLU-Pro / GPQA / BBH / HellaSwag / WinoGrande …)
    also need ≥ 50 % of the option text to match.
  - States under 8 words are recorded as "weak" and not counted.
- **Coverage.** The scan covered every training file of every stage of this lineage, plus sibling and ancestor corpora.
  The per-file scan outputs (line counts, file sha256, hit counts) are summarized in the `scans` field of
  trained-on-di-ids.json.
- **Blocklist.** Since 2026-09-27 every Decision Index item is on a permanent blocklist, stored as fingerprints, not
  text. The data acceptance gate fails any training row that hits it.
- **Items the scan cannot see.** Items without free text cannot be checked this way: CLINC150 short intents, POP909 and
  cfcolor. Nothing in the pretraining data of the base model was scanned.

## 4. What we did not do

- We never used a Decision Index test split, the kit's suite rows or the sample rows for training, selection or
  calibration.
- We never tuned the prompt, the readout or the effort policy per benchmark. The knockout for more than 26 options is one
  rule for every question.
- We selected checkpoints on our own holdouts only. The Decision Index was read once per candidate checkpoint.
- One disclosure: the benchmarks that stage 4 covers were chosen from our Decision Index sample reads, as the
  benchmarks where we trailed Jev most. That choice used scores only. No items were used, and the stage has no
  checkpoint selection.
