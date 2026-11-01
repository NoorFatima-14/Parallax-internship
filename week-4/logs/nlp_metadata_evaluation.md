# NLP Metadata Evaluation

All numbers come from labels that a person created (see labels/GUIDELINES.md). Intervals are 95% Wilson intervals: with small samples they are wide, and that is the honest picture.

## Topic quality (your ratings)

31 topics rated (the largest first).

| Rating | Topics | Share of rated topics (95% interval) | Documents in these topics |
|---|---|---|---|
| good | 27 | 87% (71% to 95%) | 4410 (67%) |
| mixed | 4 | 13% (5% to 29%) | 2153 (33%) |
| poor | 0 | 0% (0% to 11%) | 0 (0%) |

Does the automatic score (NPMI) agree with your ratings?

| Rating | Mean NPMI |
|---|---|
| good | 0.232 |
| mixed | 0.090 |
| poor | - |

Names you gave to topics: 0: statistical learning and optimization theory; 1: LLM agents and agent safety; 2: 3D vision and scene reconstruction; 3: evaluation, judging and decision quality; 4: speech, audio and multilingual communication; 5: robotics, action planning and manipulation; 6: multimodal vision-language models; 7: model quantization and memory-efficient inference; 8: reinforcement learning and reward optimization; 9: video understanding and temporal modeling; 10: language, semantics and multilingual NLP; 11: LLM personas and social behavior; 12: time-series forecasting; 13: diffusion language models and decoding; 14: visual reasoning in vision-language models; 15: continual learning and adaptation; 16: LLM knowledge and policy distillation; 17: medical image analysis and segmentation; 18: mathematical reasoning in language models; 19: clinical AI and medical decision support; 20: retrieval-augmented generation and data querying; 21: Bayesian and multi-objective optimization; 22: long-term memory and retrieval for agents; 23: molecular, protein and biological modeling; 24: attention mechanisms and efficient sequence modeling; 25: remote sensing and agricultural vision; 26: anomaly detection; 27: image editing and diffusion models; 28: federated learning; 29: autonomous driving

## Topic assignment accuracy (your judgement)

| Group | Documents | Correct (y) | Correct or partly (y or p) |
|---|---|---|---|
| all | 60 | 58% (46% to 70%) | 83% (72% to 91%) |
| random | 45 | 56% (41% to 69%) | 84% (71% to 92%) |
| short | 15 | 67% (42% to 85%) | 80% (55% to 93%) |

'short' documents are the shortest 10% of the corpus. They were not used to fit the topic model.

## Dataset-mention extraction (your labels)

50 documents labeled, 33 dataset mentions in total, 24 documents mention at least one dataset.

**all labeled documents** (50 documents)

| System | Precision | Recall | F1 | Correct | Wrong | Missed | Precision interval | Recall interval |
|---|---|---|---|---|---|---|---|---|
| spacy_baseline | 0.09 | 0.48 | 0.16 | 16 | 156 | 17 | 6% to 15% | 33% to 65% |
| seed_list | 0.57 | 0.12 | 0.20 | 4 | 3 | 29 | 25% to 84% | 5% to 27% |
| seed_plus_patterns | 0.56 | 0.45 | 0.50 | 15 | 12 | 18 | 37% to 72% | 30% to 62% |

**random documents only** (30 documents)

| System | Precision | Recall | F1 | Correct | Wrong | Missed | Precision interval | Recall interval |
|---|---|---|---|---|---|---|---|---|
| spacy_baseline | 0.03 | 0.27 | 0.06 | 3 | 87 | 8 | 1% to 9% | 10% to 57% |
| seed_list | 0.00 | 0.00 | 0.00 | 0 | 1 | 11 | 0% to 79% | 0% to 26% |
| seed_plus_patterns | 0.33 | 0.09 | 0.14 | 1 | 2 | 10 | 6% to 79% | 2% to 38% |

## Mistakes of the best system (seed_plus_patterns)

Dataset names are shown in normalized form (lower case, no spaces or hyphens).

Wrong (the system found it, you did not):

- soc  (http://arxiv.org/abs/2608.10777v1)
- englishlanguage  (http://arxiv.org/abs/2609.37713v2)
- dflash  (http://arxiv.org/abs/2610.01439v1)
- dspark  (http://arxiv.org/abs/2610.01439v1)
- mimiciv  (http://arxiv.org/abs/2609.40108v3)
- spatialai  (http://arxiv.org/abs/2609.32813v1)
- imagenet  (http://arxiv.org/abs/2609.34732v1)
- csp  (http://arxiv.org/abs/2610.01315v1)
- gpqa  (http://arxiv.org/abs/2610.01042v1)
- trajectoryrri  (http://arxiv.org/abs/2610.00571v1)
- istd  (http://arxiv.org/abs/2610.05918v1)
- radiodinos16  (http://arxiv.org/abs/2609.33769v1)

Missed (you found it, the system did not):

- pathovernier  (http://arxiv.org/abs/2609.34277v1)
- chime6  (http://arxiv.org/abs/2609.39344v1)
- chime8notsofar  (http://arxiv.org/abs/2609.39344v1)
- abtbuy  (http://arxiv.org/abs/2609.37713v2)
- billigerdeproducts  (http://arxiv.org/abs/2609.37713v2)
- wdcproducts  (http://arxiv.org/abs/2609.37713v2)
- ddxplus  (http://arxiv.org/abs/2610.04542v1)
- traveluav  (http://arxiv.org/abs/2609.33581v1)
- usaiquant  (http://arxiv.org/abs/2609.32813v1)
- crosslingualunlearningtensor  (http://arxiv.org/abs/2609.40286v1)
- cod  (http://arxiv.org/abs/2610.01315v1)
- mpds  (http://arxiv.org/abs/2610.01315v1)
- genai4idnbenchmark30  (http://arxiv.org/abs/2610.05633v1)
- libero  (http://arxiv.org/abs/2610.06280v1)
- liberoplus  (http://arxiv.org/abs/2610.06280v1)
- dl3dv  (http://arxiv.org/abs/2610.03192v1)
- mipnerf360  (http://arxiv.org/abs/2610.03192v1)
- alfworld  (http://arxiv.org/abs/2610.00372v1)

