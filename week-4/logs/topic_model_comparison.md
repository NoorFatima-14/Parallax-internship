# Topic Model Comparison (latest run)

- Documents: 6563. Short documents (under 100 words): 71. They are not used to fit BERTopic and are assigned to the nearest topic afterwards.
- NPMI coherence: -1 to 1, higher is better. Diversity: share of distinct words among all topics' top words.
- Core share: share of documents that belong to a cluster found by the model. The rest (outliers and short documents) are assigned to the nearest topic by their vectors and marked `topic_core = False`.

| Method | Topics | Core share | Median topic size | NPMI | Diversity | Seconds |
|---|---|---|---|---|---|---|
| bertopic | 31 | 0.634 | 139 | 0.2141 | 0.8258 | 107.0 |
| lda | 25 | 1.0 | 73 | 0.0833 | 0.908 | 59.7 |

## Topics (BERTopic, largest first)

| Id | Docs | Label | NPMI |
|---|---|---|---|
| 0 | 1543 | learning, data, bound, error | 0.105 |
| 1 | 661 | agent, agents, harness, execution | 0.201 |
| 2 | 387 | 3d, motion, scene, reconstruction | 0.246 |
| 3 | 344 | jev, answer, answers, accuracy | 0.181 |
| 4 | 277 | speech, audio, languages, language | 0.212 |
| 5 | 270 | action, robot, world, manipulation | 0.296 |
| 6 | 244 | visual, multimodal, tokens, vision-language | 0.206 |
| 7 | 221 | quantization, cache, compression, attention | 0.116 |
| 8 | 209 | reward, policy, rewards, reinforcement | 0.308 |
| 9 | 166 | video, temporal, frames, videos | 0.202 |
| 10 | 156 | lexical, sentence, language, multilingual | -0.105 |
| 11 | 153 | social, persona, llms, user | 0.22 |
| 12 | 153 | forecasting, series, forecast, time series | 0.368 |
| 13 | 145 | decoding, diffusion, tokens, diffusion language | 0.307 |
| 14 | 142 | visual, reasoning, image, evidence | 0.223 |
| 15 | 139 | forgetting, continual, learning, continual learning | 0.311 |
| 16 | 138 | teacher, student, distillation, on-policy | 0.496 |
| 17 | 135 | segmentation, clinical, image, tumor | 0.29 |
| 18 | 128 | reasoning, mathematical, answer, language models | 0.148 |
| 19 | 110 | clinical, medical, patient, evidence | 0.181 |
| 20 | 106 | retrieval, query, knowledge, schema | 0.163 |
| 21 | 99 | optimization, bayesian optimization, multi-objective, bayesian | 0.159 |
| 22 | 91 | memory, memories, long-term, retrieval | 0.262 |
| 23 | 86 | molecular, protein, sequence, chemical | 0.147 |
| 24 | 80 | attention, linear attention, linear, softmax | 0.211 |
| 25 | 75 | crop, hyperspectral, satellite, imagery | -0.181 |
| 26 | 72 | anomaly, anomaly detection, detection, anomalies | 0.406 |
| 27 | 69 | editing, image, style, edit | 0.234 |
| 28 | 68 | federated, federated learning, clients, client | 0.352 |
| 29 | 54 | driving, autonomous driving, autonomous, vehicle | 0.352 |
| 30 | 42 | quantum, classical, circuit, kernel | 0.021 |
