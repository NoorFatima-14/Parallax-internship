# Hallucination Check Test

8 questions, real passages from the database, wrong answers written on purpose. No LLM calls.

| Case | Blocked | Warned | Clean | Expected | Result |
|---|---|---|---|---|---|
| true statement (control) | 0/8 | 0/8 | 8/8 | clean | pass |
| invented number | 8/8 | 0/8 | 0/8 | blocked | pass |
| invalid citation | 8/8 | 0/8 | 0/8 | blocked | pass |
| unrelated facts | 8/8 | 0/8 | 0/8 | blocked | pass |
| answer to another question | 7/8 | 0/8 | 1/8 | blocked | MISSED 1 |
| true + invented sentence | 0/8 | 8/8 | 0/8 | blocked or warned | flagged 8/8 (not counted) |
| generic on-topic text | 7/8 | 1/8 | 0/8 | - | information only |

Cases that failed their expectation:

- answer to another question (clean) | How is uncertainty estimated in deep neural networks? | reasons: []
  - Uncertainty proves informative at all three spatial scales, remains informative on an external multi-centre dataset under distribution shift, and supports OOD detection. [1]
