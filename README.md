# JusDef — v1 workshop code (archived)

> **This repository is superseded. Active work is at
> [`oziofficial5/jusdefv2`](https://github.com/oziofficial5/jusdefv2).**
>
> Everything released with the doctoral thesis — the corpus, the trained
> detector, the full experimental code, and the per-seed logs backing every
> reported number — lives there. This repository is kept for provenance only.

---

## What this is

The original implementation of **JusDef v1**, the architecture reported in the
ANNPR 2026 workshop paper:

> Khaliq, A. A., Montanelli, S., Dalianis, H., Naqvi, N. (2026). *JusDef:
> Defeasible Message Passing for Exception-Aware Legal Document
> Classification.* Artificial Neural Networks in Pattern Recognition, 12th IAPR
> TC3 Workshop, ANNPR 2026. Lecture Notes in Artificial Intelligence, Springer
> Nature Switzerland.
> [doi:10.1007/978-3-032-39028-8_6](https://doi.org/10.1007/978-3-032-39028-8_6)

JusDef v1 is a heterogeneous graph neural network over legal documents. Sections,
concepts, authorities and labels are typed nodes; each operator occurrence is a
typed edge carrying one of four labels (AFF, NEG, EXC, OVR). A **Defeasible
Message Passing** layer lets messages defeat one another under a priority order
drawn from defeasible reasoning theory, using a hard binary defeat gate trained
through a straight-through estimator.

```
src/model/
  jusdef.py            the v1 model
  dmp_layer.py         Defeasible Message Passing, hard defeat gate + STE
  authority_scorer.py  scalar authority priority on each mention edge
  baselines.py         R-GCN baseline
```

---

## Why it is archived, and what happened to the result

The workshop paper reports a gain for the hard-defeat architecture on an
exception-dependent label subset. **That evaluation tuned its decision threshold
on test labels.**

An audit of this code found that the published threshold-tuning routine, in some
configurations, optimised the threshold on the test split rather than the
validation split. Under the corrected protocol the reported advantage does not
survive, and the corrected v2 architecture falls nine macro-F1 points behind the
R-GCN baseline it was built to extend.

The doctoral thesis reports that correction in full, diagnoses the shortfall
through a hypothesis-falsification chain over five candidate explanations, and
redesigns the aggregation layer in response. None of that work is in this
repository.

This is recorded here rather than quietly dropped, because anyone finding this
code from the ANNPR paper should know the status of the number it reports.

---

## Where to go instead

| You want | Go to |
|---|---|
| The corpus, guidelines and IAA files | [`jusdefv2/data/annotations/`](https://github.com/oziofficial5/jusdefv2/tree/thesis-main/data/annotations) |
| The trained operator detector | [`jusdefv2/outputs/checkpoints/`](https://github.com/oziofficial5/jusdefv2) |
| The corrected architectures (JusDef-SP, JusDef-RR) | [`jusdefv2/src/model/`](https://github.com/oziofficial5/jusdefv2) |
| The three-arm permutation control | [`jusdefv2`](https://github.com/oziofficial5/jusdefv2#the-permutation-control) |
| Per-seed logs for every reported number | [`jusdefv2/outputs/logs/`](https://github.com/oziofficial5/jusdefv2) |

---

## Contact

Awais Abdul Khaliq, Dipartimento di Informatica "Giovanni Degli Antoni",
Università degli Studi di Milano.
ORCID [0000-0002-3439-6256](https://orcid.org/0000-0002-3439-6256)
