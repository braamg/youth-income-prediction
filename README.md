# Predicting youth employment in South Africa

**Predictive Insights Youth Income Prediction Challenge, Zindi, November 2026**

**Result:** ROC AUC 0.8743 on the private leaderboard (0.8648 public), 1st when the competition closed. The placing is provisional: Zindi's code review is still open.

Built with PriorLabs-TabPFN.

> **About this page.** It is a personal account of how I approached the problem. The competition data belongs to Zindi and the competition host, and the rules assign copyright in a top-three solution to Zindi. So this page has no data, no statistics or charts drawn from the data, and no code. The scores are those of my own models.

## The task

Given answers from a labour-market survey, estimate how likely each young person is to be in work some time later. Organisations that support young job-seekers can use such a ranking to decide where help is needed most.

Entries were scored by ROC AUC, so only the ordering of the predictions counted. Roughly a fifth of the test set fed the public leaderboard and the rest decided the final ranking. The rules limited entrants to the supplied data and open-source packages, allowed openly available pretrained models, and ruled out AutoML.

## Making the local score trustworthy

A small public leaderboard is a noisy guide, and the gains I was chasing were smaller than its noise. I put the first effort into a validation setup I could believe:

- **Reproduce a known score.** I rebuilt a baseline with a published result and checked which metric made my local number line up with it.
- **Look for drift.** A classifier trained to separate training rows from test rows did no better than chance.
- **Compare like with like.** Candidates were always evaluated on the same cross-validation splits as the model they were meant to replace.
- **Allow for overlapping folds.** Cross-validation folds reuse most of their training rows, which makes an ordinary paired t-test too optimistic. I used the Nadeau and Bengio correction.
- **Ask for more than one win.** The standard I set was that an idea should hold up on a second set of splits, in an ablation, and in a perturbation test. Not everything met all three; the main exception is described below.

## From boosted trees to a tabular foundation model

| Stage | Cross-validated AUC |
|---|---|
| Default gradient boosting | 0.849 |
| Strongest blend of boosted trees | 0.861 |
| TabPFN v2 without extra features | 0.873 |
| Final model (5 folds) | 0.878 |

I began with boosted trees. Extra features, tuning and blending moved them forward, but slowly. Nearest neighbours, a neural network, a hierarchical regression and a second foundation model were weaker and added nothing useful to a blend.

TabPFN v2 was the turning point. It is a transformer trained in advance on synthetic tables, and it makes predictions by reading labelled examples as context. On a dataset of a few thousand rows it outperformed everything I had tuned by hand, on a laptop without a GPU.

Three ideas took it further:

1. **Treat the survey wave as a label rather than a number,** because conditions change between waves in ways that don't follow a straight line.
2. **Borrow from labour economics.** How long a person has been in their present situation affects what happens next, and the pattern depends on what that situation is. I turned this into one additional input, estimated without using held-out outcomes.
3. **Average over variety.** The final predictions combine several published TabPFN checkpoints and random seeds.

Larger ensembles, probability recalibration, stacking and pseudo-labelling did not earn a place.

## Choosing two entries

Only the better of two chosen entries counts at the end, which makes the choice a hedging problem. I fixed the acceptance tests before looking at the results.

| Entry | Public AUC | Private AUC |
|---|---|---|
| Most accurate in cross-validation | 0.8648 | 0.8743 |
| Simpler and more conservative | 0.8627 | 0.8732 |

The stronger model passed its tests against the version immediately before it. Its advantage over the simpler model was a different matter: it disappeared in the perturbation test. My own rule, taken literally, would have excluded it. I made a written exception, documented the weakness in my submission, and used the simpler model as the second entry.

## Reproducibility

The final pipeline runs from a single command with pinned packages, seeds and model checkpoints. Starting from a clean copy in a new environment, it regenerated both entries exactly.

## What I learned

1. **Earn trust in the validation before trusting a model.** It settled every decision the leaderboard was too noisy to settle.
2. **Test robustness against a fixed reference.** Checking each change only against the last one lets small weaknesses add up unnoticed.
3. **Write the rules down first, and say so when you break one.**
4. **Use the second entry to cover the first one's main risk.**
5. **Tabular foundation models are a serious option for small datasets.**

## Licence

The text of this page is available under [CC BY 4.0](LICENSE). That covers this text only, and grants no rights in the competition data or the submitted solution. TabPFN v2 is the work of Prior Labs and is used under the Prior Labs License v1.1.
