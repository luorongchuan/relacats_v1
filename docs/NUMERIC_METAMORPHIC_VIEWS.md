# Numeric Metamorphic Views for GSM8K / SVAMP

This module is the conservative first-stage implementation for extending
RelaCaTS relation witnesses to scalar math word problems.

## Why these transforms

The relation witness should perturb shortcut-sensitive input features while
keeping the mathematical task valid.  Therefore this implementation does not
change the final answer by an artificial `+1` or `x2` rule.  Instead every
accepted view is an invariant relation with

```text
phi_g(T) = T.
```

The initial library contains:

1. `identity`: unchanged problem.
2. `layout_wrapper`: embeds the exact original problem inside a different
   structural wrapper.
3. `number_representation`: replaces one safe standalone integer with an
   equivalent English number phrase, e.g. `48 -> forty-eight`.
4. `equivalent_quantity`: replaces one safe standalone integer by a trivial
   arithmetic identity, e.g. `48 -> (47 + 1)`.

The implementation is deliberately conservative.  Decimals, currency,
percentages, ratios, and other ambiguous numeric forms are not modified by the
automatic numeric edits.  If a relation cannot be certified, it is skipped.

## Step 1: run unit tests

```bash
cd /home/luorongchuan/workspace_135/RelaCaTS

python -m pytest -q \
  relacats_v1/tests/test_numeric_metamorphic_views.py
```

## Step 2: CPU-only coverage audit

Do this before any GPU generation:

```bash
cd /home/luorongchuan/workspace_135/RelaCaTS

bash relacats_v1/scripts/13_audit_numeric_metamorphic_views.sh
```

Outputs:

```text
relacats_v1/outputs/numeric_metamorphic_candidates/
  summary.json
  gsm8k/
    candidates.jsonl
    preview.jsonl
    summary.json
  svamp/
    candidates.jsonl
    preview.jsonl
    summary.json
```

The most important fields are:

```text
four_view_coverage
view_count_histogram
certification_failures
```

`certification_failures` should always be zero.  A question may have fewer
than four views if no safe integer transform is available; the generator does
not force an unsafe edit.

## Step 3: inspect previews manually

Inspect at least 20 examples from each dataset:

```bash
sed -n '1,5p' \
  relacats_v1/outputs/numeric_metamorphic_candidates/gsm8k/preview.jsonl

sed -n '1,5p' \
  relacats_v1/outputs/numeric_metamorphic_candidates/svamp/preview.jsonl
```

Check that:

- the original mathematical meaning is unchanged;
- the expected numeric answer is unchanged;
- number-word edits are grammatical enough for the task;
- equivalent expressions do not interact badly with surrounding punctuation;
- no units, percentages, decimals, dates, or IDs are accidentally changed.

## Step 4: only then integrate with teacher generation

The current commit intentionally separates *relation construction/audit* from
expensive teacher generation.  Once coverage and manual precision are
satisfactory, the existing `generate_relational_data.py` can be changed from
numeric `1 x 32 identity` to certified numeric relation sampling.  This avoids
spending GPU time before the transformation library is validated.

For the formal experiment, keep the original 32-identity pools as the CaTS
baseline.  The new numeric relation pool should be generated in a new output
root so the baseline is never overwritten.
