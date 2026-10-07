# EXP-XXX — Short experiment title

```yaml
status: planned | running | completed | failed | abandoned | superseded
created: YYYY-MM-DD
started: YYYY-MM-DD or null
completed: YYYY-MM-DD or null
owner: codex/user/other
hypothesis: one falsifiable sentence
```

## Question

State exactly what this experiment distinguishes from existing explanations.

## Inputs

```yaml
dataset_manifest:
pair_count:
speaker_count:
classes:
source_audio_paths:
preprocessing:
exclusions:
```

## Models and environment

```yaml
model_or_encoder:
checkpoint:
code_commit:
local_source_modifications:
python_environment:
device:
```

## Protocol

```yaml
feature_definition:
pooling:
split_protocol:
train_validation_test_counts:
seed:
probe_or_detector:
hyperparameters:
```

## Exact command

```bash
# Paste the actual executed command, not a reconstructed approximation.
```

## Outputs

```yaml
output_root:
console_log:
manifest:
config:
report:
other_artifacts:
```

## Validation

- Expected versus actual output count:
- Decode/finite/clipping checks:
- Embedding shape/dtype:
- Split leakage checks:
- Warnings/failures:

## Results

Record primary and secondary metrics. Include class order for every confusion matrix.

## Interpretation

Separate:

```text
verified observation:
scientific interpretation:
alternative explanations:
unsupported claims:
```

## Relationship to previous work

```yaml
depends_on:
supersedes:
superseded_by:
```

## Follow-up

State the next decision or experiment, not merely "more analysis".
