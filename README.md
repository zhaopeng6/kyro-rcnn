# Chromosome segmentation

This repository begins the reproducible Mask R-CNN study described in `planning.md`.

Phase 1 currently provides verified adapters for the local datasets:

- AutoKary: LabelMe polygon instances, rasterized into real masks.
- Lowq: Pascal VOC XML instances, used as box-only detection supervision. The adapter never fabricates masks from rectangles.

Experiment B uses `lowq54816/single_chromosomes_object` (not the 24-class Lowq variant). During dataset construction, an XML annotation is skipped when its referenced JPEG is absent; the resulting dataset exposes the skipped paths as `skipped_missing_images` instead of failing during a training epoch.

Phase 2 adds Experiment A: a ResNet-50-FPN Mask R-CNN with genuine polygon masks, aspect-ratio-preserving torchvision resizing, box/mask-safe horizontal flip augmentation, optimizer/checkpoint persistence, and transparent greedy mask-IoU validation metrics. The metrics are intentionally labelled `*_proxy`; COCO AP integration is not yet implemented.

Experiment B uses isolated `MixedSupervisionRoIHeads`: polygon batches get all detection and mask losses; Lowq box-only batches receive RPN/classification/box-regression losses and never execute the mask-loss branch. `scripts/train_mixed.py` samples homogeneous batches and records actual source-step counts.

`training.batch_size` and `training.validation_batch_size` are configuration values, not code constants. Each training step prints a JSON metric record and writes it to `outputs/<experiment>/metrics.jsonl`; with the provided configs a held-out polygon validation-loss batch is also measured every step. Set `validation_interval` higher to reduce validation cost, and use `tail -f outputs/experiment_a/metrics.jsonl` in another terminal to watch it. `optimizer` accepts `sgd` or `adamw`.

## TensorBoard

Install the optional UI dependency, then start TensorBoard in a second terminal:

```bash
python -m pip install tensorboard
tensorboard --logdir outputs --port 6006
```

Open `http://localhost:6006`. TensorBoard is enabled by default in both experiment configs and writes `train/total_loss`, every individual training loss, `train/gradient_norm`, `validation/loss`, and (for Experiment B) source-specific training-step counters. Change `training.tensorboard.enabled` to `false` to run without it.

## Checkpointing and resume

Training atomically updates `outputs/<experiment>/last.pt` every `training.checkpoint_interval` optimizer steps (default: 10), and again on the final requested step. It includes the model, optimizer, completed global step, and RNG state. Continue an interrupted run by keeping the same config and increasing the requested total steps:

```bash
python scripts/train.py --config configs/experiment_a.yaml --steps 1000 \
  --resume outputs/experiment_a/last.pt
```

Use `--checkpoint-interval 1` if losing up to the default ten recent steps is unacceptable; frequent checkpoints consume extra I/O and storage. Experiment B additionally restores its recorded polygon/box optimizer-step counts.

Both baseline configs fine-tune torchvision's default COCO-pretrained Mask R-CNN weights (`model.pretrained: true`). The first run may download those weights into the local torch cache. The class-specific prediction heads are replaced for the project model, while transferable backbone/FPN/RPN and mask features are retained. Fine-tuning uses AdamW at `0.0001`; do not resume a checkpoint created with the old randomly initialized configuration.

The datasets are grayscale, but the loader intentionally replicates each grayscale image into three identical channels. This is the compatible way to use the RGB-pretrained Mask R-CNN backbone; it does not invent colour information. Pretrained runs retain torchvision's three-ratio RPN anchor layout so its RPN weights remain loadable.

## Experiment C: BoxInst-style weak mask supervision

`configs/experiment_c.yaml` enables an isolated weak loss for Lowq box batches: a mask-projection constraint and local grayscale-affinity loss. It never converts a rectangle to a foreground target mask. This ROI-head adaptation differs from the official BoxInst/CondInst implementation, so report it as “BoxInst-style” rather than official BoxInst. Train it with `python scripts/train.py --config configs/experiment_c.yaml` and compare it against the same Experiment A/B evaluation settings.

Review Experiment-D pseudo masks before using them for training:

```bash
python scripts/view_pseudo_masks.py --manifest outputs/experiment_d_pseudomasks/manifest.json \
  --output-dir outputs/experiment_d_pseudomasks/previews
```

Lowq/VOC boxes are converted from `xyxy` to micro-sam's `yxyx` prompt convention inside `generate_pseudo_masks.py`; preserve this conversion when changing prompt code.

## Phase 4: scale robustness

Run the controlled held-out scale benchmark after training:

```bash
python scripts/scale_benchmark.py --config configs/experiment_a.yaml \
  --checkpoint outputs/experiment_a_pretrained/last.pt
```

It uniformly resizes images, masks, and boxes at every configured scale without padding/cropping; saves JSON/CSV, optional metric plots, and GT/prediction failure panels in `outputs/<experiment>/scale_benchmark/`. By default it randomly samples two validation images separately for each factor, recording the selected indices and seed in JSON; set `scale_test.max_images: null` for the full held-out set. Metrics distinguish bounding-box overlap from actual polygon-mask overlap and label split/merge results as heuristic candidates. Compare completed reports fairly with:

```bash
python scripts/compare_experiments.py \
  --experiment-a outputs/experiment_a_pretrained/scale_benchmark/scale_metrics.json \
  --experiment-b outputs/experiment_b_pretrained/scale_benchmark/scale_metrics.json
```

Training uses configured uniform content-scale augmentation, horizontal/vertical flips, fixed-canvas translation, and Gaussian noise. Geometric transforms update instance masks first, then derive tight boxes/areas from the transformed masks. The scale benchmark defaults to `internal_resize_policy: none`, which disables torchvision's normally hidden min/max resize so the requested test scale reaches FPN unchanged. This is more memory intensive than ordinary inference; use `--max-images` for a short MPS smoke run before evaluating all factors.

Run the inspection report (it does not alter source data):

```bash
pip install -r requirements.txt
python scripts/inspect_datasets.py --config configs/experiment_a.yaml
pytest -q
python scripts/train.py --config configs/experiment_a.yaml --steps 10
python scripts/evaluate.py --config configs/experiment_a.yaml --checkpoint outputs/experiment_a/last.pt
python scripts/train.py --config configs/experiment_b.yaml --steps 100
```

Outputs are written to `outputs/inspection/`. The initial configs use class-agnostic foreground detection (background `0`, chromosome `1`) while preserving original labels as metadata. Lowq's `A1`, etc. labels are intentionally not mapped to karyotype classes, because they represent a different convention and must not be guessed.

Training, mixed-loss routing, and evaluation will be added only after dataset report and geometry validation are reviewed.

The generated report found 548 AutoKary training images (23,478 polygon instances; five invalid shapes) and 5,000 Lowq 24-class XML files (229,852 box instances). These are inspection facts, not training or evaluation results.

## MVP: segment → type → karyogram

`configs/mvp_karyogram.yaml` pauses Experiment C/D work and adds a practical
three-stage workflow around Experiment A:

1. Experiment-A Mask R-CNN detects each chromosome and outputs its mask.
2. A separate 24-way ResNet-18 classifies each **mask-isolated** crop as
   `1`–`22`, `X`, or `Y`. AutoKary labels `23` and `24` map to `X` and `Y`;
   invalid labels such as `-1` are excluded from classifier training.
3. All accepted crops are rendered in conventional chromosome order. The MVP
   does not force two items per class, so potential extra/missing detections
   remain visible for review.

Train the type classifier independently of Experiment A:

```bash
python scripts/train_chromosome_classifier.py --config configs/mvp_karyogram.yaml
```

The default MVP config uses PEFT LoRA on the ImageNet-pretrained ViT-B/16.
The original ViT remains frozen; rank-16 LoRA adapters on the attention
query/value projections and the new 24-class head are trained. PEFT's library
default rank is 8; rank 16 is the initial project setting. This requires
`transformers` and `peft` (included in `requirements.txt`) and uses ViT's
224×224, mean/std 0.5 preprocessing. The saved checkpoint includes only the
LoRA adapter/head weights plus optimizer state; inference reloads the matching
pretrained base ViT automatically. For full fine-tuning instead, set
`finetuning_mode: full`; then `freeze_backbone_epochs` controls the optional
head-only warm-up. To use the faster baseline, set
`classifier_training.architecture: resnet18` and `finetuning_mode: full`.

Transformers has used both `query`/`value` and `q_proj`/`v_proj` names for ViT
attention projections across releases. The LoRA builder detects either naming
scheme and prints the selected targets before training.

When LoRA initializes, Transformers may report pretrained `pooler` weights as
unused and the new 24-class `classifier` weights as missing/newly initialized.
That is expected: the source ViT checkpoint has a different classification
head, which this project deliberately replaces. The run should continue after
that message; an exception is not expected.

The classifier writes its configured `output_dir` checkpoint (the ViT default
is `outputs/chromosome_classifier_vit_b16_lora/last.pt`) after every
completed epoch, including model and AdamW optimizer state. Resume it by
setting a larger total epoch count and passing that checkpoint:

```bash
python scripts/train_chromosome_classifier.py --config configs/mvp_karyogram.yaml \
  --epochs 40 --resume outputs/chromosome_classifier_vit_b16_lora/last.pt
```

For example, a run stopped after epoch 12 continues at epoch 13 and finishes
at epoch 40. Older classifier checkpoints created before resume support lack
optimizer state and must be used only for inference or replaced by a new run.
For a Colab run, the same checkpoint can be supplied in the config as
`classifier_training.resume_checkpoint`; command-line `--resume` takes
precedence. Mount Drive and point either setting to the Drive checkpoint. The
epoch limit is always a total target, not additional epochs.

Classifier TensorBoard logs are written to
`outputs/chromosome_classifier_vit_b16_lora/tensorboard` by default. They include training
and validation loss/accuracy, learning rate, and validation accuracy for each
of the 24 chromosome types. Start the UI while training with:

```bash
tensorboard --logdir outputs/chromosome_classifier_vit_b16_lora/tensorboard --port 6006
```

Open `http://localhost:6006`. Install the optional dependency first if needed:
`python -m pip install tensorboard`. Set
`classifier_training.tensorboard.enabled: false` to disable this logging.

Then run a metaphase image through the full pipeline:

```bash
python scripts/build_karyogram.py --config configs/mvp_karyogram.yaml \
  --image 'dataset/Zhaopeng_project/original pics/chromosome-10s-15.png'
```

The output directory `outputs/karyogram_mvp/<image-name>/` contains
`karyogram.png`, one mask-isolated crop per accepted detection in `segments/`,
and `predictions.json` with segmentation/type confidence. This is an MVP
classification workflow, not a clinical karyotyping result; inspect the
segments and confidence values before interpreting counts or abnormalities.

### Kaggle / external dataset location

All experiment and MVP configs support one dataset base directory:

```yaml
paths:
  data_root: .
```

Every *relative* entry under `datasets` is joined to `paths.data_root`; an
absolute dataset entry is used unchanged. For example, if the Kaggle input
folder contains the repository's `dataset/` directory, change only this:

```yaml
paths:
  data_root: /kaggle/input/my-chromosome-data
```

Then the existing `dataset/Autokary2022_1600x1600/...` entries resolve to
`/kaggle/input/my-chromosome-data/dataset/Autokary2022_1600x1600/...`.
If your Kaggle input itself contains `Autokary2022_1600x1600/` rather than a
parent `dataset/` directory, set `data_root` to that input directory and remove
the leading `dataset/` from the affected `datasets` paths. Keep outputs outside
the read-only Kaggle input, for example under `/kaggle/working/`.

Before a Kaggle classifier run, validate the mounted paths without loading a
model or GPU:

```bash
python scripts/train_chromosome_classifier.py \
  --config configs/mvp_karyogram.yaml --check-data
```

It prints the resolved train/validation directories and labelled crop counts;
the expected local counts are 23,465 train and 2,873 validation crops.
