# Codex Task: Build a Multi-Scale Chromosome Instance Segmentation Pipeline with Mixed Polygon and Bounding-Box Supervision

## 1. Project Background

I am developing an automated human chromosome segmentation and classification system for metaphase microscopy images.

The ultimate goal is to:

1. Detect individual chromosomes, including touching and overlapping chromosomes.
2. Generate accurate instance segmentation masks.
3. Extract chromosome crops for downstream DINOv2 classification into chromosome classes 1–22, X, and Y.
4. Eventually reconstruct a karyotype and detect chromosome abnormalities.

The immediate task is **instance segmentation**, not downstream classification.

I have two datasets under ./dataset folder:

**Dataset A: AutoKary2022**

- Relatively small dataset.
- Contains microscopy images with LabelMe JSON polygon annotations.
- Polygon annotations may represent individual chromosomes, including some overlapping chromosomes.
- Image dimensions and aspect ratios vary.
- Physical microscope scale and magnification metadata are unavailable or inconsistent.

**Dataset B: Lowq54816 dataset**

- Significantly more microscopy images.
- Contains chromosome annotations using rectangular bounding boxes.
- Many chromosome boxes overlap and may include pixels belonging to neighboring chromosomes.
- Image dimensions, aspect ratios, and apparent chromosome sizes vary.
- No reliable physical scale metadata.

Both datasets contain healthy human chromosome images, with chromosome class labels available where provided by the source annotations.

Important constraints:

- Do not assume chromosomes are straight.
- Do not estimate magnification using chromosome length.
- Do not estimate chromosome area from rectangular bounding-box area.
- Do not assume a rectangle gives an accurate pixel mask.
- Do not assume all images have the same dimensions or aspect ratio.
- Do not stretch images to fixed square dimensions.
- Avoid relying on absolute microscope magnification.
- Preserve individual chromosome identities where annotations permit.
- Handle partially overlapping and touching chromosomes.
- Do not use downstream chromosome classification predictions as prerequisites for segmentation.

## 2. Overall Objective

Implement a reproducible PyTorch training and evaluation pipeline based on:

**Mask R-CNN with ResNet-50-FPN and mixed polygon/box supervision.**

The initial implementation must support two experiments:

### Experiment A — Polygon-only baseline

Train Mask R-CNN using Dataset A.

Use full detection and instance-mask supervision.

### Experiment B — Mixed-supervision baseline

Train Mask R-CNN using both datasets.

- Dataset A provides detection and mask supervision.
- Dataset B provides detection supervision only.
- The mask loss must be calculated only for samples with valid polygon masks.
- Bounding-box-only samples must never receive fabricated ground-truth masks.

The model should learn chromosome detection from both datasets while learning instance-mask prediction from Dataset A.

Design the code so additional experiments can be added without restructuring the entire project.

### Future experiments

Implement as optional later stages:

- Experiment C: BoxInst-style box-supervised instance-mask loss.
- Experiment D: μSAM-assisted mask pseudo-label generation.

Do not implement these advanced stages before A and B are working, unless their implementation can be cleanly isolated without delaying or destabilizing the baseline.

## 3. First Step: Inspect the Existing Repository and Dataset Formats

Before writing the training pipeline:

1. Inspect the current repository structure.
2. Find existing dataset loaders, polygon parsing utilities, trained models, preprocessing scripts, and relevant configuration files.
3. Inspect representative annotation files from both datasets.
4. Determine actual image paths and annotation directory structures.
5. Identify chromosome label naming conventions.
6. Determine whether polygon annotations represent separate instances or a combined foreground mask.
7. Check how overlapping chromosomes are annotated.
8. Identify whether Dataset B uses LabelMe rectangles, Pascal VOC XML, COCO JSON, or another format.

Reuse correct existing utilities where practical.

Do not invent filenames, directory structures, annotation formats, or label mappings.

If data is not available locally, implement configurable adapters with clear documented assumptions and create minimal synthetic fixtures for testing.

Before building the model, produce a short dataset inspection report summarizing:

- Number of images in each dataset.
- Number of annotations.
- Distribution of image widths and heights.
- Image aspect-ratio distribution.
- Chromosome bounding-box dimensions.
- Class-label frequencies.
- Number of images with overlapping boxes.
- Invalid or missing annotations.
- Potential patient-level identifiers.

Save this report as machine-readable JSON and a human-readable Markdown summary.

## 4. Recommended Project Structure

Adapt this structure to the existing repository rather than overwriting existing work.

```text
chromosome_segmentation/
├── configs/
│   ├── experiment_a.yaml
│   ├── experiment_b.yaml
│   ├── experiment_c.yaml
│   └── experiment_d.yaml
├── src/
│   ├── datasets/
│   │   ├── autokary.py
│   │   ├── box_dataset.py
│   │   ├── annotations.py
│   │   ├── combined_dataset.py
│   │   ├── transforms.py
│   │   └── collate.py
│   ├── models/
│   │   ├── mask_rcnn.py
│   │   └── loss_router.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── sampler.py
│   │   └── checkpoints.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── scale_robustness.py
│   │   └── visualization.py
│   └── utils/
│       ├── config.py
│       └── reproducibility.py
├── scripts/
│   ├── inspect_datasets.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── compare_experiments.py
├── tests/
├── outputs/
├── requirements.txt
└── README.md
```

Use relative or configurable paths. Never hardcode machine-specific dataset locations.

## 5. Annotation Parsing

### Dataset A: Polygon annotations

Convert each individual chromosome polygon into:

- Binary instance mask.
- Tight bounding box calculated from the mask.
- Numeric class label.
- Image identifier.
- Instance identifier.

For each image, produce a torchvision-style target dictionary:

```python
{
    "boxes": FloatTensor[N, 4],
    "labels": Int64Tensor[N],
    "masks": UInt8Tensor[N, H, W],
    "image_id": ...,
    "area": FloatTensor[N],
    "iscrowd": Int64Tensor[N]
}
```

Use `xyxy` box coordinates.

Verify that polygons are rasterized correctly using the original image coordinate system.

Inspect cases involving chromosome overlap. If polygons specify distinct target objects, preserve their separate instance masks, even when their geometric supports overlap.

If annotations do not define which chromosome owns ambiguous crossing pixels, document this limitation rather than inventing instance identities.

Handle invalid polygons, out-of-range coordinates, empty masks, and partially truncated chromosomes.

### Dataset B: Rectangular bounding boxes

Parse each box into:

- Bounding-box coordinates.
- Numeric class label if available.
- Image identifier.
- Instance identifier.

Do not convert rectangles to ground-truth segmentation masks.

Each sample must include an explicit supervision-type indicator, such as:

```python
supervision_type = "box"
```

Dataset A uses:

```python
supervision_type = "polygon"
```

A box-only sample must still provide the required detection target fields, including boxes, labels, image ID, areas, and crowd flags.

Implement a consistent chromosome label mapping across datasets. Handle differing label conventions explicitly.

For segmentation, support a configurable class-agnostic model as the default:

- Background: class 0.
- Chromosome: class 1.

Preserve original 1–22/X/Y class metadata for downstream use.

A separate configurable class-aware detection experiment may be added later, but do not let chromosome classification complexity interfere with the initial segmentation benchmark.

## 6. Preprocessing and Geometric Transformations

Images have different dimensions and aspect ratios.

Do not warp every image to a fixed square.

Use aspect-ratio-preserving resizing:

\[
s=\min\left(\frac{L_{\max}}{\max(H,W)},\frac{L_{\min}}{\min(H,W)}\right)
\]

This is one possible resize policy; make the implementation configurable, and verify that it meets the intended minimum- and maximum-side constraints.

If the constraints cannot both be satisfied, prioritize avoiding distortion and document the resulting size.

Image dimensions should remain compatible with the model.

Where padding is required, preserve original image geometry and record the transformation parameters.

Maintain an invertible transform record for:

- Uniform resizing.
- Padding.
- Cropping.
- Horizontal and vertical flips.
- Rotation.

All applicable transformations must update boxes and masks consistently.

Use suitable interpolation:

- Bilinear or bicubic for microscopy images.
- Nearest-neighbor for instance masks.

Do not apply inappropriate color augmentations that destroy chromosome banding information.

Support grayscale microscopy images by converting to three-channel input where necessary, while retaining grayscale-friendly normalization.

Do not independently normalize each chromosome crop during segmentation training.

## 7. Scale Augmentation

Scale robustness is a primary research objective.

Implement random uniform scaling as a configurable augmentation.

Initial experimental range:

```yaml
augmentation:
  scale:
    enabled: true
    min: 0.65
    max: 1.50
```

Treat these numbers as hypotheses, not fixed optimal settings.

Sample scales independently per training image.

Use scaling, cropping, or padding policies that produce valid detection targets.

Avoid filtering out small chromosomes unintentionally after downscaling.

If crops truncate chromosomes, either handle them consistently with the annotation policy or exclude invalid truncated instances.

Also support:

- Rotation.
- Horizontal/vertical flipping.
- Moderate brightness and contrast changes.
- Optional Gaussian noise.
- Optional blur augmentation.

Augmentation should reflect realistic microscopy variability.

Provide separate augmentation presets:

- `baseline`
- `scale_strong`
- `scale_moderate`

Ensure experiments A and B use identical augmentation configurations when directly comparing the effect of additional box data.

Do not introduce stronger augmentation into Experiment B alone.

## 8. Model Architecture

Use torchvision Mask R-CNN with a ResNet-50-FPN backbone.

Start from pretrained weights when compatible with the environment and experiment requirements.

The FPN provides multi-resolution features for detecting chromosomes with different apparent pixel sizes.

Configure:

- Number of detection classes.
- Region proposal network anchor sizes.
- Anchor aspect ratios.
- ROI pooling settings.
- Mask prediction head.
- Detection score threshold.
- Non-maximum suppression threshold.
- Maximum detections per image.

Pay special attention to narrow, elongated chromosome shapes and densely overlapping boxes.

Default anchors may not be ideal.

Inspect annotation statistics and provide an option to tune anchors using observed box distributions.

Ensure the model can handle overlapping chromosomes without treating every overlapping box pair as duplicates.

Evaluate NMS behavior carefully because aggressive NMS may suppress genuinely distinct crossing chromosomes.

For the initial version, use one chromosome foreground class.

The output for each predicted instance should include:

```python
{
    "box": [x1, y1, x2, y2],
    "score": float,
    "mask": HxW probability map
}
```

Keep output coordinates compatible with the original input image.

## 9. Mixed-Supervision Training

This is the critical engineering requirement.

Standard torchvision Mask R-CNN training expects instance masks for segmentation loss.

A naive approach that inserts zero masks for Dataset B is incorrect.

Implement explicit loss routing based on supervision type.

### Polygon batch

For Dataset A:

```text
Loss = RPN losses
     + ROI classification loss
     + ROI box regression loss
     + instance mask loss
```

### Box batch

For Dataset B:

```text
Loss = RPN losses
     + ROI classification loss
     + ROI box regression loss
```

Do not calculate a mask loss for Dataset B.

Implement this without silently modifying the mathematical meaning of any loss.

Preferred implementation:

- Use homogeneous supervision-type mini-batches initially.
- Adapt torchvision's ROI-head loss calculation so mask supervision is explicitly disabled for box-only batches.
- Keep detection losses active on both datasets.
- Keep mask-head parameters trainable on polygon batches.
- Do not run dummy mask targets through the normal mask loss.

Avoid relying on an exception handler or replacing missing masks with fabricated tensors.

If modifying torchvision internals, isolate the changes in a clearly documented custom module. Avoid broad monkey patches.

Write tests showing that:

1. A polygon batch produces detection and mask losses.
2. A box batch produces detection losses but no mask loss.
3. Mask-head parameters receive gradients on polygon batches.
4. Box-only batches do not inadvertently produce mask-head gradients.
5. Model parameters remain finite after mixed training steps.

## 10. Sampling and Dataset Balance

Dataset B is much larger than Dataset A.

Naively sampling from the concatenated dataset would cause polygon batches to become rare, weakening mask learning.

Implement configurable sampling ratios.

Start with:

```yaml
training:
  polygon_batch_probability: 0.5
  box_batch_probability: 0.5
```

This means that half of training steps should use polygon-labeled batches and half should use box-labeled batches.

Try alternative ratios later:

- 25% polygon / 75% box
- 50% polygon / 50% box
- 75% polygon / 25% box

Report the number of optimizer steps spent on each supervision type.

The training loop must define epochs or total training steps consistently when using weighted sampling or repeating a smaller dataset.

Do not silently oversample Dataset A without recording the effective sampling distribution.

Ensure samples from different patients or duplicate images cannot leak between training and validation splits.

## 11. Training Strategy

Support both a one-stage and a two-stage schedule.

### One-stage joint training

Train the shared detection and mask network using mixed supervision batches from the beginning.

### Two-stage training

Stage 1:

Train detection using both polygon and box datasets.

Stage 2:

Enable mask-head training using the polygon dataset while continuing detection supervision.

Optionally use a lower learning rate for the backbone during Stage 2.

Compare the two strategies rather than assuming one is superior.

Provide configurable:

- Batch size.
- Number of workers.
- Learning rate.
- Weight decay.
- Scheduler.
- Gradient clipping.
- Mixed precision.
- Total optimizer steps.
- Validation interval.
- Random seed.
- Early stopping.
- Checkpoint frequency.

Support CUDA, MPS and CPU execution, with CPU execution as default.

Use mixed precision where supported and stable.

Save both last and best checkpoints, along with optimizer and scheduler state for resuming.

Do not claim training has completed unless it actually ran.

## 12. Validation and Metrics

Use polygon-labeled validation images to measure instance segmentation accuracy.

At minimum compute:

**Detection**

- AP50 and AP75.
- Overall detection AP.
- Recall.
- Detection performance by chromosome size.

**Instance segmentation**

- Mask AP50 and AP75.
- Overall mask AP.
- Instance-level Dice.
- Instance-level IoU.
- Instance recall.

Match predicted instances to ground-truth instances using an explicit one-to-one matching policy, such as Hungarian matching based on mask IoU, with configurable acceptance thresholds.

Distinguish instance-level metrics from foreground semantic segmentation metrics.

**Chromosome-specific error analysis**

Measure or estimate:

- Missed chromosomes.
- Duplicate predictions.
- Split predictions.
- Merged predictions.
- Overlapping chromosome performance.
- Isolated chromosome performance.

Define split and merge criteria explicitly and make clear that their attribution may be ambiguous when ground-truth annotations themselves are incomplete.

For scale evaluation, group chromosomes by observed instance-mask area in the original image, but do not interpret pixel area as calibrated physical size.

Compute metrics separately for small, medium, and large apparent chromosomes.

Also report performance by dataset source, patient, and image-resolution group.

## 13. Scale Robustness Benchmark

Implement a controlled test using polygon-labeled validation images.

Apply synthetic scale factors:

```yaml
scale_test:
  factors: [0.6, 0.75, 1.0, 1.25, 1.5, 1.75]
```

For each scale:

1. Transform the input microscopy image.
2. Transform ground-truth masks and boxes consistently.
3. Run inference.
4. Map predictions into a common coordinate system where appropriate.
5. Compute instance segmentation metrics.

Avoid comparisons that include invalid padded or cropped regions.

Save plots of:

- Mask AP versus scale factor.
- Instance Dice versus scale factor.
- Instance recall versus scale factor.
- Split/merge error rates versus scale factor.

Report the metric at scale 1.0 and an aggregate robustness score across the tested scales.

Also investigate whether predictions are geometrically consistent after rescaling.

Do not confuse consistency with accuracy: a model can consistently produce incorrect masks.

## 14. Overlapping Chromosome Evaluation

Overlapping chromosomes are a primary failure mode.

Using polygon annotations, identify ground-truth instance pairs that overlap geometrically.

Also identify high-overlap bounding-box pairs.

Keep these definitions separate:

- Actual instance-mask overlap.
- Bounding-box overlap.
- Pixel-level contact or crossing.

Bounding-box overlap alone does not prove that two chromosomes physically overlap.

Measure segmentation performance separately for:

- Isolated instances.
- Instances with overlapping bounding boxes.
- Instances with intersecting polygon masks, where valid annotations support this distinction.

Save representative failure visualizations.

For each output image, render:

1. Original metaphase image.
2. Ground-truth chromosome masks.
3. Predicted instance masks.
4. Bounding boxes.
5. Predicted confidence scores.
6. Optional matching between prediction and ground truth.

Keep overlapping instances visually distinguishable.

## 15. Experiment C: BoxInst-Style Weak Mask Supervision

Implement only after A and B are validated.

Investigate BoxInst's official implementation and paper before coding.

Use box-supervised losses based on:

- Mask projection constraints.
- Local pairwise affinity.

Do not treat the rectangular box as a filled chromosome mask.

Design the weak-mask losses so they can be disabled individually.

Handle overlap ambiguities carefully.

For candidate regions where multiple chromosome boxes substantially overlap, consider masking ambiguous pixels from pairwise supervision or reducing their weights.

Compare:

- Experiment B: detection-only supervision from boxes.
- Experiment C: detection plus weak mask supervision from boxes.

Use the same training budget and backbone when possible.

Document differences from the official BoxInst formulation, including any limitations caused by integrating it into Mask R-CNN.

## 16. Experiment D: μSAM Pseudo-Mask Generation

Implement this as an optional independent pipeline.

Workflow:

1. Fine-tune a microscopy-appropriate SAM model using Dataset A's polygon masks.
2. For Dataset B, use each bounding box as a prompt.
3. Generate one or more candidate masks.
4. Score candidate masks for reliability.
5. Reject low-confidence or inconsistent masks.
6. Save accepted masks and confidence metadata.
7. Fine-tune the segmentation model using accepted pseudo-masks.

Potential filtering signals:

- Predicted mask confidence where exposed by the model.
- Agreement with the annotated bounding-box extent.
- Consistency under image transformations.
- Excessive leakage into neighboring regions.
- Severe overlap with neighboring predicted masks.

A box-matching score alone must not be considered proof that the correct chromosome was segmented.

Preserve original annotations separately from generated pseudo-labels.

Do not overwrite ground-truth data.

Provide visualization utilities for manual review.

## 17. Data Leakage and Reproducibility

Use patient-level train/validation/test splits whenever patient identifiers are available.

If patient identifiers are missing, investigate filename grouping and near-duplicate images.

Do not assume filenames always provide reliable patient identity.

Check whether the same image or near-duplicate images occur in both datasets.

Do not allow the same patient or near-duplicate image to appear in both train and test splits.

Save split manifests so all experiments evaluate on identical held-out data.

Set random seeds for Python, NumPy, PyTorch, sampling, and augmentation as far as possible.

Report software and hardware versions.

Do not claim full determinism where GPU operations are nondeterministic.

## 18. Command-Line Interface

Provide commands similar to:

```bash
python scripts/inspect_datasets.py \
    --config configs/experiment_a.yaml
```

```bash
python scripts/prepare_data.py \
    --config configs/experiment_b.yaml
```

```bash
python scripts/train.py \
    --config configs/experiment_a.yaml
```

```bash
python scripts/train.py \
    --config configs/experiment_b.yaml
```

```bash
python scripts/evaluate.py \
    --config configs/experiment_b.yaml \
    --checkpoint outputs/experiment_b/best.pt
```

```bash
python scripts/compare_experiments.py \
    --experiments experiment_a experiment_b
```

```bash
python scripts/predict.py \
    --checkpoint outputs/experiment_b/best.pt \
    --input path/to/metaphase.png \
    --output outputs/predictions/
```

The exact CLI may differ if there is an existing project convention, but keep it consistent and document it.

## 19. Testing Requirements

Write automated tests for:

- LabelMe polygon parsing.
- Rectangle parsing.
- Polygon rasterization.
- Consistent class mappings.
- Invalid annotation handling.
- Geometric augmentation correctness.
- Aspect-ratio preservation.
- Mask and box coordinate alignment.
- Mixed-supervision loss routing.
- Model forward and backward passes.
- Empty annotations and empty predictions.
- Instance matching.
- Metric calculations.
- Prediction coordinate restoration.
- Checkpoint saving and loading.

Use synthetic test images where actual datasets are unavailable.

Run a short smoke test with a small number of images before attempting full training.

Verify the pipeline can overfit a tiny polygon-labeled dataset, which is a useful diagnostic for annotation and loss implementation problems.

## 20. Deliverables

Produce:

1. Working data loaders for both datasets.
2. Annotation format adapters.
3. Data inspection report.
4. Configurable Mask R-CNN training pipeline.
5. Correct mixed-supervision loss routing.
6. Reproducible experiment configurations.
7. Inference pipeline.
8. Instance segmentation evaluation.
9. Scale robustness benchmark.
10. Overlap-specific error analysis.
11. Visualization tools.
12. Unit tests.
13. README with installation and execution instructions.
14. An experiment comparison report template.

The README should clearly explain the difference between polygon supervision, box-only detection supervision, and optional weak mask supervision.

## 21. Implementation Priorities

Implement incrementally in the following order.

**Phase 1 — Data validation**

Inspect both datasets, build annotation adapters, and verify geometry using visualization and tests.

**Phase 2 — Experiment A**

Train and evaluate the polygon-only Mask R-CNN baseline.

**Phase 3 — Experiment B**

Implement mixed supervision and train the model using polygon and box datasets together.

**Phase 4 — Scale robustness**

Add controlled scale evaluation, error analysis, and experiment comparison.

**Phase 5 — Optional advanced experiments**

Add BoxInst-style supervision and μSAM pseudo-label generation if the initial results justify them.

Do not build all components in one large unverified change.

At each phase:

- Implement code.
- Add tests.
- Run the relevant tests.
- Resolve failures.
- Document results.
- Proceed to the next phase.

If datasets or GPU resources are unavailable, complete the implementation and perform CPU-based synthetic smoke tests. Clearly state what was and was not executed.

## 22. Success Criteria

The project is successful when:

- Both annotation types can be loaded correctly.
- No fabricated instance masks are created from bounding boxes.
- Polygon images train both detection and mask heads.
- Box-only images train detection components without invalid mask losses.
- Variable image dimensions and aspect ratios are handled correctly.
- Overlapping chromosome instances can be represented independently.
- Training and inference are reproducible.
- Experiments A and B can be fairly compared.
- Scale robustness is quantitatively measured.
- Segmentation outputs can be consumed by the existing DINOv2 classification pipeline.

The primary research question is:

**Does additional bounding-box supervision improve scale robustness and chromosome instance segmentation quality compared with training only on the smaller polygon-annotated dataset?**

Secondary questions include whether weak mask supervision and pretrained microscopy foundation models provide further improvements.

## 23. Execution Instructions

Begin by inspecting the repository and available datasets.

Make reasonable engineering decisions when implementation details are unspecified, and document them.

Prioritize code correctness, clean abstractions, and measurable experiments over architectural complexity.

Do not invent evaluation results.

Do not train using test data.

Do not modify or delete original dataset files.

Do not overwrite existing model checkpoints unless explicitly configured.

Implement the pipeline, run available tests, and provide a concise summary of completed functionality, remaining limitations, and exact commands for running the experiments.
