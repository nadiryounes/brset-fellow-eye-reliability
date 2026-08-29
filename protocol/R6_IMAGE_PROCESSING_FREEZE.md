# R6 Image Processing Freeze

## Objective

Preserve the full retinal field and OD/OS direction while supplying deterministic 224×224 RGB inputs to the frozen ConvNeXt-Tiny encoder. Training and evaluation transforms are identical because geometric or color augmentation could alter the quality and laterality mechanisms under study.

## Frozen decode and geometry pipeline

1. Read the original JPEG without modifying it.
2. Apply EXIF orientation using an EXIF-transpose operation.
3. Convert to three-channel sRGB/RGB.
4. Preserve aspect ratio and the entire decoded frame.
5. Symmetrically pad the shorter dimension with RGB `(0,0,0)` to form a square. If an odd number of pixels is required, place the extra pixel on the right or bottom.
6. Resize the padded square to 224×224 with bilinear interpolation and antialiasing enabled.
7. Convert channel values to `[0,1]`.
8. Normalize with ImageNet mean `[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`.

There is one resolution: 224×224. No resolution selection occurs after predictions.

## Crop and aspect-ratio policy

- No center crop, random crop, circular fundus crop, optic-disc crop, or macular crop.
- No direct aspect-ratio warp.
- No removal of black borders or field-dependent pixels.

The official ConvNeXt weights document a resize-to-236/center-crop-to-224 inference transform. R6 deliberately replaces that geometric step with full-frame letterboxing because center cropping could remove precisely the image-field information being studied. The ImageNet normalization is retained. This deviation must be reported.

## Augmentation policy

| Augmentation | Frozen decision | Reason |
|---|---|---|
| Horizontal flip | Prohibited | Would exchange nasal/temporal orientation and confuse the signed OD/OS question |
| Vertical flip | Prohibited | Anatomically implausible and changes orientation |
| Rotation | Prohibited | Can create artificial field loss/padding and complicate laterality |
| Random resized crop | Prohibited | Alters image field and can hide endpoint evidence |
| Translation/shear/perspective | Prohibited | Alters field geometry |
| Blur/sharpen | Prohibited | Directly changes focus, a primary explanatory variable |
| Brightness/contrast/saturation/hue jitter | Prohibited | Changes human quality mechanisms and acquisition appearance |
| CLAHE/histogram normalization | Prohibited | Rewrites illumination/color structure |
| Test-time augmentation | Prohibited | Adds a prediction/uncertainty method not in the registry |

The absence of augmentation is a scientific-control decision. It is not changed if development performance is lower than expected.

## Laterality audit

- Right eye is `exam_eye=1`/OD and left eye is `exam_eye=2`/OS under the official v1.0.2 mapping.
- No image reflection is permitted at any stage.
- Image arrays do not receive a laterality marker or text overlay.
- Laterality enters only the reliability/mechanistic tabular specification where frozen.
- Pair ordering is always `(R,L)`, so \(A=z_R-z_L\).

## Failure handling

R0 found all expected files and verified release hashes. Before R7 extraction, a decode-only preflight may confirm that each eligible file opens and has positive dimensions; it may not generate embeddings or predictions. If an eligible image fails decoding or its checksum fails, stop and issue a protocol deviation. Do not replace it, repair the raw JPEG, select a different image, or move the patient across partitions.

## Reproducibility record required in R7

- image library and version;
- interpolation implementation;
- exact preprocessing code hash;
- eligible image count and decode failures;
- no-augmentation assertion;
- laterality mapping assertion;
- normalized tensor shape and dtype;
- no image copies written into the project repository.
