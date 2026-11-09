# Literature Review and Method Selection

*MA6216 Final Project 3: Industrial Visual Anomaly Detection on MVTec AD (Task 1)*

**Summary.** Industrial anomaly detection learns what *normal* products look like and flags deviations, because defects are too rare and too varied to learn in a supervised way. We review seven families of methods and pick two that rest on different principles. **PaDiM** [7] is the baseline: a *parametric* model that fits a Gaussian to pretrained features at every image position and scores test patches with the Mahalanobis distance. **PatchCore** [8] is the main method: a *non-parametric* model that keeps a coreset-subsampled memory bank of locally aware patch features and scores test patches by nearest-neighbour distance. Neither needs gradient-based training. Both produce image scores and pixel anomaly maps, and both are close to the state of the art on MVTec AD. For the bonus, we extend PatchCore with a DINOv2 backbone, multi-scale feature fusion and better post-processing.

---

## 1. Problem definition

### 1.1 Industrial inspection as one-class learning
Automated visual inspection must reject products with scratches, cracks, contamination, missing or misplaced parts, deformations, or abnormal textures. Supervised defect classification is a poor fit for three reasons:

- **Defects are rare.** A mature production line may produce only a handful of defective parts per thousand. Collecting a balanced labelled set is expensive and slow, and pixel-accurate masks cost even more.
- **Defects are diverse and open-ended.** The set of possible defects has no fixed boundary. A classifier trained on known defect types can miss a new failure mode, such as a new contaminant or a crack in an unseen location, which is exactly the case that matters most.
- **Normal data is cheap.** Good parts are plentiful and their appearance is fairly consistent, although lighting, pose and natural texture variation still change it.

The task is therefore posed as **unsupervised (one-class) anomaly detection**. A model sees only defect-free training images and learns a representation of normality. At test time it assigns an *anomaly score* that measures how far an input deviates from that representation. MVTec AD [1, 2] formalises this setting: 15 categories (5 textures, 10 objects), 5,354 high-resolution images, training sets that contain **only good images**, and test sets that mix good images with 73 defect types annotated with pixel-precise masks. We use three object categories (bottle, capsule, screw) and three texture categories (carpet, tile, leather).

Our protocol enforces the one-class assumption strictly. Models are fitted on 90 % of `train/good`. The other 10 % is a held-out **normal-only validation split**, which is the only data used to set decision thresholds and normalise scores. Test labels and masks are used only to compute metrics.

### 1.2 Image-level vs pixel-level tasks
- **Image-level detection (classification):** output one score $s(x)\in\mathbb{R}$ per image. Thresholding it gives a normal/defective decision, which drives accept/reject on the line.
- **Pixel-level localisation (segmentation):** output an anomaly map $A(x)\in\mathbb{R}^{H\times W}$. The map tells an operator *where* the defect is, supports root-cause analysis, and makes the decision auditable.

Most modern methods compute a dense map first and derive the image score from it, for example as its maximum. The two tasks are linked but not identical. A method can rank images well and still localise poorly, or the reverse. This is why we report metrics for both (Section 5).

---

## 2. Taxonomy of approaches

Table 1 lists published MVTec AD results (mean over the 15 categories, in %, as self-reported by each paper). Papers differ in input resolution, cropping, backbone and evaluation code, so differences of about 1 point are **not** significant. "–" means the paper does not report the metric on MVTec AD, or we could not verify the value. "≈" marks values taken from secondary tables.

**Table 1.** Representative methods and their reported MVTec AD results.

| Family | Method (venue) | Image AUROC | Pixel AUROC | AU-PRO |
|---|---|---|---|---|
| Reconstruction | SSIM-AE [3] (VISAPP'19), GANomaly [4] (ACCV'18) | – | – | – |
| Embedding + density / memory | SPADE [6] (arXiv'20) | 85.5 | 96.5 | 91.7 |
| | PaDiM, WRN50-Rd550 [7] (ICPR-W'20) | 95.3 | 97.5 | 92.1 |
| | PatchCore-25 %, WRN50 [8] (CVPR'22) | 99.1 | 98.1 | 93.4 |
| Knowledge distillation | Uninformed Students [9] (CVPR'20) | – | – | ≈91.4 |
| | STFPM [10] (BMVC'21) | 95.5 | 97.0 | 92.1 |
| | RD4AD [11] (CVPR'22) | 98.5 | 97.8 | 93.9 |
| | EfficientAD-S / -M [12] (WACV'24) | 98.8 / 99.1 | – | – |
| Normalising flows | DifferNet [13] (WACV'21) | ≈94.9 | – | – |
| | CFlow-AD [14] (WACV'22) | ≈98.3 | ≈98.6 | ≈94.6 |
| | FastFlow, WRN50 / CaiT [15] (arXiv'21) | 99.3 / 99.4 | 98.1 / 98.5 | – |
| Synthetic anomalies | CutPaste, 3-way [16] (CVPR'21) | 95.2 | 96.0 | – |
| | DRAEM [17] (ICCV'21) | 98.0 | 97.3 | – |
| | SimpleNet [18] (CVPR'23) | 99.6 | 98.1 | – |
| Foundation / zero-few-shot | WinCLIP, 0-shot [19] (CVPR'23) | 91.8 | 85.1 | – |
| | WinCLIP+, 1-shot [19] | 93.1 | 95.2 | – |
| | AnomalyDINO, 1-shot [20] (WACV'25) | 96.6 | – | – |
| Unified multi-class | UniAD, one model for 15 classes [23] (NeurIPS'22) | 96.5 | 96.8 | – |

### 2.1 Reconstruction-based methods
**Principle.** Train a generative model (an autoencoder, a VAE [5] or a GAN) to reconstruct normal images. Because it never sees defects, it should fail to reproduce them. The reconstruction error $\lVert x-\hat x\rVert$, computed per pixel or with a perceptual or SSIM metric, serves as the anomaly map. **SSIM-AE** [3] replaces the per-pixel $\ell_2$ loss with a structural-similarity loss, which captures local structure and contrast instead of exact intensities. **GANomaly** [4] uses an encoder–decoder–encoder generator and measures anomaly in the latent space.

**Pros.** Intuitive and fully unsupervised. The anomaly map is directly interpretable as a pixel residual.
**Cons.** Networks *generalise too well*: they often reconstruct defects too (the "identity shortcut"), especially on textures. Fine high-frequency textures are reconstructed blurrily, which produces false positives. Error metrics in pixel space respond poorly to semantic anomalies. The MVTec AD papers [1, 2] showed that these baselines are clearly weaker than feature-based methods, so modern work uses reconstruction mainly as a component, for example in DRAEM (Section 2.5) or UniAD's feature reconstruction (Section 2.7).

### 2.2 Feature embedding with density estimation or a memory bank
**Principle.** Skip learning from scratch. Extract **pretrained deep features** (for example from an ImageNet-trained WideResNet-50 [24]) and model the distribution of normal features, either *parametrically* (fit a density) or *non-parametrically* (store examples and use nearest neighbours). Rippel et al. [26] showed that a single multivariate Gaussian over global pretrained features is already a strong image-level detector. The methods below extend this idea to **patches**, which yields localisation.

- **SPADE** [6]: kNN on global features selects the closest normal images. Then a pixel-wise kNN on multi-layer feature pyramids against those images produces the anomaly map. It is simple but slow at test time, because it needs many dense kNN searches.
- **PaDiM** [7]: fits one multivariate Gaussian $\mathcal N(\mu_{ij},\Sigma_{ij})$ **per spatial position** $(i,j)$ over concatenated multi-layer features. The score is the Mahalanobis distance. Details are in Section 3.1.
- **PatchCore** [8]: stores locally aware patch features from all training images in a **memory bank**, reduces the bank with greedy **coreset subsampling**, and scores test patches by their **nearest-neighbour distance**. Details are in Section 3.2.

**Pros.** No (or minimal) training. Very strong accuracy. Pretrained features are robust to nuisance variation.
**Cons.** Accuracy depends on the backbone. Inference cost and memory grow with the memory bank (PatchCore) or with the feature dimension squared (PaDiM). Features trained on ImageNet classification are not tuned to subtle industrial defects.

### 2.3 Knowledge distillation (teacher–student)
**Principle.** A frozen pretrained *teacher* produces features. A *student* is trained on normal images only to regress the teacher's outputs. On normal inputs the two agree. On anomalies, which the student never saw, they disagree, and the discrepancy per location is the anomaly map.

- **Uninformed Students** [9]: an ensemble of students trained on patch descriptors. Anomaly is regression error plus predictive uncertainty. This paper also introduced the PRO metric.
- **STFPM** [10]: matches teacher and student feature pyramids at several layers in one forward pass.
- **RD4AD** [11]: *reverse* distillation. The student is a decoder that reconstructs the teacher's multi-scale features from a compact one-class bottleneck embedding. This stops the student from simply copying anomalies.
- **EfficientAD** [12]: a lightweight patch-description network distilled from a pretrained network, plus an autoencoder for logical anomalies. It reaches high accuracy at millisecond-level latency (reported as ≈2–5 ms per image on an RTX A6000).

**Pros.** Fast single-pass inference with a compact model, which suits real-time lines. Good localisation.
**Cons.** Requires training, so there are hyper-parameters, convergence issues and a risk that the student generalises to anomalies. Some designs are sensitive to the choice of teacher.

### 2.4 Normalising flows
**Principle.** A normalising flow is an invertible network that maps the feature distribution to a simple base density, usually a standard Gaussian, so the exact likelihood $p(f)$ is tractable. Low likelihood means anomalous. The flow is applied to pretrained features, not raw pixels.

- **DifferNet** [13]: a flow on multi-scale *image-level* features. It detects but does not localise.
- **CFlow-AD** [14]: conditional flows on *patch* features with positional encoding, at several scales, giving localisation.
- **FastFlow** [15]: 2-D fully convolutional flows that keep spatial structure. It works with CNN and ViT (DeiT, CaiT) backbones.

**Pros.** Principled likelihood that is more expressive than a single Gaussian (it can model multi-modal normal data). Strong accuracy.
**Cons.** Requires training, and flows can be unstable. More engineering is needed. The likelihood can be poorly calibrated across categories.

### 2.5 Synthetic anomalies and self-supervision
**Principle.** Turn one-class learning into a supervised proxy task by **synthesising pseudo-anomalies** from normal images. A classifier or segmenter learns to separate normal images from the synthetic defects.

- **CutPaste** [16]: cuts a patch from an image and pastes it elsewhere. It learns a representation that separates normal from cut-pasted images, then fits a Gaussian density on it.
- **DRAEM** [17]: blends Perlin-noise-shaped texture patches into images. A reconstruction subnetwork restores the normal image, and a discriminative subnetwork segments the difference.
- **SimpleNet** [18]: adds Gaussian noise to pretrained patch features (*feature-space* anomalies), adapts the features, and trains a small discriminator. It is very fast (77 FPS reported on an RTX 3080 Ti).

**Pros.** Directly optimises a discriminative objective. Strong localisation when the synthetic defects resemble real ones.
**Cons.** Performance depends on how realistic the synthetic anomalies are, which risks overfitting to the synthesis procedure. It requires training and extra hyper-parameters.

### 2.6 Foundation models and zero- or few-shot detection
**Principle.** Large pretrained models such as CLIP [22] or self-supervised DINOv2 [21] provide general-purpose features, which make anomaly detection possible with **no or very few** normal images.

- **WinCLIP** [19]: zero-shot. It compares CLIP embeddings of sliding windows with text prompts for "normal" and "damaged" states. WinCLIP+ adds a few normal reference images.
- **AnomalyDINO** [20]: a PatchCore-style patch-level kNN on **DINOv2** features, with masking and no training. With *one* normal image it reaches 96.6 % image AUROC on MVTec AD, which shows how strong DINOv2 patch features are.

**Pros.** Little or no data needed. Fast deployment of new products. Language prompts in the CLIP case.
**Cons.** Zero-shot accuracy still lags full-shot methods (Table 1). Large ViT backbones are expensive. Low patch resolution (for example a 16×16 grid at 224 px with patch size 14) limits localisation of small defects.

### 2.7 Unified multi-class models
**Principle.** Train **one** model for all categories instead of one per category, which simplifies deployment. **UniAD** [23] reconstructs pretrained features with a transformer. It adds a layer-wise query decoder, neighbour-masked attention and feature jittering to avoid the identity shortcut.

**Pros.** A single model to maintain. Scales to many product types.
**Cons.** Accuracy is somewhat lower than the best per-category models (96.5 % vs ≈99 % image AUROC). Training is required. Our project fits models per category, so this family is out of scope. We note it for completeness.

---

## 3. Method selection

The assignment asks for at least two methods that ideally rest on **different principles**. We choose **PaDiM** as the baseline and **PatchCore** as the main method. Both use the same frozen ImageNet WideResNet-50-2 [24] features, so the comparison isolates *how normality is modelled*: a **parametric per-position density** against a **non-parametric nearest-neighbour memory**.

### 3.1 Baseline: PaDiM (parametric Gaussian, Mahalanobis distance)
1. **Features.** For each training image, take the activations of `layer1`, `layer2` and `layer3`, upsample them to the `layer1` grid (56×56 for a 224 px input), and concatenate them into a $D$ = 1,792-dim vector per position. Randomly select $d$ = 550 dimensions, which the paper found nearly as good as PCA and much cheaper.
2. **Fit.** At every position $(i,j)$, estimate a Gaussian from the $N$ training vectors:
   $\mu_{ij}=\frac1N\sum_n f^n_{ij}$ and $\Sigma_{ij}=\frac1{N-1}\sum_n (f^n_{ij}-\mu_{ij})(f^n_{ij}-\mu_{ij})^\top+\varepsilon I$.
   The term $\varepsilon I$ (with $\varepsilon$ = 0.01) is essential. With $N\approx190$ images and $d$ = 550 the sample covariance is rank-deficient.
3. **Score.** The anomaly map is the Mahalanobis distance $M(f_{ij})=\sqrt{(f_{ij}-\mu_{ij})^\top\Sigma_{ij}^{-1}(f_{ij}-\mu_{ij})}$. It is upsampled and Gaussian-smoothed. The image score is the map maximum.

**Key property.** The model is *position-specific*: it assumes that the same location in every image shows the same part of the product. This holds for aligned objects, but the assumption is wrong under pose variation.

### 3.2 Main method: PatchCore (non-parametric kNN on a coreset memory bank)
1. **Locally aware patch features.** Take mid-level features from `layer2` and `layer3`. Mid-level features avoid both the generic low-level edges of `layer1` and the ImageNet-specific semantics of deep layers. Each feature vector is aggregated over its 3×3 neighbourhood by adaptive average pooling, which enlarges the receptive field without losing resolution. Then upsample `layer3` to the `layer2` grid (28×28), concatenate, and project to 1,024 dimensions.
2. **Memory bank and coreset.** Collect all patch features of all training images, $\mathcal M$ (≈190 × 784 ≈ 150k vectors per category). Greedy **k-center coreset** selection [25] keeps a subset $\mathcal M_C$ (here 10 %) that minimises the maximum distance from any discarded patch to its nearest kept patch. A random projection to 128 dimensions speeds up the selection. The coreset keeps the *coverage* of the normal feature space, including rare modes, which random subsampling would drop.
3. **Score.** Each test patch $m$ gets the score $s(m)=\min_{n\in\mathcal M_C}\lVert m-n\rVert_2$, which gives the anomaly map. The image score is the maximum patch score, re-weighted by how isolated its nearest neighbour $m^\*$ is among its own $b$ = 9 neighbours in $\mathcal M_C$. This reduces scores for test patches that fall in sparse but normal regions.

**Key property.** The model is *position-agnostic* and *multi-modal*. A test patch only needs to resemble *some* normal patch from *any* image at *any* location.

### 3.3 Why the two methods represent different principles

| Aspect | PaDiM | PatchCore |
|---|---|---|
| Model of normality | Parametric: one Gaussian per position | Non-parametric: set of exemplar patches |
| Distance | Mahalanobis (to a distribution) | Euclidean nearest neighbour (to a sample) |
| Spatial assumption | Position-specific (needs alignment) | Position-agnostic |
| Multi-modality | Unimodal per position | Arbitrary (memory covers all modes) |
| "Training" | One pass: means and covariances | One pass: feature extraction and coreset |

### 3.4 Justification

- **Accuracy.** PatchCore is among the most accurate full-shot methods on MVTec AD: 99.1 % image AUROC, 98.1 % pixel AUROC and 93.4 % AU-PRO [8]. That is on par with methods that need training (Table 1). PaDiM (95.3 / 97.5 / 92.1 [7]) is a strong but clearly lower baseline. The gap is large enough to analyse meaningfully.
- **Complexity.** Neither method uses back-propagation, losses or schedules. Each is a few hundred lines of code with a handful of interpretable hyper-parameters (layers, $d$, $\varepsilon$, coreset ratio, $k$). This makes failure analysis straightforward.
- **Compute.** Fitting is a single forward pass over about 200 images plus covariance estimation (PaDiM) or coreset selection (PatchCore). Each takes seconds to minutes on one GPU. At inference, PaDiM costs $O(HW\,d^2)$ per image. PatchCore costs one kNN search against $|\mathcal M_C|$, which the coreset and GPU matrix multiplication (or FAISS) keep cheap.
- **Memory.** PaDiM stores a $d\times d$ inverse covariance for every position: $56^2\times550^2\times4$ B ≈ **3.8 GB** (float32) per category. This is a known drawback of PaDiM. PatchCore with a 10 % coreset stores about $15\text{k}\times1024\times4$ B ≈ **60 MB**. PatchCore is therefore more accurate *and* lighter in memory, and the coreset ratio offers an explicit accuracy–cost trade-off.
- **No training required.** Fast setup per product, no risk of training divergence, and reproducible results. This fits a course project with a shared GPU.
- **Industrial applicability.** Both need only normal images. PatchCore's memory bank can be *updated incrementally*: new normal variation, such as a new batch of material, can be added without retraining. Its nearest neighbour also makes decisions explainable, because each anomalous patch can be shown next to the most similar normal patch. PaDiM's Gaussians give a calibrated-looking distance but must be refitted when the data changes.

### 3.5 Suitability for our categories

| Category | Type | Characteristics | Typical defects (MVTec AD) | Expectation |
|---|---|---|---|---|
| bottle | object | Top view, centred, roughly rotation-symmetric, uniform background | broken_large, broken_small, contamination | Well aligned, so PaDiM's per-position Gaussians fit well. Both methods should be strong. |
| capsule | object | Elongated, roughly aligned but with small shifts and rotations. Printed "500" imprint. | crack, faulty_imprint, poke, scratch, squeeze | Small, subtle defects. Misalignment smears PaDiM's position statistics (imprint pixels can land on body positions). PatchCore is more robust. |
| screw | object | **Arbitrary rotation and position**, thin structure, fine thread | manipulated_front, scratch_head, scratch_neck, thread_side, thread_top | Hardest case for PaDiM. With pose varying, each position's Gaussian mixes screw and background, so the covariance inflates and sensitivity drops. PatchCore's position-agnostic kNN is the better fit, though tiny thread defects stay difficult. |
| carpet | texture | Stationary woven texture | color, cut, hole, metal_contamination, thread | Statistics are the same at every position, so position-specificity is unnecessary but harmless. Both should do well. The PatchCore memory bank is highly redundant and compresses well under the coreset. |
| tile | texture | Stationary but irregular random pattern with high normal variance | crack, glue_strip, gray_stroke, oil, rough | High normal variability raises false positives. A multi-modal memory may help. |
| leather | texture | Stationary, fine grain | color, cut, fold, glue, poke | Subtle colour and fold defects. Both should do well. |

In summary, **texture categories** are *stationary*: normal statistics do not depend on position, so both approaches are viable and the main difficulty is subtle low-contrast defects. **Object categories** break PaDiM's alignment assumption to different degrees (bottle < capsule < screw). We therefore expect the PaDiM–PatchCore gap to be largest on screw and capsule, which is a testable hypothesis for Task 4.

---

## 4. Motivation for the bonus enhancements (PatchCore)

We propose three enhancements, each targeting a different stage of the pipeline. Each will be compared against the WRN50 PatchCore baseline under the same protocol. All choices are fixed in advance or set on the normal validation split, never on test results.

### 4.1 Stronger backbone: DINOv2
PatchCore can only be as good as its patch features. ImageNet-*supervised* CNN features are tuned for object categories, not for low-level surface integrity. **DINOv2** [21] is a ViT trained with large-scale self-supervision (LVD-142M). Its patch tokens have strong local semantics and correspondence properties, so similar parts of different objects map to similar features, even across pose changes. AnomalyDINO [20] shows that a plain patch-kNN on DINOv2 features is already highly competitive, even in the few-shot regime. We expect gains mainly on **capsule and screw**, where pose-robust, part-aware features should make nearest-neighbour matching more reliable.

Trade-offs to examine: patch size 14 gives a coarse grid (16×16 at 224 px), which hurts small defects. Raising the input resolution (for example 448 px gives a 32×32 grid) recovers resolution but increases compute and memory-bank size. We have cached weights for ViT-S/14 and ViT-B/14.

### 4.2 Multi-scale feature extraction and fusion
Defects range from a few pixels (screw thread, capsule poke) to large regions (glue strip, broken bottle). One feature level has one effective receptive field. Fusing levels (adding `layer1` for fine detail or `layer4` for context, or fusing several DINOv2 blocks) or input resolutions (an image pyramid) lets each scale catch the defects it is suited to. Fusion can happen **at the feature level** (concatenation before the memory bank, as in PatchCore's `layer2` + `layer3`) or **at the score level** (separate memory banks per scale, then normalised maps are combined). Score-level fusion keeps each memory bank small and makes the contribution of each scale interpretable.

### 4.3 Post-processing: aggregation, normalisation and smoothing
- **Image-score aggregation.** The maximum of the map is sensitive to a single noisy patch. The **mean of the top-k %** pixels is more robust for large or diffuse defects, but it can dilute tiny ones. Comparing both shows which defect types each one favours.
- **Score normalisation.** Raw kNN distances have different scales across categories and backbones. **z-score normalisation** with the mean and standard deviation of *normal validation* scores makes thresholds comparable and enables fusion across scales or backbones. It is a prerequisite for 4.2 and for a single threshold rule (a quantile of the validation scores) applied to every category.
- **Smoothing.** Patch maps are coarse (28×28 or 16×16). Bilinear upsampling followed by Gaussian smoothing ($\sigma$ = 4 px, as in [8]) suppresses isolated patch noise and gives smoother, more region-like maps, which mainly improves AU-PRO. Too large a $\sigma$ blurs small defects away, so it must be checked with care.

---

## 5. Evaluation metrics

| Metric | Level | Threshold-free? | What it measures |
|---|---|---|---|
| Image AUROC | image | yes | Probability that a random defective image scores higher than a random good one |
| F1 (at our threshold) | image | no | Precision–recall balance of the *deployed* decision rule |
| Pixel AUROC | pixel | yes | Pixel-wise ranking of defect pixels over normal pixels, pooled over all test images |
| AU-PRO @ 30 % FPR | region | yes (integrated) | Mean per-defect-region overlap, integrated over FPR in [0, 0.3] |

- **Image AUROC** is the standard detection metric. It is threshold-free and insensitive to the good/defective ratio, so it is comparable across categories and papers.
- **F1-score** evaluates an actual accept/reject decision. Our threshold is a high quantile of scores on the **normal validation split**, which is a realistic setup because no defects are needed to set it. We also report **F1-max**, the best F1 over all thresholds, but only as an *oracle* upper bound: it uses test labels and cannot be achieved in deployment.
- **Pixel AUROC** pools all pixels. Because the large majority of pixels are normal and large defects contain many pixels, the metric is **dominated by large defects**. Missing a tiny defect barely changes it, and false-positive rates that look low in relative terms can still mean many false pixels in absolute terms. As a result, many methods saturate above 97 % (Table 1).
- **AU-PRO** [9, 2] fixes this. For each threshold, compute the overlap $|R\cap P|/|R|$ for **each connected ground-truth region** $R$ separately and *average over regions*. A 20-pixel scratch then counts as much as a 20,000-pixel broken area. Plot PRO against the pixel FPR on normal pixels, integrate up to **FPR = 30 %**, and normalise so a perfect map scores 1. Maps that need more than 30 % false-positive pixels to find a defect are useless in practice, so that part of the curve is excluded. AU-PRO is therefore the fairer localisation metric for **small defects**, which are common in screw, capsule and leather.

---

## 6. References

[1] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 9592–9600.

[2] P. Bergmann, K. Batzner, M. Fauser, D. Sattlegger, and C. Steger, "The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," *International Journal of Computer Vision*, vol. 129, no. 4, pp. 1038–1059, 2021.

[3] P. Bergmann, S. Löwe, M. Fauser, D. Sattlegger, and C. Steger, "Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders," in *Proc. Int. Joint Conf. Computer Vision, Imaging and Computer Graphics Theory and Applications (VISAPP)*, 2019, pp. 372–380.

[4] S. Akçay, A. Atapour-Abarghouei, and T. P. Breckon, "GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training," in *Proc. Asian Conf. Computer Vision (ACCV)*, 2018, pp. 622–637.

[5] D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2014.

[6] N. Cohen and Y. Hoshen, "Sub-Image Anomaly Detection with Deep Pyramid Correspondences," arXiv:2005.02357, 2020.

[7] T. Defard, A. Setkov, A. Loesch, and R. Audigier, "PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization," in *ICPR International Workshops and Challenges*, LNCS vol. 12664, Springer, 2021, pp. 475–489. arXiv:2011.08785.

[8] K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, and P. Gehler, "Towards Total Recall in Industrial Anomaly Detection," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 14318–14328.

[9] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, "Uninformed Students: Student–Teacher Anomaly Detection with Discriminative Latent Embeddings," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 4183–4192.

[10] G. Wang, S. Han, E. Ding, and D. Huang, "Student-Teacher Feature Pyramid Matching for Anomaly Detection," in *Proc. British Machine Vision Conf. (BMVC)*, 2021. arXiv:2103.04257.

[11] H. Deng and X. Li, "Anomaly Detection via Reverse Distillation from One-Class Embedding," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 9737–9746.

[12] K. Batzner, L. Heckler, and R. König, "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies," in *Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV)*, 2024, pp. 128–138.

[13] M. Rudolph, B. Wandt, and B. Rosenhahn, "Same Same but DifferNet: Semi-Supervised Defect Detection with Normalizing Flows," in *Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV)*, 2021, pp. 1907–1916.

[14] D. Gudovskiy, S. Ishizaka, and K. Kozuka, "CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows," in *Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV)*, 2022, pp. 98–107.

[15] J. Yu, Y. Zheng, X. Wang, W. Li, Y. Wu, R. Zhao, and L. Wu, "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows," arXiv:2111.07677, 2021.

[16] C.-L. Li, K. Sohn, J. Yoon, and T. Pfister, "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 9664–9674.

[17] V. Zavrtanik, M. Kristan, and D. Skočaj, "DRAEM — A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2021, pp. 8330–8339.

[18] Z. Liu, Y. Zhou, Y. Xu, and Z. Wang, "SimpleNet: A Simple Network for Image Anomaly Detection and Localization," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2023, pp. 20402–20411.

[19] J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Ravichandran, and O. Dabeer, "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2023, pp. 19606–19616.

[20] S. Damm, M. Laszkiewicz, J. Lederer, and A. Fischer, "AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2," in *Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV)*, 2025, pp. 1319–1329.

[21] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, *et al.*, "DINOv2: Learning Robust Visual Features without Supervision," *Transactions on Machine Learning Research (TMLR)*, 2024.

[22] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, "Learning Transferable Visual Models From Natural Language Supervision," in *Proc. Int. Conf. Machine Learning (ICML)*, 2021, pp. 8748–8763.

[23] Z. You, L. Cui, Y. Shen, K. Yang, X. Lu, Y. Zheng, and X. Le, "A Unified Model for Multi-class Anomaly Detection," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

[24] S. Zagoruyko and N. Komodakis, "Wide Residual Networks," in *Proc. British Machine Vision Conf. (BMVC)*, 2016.

[25] O. Sener and S. Savarese, "Active Learning for Convolutional Neural Networks: A Core-Set Approach," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2018.

[26] O. Rippel, P. Mertens, and D. Merhof, "Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection," in *Proc. Int. Conf. Pattern Recognition (ICPR)*, 2020, pp. 6726–6733.

*Note on numbers.* Table 1 values come from the cited papers' own MVTec AD tables (15-category means). Values marked "≈" were cross-checked against secondary tables only. Per-category numbers for our six categories will come from our own runs (Task 4), not from the literature.
