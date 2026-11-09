# Final Project 3 — Industrial Visual Anomaly Detection

## 1. Project Overview

In this project, your group will investigate and develop an industrial visual anomaly detection system using the MVTec AD dataset. Industrial anomaly detection aims to identify defective products or abnormal visual patterns, especially when examples of defects are limited or unavailable during training.

Unlike conventional supervised defect classification, anomaly detection typically learns the characteristics of normal products and identifies deviations from these learned normal patterns.

Examples of industrial anomalies include:

- Scratches, cracks, and surface damage
- Missing or incorrectly positioned components
- Contamination and foreign materials
- Structural deformation
- Abnormal textures or patterns

Your group will conduct literature research, select suitable anomaly detection techniques, implement and compare different methods, and evaluate their suitability for industrial inspection.

## 2. Required Tasks

### Task 1 — Literature Review and Method Selection

Conduct a literature review of industrial anomaly detection techniques. Your group should investigate the basic principles of different approaches.

Your group must select at least two anomaly detection methods for implementation and comparison. The two methods should ideally represent different technical principles. A simple baseline and a more advanced approach are acceptable.

**Required activities:**

- Review relevant academic publications and existing techniques.
- Explain the underlying principle of each selected method.
- Discuss why these methods may be suitable for your selected MVTec AD categories.
- Justify the final selection based on accuracy, complexity, computational requirements, or industrial applicability.

### Task 2 — Dataset Preparation and Analysis

All groups must use the MVTec Anomaly Detection (MVTec AD) dataset.

To download, fill in the short form on the MVTec AD page. Single categories (about 0.1–0.7 GB each) can be downloaded instead of the whole dataset (about 4.9 GB).

- **Licence:** CC BY-NC-SA 4.0 (no commercial use).
- **Cite:**
  - P. Bergmann, M. Fauser, D. Sattlegger and C. Steger, CVPR 2019
  - P. Bergmann, K. Batzner, M. Fauser, D. Sattlegger and C. Steger, IJCV 2021

**Required activities:**

- Select at least four product categories from MVTec AD. It is recommended to choose both object categories (e.g., bottle, metal nut, capsule) and texture categories (e.g., carpet, tile, leather).
- Implement suitable preprocessing, such as image resizing, normalization, or other justified operations.

You should discuss how characteristics such as surface texture, object geometry, background variation, and defect size may influence anomaly detection performance.

### Task 3 — Anomaly Detection Implementation

Implement your selected anomaly detection methods (at least two) using the normal training images. The system should learn a representation of normal appearance and use this representation to estimate how abnormal a new image is. Defect labels and test annotations must not be used to train or tune the methods or to set the decision threshold.

In addition to identifying whether a product is abnormal, the system should determine where the defect is located.

For your selected methods, implement anomaly localization using a pixel-level or region-level anomaly map.

**Required activities:**

- Implement at least two methods.
- Train or adapt the methods using normal images, generate anomaly scores for unseen test images and anomaly heatmaps for test images.
- Classify test images as normal or anomalous using an appropriate decision rule.
- Overlay the predicted anomalous regions on the original images and compare the predicted regions with MVTec AD ground-truth defect masks.

Explain how the models identify deviations from normal appearance and analyze examples of successful localization and incorrect predictions.

### Task 4 — Quantitative Evaluation and Comparative Analysis

Conduct quantitative experiments to evaluate and compare the selected anomaly detection methods. The evaluation should cover two aspects.

#### A. Image-level anomaly detection

Determine whether an image represents a normal or defective product.

Suitable metrics include:

- Image-level AUROC
- F1-score

#### B. Pixel-level anomaly localization

Determine how accurately the method identifies defective image regions.

Suitable metrics include:

- Pixel-level AUROC
- Region-level overlap metrics, such as AU-PRO (the normalized area under the per-region-overlap curve up to a false positive rate of 30 %)

At minimum, report image-level AUROC for both methods and a quantitative localization metric for each method that produces anomaly maps.

## 3. Final Presentation and Deliverables

Your final presentation should include:

- **Problem definition:** What industrial inspection problem does anomaly detection address?
- **Dataset analysis:** Which MVTec AD categories were selected, and what types of anomalies do they contain?
- **Literature review:** What anomaly detection approaches were investigated?
- **Method selection:** Why were the selected methods chosen?
- **Method principles and implementation:** How do the methods learn normal appearance and detect anomalies?
- **Experimental evaluation:** Present quantitative results and comparison.
- **Failure-case analysis:** Explain why certain anomalies were incorrectly detected or missed.
- **Conclusions:** Discuss the advantages, limitations, and industrial applicability of the evaluated approaches.

## 4. Bonus Tasks — Additional Marks

Propose a meaningful enhancement to one of the selected methods.

Possible directions include but not limited to:

- Use better vision foundation model
- Multi-scale feature extraction and feature fusion
- Improve post-processing such as better anomaly scoring or more accurate anomaly localization

The improvement should be evaluated against the selected method's baseline to demonstrate its effectiveness.
