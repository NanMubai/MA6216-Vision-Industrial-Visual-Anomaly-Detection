# Experiment comparison

Mean cells show category counts [n=...]; missing results are shown as —.

image_f1 uses normal-validation thresholds; image_f1_max_oracle is an oracle reference.

## Run means

| run | image_auroc | image_f1 | image_f1_max_oracle | pixel_auroc | aupro | fit_seconds | infer_ms_per_image | infer_ms_per_image_e2e | peak_gpu_mem_gb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore_wrn50 | 0.9874 [n=6] | 0.9189 [n=6] | 0.9842 [n=6] | 0.9853 [n=6] | 0.9308 [n=6] | 3.6353 [n=6] | 2.7905 [n=6] | 9.2629 [n=6] | 1.5023 [n=6] |
| patchcore_dinov2_vitb14 | 0.9749 [n=6] | 0.9145 [n=6] | 0.9802 [n=6] | 0.9862 [n=6] | 0.9314 [n=6] | 6.6474 [n=6] | 12.9116 [n=6] | 19.4677 [n=6] | 3.3305 [n=6] |
| patchcore_dinov2_vits14 | 0.9678 [n=6] | 0.9281 [n=6] | 0.9730 [n=6] | 0.9860 [n=6] | 0.9290 [n=6] | 5.2864 [n=6] | 6.3268 [n=6] | 12.7990 [n=6] | 2.2156 [n=6] |
| patchcore_wrn50_dinov2 | 0.9646 [n=6] | 0.9061 [n=6] | 0.9716 [n=6] | 0.9828 [n=6] | 0.9141 [n=6] | 5.6790 [n=6] | 13.7702 [n=6] | 20.3526 [n=6] | 3.0805 [n=6] |

## image_auroc

| category | patchcore_wrn50 | patchcore_dinov2_vitb14 | patchcore_dinov2_vits14 | patchcore_wrn50_dinov2 |
| --- | ---: | ---: | ---: | ---: |
| bottle | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| capsule | 0.9777 | 0.9713 | 0.9900 | 0.9793 |
| screw | 0.9705 | 0.8791 | 0.8223 | 0.8094 |
| carpet | 0.9872 | 0.9996 | 0.9996 | 1.0000 |
| tile | 0.9892 | 0.9993 | 0.9949 | 0.9989 |
| leather | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| mean (objects) | 0.9827 [n=3] | 0.9501 [n=3] | 0.9374 [n=3] | 0.9295 [n=3] |
| mean (textures) | 0.9921 [n=3] | 0.9996 [n=3] | 0.9982 [n=3] | 0.9996 [n=3] |
| mean (all) | 0.9874 [n=6] | 0.9749 [n=6] | 0.9678 [n=6] | 0.9646 [n=6] |

## image_f1

| category | patchcore_wrn50 | patchcore_dinov2_vitb14 | patchcore_dinov2_vits14 | patchcore_wrn50_dinov2 |
| --- | ---: | ---: | ---: | ---: |
| bottle | 1.0000 | 0.9921 | 1.0000 | 1.0000 |
| capsule | 0.8360 | 0.8601 | 0.8526 | 0.8022 |
| screw | 0.8837 | 0.6740 | 0.7385 | 0.6630 |
| carpet | 0.9355 | 0.9780 | 0.9834 | 0.9889 |
| tile | 0.8947 | 0.9825 | 0.9940 | 0.9822 |
| leather | 0.9634 | 1.0000 | 1.0000 | 1.0000 |
| mean (objects) | 0.9066 [n=3] | 0.8421 [n=3] | 0.8637 [n=3] | 0.8217 [n=3] |
| mean (textures) | 0.9312 [n=3] | 0.9868 [n=3] | 0.9925 [n=3] | 0.9904 [n=3] |
| mean (all) | 0.9189 [n=6] | 0.9145 [n=6] | 0.9281 [n=6] | 0.9061 [n=6] |

## image_f1_max_oracle

| category | patchcore_wrn50 | patchcore_dinov2_vitb14 | patchcore_dinov2_vits14 | patchcore_wrn50_dinov2 |
| --- | ---: | ---: | ---: | ---: |
| bottle | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| capsule | 0.9772 | 0.9820 | 0.9820 | 0.9730 |
| screw | 0.9633 | 0.9106 | 0.8676 | 0.8623 |
| carpet | 0.9770 | 0.9944 | 0.9944 | 1.0000 |
| tile | 0.9880 | 0.9941 | 0.9940 | 0.9940 |
| leather | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| mean (objects) | 0.9801 [n=3] | 0.9642 [n=3] | 0.9499 [n=3] | 0.9451 [n=3] |
| mean (textures) | 0.9883 [n=3] | 0.9962 [n=3] | 0.9961 [n=3] | 0.9980 [n=3] |
| mean (all) | 0.9842 [n=6] | 0.9802 [n=6] | 0.9730 [n=6] | 0.9716 [n=6] |

## pixel_auroc

| category | patchcore_wrn50 | patchcore_dinov2_vitb14 | patchcore_dinov2_vits14 | patchcore_wrn50_dinov2 |
| --- | ---: | ---: | ---: | ---: |
| bottle | 0.9867 | 0.9915 | 0.9920 | 0.9906 |
| capsule | 0.9888 | 0.9905 | 0.9906 | 0.9901 |
| screw | 0.9932 | 0.9688 | 0.9670 | 0.9510 |
| carpet | 0.9898 | 0.9956 | 0.9962 | 0.9952 |
| tile | 0.9602 | 0.9786 | 0.9756 | 0.9784 |
| leather | 0.9930 | 0.9923 | 0.9947 | 0.9916 |
| mean (objects) | 0.9896 [n=3] | 0.9836 [n=3] | 0.9832 [n=3] | 0.9773 [n=3] |
| mean (textures) | 0.9810 [n=3] | 0.9888 [n=3] | 0.9888 [n=3] | 0.9884 [n=3] |
| mean (all) | 0.9853 [n=6] | 0.9862 [n=6] | 0.9860 [n=6] | 0.9828 [n=6] |

## aupro

| category | patchcore_wrn50 | patchcore_dinov2_vitb14 | patchcore_dinov2_vits14 | patchcore_wrn50_dinov2 |
| --- | ---: | ---: | ---: | ---: |
| bottle | 0.9493 | 0.9620 | 0.9650 | 0.9557 |
| capsule | 0.9313 | 0.9454 | 0.9398 | 0.9400 |
| screw | 0.9627 | 0.8452 | 0.8356 | 0.7677 |
| carpet | 0.9503 | 0.9801 | 0.9840 | 0.9765 |
| tile | 0.8192 | 0.8782 | 0.8654 | 0.8701 |
| leather | 0.9722 | 0.9775 | 0.9843 | 0.9749 |
| mean (objects) | 0.9478 [n=3] | 0.9175 [n=3] | 0.9135 [n=3] | 0.8878 [n=3] |
| mean (textures) | 0.9139 [n=3] | 0.9452 [n=3] | 0.9446 [n=3] | 0.9405 [n=3] |
| mean (all) | 0.9308 [n=6] | 0.9314 [n=6] | 0.9290 [n=6] | 0.9141 [n=6] |
