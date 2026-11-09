# Experiment comparison

Mean cells show category counts [n=...]; missing results are shown as —.

image_f1 uses normal-validation thresholds; image_f1_max_oracle is an oracle reference.

## Run means

| run | image_auroc | image_f1 | image_f1_max_oracle | pixel_auroc | aupro | fit_seconds | infer_ms_per_image | infer_ms_per_image_e2e | peak_gpu_mem_gb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| padim_wrn50 | 0.9525 [n=6] | 0.7082 [n=6] | 0.9705 [n=6] | 0.9779 [n=6] | 0.9213 [n=6] | 6.1490 [n=6] | 3.5849 [n=6] | 10.0975 [n=6] | 1.7198 [n=6] |
| patchcore_wrn50 | 0.9874 [n=6] | 0.9189 [n=6] | 0.9842 [n=6] | 0.9853 [n=6] | 0.9308 [n=6] | 3.6301 [n=6] | 2.7824 [n=6] | 9.3113 [n=6] | 1.5023 [n=6] |

## image_auroc

| category | padim_wrn50 | patchcore_wrn50 |
| --- | ---: | ---: |
| bottle | 0.9992 | 1.0000 |
| capsule | 0.9023 | 0.9777 |
| screw | 0.8336 | 0.9705 |
| carpet | 0.9988 | 0.9872 |
| tile | 0.9812 | 0.9892 |
| leather | 1.0000 | 1.0000 |
| mean (objects) | 0.9117 [n=3] | 0.9827 [n=3] |
| mean (textures) | 0.9933 [n=3] | 0.9921 [n=3] |
| mean (all) | 0.9525 [n=6] | 0.9874 [n=6] |

## image_f1

| category | padim_wrn50 | patchcore_wrn50 |
| --- | ---: | ---: |
| bottle | 0.9767 | 1.0000 |
| capsule | 0.7869 | 0.8360 |
| screw | 0.2222 | 0.8837 |
| carpet | 0.9832 | 0.9355 |
| tile | 0.2857 | 0.8947 |
| leather | 0.9945 | 0.9634 |
| mean (objects) | 0.6620 [n=3] | 0.9066 [n=3] |
| mean (textures) | 0.7545 [n=3] | 0.9312 [n=3] |
| mean (all) | 0.7082 [n=6] | 0.9189 [n=6] |

## image_f1_max_oracle

| category | padim_wrn50 | patchcore_wrn50 |
| --- | ---: | ---: |
| bottle | 0.9921 | 1.0000 |
| capsule | 0.9686 | 0.9772 |
| screw | 0.9048 | 0.9633 |
| carpet | 0.9944 | 0.9770 |
| tile | 0.9634 | 0.9880 |
| leather | 1.0000 | 1.0000 |
| mean (objects) | 0.9552 [n=3] | 0.9801 [n=3] |
| mean (textures) | 0.9859 [n=3] | 0.9883 [n=3] |
| mean (all) | 0.9705 [n=6] | 0.9842 [n=6] |

## pixel_auroc

| category | padim_wrn50 | patchcore_wrn50 |
| --- | ---: | ---: |
| bottle | 0.9816 | 0.9867 |
| capsule | 0.9850 | 0.9888 |
| screw | 0.9835 | 0.9932 |
| carpet | 0.9893 | 0.9898 |
| tile | 0.9386 | 0.9602 |
| leather | 0.9894 | 0.9930 |
| mean (objects) | 0.9834 [n=3] | 0.9896 [n=3] |
| mean (textures) | 0.9724 [n=3] | 0.9810 [n=3] |
| mean (all) | 0.9779 [n=6] | 0.9853 [n=6] |

## aupro

| category | padim_wrn50 | patchcore_wrn50 |
| --- | ---: | ---: |
| bottle | 0.9486 | 0.9493 |
| capsule | 0.9144 | 0.9313 |
| screw | 0.9350 | 0.9627 |
| carpet | 0.9605 | 0.9503 |
| tile | 0.7990 | 0.8192 |
| leather | 0.9705 | 0.9722 |
| mean (objects) | 0.9327 [n=3] | 0.9478 [n=3] |
| mean (textures) | 0.9100 [n=3] | 0.9139 [n=3] |
| mean (all) | 0.9213 [n=6] | 0.9308 [n=6] |
