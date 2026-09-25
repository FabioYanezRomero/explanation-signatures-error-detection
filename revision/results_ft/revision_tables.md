## Table A (gold_label): stratified logistic regression on the four dimensions, errors w.r.t. the gold label

| Dataset | Explainer | Graph | Majority acc. | Accuracy | Bal. acc. | AUROC | AUPRC (err) | MCC | Recall (err) |
|---|---|---|---|---|---|---|---|---|---|
| AG News | SubgraphX | Constituency | 0.935 | 0.864 | 0.745 | 0.824 | 0.300 | 0.339 | 0.609 |
| AG News | SubgraphX | Syntactic | 0.936 | 0.839 | 0.772 | 0.847 | 0.288 | 0.342 | 0.696 |
| AG News | SubgraphX | Window | 0.937 | 0.817 | 0.777 | 0.846 | 0.297 | 0.329 | 0.730 |
| AG News | SubgraphX | Skip-gram | 0.937 | 0.791 | 0.767 | 0.853 | 0.291 | 0.304 | 0.741 |
| AG News | GraphSVX | Constituency | 0.935 | 0.847 | 0.761 | 0.837 | 0.319 | 0.340 | 0.661 |
| AG News | GraphSVX | Syntactic | 0.936 | 0.818 | 0.789 | 0.853 | 0.296 | 0.344 | 0.756 |
| AG News | GraphSVX | Window | 0.937 | 0.821 | 0.772 | 0.851 | 0.323 | 0.327 | 0.715 |
| AG News | GraphSVX | Skip-gram | 0.937 | 0.822 | 0.780 | 0.861 | 0.343 | 0.335 | 0.732 |
| AG News | TokenSHAP (BERT) | Tokens | 0.938 | 0.874 | 0.782 | 0.864 | 0.364 | 0.383 | 0.677 |
| SST-2 | SubgraphX | Constituency | 0.911 | 0.846 | 0.748 | 0.805 | 0.385 | 0.371 | 0.628 |
| SST-2 | SubgraphX | Syntactic | 0.904 | 0.812 | 0.736 | 0.819 | 0.327 | 0.339 | 0.643 |
| SST-2 | SubgraphX | Window | 0.913 | 0.810 | 0.723 | 0.798 | 0.299 | 0.309 | 0.618 |
| SST-2 | SubgraphX | Skip-gram | 0.911 | 0.790 | 0.723 | 0.795 | 0.291 | 0.300 | 0.641 |
| SST-2 | GraphSVX | Constituency | 0.911 | 0.789 | 0.734 | 0.824 | 0.326 | 0.312 | 0.667 |
| SST-2 | GraphSVX | Syntactic | 0.904 | 0.807 | 0.713 | 0.818 | 0.324 | 0.307 | 0.595 |
| SST-2 | GraphSVX | Window | 0.913 | 0.820 | 0.741 | 0.800 | 0.289 | 0.336 | 0.645 |
| SST-2 | GraphSVX | Skip-gram | 0.911 | 0.825 | 0.730 | 0.810 | 0.288 | 0.331 | 0.615 |
| SST-2 | TokenSHAP (BERT) | Tokens | 0.916 | 0.810 | 0.722 | 0.813 | 0.260 | 0.302 | 0.616 |

## Table A (teacher_agreement): stratified logistic regression on the four dimensions, disagreement with the BERT teacher

| Dataset | Explainer | Graph | Majority acc. | Accuracy | Bal. acc. | AUROC | AUPRC (err) | MCC | Recall (err) |
|---|---|---|---|---|---|---|---|---|---|
| AG News | SubgraphX | Constituency | 0.981 | 0.941 | 0.925 | 0.962 | 0.356 | 0.438 | 0.908 |
| AG News | SubgraphX | Syntactic | 0.981 | 0.931 | 0.930 | 0.974 | 0.385 | 0.415 | 0.929 |
| AG News | SubgraphX | Window | 0.982 | 0.939 | 0.933 | 0.980 | 0.388 | 0.433 | 0.926 |
| AG News | SubgraphX | Skip-gram | 0.981 | 0.942 | 0.943 | 0.982 | 0.432 | 0.459 | 0.944 |
| AG News | GraphSVX | Constituency | 0.981 | 0.944 | 0.919 | 0.975 | 0.402 | 0.440 | 0.894 |
| AG News | GraphSVX | Syntactic | 0.981 | 0.924 | 0.916 | 0.973 | 0.392 | 0.391 | 0.908 |
| AG News | GraphSVX | Window | 0.982 | 0.939 | 0.947 | 0.983 | 0.508 | 0.444 | 0.956 |
| AG News | GraphSVX | Skip-gram | 0.981 | 0.941 | 0.942 | 0.983 | 0.454 | 0.454 | 0.944 |
| AG News | TokenSHAP (BERT) | Tokens | 0.938 | 0.874 | 0.782 | 0.864 | 0.364 | 0.383 | 0.677 |
| SST-2 | SubgraphX | Constituency | 0.984 | 0.929 | 0.788 | 0.941 | 0.306 | 0.274 | 0.643 |
| SST-2 | SubgraphX | Syntactic | 0.972 | 0.932 | 0.884 | 0.900 | 0.391 | 0.448 | 0.833 |
| SST-2 | SubgraphX | Window | 0.982 | 0.933 | 0.905 | 0.907 | 0.291 | 0.400 | 0.875 |
| SST-2 | SubgraphX | Skip-gram | 0.982 | 0.938 | 0.907 | 0.962 | 0.330 | 0.413 | 0.875 |
| SST-2 | GraphSVX | Constituency | 0.984 | 0.917 | 0.818 | 0.884 | 0.169 | 0.280 | 0.714 |
| SST-2 | GraphSVX | Syntactic | 0.972 | 0.931 | 0.863 | 0.901 | 0.387 | 0.427 | 0.792 |
| SST-2 | GraphSVX | Window | 0.982 | 0.935 | 0.905 | 0.887 | 0.472 | 0.403 | 0.875 |
| SST-2 | GraphSVX | Skip-gram | 0.982 | 0.916 | 0.835 | 0.947 | 0.243 | 0.309 | 0.750 |
| SST-2 | TokenSHAP (BERT) | Tokens | 0.916 | 0.810 | 0.722 | 0.813 | 0.260 | 0.302 | 0.616 |

## Table A2: paper's LR script (teacher target), stratified by PREDICTED class, 200 bootstraps

| Dataset | Explainer | Graph | Pooled acc. | Per-pred-class acc. | SD | Bootstrap acc. | Bootstrap SD |
|---|---|---|---|---|---|---|---|
| AG News | SubgraphX | Constituency | 0.941 | 0.946 | 0.023 | 0.953 | 0.020 |
| AG News | SubgraphX | Syntactic | 0.929 | 0.943 | 0.026 | 0.953 | 0.021 |
| AG News | SubgraphX | Window | 0.939 | 0.951 | 0.025 | 0.960 | 0.020 |
| AG News | SubgraphX | Skip-gram | 0.944 | 0.954 | 0.022 | 0.962 | 0.018 |
| AG News | GraphSVX | Constituency | 0.944 | 0.944 | 0.023 | 0.952 | 0.020 |
| AG News | GraphSVX | Syntactic | 0.924 | 0.943 | 0.026 | 0.951 | 0.023 |
| AG News | GraphSVX | Window | 0.939 | 0.949 | 0.024 | 0.959 | 0.020 |
| AG News | GraphSVX | Skip-gram | 0.941 | 0.951 | 0.022 | 0.959 | 0.019 |
| AG News | TokenSHAP (BERT) | Tokens | 0.874 | 0.872 | 0.042 | 0.869 | 0.038 |
| SST-2 | SubgraphX | Constituency | 0.919 | 0.940 | 0.020 | 0.956 | 0.016 |
| SST-2 | SubgraphX | Syntactic | 0.935 | 0.939 | 0.028 | 0.954 | 0.015 |
| SST-2 | SubgraphX | Window | 0.922 | 0.932 | 0.029 | 0.947 | 0.019 |
| SST-2 | SubgraphX | Skip-gram | 0.924 | 0.924 | 0.045 | 0.947 | 0.034 |
| SST-2 | GraphSVX | Constituency | 0.917 | 0.951 | 0.024 | 0.962 | 0.016 |
| SST-2 | GraphSVX | Syntactic | 0.931 | 0.929 | 0.038 | 0.944 | 0.018 |
| SST-2 | GraphSVX | Window | 0.935 | 0.953 | 0.028 | 0.964 | 0.020 |
| SST-2 | GraphSVX | Skip-gram | 0.916 | 0.922 | 0.032 | 0.944 | 0.026 |
| SST-2 | TokenSHAP (BERT) | Tokens | 0.813 | 0.800 | 0.083 | 0.827 | 0.061 |

## Table B (gold_label): nested models, AUROC of error detection

| Dataset | Explainer | Graph | Confidence | Explanation | Conf.+Expl. | Delta AUROC [95% CI] |
|---|---|---|---|---|---|---|
| AG News | SubgraphX | Constituency | 0.820 | 0.824 | 0.830 | +0.011 [-0.006, +0.029] |
| AG News | SubgraphX | Syntactic | 0.827 | 0.847 | 0.847 | +0.020 [+0.007, +0.033] |
| AG News | SubgraphX | Window | 0.821 | 0.846 | 0.845 | +0.025 [+0.010, +0.041] |
| AG News | SubgraphX | Skip-gram | 0.815 | 0.853 | 0.854 | +0.038 [+0.022, +0.054] |
| AG News | GraphSVX | Constituency | 0.820 | 0.837 | 0.843 | +0.024 [+0.005, +0.043] |
| AG News | GraphSVX | Syntactic | 0.827 | 0.853 | 0.853 | +0.026 [+0.012, +0.040] |
| AG News | GraphSVX | Window | 0.821 | 0.851 | 0.851 | +0.030 [+0.016, +0.046] |
| AG News | GraphSVX | Skip-gram | 0.815 | 0.861 | 0.861 | +0.046 [+0.031, +0.062] |
| AG News | TokenSHAP (BERT) | Tokens | 0.887 | 0.863 | 0.885 | -0.001 [-0.007, +0.005] |
| SST-2 | SubgraphX | Constituency | 0.829 | 0.805 | 0.814 | -0.016 [-0.043, +0.007] |
| SST-2 | SubgraphX | Syntactic | 0.850 | 0.820 | 0.835 | -0.015 [-0.048, +0.019] |
| SST-2 | SubgraphX | Window | 0.833 | 0.798 | 0.808 | -0.025 [-0.055, -0.000] |
| SST-2 | SubgraphX | Skip-gram | 0.846 | 0.794 | 0.816 | -0.029 [-0.067, +0.004] |
| SST-2 | GraphSVX | Constituency | 0.829 | 0.824 | 0.829 | +0.000 [-0.028, +0.033] |
| SST-2 | GraphSVX | Syntactic | 0.850 | 0.818 | 0.831 | -0.019 [-0.044, +0.006] |
| SST-2 | GraphSVX | Window | 0.833 | 0.799 | 0.801 | -0.032 [-0.081, +0.007] |
| SST-2 | GraphSVX | Skip-gram | 0.846 | 0.810 | 0.828 | -0.018 [-0.053, +0.013] |
| SST-2 | TokenSHAP (BERT) | Tokens | 0.835 | 0.813 | 0.834 | -0.001 [-0.027, +0.028] |

## Table B (teacher_agreement): nested models, AUROC of error detection

| Dataset | Explainer | Graph | Confidence | Explanation | Conf.+Expl. | Delta AUROC [95% CI] |
|---|---|---|---|---|---|---|
| AG News | SubgraphX | Constituency | 0.974 | 0.962 | 0.963 | -0.011 [-0.035, +0.005] |
| AG News | SubgraphX | Syntactic | 0.972 | 0.973 | 0.974 | +0.001 [-0.006, +0.007] |
| AG News | SubgraphX | Window | 0.976 | 0.980 | 0.980 | +0.004 [+0.001, +0.007] |
| AG News | SubgraphX | Skip-gram | 0.982 | 0.982 | 0.982 | -0.000 [-0.002, +0.002] |
| AG News | GraphSVX | Constituency | 0.974 | 0.975 | 0.975 | +0.002 [-0.004, +0.008] |
| AG News | GraphSVX | Syntactic | 0.972 | 0.972 | 0.973 | +0.001 [-0.004, +0.006] |
| AG News | GraphSVX | Window | 0.976 | 0.983 | 0.983 | +0.007 [+0.004, +0.010] |
| AG News | GraphSVX | Skip-gram | 0.982 | 0.983 | 0.983 | +0.001 [-0.001, +0.004] |
| AG News | TokenSHAP (BERT) | Tokens | 0.887 | 0.863 | 0.885 | -0.001 [-0.007, +0.005] |
| SST-2 | SubgraphX | Constituency | 0.906 | 0.943 | 0.939 | +0.033 [-0.034, +0.144] |
| SST-2 | SubgraphX | Syntactic | 0.946 | 0.900 | 0.925 | -0.022 [-0.055, +0.006] |
| SST-2 | SubgraphX | Window | 0.907 | 0.908 | 0.898 | -0.010 [-0.035, +0.011] |
| SST-2 | SubgraphX | Skip-gram | 0.903 | 0.962 | 0.959 | +0.057 [-0.018, +0.191] |
| SST-2 | GraphSVX | Constituency | 0.906 | 0.884 | 0.884 | -0.022 [-0.044, -0.002] |
| SST-2 | GraphSVX | Syntactic | 0.946 | 0.902 | 0.914 | -0.033 [-0.082, -0.001] |
| SST-2 | GraphSVX | Window | 0.907 | 0.887 | 0.882 | -0.030 [-0.190, +0.062] |
| SST-2 | GraphSVX | Skip-gram | 0.903 | 0.947 | 0.942 | +0.040 [-0.020, +0.155] |
| SST-2 | TokenSHAP (BERT) | Tokens | 0.835 | 0.813 | 0.834 | -0.001 [-0.027, +0.028] |

## Table C (gold_label): quadrant association with correctness (Dimensions 3 and 4)

| Dataset | Explainer | Graph | JS (D3) | Cramer V (D3) | old Sep. (D3) | JS (D4) | Cramer V (D4) | old Sep. (D4) |
|---|---|---|---|---|---|---|---|---|
| AG News | SubgraphX | Constituency | 0.045 | 0.185 | 6.456 | 0.000 | 0.001 | 0.028 |
| AG News | SubgraphX | Syntactic | 0.045 | 0.202 | 6.967 | 0.000 | 0.007 | 0.237 |
| AG News | SubgraphX | Window | 0.068 | 0.250 | 8.591 | 0.021 | 0.089 | 3.063 |
| AG News | SubgraphX | Skip-gram | 0.032 | 0.175 | 6.003 | 0.004 | 0.034 | 1.184 |
| AG News | GraphSVX | Constituency | 0.087 | 0.286 | 10.000 | 0.023 | 0.133 | 4.652 |
| AG News | GraphSVX | Syntactic | 0.043 | 0.203 | 7.008 | 0.001 | 0.026 | 0.900 |
| AG News | GraphSVX | Window | 0.039 | 0.206 | 7.070 | 0.017 | 0.110 | 3.776 |
| AG News | GraphSVX | Skip-gram | 0.023 | 0.151 | 5.178 | 0.008 | 0.078 | 2.688 |
| AG News | TokenSHAP (BERT) | Tokens | 0.086 | 0.240 | 8.171 | 0.034 | 0.129 | 4.409 |
| SST-2 | SubgraphX | Constituency | 0.029 | 0.200 | 8.073 | 0.014 | 0.076 | 3.056 |
| SST-2 | SubgraphX | Syntactic | 0.057 | 0.233 | 9.734 | 0.003 | 0.025 | 1.035 |
| SST-2 | SubgraphX | Window | 0.048 | 0.192 | 7.649 | 0.005 | 0.058 | 2.332 |
| SST-2 | SubgraphX | Skip-gram | 0.004 | 0.040 | 1.623 | 0.006 | 0.050 | 2.011 |
| SST-2 | GraphSVX | Constituency | 0.060 | 0.267 | 10.759 | 0.000 | nan | 0.000 |
| SST-2 | GraphSVX | Syntactic | 0.084 | 0.279 | 11.647 | 0.003 | 0.025 | 1.035 |
| SST-2 | GraphSVX | Window | 0.028 | 0.183 | 7.299 | 0.000 | nan | 0.000 |
| SST-2 | GraphSVX | Skip-gram | 0.033 | 0.209 | 8.446 | 0.000 | nan | 0.000 |
| SST-2 | TokenSHAP (BERT) | Tokens | 0.054 | 0.179 | 7.004 | 0.024 | 0.123 | 4.812 |

## Table C (teacher_agreement): quadrant association with correctness (Dimensions 3 and 4)

| Dataset | Explainer | Graph | JS (D3) | Cramer V (D3) | old Sep. (D3) | JS (D4) | Cramer V (D4) | old Sep. (D4) |
|---|---|---|---|---|---|---|---|---|
| AG News | SubgraphX | Constituency | 0.183 | 0.306 | 5.842 | 0.094 | 0.090 | 1.718 |
| AG News | SubgraphX | Syntactic | 0.205 | 0.365 | 6.964 | 0.022 | 0.033 | 0.633 |
| AG News | SubgraphX | Window | 0.347 | 0.454 | 8.512 | 0.018 | 0.038 | 0.715 |
| AG News | SubgraphX | Skip-gram | 0.151 | 0.349 | 6.732 | 0.115 | 0.095 | 1.834 |
| AG News | GraphSVX | Constituency | 0.267 | 0.397 | 7.575 | 0.005 | 0.030 | 0.569 |
| AG News | GraphSVX | Syntactic | 0.229 | 0.424 | 8.089 | 0.001 | 0.010 | 0.192 |
| AG News | GraphSVX | Window | 0.248 | 0.491 | 9.214 | 0.003 | 0.021 | 0.396 |
| AG News | GraphSVX | Skip-gram | 0.143 | 0.382 | 7.373 | 0.003 | 0.024 | 0.464 |
| AG News | TokenSHAP (BERT) | Tokens | 0.086 | 0.240 | 8.171 | 0.034 | 0.129 | 4.409 |
| SST-2 | SubgraphX | Constituency | 0.069 | 0.261 | 4.647 | 0.063 | 0.061 | 1.090 |
| SST-2 | SubgraphX | Syntactic | 0.354 | 0.461 | 10.656 | 0.003 | 0.013 | 0.296 |
| SST-2 | SubgraphX | Window | 0.295 | 0.332 | 6.311 | 0.012 | 0.051 | 0.963 |
| SST-2 | SubgraphX | Skip-gram | 0.024 | 0.093 | 1.772 | 0.184 | 0.093 | 1.766 |
| SST-2 | GraphSVX | Constituency | 0.203 | 0.313 | 5.566 | 0.000 | nan | 0.000 |
| SST-2 | GraphSVX | Syntactic | 0.516 | 0.487 | 11.259 | 0.012 | 0.080 | 1.852 |
| SST-2 | GraphSVX | Window | 0.217 | 0.383 | 7.266 | 0.000 | nan | 0.000 |
| SST-2 | GraphSVX | Skip-gram | 0.137 | 0.275 | 5.227 | 0.000 | nan | 0.000 |
| SST-2 | TokenSHAP (BERT) | Tokens | 0.054 | 0.179 | 7.004 | 0.024 | 0.123 | 4.812 |

## Table D (gold_label): calibration (ECE) and the high-confidence regime (confidence > 0.9)

| Dataset | Explainer | Graph | ECE | Errors | Share of errors conf>0.9 | Errors conf>0.9 | AUROC conf. | AUROC conf.+expl. | Delta AUROC [CI] | Delta AUPRC [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| AG News | SubgraphX | Constituency | 0.044 | 496 | 0.609 | 302 | 0.773 | 0.786 | +0.012 [-0.017, +0.042] | +0.005 [-0.017, +0.026] |
| AG News | SubgraphX | Syntactic | 0.041 | 484 | 0.562 | 272 | 0.774 | 0.800 | +0.027 [+0.001, +0.051] | -0.005 [-0.028, +0.015] |
| AG News | SubgraphX | Window | 0.043 | 478 | 0.613 | 293 | 0.782 | 0.802 | +0.019 [-0.004, +0.042] | +0.000 [-0.022, +0.023] |
| AG News | SubgraphX | Skip-gram | 0.042 | 478 | 0.611 | 292 | 0.784 | 0.808 | +0.023 [+0.003, +0.044] | +0.007 [-0.013, +0.031] |
| AG News | GraphSVX | Constituency | 0.044 | 496 | 0.609 | 302 | 0.773 | 0.798 | +0.024 [-0.002, +0.050] | +0.023 [-0.002, +0.051] |
| AG News | GraphSVX | Syntactic | 0.041 | 484 | 0.562 | 272 | 0.774 | 0.799 | +0.025 [+0.003, +0.047] | -0.021 [-0.043, -0.004] |
| AG News | GraphSVX | Window | 0.043 | 478 | 0.613 | 293 | 0.782 | 0.805 | +0.022 [+0.003, +0.041] | -0.001 [-0.027, +0.027] |
| AG News | GraphSVX | Skip-gram | 0.042 | 478 | 0.611 | 292 | 0.784 | 0.818 | +0.033 [+0.014, +0.053] | +0.015 [-0.011, +0.045] |
| AG News | TokenSHAP (BERT) | Tokens | 0.010 | 471 | 0.306 | 144 | 0.809 | 0.795 | -0.013 [-0.029, +0.004] | +0.000 [-0.017, +0.021] |
| SST-2 | SubgraphX | Constituency | 0.049 | 78 | 0.487 | 38 | 0.802 | 0.724 | -0.078 [-0.137, -0.028] | +0.007 [-0.040, +0.077] |
| SST-2 | SubgraphX | Syntactic | 0.056 | 84 | 0.512 | 43 | 0.813 | 0.795 | -0.018 [-0.071, +0.032] | +0.009 [-0.054, +0.075] |
| SST-2 | SubgraphX | Window | 0.051 | 76 | 0.566 | 43 | 0.818 | 0.796 | -0.023 [-0.086, +0.026] | -0.016 [-0.081, +0.056] |
| SST-2 | SubgraphX | Skip-gram | 0.053 | 78 | 0.513 | 40 | 0.837 | 0.814 | -0.023 [-0.081, +0.028] | -0.013 [-0.081, +0.045] |
| SST-2 | GraphSVX | Constituency | 0.049 | 78 | 0.487 | 38 | 0.802 | 0.754 | -0.050 [-0.095, -0.008] | -0.040 [-0.079, -0.006] |
| SST-2 | GraphSVX | Syntactic | 0.056 | 84 | 0.512 | 43 | 0.813 | 0.815 | +0.002 [-0.056, +0.058] | -0.020 [-0.065, +0.020] |
| SST-2 | GraphSVX | Window | 0.051 | 76 | 0.566 | 43 | 0.818 | 0.784 | -0.035 [-0.099, +0.016] | -0.011 [-0.091, +0.071] |
| SST-2 | GraphSVX | Skip-gram | 0.053 | 78 | 0.513 | 40 | 0.837 | 0.804 | -0.033 [-0.096, +0.016] | +0.016 [-0.057, +0.100] |
| SST-2 | TokenSHAP (BERT) | Tokens | 0.039 | 73 | 0.438 | 32 | 0.793 | 0.797 | +0.005 [-0.059, +0.066] | +0.051 [-0.025, +0.157] |

## Table D (teacher_agreement): calibration (ECE) and the high-confidence regime (confidence > 0.9)

| Dataset | Explainer | Graph | ECE | Errors | Share of errors conf>0.9 | Errors conf>0.9 | AUROC conf. | AUROC conf.+expl. | Delta AUROC [CI] | Delta AUPRC [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| AG News | SubgraphX | Constituency | 0.003 | 141 | 0.099 | 14 | 0.909 | 0.888 | -0.023 [-0.200, +0.141] | -0.004 [-0.039, +0.018] |
| AG News | SubgraphX | Syntactic | 0.006 | 141 | 0.113 | 16 | 0.958 | 0.926 | -0.031 [-0.081, +0.009] | +0.110 [-0.001, +0.280] |
| AG News | SubgraphX | Window | 0.005 | 136 | 0.103 | 14 | 0.978 | 0.962 | -0.017 [-0.056, +0.006] | +0.008 [-0.025, +0.057] |
| AG News | SubgraphX | Skip-gram | 0.006 | 144 | 0.069 | 10 | 0.981 | 0.976 | -0.005 [-0.023, +0.009] | -0.007 [-0.053, +0.029] |
| AG News | GraphSVX | Constituency | 0.003 | 141 | 0.099 | 14 | 0.909 | 0.920 | +0.010 [-0.098, +0.147] | +0.036 [-0.046, +0.174] |
| AG News | GraphSVX | Syntactic | 0.006 | 141 | 0.113 | 16 | 0.958 | 0.930 | -0.028 [-0.064, +0.003] | -0.005 [-0.033, +0.024] |
| AG News | GraphSVX | Window | 0.005 | 136 | 0.103 | 14 | 0.978 | 0.935 | -0.042 [-0.113, +0.009] | -0.001 [-0.104, +0.070] |
| AG News | GraphSVX | Skip-gram | 0.006 | 144 | 0.069 | 10 | 0.981 | 0.904 | -0.073 [-0.246, +0.011] | +0.031 [-0.035, +0.163] |
| AG News | TokenSHAP (BERT) | Tokens | 0.010 | 471 | 0.306 | 144 | 0.809 | 0.795 | -0.013 [-0.029, +0.004] | +0.000 [-0.017, +0.021] |
| SST-2 | SubgraphX | Constituency | 0.025 | 14 | 0.071 | 1 | nan | nan | n/a | n/a |
| SST-2 | SubgraphX | Syntactic | 0.013 | 24 | 0.125 | 3 | nan | nan | n/a | n/a |
| SST-2 | SubgraphX | Window | 0.021 | 16 | 0.062 | 1 | nan | nan | n/a | n/a |
| SST-2 | SubgraphX | Skip-gram | 0.021 | 16 | 0.125 | 2 | nan | nan | n/a | n/a |
| SST-2 | GraphSVX | Constituency | 0.025 | 14 | 0.071 | 1 | nan | nan | n/a | n/a |
| SST-2 | GraphSVX | Syntactic | 0.013 | 24 | 0.125 | 3 | nan | nan | n/a | n/a |
| SST-2 | GraphSVX | Window | 0.021 | 16 | 0.062 | 1 | nan | nan | n/a | n/a |
| SST-2 | GraphSVX | Skip-gram | 0.021 | 16 | 0.125 | 2 | nan | nan | n/a | n/a |
| SST-2 | TokenSHAP (BERT) | Tokens | 0.039 | 73 | 0.438 | 32 | 0.793 | 0.797 | +0.005 [-0.059, +0.066] | +0.051 [-0.025, +0.157] |
