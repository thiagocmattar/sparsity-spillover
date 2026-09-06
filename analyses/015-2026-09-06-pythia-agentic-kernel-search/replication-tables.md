# Run 025 fresh-process and hardware-transfer results

The replication unit is a fresh Python process. RTX summaries use the median of three eager-only process estimates. H100 summaries use the median of eager-only repeats 2 and 3; repeat 1 intentionally initialized and measured CUDA graphs and is retained only in the graph/sensitivity records. Each process estimate is itself a paired geometric mean over 80 native/candidate timings on 16 fixed input blocks.

## Frozen winner measurements

| GPU | size | family | kappa | loss | R_model | process speedup range; median | n | qualified |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A0 | - | 5.208573 | 0.0000% | [0.8296, 0.9827]; **0.9209x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A1-H | - | 5.269634 | 2.7141% | [0.9838, 0.9939]; **0.9916x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A4-OL1 | 0 | 5.458276 | 7.8100% | [1.0523, 1.0546]; **1.0538x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A4-OL1 | 0.01 | 5.458309 | 8.5867% | [1.0492, 1.0580]; **1.0544x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A4-OL1 | 0.05 | 5.489760 | 10.4564% | [1.0500, 1.0625]; **1.0536x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A4-OL1 | 0.1 | 5.548254 | 11.3943% | [1.0479, 1.0555]; **1.0503x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A4-OL1 | 0.5 | 6.037982 | 12.7134% | [1.0473, 1.0639]; **1.0489x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A7-OL1 | 0 | 5.480181 | 7.0542% | [1.0472, 1.0575]; **1.0489x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A7-OL1 | 0.01 | 5.475801 | 7.7234% | [1.0379, 1.0519]; **1.0400x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A7-OL1 | 0.05 | 5.462800 | 9.8630% | [1.0418, 1.0478]; **1.0438x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A7-OL1 | 0.1 | 5.429488 | 11.7968% | [1.0447, 1.0608]; **1.0468x** | 3 | no |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A7-OL1 | 0.5 | 5.829407 | 27.4827% | [1.0507, 1.0522]; **1.0521x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A0 | - | 4.099766 | 0.0005% | [0.9991, 1.0092]; **1.0043x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A1-H | - | 4.222750 | 10.0658% | [0.9457, 0.9786]; **0.9641x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A4-OL1 | 0 | 4.805361 | 25.6725% | [0.7512, 0.9034]; **0.7515x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A4-OL1 | 0.01 | 4.822850 | 26.2242% | [0.7628, 0.7956]; **0.7689x** | 3 | no |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A4-OL1 | 0.05 | 4.874177 | 28.4900% | [0.8201, 0.9196]; **0.9126x** | 3 | no |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A4-OL1 | 0.1 | 4.943887 | 30.5743% | [0.8983, 1.0428]; **0.9739x** | 3 | no |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A4-OL1 | 0.5 | 5.389480 | 35.5962% | [1.0165, 1.0375]; **1.0331x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A7-OL1 | 0 | 4.941206 | 23.5624% | [0.9439, 0.9972]; **0.9957x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A7-OL1 | 0.01 | 4.936396 | 24.2263% | [0.9698, 0.9939]; **0.9751x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A7-OL1 | 0.05 | 4.963656 | 26.5239% | [0.9779, 0.9939]; **0.9803x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A7-OL1 | 0.1 | 4.950079 | 28.6938% | [0.9883, 1.0043]; **1.0039x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A7-OL1 | 0.5 | 5.215925 | 40.6019% | [1.0032, 1.0150]; **1.0082x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 410M | A0 | - | 4.547456 | 0.0110% | [0.9452, 0.9471]; **0.9456x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 410M | A1-H | - | 4.651294 | 21.0497% | [0.9429, 0.9450]; **0.9448x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 410M | A4-OL1 | 0 | 5.692075 | 38.3387% | [0.9441, 0.9482]; **0.9478x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 410M | A4-OL1 | 0.01 | 5.691362 | 39.1492% | [0.9478, 0.9487]; **0.9478x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 410M | A4-OL1 | 0.05 | 5.678925 | 42.1245% | [0.9477, 0.9486]; **0.9485x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 410M | A4-OL1 | 0.1 | 5.678887 | 45.6880% | [0.9482, 0.9494]; **0.9486x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 410M | A4-OL1 | 0.5 | 5.190966 | 71.5914% | [1.0189, 1.0210]; **1.0192x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 410M | A7-OL1 | 0 | 5.426296 | 41.1215% | [0.9479, 0.9493]; **0.9488x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 410M | A7-OL1 | 0.01 | 5.433138 | 41.4781% | [0.9534, 0.9538]; **0.9537x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 410M | A7-OL1 | 0.05 | 5.467853 | 44.0532% | [0.9530, 0.9541]; **0.9535x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 410M | A7-OL1 | 0.1 | 5.490749 | 48.9976% | [0.9531, 0.9542]; **0.9537x** | 3 | yes |
| NVIDIA RTX PRO 4500 Blackwell | 410M | A7-OL1 | 0.5 | 5.120692 | 80.6155% | [1.0150, 1.0179]; **1.0175x** | 3 | yes |
| NVIDIA H100 NVL | 14M | A0 | - | 5.208573 | 0.0000% | [0.7864, 0.7911]; **0.7888x** | 2 | yes |
| NVIDIA H100 NVL | 14M | A1-H | - | 5.269634 | 2.7141% | [0.9911, 0.9918]; **0.9914x** | 2 | yes |
| NVIDIA H100 NVL | 14M | A4-OL1 | 0 | 5.458276 | 7.8100% | [1.0681, 1.0711]; **1.0696x** | 2 | yes |
| NVIDIA H100 NVL | 14M | A4-OL1 | 0.5 | 6.037982 | 12.7134% | [1.0678, 1.0698]; **1.0688x** | 2 | yes |
| NVIDIA H100 NVL | 14M | A7-OL1 | 0 | 5.480181 | 7.0542% | [1.0679, 1.0772]; **1.0725x** | 2 | yes |
| NVIDIA H100 NVL | 14M | A7-OL1 | 0.5 | 5.829407 | 27.4827% | [1.0596, 1.0655]; **1.0626x** | 2 | yes |
| NVIDIA H100 NVL | 70M | A0 | - | 4.099766 | 0.0005% | [0.9943, 0.9991]; **0.9967x** | 2 | yes |
| NVIDIA H100 NVL | 70M | A1-H | - | 4.222750 | 10.0658% | [0.9372, 0.9384]; **0.9378x** | 2 | yes |
| NVIDIA H100 NVL | 70M | A4-OL1 | 0 | 4.805361 | 25.6725% | [0.8480, 0.8599]; **0.8539x** | 2 | yes |
| NVIDIA H100 NVL | 70M | A4-OL1 | 0.5 | 5.389480 | 35.5962% | [1.0434, 1.0469]; **1.0452x** | 2 | yes |
| NVIDIA H100 NVL | 70M | A7-OL1 | 0 | 4.941206 | 23.5624% | [0.9923, 0.9925]; **0.9924x** | 2 | yes |
| NVIDIA H100 NVL | 70M | A7-OL1 | 0.5 | 5.215925 | 40.6019% | [1.0102, 1.0109]; **1.0105x** | 2 | yes |
| NVIDIA H100 NVL | 410M | A0 | - | 4.547456 | 0.0110% | [0.8706, 0.8731]; **0.8718x** | 2 | yes |
| NVIDIA H100 NVL | 410M | A1-H | - | 4.651294 | 21.0497% | [0.8792, 0.8842]; **0.8817x** | 2 | yes |
| NVIDIA H100 NVL | 410M | A4-OL1 | 0 | 5.692075 | 38.3387% | [0.9278, 0.9284]; **0.9281x** | 2 | yes |
| NVIDIA H100 NVL | 410M | A4-OL1 | 0.5 | 5.190966 | 71.5914% | [0.9238, 0.9259]; **0.9248x** | 2 | yes |
| NVIDIA H100 NVL | 410M | A7-OL1 | 0 | 5.426296 | 41.1215% | [0.9280, 0.9293]; **0.9287x** | 2 | yes |
| NVIDIA H100 NVL | 410M | A7-OL1 | 0.5 | 5.120692 | 80.6155% | [0.9326, 0.9327]; **0.9326x** | 2 | yes |

## Descriptive associations

| GPU | size | stratum | n | excluded | slope / +10 pp R_model | R2 | Spearman rho | LOO slope range |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| NVIDIA RTX PRO 4500 Blackwell | 14M | primary-qualified | 11 | 1 | +0.0347x | 0.3468 | +0.5727 | [+0.0132, +0.0959] |
| NVIDIA RTX PRO 4500 Blackwell | 14M | H100-matched-sentinels | 6 | 0 | +0.0350x | 0.4060 | +0.8286 | [+0.0147, +0.1014] |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A4-OL1-qualified | 5 | 0 | -0.0109x | 0.8031 | -0.9000 | [-0.0141, -0.0088] |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A7-OL1-qualified | 4 | 0 | +0.0039x | 0.4934 | +0.4000 | [-0.0096, +0.0056] |
| NVIDIA RTX PRO 4500 Blackwell | 70M | primary-qualified | 9 | 3 | +0.0036x | 0.0027 | +0.4667 | [-0.0040, +0.0245] |
| NVIDIA RTX PRO 4500 Blackwell | 70M | H100-matched-sentinels | 6 | 0 | +0.0013x | 0.0004 | +0.4286 | [-0.0115, +0.0245] |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A7-OL1-qualified | 5 | 0 | +0.0142x | 0.4667 | +0.7000 | [+0.0105, +0.0268] |
| NVIDIA RTX PRO 4500 Blackwell | 410M | primary-qualified | 12 | 0 | +0.0105x | 0.6358 | +0.8671 | [+0.0081, +0.0157] |
| NVIDIA RTX PRO 4500 Blackwell | 410M | H100-matched-sentinels | 6 | 0 | +0.0108x | 0.7780 | +0.8857 | [+0.0091, +0.0150] |
| NVIDIA RTX PRO 4500 Blackwell | 410M | A4-OL1-qualified | 5 | 0 | +0.0225x | 0.9608 | +1.0000 | [+0.0012, +0.0235] |
| NVIDIA RTX PRO 4500 Blackwell | 410M | A7-OL1-qualified | 5 | 0 | +0.0172x | 0.9732 | +0.7000 | [+0.0034, +0.0178] |
| NVIDIA H100 NVL | 14M | primary-qualified | 6 | 0 | +0.0649x | 0.3206 | +0.4286 | [+0.0154, +0.2037] |
| NVIDIA H100 NVL | 70M | primary-qualified | 6 | 0 | +0.0100x | 0.0511 | +0.4286 | [+0.0004, +0.0326] |
| NVIDIA H100 NVL | 410M | primary-qualified | 6 | 0 | +0.0075x | 0.6999 | +0.8286 | [+0.0058, +0.0087] |

## Matched hardware transfer

| size | family | kappa | R_model | RTX median | H100 median | H100 - RTX | both qualified |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 14M | A0 | - | 0.0000% | 0.9209x | 0.7888x | -0.1321x | yes |
| 14M | A1-H | - | 2.7141% | 0.9916x | 0.9914x | -0.0002x | yes |
| 14M | A4-OL1 | 0 | 7.8100% | 1.0538x | 1.0696x | +0.0158x | yes |
| 14M | A4-OL1 | 0.5 | 12.7134% | 1.0489x | 1.0688x | +0.0199x | yes |
| 14M | A7-OL1 | 0 | 7.0542% | 1.0489x | 1.0725x | +0.0236x | yes |
| 14M | A7-OL1 | 0.5 | 27.4827% | 1.0521x | 1.0626x | +0.0105x | yes |
| 70M | A0 | - | 0.0005% | 1.0043x | 0.9967x | -0.0076x | yes |
| 70M | A1-H | - | 10.0658% | 0.9641x | 0.9378x | -0.0264x | yes |
| 70M | A4-OL1 | 0 | 25.6725% | 0.7515x | 0.8539x | +0.1024x | yes |
| 70M | A4-OL1 | 0.5 | 35.5962% | 1.0331x | 1.0452x | +0.0121x | yes |
| 70M | A7-OL1 | 0 | 23.5624% | 0.9957x | 0.9924x | -0.0033x | yes |
| 70M | A7-OL1 | 0.5 | 40.6019% | 1.0082x | 1.0105x | +0.0024x | yes |
| 410M | A0 | - | 0.0110% | 0.9456x | 0.8718x | -0.0738x | yes |
| 410M | A1-H | - | 21.0497% | 0.9448x | 0.8817x | -0.0631x | yes |
| 410M | A4-OL1 | 0 | 38.3387% | 0.9478x | 0.9281x | -0.0197x | yes |
| 410M | A4-OL1 | 0.5 | 71.5914% | 1.0192x | 0.9248x | -0.0944x | yes |
| 410M | A7-OL1 | 0 | 41.1215% | 0.9488x | 0.9287x | -0.0201x | yes |
| 410M | A7-OL1 | 0.5 | 80.6155% | 1.0175x | 0.9326x | -0.0848x | yes |

## H100 same-R_model starting-point comparisons

| size | family | kappa | R_model | P0 speedup | winner speedup | ratio | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 14M | A0 | - | 0.0000% | 0.6295x | 0.7861x | 1.2486x | timed but at least one numerical gate failed |
| 14M | A1-H | - | 2.7141% | - | 0.9930x | - | pre-timing numerical gate stopped at least one implementation |
| 14M | A4-OL1 | 0 | 7.8100% | - | - | - | pre-timing numerical gate stopped at least one implementation |
| 14M | A4-OL1 | 0.5 | 12.7134% | 1.1511x | 1.0622x | 0.9227x | both qualified |
| 14M | A7-OL1 | 0 | 7.0542% | - | - | - | pre-timing numerical gate stopped at least one implementation |
| 14M | A7-OL1 | 0.5 | 27.4827% | - | - | - | pre-timing numerical gate stopped at least one implementation |
| 410M | A0 | - | 0.0110% | - | 0.8791x | - | pre-timing numerical gate stopped at least one implementation |
| 410M | A1-H | - | 21.0497% | - | 0.8652x | - | pre-timing numerical gate stopped at least one implementation |
| 410M | A4-OL1 | 0 | 38.3387% | - | 0.9248x | - | pre-timing numerical gate stopped at least one implementation |
| 410M | A4-OL1 | 0.5 | 71.5914% | - | 0.9196x | - | pre-timing numerical gate stopped at least one implementation |
| 410M | A7-OL1 | 0 | 41.1215% | - | 0.9294x | - | pre-timing numerical gate stopped at least one implementation |
| 410M | A7-OL1 | 0.5 | 80.6155% | - | 0.9293x | - | pre-timing numerical gate stopped at least one implementation |
| 70M | A0 | - | 0.0005% | - | 0.9986x | - | pre-timing numerical gate stopped at least one implementation |
| 70M | A1-H | - | 10.0658% | 0.7030x | 0.9307x | 1.3239x | timed but at least one numerical gate failed |
| 70M | A4-OL1 | 0 | 25.6725% | - | 0.8476x | - | pre-timing numerical gate stopped at least one implementation |
| 70M | A4-OL1 | 0.5 | 35.5962% | 1.0507x | 1.0491x | 0.9984x | timed but at least one numerical gate failed |
| 70M | A7-OL1 | 0 | 23.5624% | - | 0.9962x | - | pre-timing numerical gate stopped at least one implementation |
| 70M | A7-OL1 | 0.5 | 40.6019% | - | 1.0138x | - | pre-timing numerical gate stopped at least one implementation |

## Component probes

| GPU | size | family | kappa | component | sites | component speedup | full-policy median |
| --- | --- | --- | ---: | --- | --- | ---: | ---: |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A0 | - | attention_projection | a;z | 1.0112x | 0.9209x |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A0 | - | ffn | h;m | 0.9639x | 0.9209x |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A1-H | - | ffn | h | 1.0114x | 0.9916x |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A4-OL1 | 0 | attention_projection | a;z | 1.0236x | 1.0538x |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A4-OL1 | 0 | ffn | h;m | 1.0196x | 1.0538x |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A4-OL1 | 0.5 | attention_projection | a;z | 1.0140x | 1.0489x |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A4-OL1 | 0.5 | ffn | h;m | 1.0127x | 1.0489x |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A7-OL1 | 0 | attention_projection | a;z | 1.0137x | 1.0489x |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A7-OL1 | 0 | ffn | h;m | 1.0089x | 1.0489x |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A7-OL1 | 0.5 | attention_projection | a;z | 1.0161x | 1.0521x |
| NVIDIA RTX PRO 4500 Blackwell | 14M | A7-OL1 | 0.5 | ffn | h;m | 1.0145x | 1.0521x |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A1-H | - | ffn | h | 0.9594x | 0.9641x |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A4-OL1 | 0 | attention_projection | a;z | 1.0081x | 0.7515x |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A4-OL1 | 0 | ffn | h;m | 0.9545x | 0.7515x |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A4-OL1 | 0.5 | attention_projection | a;z | 1.0164x | 1.0331x |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A4-OL1 | 0.5 | ffn | h;m | 1.0078x | 1.0331x |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A7-OL1 | 0 | attention_projection | z | 1.0062x | 0.9957x |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A7-OL1 | 0 | ffn | h | 0.9625x | 0.9957x |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A7-OL1 | 0.5 | attention_projection | z | 1.0073x | 1.0082x |
| NVIDIA RTX PRO 4500 Blackwell | 70M | A7-OL1 | 0.5 | ffn | h | 0.9989x | 1.0082x |
| NVIDIA H100 NVL | 14M | A0 | - | attention_projection | a;z | 1.0034x | 0.7888x |
| NVIDIA H100 NVL | 14M | A0 | - | ffn | h;m | 0.8497x | 0.7888x |
| NVIDIA H100 NVL | 14M | A1-H | - | ffn | h | 0.9916x | 0.9914x |
| NVIDIA H100 NVL | 14M | A4-OL1 | 0 | attention_projection | a;z | 1.0191x | 1.0696x |
| NVIDIA H100 NVL | 14M | A4-OL1 | 0 | ffn | h;m | 1.0177x | 1.0696x |
| NVIDIA H100 NVL | 14M | A4-OL1 | 0.5 | attention_projection | a;z | 1.0215x | 1.0688x |
| NVIDIA H100 NVL | 14M | A4-OL1 | 0.5 | ffn | h;m | 1.0205x | 1.0688x |
| NVIDIA H100 NVL | 14M | A7-OL1 | 0 | attention_projection | a;z | 1.0252x | 1.0725x |
| NVIDIA H100 NVL | 14M | A7-OL1 | 0 | ffn | h;m | 1.0246x | 1.0725x |
| NVIDIA H100 NVL | 14M | A7-OL1 | 0.5 | attention_projection | a;z | 1.0254x | 1.0626x |
| NVIDIA H100 NVL | 14M | A7-OL1 | 0.5 | ffn | h;m | 1.0126x | 1.0626x |
| NVIDIA H100 NVL | 70M | A1-H | - | ffn | h | 0.9332x | 0.9378x |
| NVIDIA H100 NVL | 70M | A4-OL1 | 0 | attention_projection | a;z | 1.0236x | 0.8539x |
| NVIDIA H100 NVL | 70M | A4-OL1 | 0 | ffn | h;m | 0.9109x | 0.8539x |
| NVIDIA H100 NVL | 70M | A4-OL1 | 0.5 | attention_projection | a;z | 1.0214x | 1.0452x |
| NVIDIA H100 NVL | 70M | A4-OL1 | 0.5 | ffn | h;m | 1.0007x | 1.0452x |
| NVIDIA H100 NVL | 70M | A7-OL1 | 0 | attention_projection | z | 1.0064x | 0.9924x |
| NVIDIA H100 NVL | 70M | A7-OL1 | 0 | ffn | h | 0.9780x | 0.9924x |
| NVIDIA H100 NVL | 70M | A7-OL1 | 0.5 | attention_projection | z | 1.0066x | 1.0105x |
| NVIDIA H100 NVL | 70M | A7-OL1 | 0.5 | ffn | h | 0.9993x | 1.0105x |

## H100 CUDA-graph audit

| role | size | family | kappa | implementation | graph speedup | validation scope | graph correct | process stage |
| --- | --- | --- | ---: | --- | ---: | --- | --- | --- |
| p0 | 14M | A0 | - | p0 | 0.2964x | complete-validation | no | failed_or_incomplete |
| p0 | 14M | A1-H | - | p0 | - | development-16 | no | failed_or_incomplete |
| p0 | 14M | A4-OL1 | 0 | p0 | - | development-16 | no | failed_or_incomplete |
| p0 | 14M | A4-OL1 | 0.5 | p0 | 0.8842x | complete-validation | yes | complete |
| p0 | 14M | A7-OL1 | 0 | p0 | - | development-16 | no | failed_or_incomplete |
| p0 | 14M | A7-OL1 | 0.5 | p0 | - | development-16 | no | failed_or_incomplete |
| p0 | 410M | A0 | - | p0 | - | development-16 | no | failed_or_incomplete |
| p0 | 410M | A1-H | - | p0 | - | development-16 | no | failed_or_incomplete |
| p0 | 410M | A4-OL1 | 0 | p0 | - | development-16 | no | failed_or_incomplete |
| p0 | 410M | A4-OL1 | 0.5 | p0 | - | development-16 | no | failed_or_incomplete |
| p0 | 410M | A7-OL1 | 0 | p0 | - | development-16 | no | failed_or_incomplete |
| p0 | 410M | A7-OL1 | 0.5 | p0 | - | development-16 | no | failed_or_incomplete |
| p0 | 70M | A0 | - | p0 | - | development-16 | no | failed_or_incomplete |
| p0 | 70M | A1-H | - | p0 | 0.4457x | complete-validation | no | failed_or_incomplete |
| p0 | 70M | A4-OL1 | 0 | p0 | - | development-16 | no | failed_or_incomplete |
| p0 | 70M | A4-OL1 | 0.5 | p0 | 0.5770x | complete-validation | no | failed_or_incomplete |
| p0 | 70M | A7-OL1 | 0 | p0 | - | development-16 | no | failed_or_incomplete |
| p0 | 70M | A7-OL1 | 0.5 | p0 | - | development-16 | no | failed_or_incomplete |
| winner | 14M | A0 | - | k013 | 0.4632x | complete-validation | no | failed_or_incomplete |
| winner | 14M | A1-H | - | k013 | 0.9118x | complete-validation | no | failed_or_incomplete |
| winner | 14M | A4-OL1 | 0 | k013 | - | development-16 | no | failed_or_incomplete |
| winner | 14M | A4-OL1 | 0.5 | k013 | 0.9733x | complete-validation | yes | complete |
| winner | 14M | A7-OL1 | 0 | k013 | - | development-16 | no | failed_or_incomplete |
| winner | 14M | A7-OL1 | 0.5 | k013 | - | development-16 | no | failed_or_incomplete |
| winner | 410M | A0 | - | k010 | 0.8782x | complete-validation | yes | complete |
| winner | 410M | A1-H | - | k010 | 0.8769x | complete-validation | yes | complete |
| winner | 410M | A4-OL1 | 0 | k010 | 0.8912x | complete-validation | yes | complete |
| winner | 410M | A4-OL1 | 0.5 | k010 | 0.9870x | complete-validation | yes | complete |
| winner | 410M | A7-OL1 | 0 | k010 | 0.8880x | complete-validation | yes | complete |
| winner | 410M | A7-OL1 | 0.5 | k010 | 0.9874x | complete-validation | yes | complete |
| winner | 70M | A0 | - | k016 | 1.0003x | complete-validation | yes | complete |
| winner | 70M | A1-H | - | k016 | 0.7909x | complete-validation | yes | complete |
| winner | 70M | A4-OL1 | 0 | k016 | 0.5478x | complete-validation | yes | complete |
| winner | 70M | A4-OL1 | 0.5 | k016 | 0.8354x | complete-validation | yes | complete |
| winner | 70M | A7-OL1 | 0 | k016 | 0.7882x | complete-validation | yes | complete |
| winner | 70M | A7-OL1 | 0.5 | k016 | 0.9602x | complete-validation | yes | complete |

Timing repetitions are systems repetitions, not additional model seeds. Component effects are full-model timings with only the named linears eligible and are not additive. QK/PV attention remained dense SDPA; `attention_projection` refers only to the sparse output/projection linears. `R_model` remains the canonical logical-product opportunity of the checkpoint, not measured runtime savings.
