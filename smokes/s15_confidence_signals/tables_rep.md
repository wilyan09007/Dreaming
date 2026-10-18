## S15 same-config repeat: the same signals on each independent draw of the same cheap model (target: draw's label == E1)
A and Ab are the same request twice; S sends temperature 0.7. gpt-5.4-nano ignores temperature (every draw is a T=1 sample); gpt-4.1-nano and qwen3-next honour it. b1 = returned-label logprob, b2 = gap to best rival label, e = student gap (does not depend on the draw's label except through the target), f1 = student label == draw's label, f3 = f1 then b2 as tie-break.
| draw | match E1 | AUROC b1 | b2 | e student gap | f1 | f3 | f3 minus b1 | f1 minus b1 |
|---|---|---|---|---|---|---|---|---|
| nano_A | 0.800 | 0.894 | 0.892 | 0.665 | 0.897 | 0.955 | +0.061 | +0.003 |
| nano_Ab | 0.785 | 0.907 | 0.905 | 0.690 | 0.895 | 0.966 | +0.059 | -0.012 |
| nano_S | 0.810 | 0.875 | 0.873 | 0.632 | 0.871 | 0.953 | +0.078 | -0.005 |
| nano: mean over the 3 draws, range | | | | | | | +0.066 [+0.059, +0.078] | -0.004 [-0.012, +0.003] |
| qnx_A | 0.795 | 0.788 | 0.788 | 0.651 | 0.916 | 0.946 | +0.158 | +0.128 |
| qnx_Ab | 0.795 | 0.775 | 0.781 | 0.652 | 0.919 | 0.957 | +0.183 | +0.145 |
| qnx_S | 0.790 | 0.796 | 0.789 | 0.663 | 0.920 | 0.948 | +0.152 | +0.124 |
| qnx: mean over the 3 draws, range | | | | | | | +0.164 [+0.152, +0.183] | +0.132 [+0.124, +0.145] |
| g41_A | 0.690 | 0.724 | 0.707 | 0.683 | 0.921 | 0.944 | +0.220 | +0.197 |
| g41_Ab | 0.685 | 0.735 | 0.724 | 0.671 | 0.918 | 0.940 | +0.205 | +0.183 |
| g41_S | 0.660 | 0.774 | 0.765 | 0.655 | 0.921 | 0.948 | +0.174 | +0.148 |
| g41: mean over the 3 draws, range | | | | | | | +0.200 [+0.174, +0.220] | +0.176 [+0.148, +0.197] |