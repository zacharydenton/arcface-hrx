# ArcFace head experiments — 2026-09-17

No head change was retained. [Diagnostic timings](head-experiments-2026-09-17.json).

A single-face GEMV prototype avoided mostly empty WMMA rows in the final
25,088-to-512 projection. It saved only about 1–2% end-to-end in paired runs,
and changed reduction order (maximum embedding difference about 2.5e-6,
cosine similarity about 0.9999999999992). Although numerically very close,
this broke the existing exact single-face/batch equality contract. The
prototype was removed; test tolerances were not relaxed.

A split-K sweep using identical frozen HRX 0.7.0 sources tested 7, 14, 28, 49
and 56 splits for batch 1 (200 samples), and 7, 14, 28 and 56 for batch 6
(150 samples). The existing 28-split path was approximately 2.38–2.40 ms for
one face and 4.32–4.39 ms for six. Other split counts did not establish an
improvement. Both the experimental environment override and the alternate
kernel were removed. Concurrent work in other ArcFace files was preserved.

Measurements use synchronized warm host timing on gfx1151, including I/O;
other GPU workloads remained running. These results do not rule out gains
from fusing alignment and normalization or improving convolution kernels.
