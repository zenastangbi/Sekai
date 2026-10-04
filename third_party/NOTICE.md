# Third-party notices

| Material | Source | License |
|---|---|---|
| `evaluation/metrics.py` | [DAVIS evaluation toolkit](https://github.com/davisvideochallenge/davis2017-evaluation), `davis2017/metrics.py` | [BSD 3-Clause](DAVIS_LICENSE) |
| `sam-causal.patch` | [SAM 3](https://github.com/facebookresearch/sam3), revision `8f0b7f4d4e7`, with Sekai's inference adaptations | [SAM License](SAM_LICENSE) |
| Seven example annotations | Sekai-authored queries, events and target-set labels over GroundMoRe source clips | [CC BY-NC 4.0](../examples/LICENSE) |

DAVIS copyright (c) 2020, DAVIS: Densely Annotated VIdeo Segmentation. All rights reserved. The `_seg2bmap` attribution to David Martin (January 2003) appears in the source. Local changes concern formatting and documentation. The complete copyright notice, conditions and disclaimer are in `DAVIS_LICENSE`.

SAM copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. Modifications copyright (c) 2026 The Sekai Authors. The patch changes frame propagation, detector outputs and checkpoint loading in eight upstream files and adds `sam3/checkpoint_loading.py`. Patched files carry modification notices; the complete applicable terms are in `SAM_LICENSE`.

Qwen3-VL and SAM model files are external dependencies distributed under their respective model terms. GroundMoRe frames and instance masks retain their source terms. Figure 4 contains third-party video frames and annotation overlays. Credits for Figure 4 and all seven examples appear in [Video sources](../README.md#video-sources).
