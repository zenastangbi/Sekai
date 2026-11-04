"""DAVIS region and boundary similarity metrics.

Copyright (c) 2020, DAVIS: Densely Annotated VIdeo Segmentation
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
Source: https://github.com/davisvideochallenge/davis2017-evaluation
Upstream file: davis2017/metrics.py
Full license: ../third_party/DAVIS_LICENSE
"""

import math

import cv2
import numpy as np


def db_eval_iou(annotation, segmentation, void_pixels=None):
    """Return region IoU for annotation and segmentation binary arrays.

    Optional void_pixels excludes pixels from intersection and union.
    Leading frame dimensions yield an array of scores.
    """
    assert annotation.shape == segmentation.shape, (
        f"Annotation({annotation.shape}) and "
        f"segmentation:{segmentation.shape} dimensions do not match."
    )
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, (
            f"Annotation({annotation.shape}) and void "
            f"pixels:{void_pixels.shape} dimensions do not match."
        )
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    inters = np.sum(
        (segmentation & annotation) & np.logical_not(void_pixels),
        axis=(-2, -1),
    )
    union = np.sum(
        (segmentation | annotation) & np.logical_not(void_pixels),
        axis=(-2, -1),
    )

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def db_eval_boundary(
    annotation, segmentation, void_pixels=None, bound_th=0.008
):
    """Return boundary F scores for 2D masks or 3D frame stacks.

    annotation and segmentation have equal shape; void_pixels marks
    excluded pixels and bound_th specifies boundary tolerance.
    """
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = (
                None
                if void_pixels is None
                else void_pixels[
                    frame_id,
                    :,
                    :,
                ]
            )
            f_res[frame_id] = f_measure(
                segmentation[
                    frame_id,
                    :,
                    :,
                ],
                annotation[frame_id, :, :],
                void_pixels_frame,
                bound_th=bound_th,
            )
    elif annotation.ndim == 2:
        f_res = f_measure(
            segmentation, annotation, void_pixels, bound_th=bound_th
        )
    else:
        raise ValueError(
            "db_eval_boundary does not support tensors with "
            f"{annotation.ndim} dimensions"
        )
    return f_res


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """Return boundary F for foreground_mask and gt_mask.

    void_pixels optionally marks excluded pixels. bound_th is a pixel
    tolerance when at least one, otherwise a fraction of image
    diagonal.
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = (
        bound_th
        if bound_th >= 1
        else np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))
    )

    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    fg_dil = cv2.dilate(
        fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8)
    )
    gt_dil = cv2.dilate(
        gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8)
    )

    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """Return a boundary map from seg at width and height.

    The boundary is offset half a pixel toward the image origin.

    David Martin <dmartin@eecs.berkeley.edu>
    January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), (
        "Cant convert %dx%d seg to %dx%d bmap." % (w, h, width, height)
    )

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap
