from functools import lru_cache

import numpy as np
import logging
import matplotlib.pyplot as plt
from motion import find_transform, warp, rgb2gray


def pre_proc(raw):
    # pad_shape = list(raw.shape)
    # pad_shape[0] += 2
    # processed = np.empty(pad_shape, np.double)
    # processed[1:-1] = raw
    # processed[0], processed[-1] = raw[0], raw[-1]
    processed = raw.astype(np.double) / 255
    return processed


def post_proc(raw):
    return (raw.clip(0, 1) * 255).astype(np.uint8)


def get_diff(img1, img2, threshold=(0.05, 0.2), transform=None):
    if transform is None:
        diff = np.sum(np.abs(img1 - img2), -1)
    else:
        assert len(img1.shape) == 3
        iw = warp(img1, transform)
        mask = warp(np.ones(img1.shape[:2]), transform)
        diff = np.sum(np.abs(iw - img2), -1) * mask

    out = np.zeros_like(diff, dtype=np.uint8)
    for t_i in threshold:
        out += diff > t_i
    return out


def loss_neg(mat1, mat2, mask, transform=None, binary=False):
    # mask is negative -> mat1 should be the same as mat2
    f = lambda x: abs(x)
    if transform is not None:
        mat1 = warp(mat1, transform)
    if binary:
        loss = np.logical_xor(mat1 > 0, mat2 > 0) * mask
    else:
        loss = f(mat1 - mat2) * mask
    return loss


def loss_pos(mat1, mat2, mask, transform=None, binary=False):
    # mask is positive -> mat1/mat2 cannot be all negative
    f = lambda x: 1 - x
    if transform is not None:
        mat1 = warp(mat1, transform)
    if binary:
        loss = np.logical_and(mat1 < 0, mat2 < 0) * mask
    else:
        loss = f(mat1) * f(mat2) * mask
    return loss


def crf(mattes, strength=0.1):
    mattes[1:-1] *= 1 - strength
    mattes[:, :, 1:-1] += (mattes[:, :, 2:] + mattes[:, :, :-2]) * (strength / 8)
    mattes[:, 1:-1] += (mattes[:, 2:] + mattes[:, :-2]) * (strength / 4)
    mattes[1:-1] += (mattes[2:] + mattes[:-2]) * (strength / 4)


def process(video, matte, motion=False):
    assert video.dtype == np.uint8
    assert video.shape[:3] == matte.shape

    @lru_cache
    def multi_trans(stride):
        stride = int(stride)
        assert stride != 0
        if stride < 0:
            return [None] * (-stride) + [t._inv_matrix for t in multi_trans(-stride)]
        if stride == 1:
            return transforms
        prev_trans = multi_trans(stride - 1)
        return [pt + nt for pt, nt in zip(prev_trans[:-1], transforms[stride - 1:])]

    def diff_video(idx1, idx2):
        return get_diff(video[idx1], video[idx2], threshold,
                        transform=multi_trans(idx2 - idx1)[idx1] if motion else None)

    def optimize(stride=1):
        if not motion:
            def change(m, ref, diff_idx):
                m -= np.maximum(diff_neg[diff_idx] * (0.2 - ref), -0.5) * 1
                m += diff_pos[diff_idx] * (1 - ref) * 0.3
                np.clip(m, -1, 1, out=m)
            diff = get_diff(video[stride:], video[:-stride], threshold)
            diff_neg = diff == 0
            diff_pos = diff == 2
            for i, matte_i in enumerate(out[stride:], stride):
                change(matte_i, out[i - stride], i - stride)

            for i in range(len(out) - stride)[::-1]:
                change(out[i], out[i + stride], i)
        else:
            def change(m, ref, diff_idx):
                m -= np.maximum(diff_neg[diff_idx] * (0.2 - ref), -0.5) * 1
                m += diff_pos[diff_idx] * (1 - ref) * 0.3
                np.clip(m, -1, 1, out=m)
            idx = list(zip(range(len(video))[stride:], range(len(video))[:-stride]))
            diff = np.stack([diff_video(idx1, idx2) for idx2, idx1 in idx])
            diff_neg = diff == 0
            diff_pos = diff == 2
            for i, matte_i in enumerate(out[stride:], stride):
                change(matte_i, warp(out[i - stride], multi_trans(stride)[i - stride]), i - stride)
            diff = np.stack([diff_video(idx1, idx2) for idx1, idx2 in idx])
            diff_neg = diff == 0
            diff_pos = diff == 2
            for i in range(len(out) - stride)[::-1]:
                change(out[i], warp(out[i + stride], multi_trans(-stride)[i + stride]), i)

    video, matte = pre_proc(video), pre_proc(matte)
    threshold = [0.15, 0.3]
    if motion:
        transforms = calc_mot(video, matte)
        # out = np.stack([diff_video(i-5, i) for i in range(5, len(video))]) / 2
        # err = np.zeros_like(video)
        # return post_proc(out), post_proc(err)

    if not motion:
        diff = get_diff(video[1:], video[:-1], threshold)
        diff_neg = diff == 0
        diff_pos = diff == 2
    err = np.zeros_like(video)

    out = matte * 2 - 1

    logging.info(f"ver3")
    np.random.seed(1)
    test_choice = [np.random.choice(len(out), 2, False) for _ in range(300)]
    for it in range(10):
        if not motion:
            neg_sum = pos_sum = 0
            for i in range(40):
                f1, f2 = test_choice[i]
                diff_f12 = diff_video(f1, f2)
                neg_sum += np.sum(loss_neg(out[f1], out[f2], diff_f12 == 0, binary=True))
                pos_sum += np.sum(loss_pos(out[f1], out[f2], diff_f12 == 2, binary=True))
            logging.info(f"Error: neg {neg_sum:.1f}, pos {pos_sum:.1f}")

            logging.info(f"Start iteration {it + 1}")
            crf(out, 0.1)
            logging.debug(f"finish crf")
            optimize(stride=round(150 ** ((9 - it) / 9)))
            logging.debug(f"finish optimize")
        else:
            logging.info(f"Start iteration {it + 1}")
            crf(out, 0.15)
            logging.debug(f"finish crf")
            optimize(stride=max(round(10 - it), 1))

    if not motion:
        for i, err_i in enumerate(err[1:], 1):
            err_i[:, :, 1] += loss_neg(out[i], out[i - 1], diff_neg[i - 1], binary=True)
            err_i[:, :, 0] += loss_pos(out[i], out[i - 1], diff_pos[i - 1], binary=True)
            logging.debug(f"Frame {i} error: neg {np.sum(err_i[:, :, 1]):.1f}, pos {np.sum(err_i[:, :, 0]):.1f}")

    # for i, err_i in enumerate(err[:-1]):
    #     err_i[:, :, 1] += loss_neg(out[i], out[i + 1], diff_neg[i], binary=True)
    #     err_i[:, :, 0] += loss_pos(out[i], out[i + 1], diff_pos[i], binary=True)
    logging.info(f"Total error: neg {np.sum(err[..., 1]):.1f}, pos {np.sum(err[..., 0]):.1f}")
    out = (out + 1) / 2
    out, err = post_proc(out), post_proc(err)
    assert len(out.shape) == 3
    assert len(err.shape) == 4
    return out, err


def calc_mot(video, matte):
    rev_matte = 1 - matte
    transforms = []
    for i in range(len(video) - 1):
        logging.debug(f"finding transform {i}")
        transforms.append(find_transform(rgb2gray(video[i]), rgb2gray(video[i + 1]), rev_matte[i], rev_matte[i + 1]))
    return transforms
