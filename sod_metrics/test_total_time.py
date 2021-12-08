import os
import time

import cv2
import torch
from tqdm import tqdm

from metrics.metric_base import Emeasure as Emeasure_base
from metrics.metric_best import Emeasure as Emeasure_best
from metrics.metric_cumsumhistogram import Emeasure as Emeasure_cumsumhistogram

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cal_em_base = Emeasure_base(only_adaptive_em=False)
cal_em_best = Emeasure_best()
cal_em_cumsumhistogram = Emeasure_cumsumhistogram()

cal_ems = dict(
    base=cal_em_base,
    best=cal_em_best,
    cumsumhistogram=cal_em_cumsumhistogram,
)


def test(pred_root, mask_root, cal_em):
    mask_name_list = sorted(os.listdir(mask_root))
    tqdm_iter = tqdm(enumerate(mask_name_list), total=len(mask_name_list), leave=False)
    for i, mask_name in tqdm_iter:
        tqdm_iter.set_description(f"te=>{i + 1} ")
        mask_array = cv2.imread(os.path.join(mask_root, mask_name), cv2.IMREAD_GRAYSCALE)
        pred_array = cv2.imread(os.path.join(pred_root, mask_name), cv2.IMREAD_GRAYSCALE)
        cal_em.step(pred_array, mask_array)
    fixed_seg_results = cal_em.get_results()['em']
    return fixed_seg_results


def main():
    pred_root = 'pred_path'
    mask_root = 'mask_path'

    times = dict()
    for name, cal_em in cal_ems.items():
        start = time.time()
        seg_results = test(pred_root=pred_root, mask_root=mask_root, cal_em=cal_em)
        end = time.time()
        print('\n', seg_results['adp'], seg_results['curve'].max(), seg_results['curve'].mean())
        times[name] = end - start
    print(times)


if __name__ == '__main__':
    main()
