import numpy as np

_EPS = 1e-16
_TYPE = np.float64


def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    gt = gt > 128
    # im2double, mapminmax
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


def _get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    return min(2 * matrix.mean(), max_value)


class Emeasure(object):
    def __init__(self):
        self.adaptive_ems = []
        self.changeable_ems = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)
        self.gt_fg_numel = np.count_nonzero(gt)
        self.gt_size = gt.shape[0] * gt.shape[1]

        changeable_ems = self.cal_changeable_em(pred, gt)
        self.changeable_ems.append(changeable_ems)
        adaptive_em = self.cal_adaptive_em(pred, gt)
        self.adaptive_ems.append(adaptive_em)

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        adaptive_em = self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold)
        return adaptive_em

    def cal_changeable_em(self, pred: np.ndarray, gt: np.ndarray) -> list:
        changeable_ems = [
            self.cal_em_with_threshold(pred, gt, threshold=threshold) for threshold in np.linspace(0, 1, 256)
        ]
        return changeable_ems

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        binarized_pred = pred >= threshold

        if self.gt_fg_numel == 0:
            binarized_pred_bg_numel = np.count_nonzero(~binarized_pred)
            enhanced_matrix_sum = binarized_pred_bg_numel
        elif self.gt_fg_numel == self.gt_size:
            binarized_pred_fg_numel = np.count_nonzero(binarized_pred)
            enhanced_matrix_sum = binarized_pred_fg_numel
        else:
            enhanced_matrix_sum = self.cal_enhanced_matrix(binarized_pred, gt)
        em = enhanced_matrix_sum / (self.gt_size - 1 + _EPS)
        return em

    def cal_enhanced_matrix(self, binarized_pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        # demeaned_pred = pred - pred.mean()
        # demeaned_gt = gt - gt.mean()
        fg_fg_numel = np.count_nonzero(binarized_pred & gt)
        fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)
        pred_bg_numel = fg_fg_numel + fg_bg_numel

        # bg_fg_numel = np.count_nonzero(~binarized_pred & gt)
        bg_fg_numel = self.gt_fg_numel - fg_fg_numel
        # bg_bg_numel = np.count_nonzero(~binarized_pred & ~gt)
        bg_bg_numel = (self.gt_size - pred_bg_numel) - bg_fg_numel

        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

        mean_pred_value = pred_bg_numel / self.gt_size
        mean_gt_value = self.gt_fg_numel / self.gt_size

        demeaned_pred_fg_value = 1 - mean_pred_value
        demeaned_pred_bg_value = 0 - mean_pred_value
        demeaned_gt_fg_value = 1 - mean_gt_value
        demeaned_gt_bg_value = 0 - mean_gt_value

        combinations = [
            (demeaned_pred_fg_value, demeaned_gt_fg_value),
            (demeaned_pred_fg_value, demeaned_gt_bg_value),
            (demeaned_pred_bg_value, demeaned_gt_fg_value),
            (demeaned_pred_bg_value, demeaned_gt_bg_value)
        ]

        results_parts = []
        for part_numel, combination in zip(parts_numel, combinations):
            # align_matrix = 2 * (demeaned_gt * demeaned_pred) / (demeaned_gt ** 2 + demeaned_pred ** 2 + _EPS)
            align_matrix_value = 2 * (combination[0] * combination[1]) / \
                                 (combination[0] ** 2 + combination[1] ** 2 + _EPS)
            # enhanced_matrix = (align_matrix + 1) ** 2 / 4
            enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
            results_parts.append(enhanced_matrix_value * part_numel)

        # enhanced_matrix = enhanced_matrix.sum()
        enhanced_matrix = sum(results_parts)
        return enhanced_matrix

    def get_results(self) -> dict:
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=_TYPE))
        changeable_em = np.mean(np.array(self.changeable_ems, dtype=_TYPE), axis=0)
        return dict(em=dict(adp=adaptive_em, curve=changeable_em))
