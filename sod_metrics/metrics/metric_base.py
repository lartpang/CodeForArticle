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
    def __init__(self, only_adaptive_em: bool = False):
        """
        only_adaptive_em: 由于计算changeable耗时较长，为了用于模型的快速验证，可以选择不计算，仅保留adaptive_em
        """
        self.adaptive_ems = []
        self.changeable_ems = None if only_adaptive_em else []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)
        self.all_fg = np.all(gt)
        self.all_bg = np.all(~gt)
        self.gt_size = gt.shape[0] * gt.shape[1]

        if self.changeable_ems is not None:
            changeable_ems = self.cal_changeable_em(pred, gt)
            self.changeable_ems.append(changeable_ems)
        adaptive_em = self.cal_adaptive_em(pred, gt)
        self.adaptive_ems.append(adaptive_em)

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        adaptive_em = self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold)
        return adaptive_em

    def cal_changeable_em(self, pred: np.ndarray, gt: np.ndarray) -> list:
        changeable_ems = [self.cal_em_with_threshold(pred, gt, threshold=th) for th in np.linspace(0, 1, 256)]
        return changeable_ems

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        binarized_pred = pred >= threshold
        if self.all_bg:
            enhanced_matrix = 1 - binarized_pred
        elif self.all_fg:
            enhanced_matrix = binarized_pred
        else:
            enhanced_matrix = self.cal_enhanced_matrix(binarized_pred, gt)
        em = enhanced_matrix.sum() / (gt.shape[0] * gt.shape[1] - 1 + _EPS)
        return em

    def cal_enhanced_matrix(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        demeaned_pred = pred - pred.mean()
        demeaned_gt = gt - gt.mean()
        align_matrix = 2 * (demeaned_gt * demeaned_pred) / (demeaned_gt ** 2 + demeaned_pred ** 2 + _EPS)
        enhanced_matrix = (align_matrix + 1) ** 2 / 4
        return enhanced_matrix

    def get_results(self) -> dict:
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=np.float64))
        if self.changeable_ems is not None:
            changeable_em = np.mean(np.array(self.changeable_ems, dtype=np.float64), axis=0)
        else:
            changeable_em = None
        return dict(em=dict(adp=adaptive_em, curve=changeable_em))
