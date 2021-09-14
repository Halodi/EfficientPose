from textwrap import fill
import numpy as np
from typing import Iterable, Union, Dict



def kabsch(reference_frame_points:np.ndarray, body_frame_points:np.ndarray) -> np.ndarray:
    """Kabsch algorithm for estimating an optimal rotation matrix between two correlated 3D point sets

    Args:
        reference_frame_points: nx3 matrix or iterable of point measurements (0 mean) in reference frame
        body_frame_points: nx3 matrix or iterable of point measurements (0 mean) in body frame

    Returns:
        - 3x3 rotation matrix (reference frame to body frame) if n>2, else None
    """

    n_ = len(reference_frame_points)
    if len(body_frame_points) != n_ or n_ <= 2: return None

    X_ = np.reshape(np.asarray(reference_frame_points), (n_,3,1))
    Y_ = np.reshape(np.asarray(body_frame_points), (n_,1,3))
    C_ = np.sum(np.matmul(X_, Y_), axis=0)

    U_, S_, Vt_ = np.linalg.svd(C_)
    D_ = np.diag(( 1.0, 1.0, np.sign(np.linalg.det(np.matmul(Vt_.T, U_.T))) ))
    R_ = np.matmul(np.matmul(Vt_.T, D_), U_.T).T

    return R_



FILTER_UNIT_VECTORS_RHS = np.eye(3).reshape(3,3,1)
FILTER_UNIT_VECTORS = np.eye(3)

def rotate_filter_unit_vectors(m:np.ndarray) -> np.ndarray:
    return np.matmul(m, FILTER_UNIT_VECTORS_RHS).reshape(len(FILTER_UNIT_VECTORS_RHS),3)

def rotational_mean(M:Iterable[np.ndarray]) -> np.ndarray:
    transformed_unit_vectors_ = np.asarray([ rotate_filter_unit_vectors(m) for m in M[:,:3,:3] ])
    tuv_mean_pos_ = np.mean(transformed_unit_vectors_, axis=0)
    return kabsch(tuv_mean_pos_, FILTER_UNIT_VECTORS)



class Filter:
    def __init__(self, args:dict):
        self._window_shape, self._min_count = ( args['width'], 4, 4 ), args['min_count']
        self._radius_sq = args['translation_radius_threshold'] * args['translation_radius_threshold']
        self.clear()

    @property
    def rejects_counter(self): return self._rejects_counter

    def clear(self):
        self._window, self._wI = np.zeros(self._window_shape), 0
        self._clear_rejects_counter()

    def _clear_rejects_counter(self): self._rejects_counter = 0
    def increment_rejects_counter(self): self._rejects_counter += 1

    def step(self, m:np.ndarray) -> bool:
        window_ = self._get_populated_window()
        if len(window_):
            r2_ = np.sum(np.square(m[:3,3] - self._translational_mean(window_)))
            if r2_ > self._radius_sq: return False

        self._window[self._wI] = np.copy(m)
        self._clear_rejects_counter()

        self._wI += 1
        if self._wI == self._window.shape[0]: self._wI = 0

        return True

    def mean(self) -> Union[np.ndarray,None]:
        window_ = self._get_populated_window()
        if len(window_) < self._min_count: return None
        elif len(window_) == 1: return window_[0]
        else:
            out_ = np.empty((4,4)); out_[3] = ( 0, 0, 0, 1 )
            out_[:3,:3], out_[:3,3] = rotational_mean(window_), self._translational_mean(window_)
            return out_

    def _get_populated_window(self) -> np.ndarray:
        return self._window[self._window[:,3,3] == 1]

    def _translational_mean(self, M:np.ndarray) -> np.ndarray:
        return np.mean(M[:,:3,3], axis=0)



class FilterMulti:
    def __init__(self, args:dict):
        self._filters = [ Filter(args) for i in range(args['max_instances_per_label']) ]
        self._max_rejects = args['clear_after_rejects']

    def step(self, M:Iterable[np.ndarray]) -> Iterable[np.ndarray]:
        if len(M):
            m_claimed_ = [ False ] * len(M)
            for filter in self._filters:
                filter_claimed_any_ = False
                for i in range(len(M)):
                    if not m_claimed_[i] and filter.step(M[i]):
                        filter_claimed_any_ = m_claimed_[i] = True

                if not filter_claimed_any_:
                    filter.increment_rejects_counter()
                    if filter.rejects_counter == self._max_rejects: filter.clear()

        filtered_poses_ = []
        for filter in self._filters:
            mean_ = filter.mean()
            if mean_ is not None:
                filtered_poses_.append(mean_)

        return filtered_poses_



class FilterMultiLabels:
    def __init__(self, args:dict, labels:Iterable[str]):
        self._filters = { label:FilterMulti(args) for label in labels }

    def __getitem__(self, label:str): return self._filters.get(label)

    def step(self, M_dict:Dict[str,Iterable[np.ndarray]]) -> Dict[str,Iterable[np.ndarray]]:
        out_ = {}
        for label, M in M_dict.items():
            filter_ = self[label]
            if filter_ is not None:
                out_[label] = filter_.step(M)

        return out_
