# these functions are copied from caiman, as caiman does not support python 3.11.8 (most likely issue, at least)
# all functions copied from caiman/base/rois.py and caiman/motion_correction.py. Release v1.11.0
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as pl
from numpy.fft import ifftshift
from scipy.optimize import linear_sum_assignment
from typing import Any, Optional


def com(A: np.ndarray, d1: int, d2: int, d3: Optional[int] = None):
    """Calculation of the center of mass for spatial components

     Args:
         A:   np.ndarray
              matrix of spatial components (d x K)

         d1:  int
              number of pixels in x-direction

         d2:  int
              number of pixels in y-direction

         d3:  int
              number of pixels in z-direction

     Returns:
         cm:  np.ndarray
              center of mass for spatial components (K x 2 or 3)
    """

    if 'csc_matrix' not in str(type(A)):
        A = scipy.sparse.csc_matrix(A)

    if d3 is None:
        Coor = np.matrix([np.outer(np.ones(d2), np.arange(d1)).ravel(),
                          np.outer(np.arange(d2), np.ones(d1)).ravel()],
                         dtype=A.dtype)
    else:
        Coor = np.matrix([
            np.outer(np.ones(d3),
                     np.outer(np.ones(d2), np.arange(d1)).ravel()).ravel(),
            np.outer(np.ones(d3),
                     np.outer(np.arange(d2), np.ones(d1)).ravel()).ravel(),
            np.outer(np.arange(d3),
                     np.outer(np.ones(d2), np.ones(d1)).ravel()).ravel()
        ],
            dtype=A.dtype)

    cm = (Coor * A / A.sum(axis=0)).T
    return np.array(cm)


def register_ROIs(A1,
                  A2,
                  dims,
                  template1=None,
                  template2=None,
                  align_flag=True,
                  D=None,
                  max_thr=0,
                  use_opt_flow=True,
                  thresh_cost=.7,
                  max_dist=10,
                  enclosed_thr=None,
                  print_assignment=False,
                  plot_results=False,
                  Cn=None,
                  cmap='viridis'):
    """
    Register ROIs across different sessions using an intersection over union 
    metric and the Hungarian algorithm for optimal matching

    Args:
        A1: ndarray or csc_matrix  # pixels x # of components
            ROIs from session 1

        A2: ndarray or csc_matrix  # pixels x # of components
            ROIs from session 2

        dims: list or tuple
            dimensionality of the FOV

        template1: ndarray dims
            template from session 1

        template2: ndarray dims
            template from session 2

        align_flag: bool
            align the templates before matching

        D: ndarray
            matrix of distances in the event they are pre-computed

        max_thr: scalar
            max threshold parameter before binarization    

        use_opt_flow: bool
            use dense optical flow to align templates

        thresh_cost: scalar
            maximum distance considered

        max_dist: scalar
            max distance between centroids

        enclosed_thr: float
            if not None set distance to at most the specified value when ground 
            truth is a subset of inferred

        print_assignment: bool
            print pairs of matched ROIs

        plot_results: bool
            create a plot of matches and mismatches

        Cn: ndarray
            background image for plotting purposes

        cmap: string
            colormap for background image

    Returns:
        matched_ROIs1: list
            indices of matched ROIs from session 1

        matched_ROIs2: list
            indices of matched ROIs from session 2

        non_matched1: list
            indices of non-matched ROIs from session 1

        non_matched2: list
            indices of non-matched ROIs from session 2

        performance:  list
            (precision, recall, accuracy, f_1 score) with A1 taken as ground truth

        A2: csc_matrix  # pixels x # of components
            ROIs from session 2 aligned to session 1

    """

    #    if 'csc_matrix' not in str(type(A1)):
    #        A1 = scipy.sparse.csc_matrix(A1)
    #    if 'csc_matrix' not in str(type(A2)):
    #        A2 = scipy.sparse.csc_matrix(A2)

    if 'ndarray' not in str(type(A1)):
        A1 = A1.toarray()
    if 'ndarray' not in str(type(A2)):
        A2 = A2.toarray()

    if template1 is None or template2 is None:
        align_flag = False

    x_grid, y_grid = np.meshgrid(np.arange(0., dims[1]).astype(
        np.float32), np.arange(0., dims[0]).astype(np.float32))

    if align_flag:     # first align ROIs from session 2 to the template from session 1
        template1 -= np.min(template1)
        template1 /= np.max(template1)
        template2 -= np.min(template2)
        template2 /= np.max(template2)

        if use_opt_flow:
            template1_norm = np.uint8(template1 * (template1 > 0) * 255)
            template2_norm = np.uint8(template2 * (template2 > 0) * 255)
            flow = cv2.calcOpticalFlowFarneback(np.uint8(template1_norm * 255), np.uint8(template2_norm * 255), None,
                                                0.5, 3, 128, 3, 7, 1.5, 0)
            x_remap = (flow[:, :, 0] + x_grid).astype(np.float32)
            y_remap = (flow[:, :, 1] + y_grid).astype(np.float32)

        else:
            raise Exception(
                "use_opt_flow=False not implemented. Would have to install caiman... Good luck with that.")

        A_2t = np.reshape(A2, dims + (-1,), order='F').transpose(2, 0, 1)
        A2 = np.stack([cv2.remap(img.astype(np.float32), x_remap,
                      y_remap, cv2.INTER_NEAREST) for img in A_2t], axis=0)
        A2 = np.reshape(A2.transpose(1, 2, 0),
                        (A1.shape[0], A_2t.shape[0]), order='F')

    A1 = np.stack([a * (a > max_thr * a.max()) for a in A1.T]).T
    A2 = np.stack([a * (a > max_thr * a.max()) for a in A2.T]).T

    if D is None:
        if 'csc_matrix' not in str(type(A1)):
            A1 = scipy.sparse.csc_matrix(A1)
        if 'csc_matrix' not in str(type(A2)):
            A2 = scipy.sparse.csc_matrix(A2)

        cm_1 = com(A1, *dims)
        cm_2 = com(A2, *dims)
        A1_tr = (A1 > 0).astype(float)
        A2_tr = (A2 > 0).astype(float)
        D = distance_masks([A1_tr, A2_tr], [cm_1, cm_2],
                           max_dist, enclosed_thr=enclosed_thr)

    matches, costs = find_matches(D, print_assignment=print_assignment)
    matches = matches[0]
    costs = costs[0]

    # store indices

    idx_tp = np.where(np.array(costs) < thresh_cost)[0]
    if len(idx_tp) > 0:
        matched_ROIs1 = matches[0][idx_tp]     # ground truth
        matched_ROIs2 = matches[1][idx_tp]     # algorithm - comp
        non_matched1 = np.setdiff1d(
            list(range(D[0].shape[0])), matches[0][idx_tp])
        non_matched2 = np.setdiff1d(
            list(range(D[0].shape[1])), matches[1][idx_tp])
        TP = np.sum(np.array(costs) < thresh_cost) * 1.
    else:
        TP = 0.
        plot_results = False
        matched_ROIs1 = []
        matched_ROIs2 = []
        non_matched1 = list(range(D[0].shape[0]))
        non_matched2 = list(range(D[0].shape[1]))

    # compute precision and recall

    FN = D[0].shape[0] - TP
    FP = D[0].shape[1] - TP
    TN = 0

    performance = dict()
    performance['recall'] = TP / (TP + FN)
    performance['precision'] = TP / (TP + FP)
    performance['accuracy'] = (TP + TN) / (TP + FP + FN + TN)
    performance['f1_score'] = 2 * TP / (2 * TP + FP + FN)

    if plot_results:
        if Cn is None:
            if template1 is not None:
                Cn = template1
            elif template2 is not None:
                Cn = template2
            else:
                Cn = np.reshape(A1.sum(1) + A2.sum(1), dims, order='F')

        masks_1 = np.reshape(A1.toarray(), dims + (-1,),
                             order='F').transpose(2, 0, 1)
        masks_2 = np.reshape(A2.toarray(), dims + (-1,),
                             order='F').transpose(2, 0, 1)
        #        try : #Plotting function
        level = 0.98
        pl.rcParams['pdf.fonttype'] = 42
        font = {'family': 'Myriad Pro', 'weight': 'regular', 'size': 10}
        pl.rc('font', **font)
        lp, hp = np.nanpercentile(Cn, [5, 95])
        pl.figure(figsize=(20, 10))
        pl.subplot(1, 2, 1)
        pl.imshow(Cn, vmin=lp, vmax=hp, cmap=cmap)
        [pl.contour(norm_nrg(mm), levels=[level], colors='w', linewidths=1)
         for mm in masks_1[matched_ROIs1]]
        [pl.contour(norm_nrg(mm), levels=[level], colors='r', linewidths=1)
         for mm in masks_2[matched_ROIs2]]
        pl.title('Matches')
        pl.axis('off')
        pl.subplot(1, 2, 2)
        pl.imshow(Cn, vmin=lp, vmax=hp, cmap=cmap)
        [pl.contour(norm_nrg(mm), levels=[level], colors='w', linewidths=1)
         for mm in masks_1[non_matched1]]
        [pl.contour(norm_nrg(mm), levels=[level], colors='r', linewidths=1)
         for mm in masks_2[non_matched2]]
        pl.title('Mismatches')
        pl.axis('off')

    return matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performance, A2


def register_multisession(A,
                          dims,
                          templates=[None],
                          align_flag=True,
                          max_thr=0,
                          use_opt_flow=True,
                          thresh_cost=.7,
                          max_dist=10,
                          enclosed_thr=None):
    """
    Register ROIs across multiple sessions using an intersection over union metric
    and the Hungarian algorithm for optimal matching. Registration occurs by 
    aligning session 1 to session 2, keeping the union of the matched and 
    non-matched components to register with session 3 and so on.

    Args:
        A: list of ndarray or csc_matrix matrices # pixels x # of components
           ROIs from each session

        dims: list or tuple
            dimensionality of the FOV

        template: list of ndarray matrices of size dims
            templates from each session

        align_flag: bool
            align the templates before matching

        max_thr: scalar
            max threshold parameter before binarization    

        use_opt_flow: bool
            use dense optical flow to align templates

        thresh_cost: scalar
            maximum distance considered

        max_dist: scalar
            max distance between centroids

        enclosed_thr: float
            if not None set distance to at most the specified value when ground 
            truth is a subset of inferred

    Returns:
        A_union: csc_matrix # pixels x # of total distinct components
            union of all kept ROIs 

        assignments: ndarray int of size # of total distinct components x # sessions
            element [i,j] = k if component k from session j is mapped to component
            i in the A_union matrix. If there is no much the value is NaN

        matchings: list of lists
            matchings[i][j] = k means that component j from session i is represented
            by component k in A_union

    """

    n_sessions = len(A)
    templates = list(templates)
    if len(templates) == 1:
        templates = n_sessions * templates

    if n_sessions <= 1:
        raise Exception('number of sessions must be greater than 1')

    A = [a.toarray() if 'ndarray' not in str(type(a)) else a for a in A]

    A_union = A[0].copy()
    matchings = []
    matchings.append(list(range(A_union.shape[-1])))

    for sess in range(1, n_sessions):
        reg_results = register_ROIs(A[sess],
                                    A_union,
                                    dims,
                                    template1=templates[sess],
                                    template2=templates[sess - 1],
                                    align_flag=align_flag,
                                    max_thr=max_thr,
                                    use_opt_flow=use_opt_flow,
                                    thresh_cost=thresh_cost,
                                    max_dist=max_dist,
                                    enclosed_thr=enclosed_thr)

        mat_sess, mat_un, nm_sess, nm_un, _, A2 = reg_results
        A_union = A2.copy()
        A_union[:, mat_un] = A[sess][:, mat_sess]
        A_union = np.concatenate(
            (A_union.toarray(), A[sess][:, nm_sess]), axis=1)
        new_match = np.zeros(A[sess].shape[-1], dtype=int)
        new_match[mat_sess] = mat_un
        new_match[nm_sess] = range(A2.shape[-1], A_union.shape[-1])
        matchings.append(new_match.tolist())

    assignments = np.empty((A_union.shape[-1], n_sessions)) * np.nan
    for sess in range(n_sessions):
        assignments[matchings[sess], sess] = range(len(matchings[sess]))

    return A_union, assignments, matchings


def distance_masks(M_s: list, cm_s: list[list], max_dist: float, enclosed_thr: Optional[float] = None) -> list:
    """
    Compute distance matrix based on an intersection over union metric. Matrix are compared in order,
    with matrix i compared with matrix i+1

    Args:
        M_s: tuples of 1-D arrays
            The thresholded A matrices (masks) to compare, output of threshold_components

        cm_s: list of list of 2-ples
            the centroids of the components in each M_s

        max_dist: float
            maximum distance among centroids allowed between components. This corresponds to a distance
            at which two components are surely disjoined

        enclosed_thr: float
            if not None set distance to at most the specified value when ground truth is a subset of inferred

    Returns:
        D_s: list of matrix distances

    Raises:
        Exception: 'Nan value produced. Error in inputs'

    """
    D_s = []

    for gt_comp, test_comp, cmgt_comp, cmtest_comp in zip(M_s[:-1], M_s[1:], cm_s[:-1], cm_s[1:]):

        # todo : better with a function that calls itself
        # not to interfere with M_s
        gt_comp = gt_comp.copy()[:, :]
        test_comp = test_comp.copy()[:, :]

        # the number of components for each
        nb_gt = np.shape(gt_comp)[-1]
        nb_test = np.shape(test_comp)[-1]
        D = np.ones((nb_gt, nb_test))

        cmgt_comp = np.array(cmgt_comp)
        cmtest_comp = np.array(cmtest_comp)
        if enclosed_thr is not None:
            gt_val = gt_comp.T.dot(gt_comp).diagonal()
        for i in range(nb_gt):
            # for each components of gt
            k = gt_comp[:, np.repeat(i, nb_test)] + test_comp
            # k is correlation matrix of this neuron to every other of the test
            for j in range(nb_test):   # for each components on the tests
                dist = np.linalg.norm(cmgt_comp[i] - cmtest_comp[j])
                # we compute the distance of this one to the other ones
                if dist < max_dist:
                    # union matrix of the i-th neuron to the jth one
                    union = k[:, j].sum()
                    # we could have used OR for union and AND for intersection while converting
                    # the matrice into real boolean before

                    # product of the two elements' matrices
                    # we multiply the boolean values from the jth omponent to the ith
                    intersection = np.array(gt_comp[:, i].T.dot(
                        test_comp[:, j]).todense()).squeeze()

                    # if we don't have even a union this is pointless
                    if union > 0:

                        # intersection is removed from union since union contains twice the overlapping area
                        # having the values in this format 0-1 is helpful for the hungarian algorithm that follows
                        D[i, j] = 1 - 1. * intersection / \
                            (union - intersection)
                        if enclosed_thr is not None:
                            if intersection == gt_val[j] or intersection == gt_val[i]:
                                D[i, j] = min(D[i, j], 0.5)
                    else:
                        D[i, j] = 1.

                    if np.isnan(D[i, j]):
                        raise Exception('Nan value produced. Error in inputs')
                else:
                    D[i, j] = 1

        D_s.append(D)
    return D_s


def apply_shifts_dft(src_freq, shifts, diffphase, is_freq=True, border_nan=True):
    """
    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Args:
        apply shifts using inverse dft
        src_freq: ndarray
            if is_freq it is fourier transform image else original image
        shifts: shifts to apply
        diffphase: comes from the register_translation output
    """

    is3D = len(src_freq.shape) == 3
    if not is_freq:
        if is3D:
            src_freq = np.fft.fftn(src_freq)
        else:
            src_freq = np.dstack([np.real(src_freq), np.imag(src_freq)])
            src_freq = cv2.dft(
                src_freq, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
            src_freq = src_freq[:, :, 0] + 1j * src_freq[:, :, 1]
            src_freq = np.array(src_freq, dtype=np.complex128, copy=False)

    if not is3D:
        nr, nc = np.shape(src_freq)
        Nr = ifftshift(np.arange(-np.fix(nr/2.), np.ceil(nr/2.)))
        Nc = ifftshift(np.arange(-np.fix(nc/2.), np.ceil(nc/2.)))
        Nc, Nr = np.meshgrid(Nc, Nr)
        Greg = src_freq * np.exp(1j * 2 * np.pi *
                                 (-shifts[0] * Nr / nr - shifts[1] * Nc / nc))
    else:
        nr, nc, nd = np.array(np.shape(src_freq), dtype=float)
        Nr = ifftshift(np.arange(-np.fix(nr / 2.), np.ceil(nr / 2.)))
        Nc = ifftshift(np.arange(-np.fix(nc / 2.), np.ceil(nc / 2.)))
        Nd = ifftshift(np.arange(-np.fix(nd / 2.), np.ceil(nd / 2.)))
        Nc, Nr, Nd = np.meshgrid(Nc, Nr, Nd)
        Greg = src_freq * np.exp(1j * 2 * np.pi *
                                 (-shifts[0] * Nr / nr - shifts[1] * Nc / nc -
                                  shifts[2] * Nd / nd))

    Greg = Greg.dot(np.exp(1j * diffphase))
    if is3D:
        new_img = np.real(np.fft.ifftn(Greg))
    else:
        Greg = np.dstack([np.real(Greg), np.imag(Greg)])
        new_img = cv2.idft(Greg)[:, :, 0]

    if border_nan is not False:
        max_w, max_h, min_w, min_h = 0, 0, 0, 0
        max_h, max_w = np.ceil(np.maximum(
            (max_h, max_w), shifts[:2])).astype(int)
        min_h, min_w = np.floor(np.minimum(
            (min_h, min_w), shifts[:2])).astype(int)
        if is3D:
            max_d = np.ceil(np.maximum(0, shifts[2])).astype(int)
            min_d = np.floor(np.minimum(0, shifts[2])).astype(int)
        if border_nan is True:
            new_img[:max_h, :] = np.nan
            if min_h < 0:
                new_img[min_h:, :] = np.nan
            new_img[:, :max_w] = np.nan
            if min_w < 0:
                new_img[:, min_w:] = np.nan
            if is3D:
                new_img[:, :, :max_d] = np.nan
                if min_d < 0:
                    new_img[:, :, min_d:] = np.nan
        elif border_nan == 'min':
            min_ = np.nanmin(new_img)
            new_img[:max_h, :] = min_
            if min_h < 0:
                new_img[min_h:, :] = min_
            new_img[:, :max_w] = min_
            if min_w < 0:
                new_img[:, min_w:] = min_
            if is3D:
                new_img[:, :, :max_d] = min_
                if min_d < 0:
                    new_img[:, :, min_d:] = min_
        elif border_nan == 'copy':
            new_img[:max_h] = new_img[max_h]
            if min_h < 0:
                new_img[min_h:] = new_img[min_h-1]
            if max_w > 0:
                new_img[:, :max_w] = new_img[:, max_w, np.newaxis]
            if min_w < 0:
                new_img[:, min_w:] = new_img[:, min_w-1, np.newaxis]
            if is3D:
                if max_d > 0:
                    new_img[:, :, :max_d] = new_img[:, :, max_d, np.newaxis]
                if min_d < 0:
                    new_img[:, :, min_d:] = new_img[:, :, min_d-1, np.newaxis]

    return new_img


def find_matches(D_s, print_assignment: bool = False) -> tuple[list, list]:
    # todo todocument

    matches = []
    costs = []
    for ii, D in enumerate(D_s):
        # we make a copy not to set changes in the original
        DD = D.copy()
        if np.sum(np.where(np.isnan(DD))) > 0:
            raise Exception('Distance Matrix contains invalid value NaN')

        # we do the hungarian
        indexes = linear_sum_assignment(DD)
        indexes2 = [(ind1, ind2) for ind1, ind2 in zip(indexes[0], indexes[1])]
        matches.append(indexes)
        DD = D.copy()
        total = []
        # we want to extract those information from the hungarian algo
        for row, column in indexes2:
            value = DD[row, column]
            total.append(value)
        costs.append(total)
        # send back the results in the format we want
    return matches, costs


def norm_nrg(a_):
    a = a_.copy()
    dims = a.shape
    a = a.reshape(-1, order='F')
    indx = np.argsort(a, axis=None)[::-1]
    cumEn = np.cumsum(a.flatten()[indx]**2)
    cumEn /= cumEn[-1]
    a = np.zeros(np.prod(dims))
    a[indx] = cumEn
    return a.reshape(dims, order='F')
