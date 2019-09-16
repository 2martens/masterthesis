# -*- coding: utf-8 -*-

#   Copyright 2018 Timon Brüning, Inga Kempfert, Anne Kunstmann, Jim Martens,
#                  Marius Pierenkemper, Yanneck Reiss
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
Functionality to evaluate results of networks.

Functions:
    get_number_gt_per_class(...): calculates the number of ground truth boxes per class
    prepare-predictions(...): prepares the predictions for further processing
    match_predictions(...): matches predictions to ground truth boxes
    get_precision_recall(...): computes the precision and recall values and returns them
    get_f1_score(...): computes the F1 score for every class
    get_mean_average_precisions(...): computes the mean average precision for each class and returns them
    get_mean_average_precision(...): computes the mean average precision over all classes and returns it
"""
from typing import Sequence, Union, Tuple, List

import numpy as np


def get_number_gt_per_class(labels: Sequence[Sequence[Sequence[int]]],
                            nr_classes: int) -> np.ndarray:
    """
    Calculates the number of ground truth boxes per class and returns result.
    
    Args:
        labels: list of labels per image
        nr_classes: number of classes

    Returns:
        numpy array with respective counts
    """
    number_gt_per_class = np.zeros(shape=(nr_classes + 1), dtype=np.int)
    label_range = range(len(labels))
    
    # iterate over images
    for i in label_range:
        boxes = np.asarray(labels[i])
        
        # iterate over boxes in image
        for j in range(boxes.shape[0]):
            class_id = int(boxes[j, 0])
            if class_id > nr_classes:
                continue
            number_gt_per_class[class_id] += 1
    
    return number_gt_per_class


def prepare_predictions(predictions: Sequence[Sequence[Sequence[Union[int, float]]]],
                        nr_classes: int) -> \
        List[List[Tuple[int, float, int, int, int, int]]]:
    """
    Prepares the predictions for further processing.
    
    Args:
        predictions: list of predictions per image
        nr_classes: number of classes

    Returns:
        list of predictions per class
    """
    results = [list() for _ in range(nr_classes + 1)]
    # index positions for bounding box coordinates
    xmin = 2
    ymin = 3
    xmax = 4
    ymax = 5
    
    for i, batch_item in enumerate(predictions):
        image_id = i
        
        for box in batch_item:
            if len(box) == 7:
                # entropy is in box list
                xmin += 1
                ymin += 1
                xmax += 1
                ymax += 1
            
            if len(box) > nr_classes:
                class_id = np.argmax(box[:-5])
                confidence = np.amax(box[:-5])
                xmin = -5
                ymin = -4
                xmax = -3
                ymax = -2
            else:
                class_id = int(box[0])
                # Round the box coordinates to reduce the required memory.
                confidence = box[1]
                
            xmin_value = round(box[xmin])
            ymin_value = round(box[ymin])
            xmax_value = round(box[xmax])
            ymax_value = round(box[ymax])
            prediction = (image_id, confidence, xmin_value, ymin_value, xmax_value, ymax_value)
            # Append the predicted box to the results list for its class.
            results[class_id].append(prediction)
    
    return results


def match_predictions(predictions: Sequence[Sequence[Tuple[int, float, int, int, int, int]]],
                      labels: Sequence[Sequence[Sequence[int]]],
                      iou_func: callable,
                      nr_classes: int,
                      iou_threshold: float,
                      border_pixels: str = "include",
                      sorting_algorithm: str = "quicksort") -> Tuple[List[np.ndarray], List[np.ndarray],
                                                                     List[np.ndarray], List[np.ndarray],
                                                                     np.ndarray, np.ndarray,
                                                                     np.ndarray, np.ndarray]:
    """
    Matches predictions to ground truth boxes.
    
    Args:
        predictions: list of predictions
        labels: list of labels per image
        iou_func: function to calculate the intersection over union
        nr_classes: number of classes
        iou_threshold: only matches higher than this value will be considered
        border_pixels:  How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxes, but not the other.
        sorting_algorithm: Which sorting algorithm the matching algorithm should use. This
            argument accepts any valid sorting algorithm for Numpy's `argsort()` function.
            You will usually want to choose between 'quicksort' (fastest and most memory efficient,
            but not stable) and 'mergesort' (slight slower and less memory efficient, but stable).
            The official Matlab evaluation algorithm uses a stable sorting algorithm, so this algorithm
            is only guaranteed to behave identically if you choose 'mergesort' as the sorting algorithm,
            but it will almost always behave identically even if you choose 'quicksort' (but no guarantees).

    Returns:
        true positives, false positives, cumulative true positives, and cumulative false positives for
            each class, open set error as defined by Miller et al, cumulative open set error,
            cumulative true positives and cumulative false positives over all classes
    """
    true_positives = [[]]  # The false positives for each class, sorted by descending confidence.
    false_positives = [[]]  # The true positives for each class, sorted by descending confidence.
    cumulative_true_positives = [[]]
    cumulative_false_positives = [[]]
    most_predictions = -1

    for class_id in range(1, nr_classes + 1):
        nr_predictions = len(predictions[class_id])
        if nr_predictions > most_predictions:
            most_predictions = nr_predictions
    
    open_set_error = np.zeros(most_predictions, dtype=np.int)
    true_positives_micro = np.zeros(most_predictions, dtype=np.int)
    false_positives_micro = np.zeros(most_predictions, dtype=np.int)

    for class_id in range(1, nr_classes + 1):
        predictions_class = predictions[class_id]
        
        # Store the matching results in these lists:
        true_pos = np.zeros(len(predictions_class),
                            dtype=np.int)  # 1 for every prediction that is a true positive, 0 otherwise
        false_pos = np.zeros(len(predictions_class),
                             dtype=np.int)  # 1 for every prediction that is a false positive, 0 otherwise

        # In case there are no predictions at all for this class, we're done here.
        if len(predictions_class) == 0:
            true_positives.append(true_pos)
            false_positives.append(false_pos)
            cumulative_true_pos = np.cumsum(true_pos)  # Cumulative sums of the true positives
            cumulative_false_pos = np.cumsum(false_pos)  # Cumulative sums of the false positives
            cumulative_true_positives.append(cumulative_true_pos)
            cumulative_false_positives.append(cumulative_false_pos)
            continue

        # Convert the predictions list for this class into a structured array so that we can sort it by confidence.

        # Create the data type for the structured array.
        preds_data_type = np.dtype([('image_id', np.int32),
                                    ('confidence', 'f4'),
                                    ('xmin', 'f4'),
                                    ('ymin', 'f4'),
                                    ('xmax', 'f4'),
                                    ('ymax', 'f4')])
        # Create the structured array
        predictions_class = np.array(predictions_class, dtype=preds_data_type)
        # Sort the detections by decreasing confidence.
        descending_indices = np.argsort(-predictions_class['confidence'], kind=sorting_algorithm)
        predictions_sorted = predictions_class[descending_indices]

        # Keep track of which ground truth boxes were already matched to a detection.
        gt_matched = {}
        
        for i in range(len(predictions_class)):
            prediction = predictions_sorted[i]
            image_id = prediction['image_id']
            # Convert the structured array element to a regular array
            pred_box = np.asarray(list(prediction[['xmin', 'ymin', 'xmax', 'ymax']]))

            # Get the relevant ground truth boxes for this prediction,
            # i.e. all ground truth boxes that match the prediction's
            # image ID and class ID.

            gt = labels[image_id]
            gt = np.asarray(gt)
            class_mask = gt[:, 0] == class_id
            gt = gt[class_mask]

            if gt.size == 0:
                # If the image doesn't contain any objects of this class,
                # the prediction becomes a false positive.
                false_pos[i] = 1
                false_positives_micro[i] += 1
                open_set_error[i] += 1
                continue

            # Compute the IoU of this prediction with all ground truth boxes of the same class.
            overlaps = iou_func(boxes1=gt[:, [1, 2, 3, 4]],
                                boxes2=pred_box,
                                coords='corners',
                                mode='element-wise',
                                border_pixels=border_pixels)

            # For each detection, match the ground truth box with the highest overlap.
            # It's possible that the same ground truth box will be matched to multiple
            # detections.
            gt_match_index = np.argmax(overlaps)
            gt_match_overlap = overlaps[gt_match_index]

            if gt_match_overlap < iou_threshold:
                # False positive, IoU threshold violated:
                # Those predictions whose matched overlap is below the threshold become
                # false positives.
                false_pos[i] = 1
                false_positives_micro[i] += 1
            else:
                if image_id not in gt_matched:
                    # True positive:
                    # If the matched ground truth box for this prediction hasn't been matched to a
                    # different prediction already, we have a true positive.
                    true_pos[i] = 1
                    true_positives_micro[i] += 1
                    gt_matched[image_id] = np.zeros(shape=(gt.shape[0]), dtype=np.bool)
                    gt_matched[image_id][gt_match_index] = True
                elif not gt_matched[image_id][gt_match_index]:
                    # True positive:
                    # If the matched ground truth box for this prediction hasn't been matched to a
                    # different prediction already, we have a true positive.
                    true_pos[i] = 1
                    true_positives_micro[i] += 1
                    gt_matched[image_id][gt_match_index] = True
                else:
                    # False positive, duplicate detection:
                    # If the matched ground truth box for this prediction has already been matched
                    # to a different prediction previously, it is a duplicate detection for an
                    # already detected object, which counts as a false positive.
                    false_pos[i] = 1
                    false_positives_micro[i] += 1
        
        true_positives.append(true_pos)
        false_positives.append(false_pos)

        cumulative_true_pos = np.cumsum(true_pos)  # Cumulative sums of the true positives
        cumulative_false_pos = np.cumsum(false_pos)  # Cumulative sums of the false positives

        cumulative_true_positives.append(cumulative_true_pos)
        cumulative_false_positives.append(cumulative_false_pos)
    
    cumulative_open_set_error = np.cumsum(open_set_error)
    cumulative_false_positives_micro = np.cumsum(false_positives_micro)
    cumulative_true_positives_micro = np.cumsum(true_positives_micro)
    
    return (
        true_positives, false_positives, cumulative_true_positives, cumulative_false_positives,
        open_set_error, cumulative_open_set_error,
        cumulative_true_positives_micro, cumulative_false_positives_micro
    )


def get_precision_recall(number_gt_per_class: np.ndarray,
                         cumulative_true_positives: Sequence[np.ndarray],
                         cumulative_false_positives: Sequence[np.ndarray],
                         cumulative_true_positives_micro: np.ndarray,
                         cumulative_false_positives_micro: np.ndarray,
                         nr_classes: int) -> Tuple[List[np.ndarray], List[np.ndarray],
                                                   np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray]:
    """
    Computes the precision and recall values and returns them.
    
    Args:
        number_gt_per_class: number of ground truth bounding boxes per class
        cumulative_true_positives: cumulative true positives per class
        cumulative_false_positives: cumulative false positives per class
        cumulative_true_positives_micro: cumulative true positives over all classes
        cumulative_false_positives_micro: cumulative false positives over all classes
        nr_classes: number of classes

    Returns:
        cumulative precisions and cumulative recalls per class,
        micro averaged precision/recall, and
        macro averaged precision/recall
    """
    cumulative_precisions = [[]]
    cumulative_recalls = [[]]
    cumulative_precision_micro = np.zeros(cumulative_true_positives_micro.shape, dtype=np.float)
    cumulative_recall_micro = np.zeros(cumulative_true_positives_micro.shape, dtype=np.float)
    cumulative_precision_macro = np.zeros_like(cumulative_precision_micro)
    cumulative_recall_macro = np.zeros_like(cumulative_recall_micro)
    total_number_gt = 0
    number_of_nonzero_classes = 0

    # Iterate over all classes.
    for class_id in range(1, nr_classes + 1):
        
        if number_gt_per_class[class_id] == 0:
            cumulative_precisions.append([])
            cumulative_recalls.append([])
            continue
    
        tp = cumulative_true_positives[class_id]
        fp = cumulative_false_positives[class_id]
    
        cumulative_precision = np.where(tp + fp > 0, tp / (tp + fp), 0)  # 1D array with shape `(num_predictions,)`
        number_gt = number_gt_per_class[class_id]
        total_number_gt += number_gt
        cumulative_recall = tp / number_gt  # 1D array with shape `(num_predictions,)`
    
        cumulative_precisions.append(cumulative_precision)
        cumulative_recalls.append(cumulative_recall)

        diff_to_largest_class = cumulative_precision_micro.shape[0] - cumulative_precision.shape[0]
        if diff_to_largest_class:
            highest_precision = cumulative_precision[-1] if cumulative_precision.shape[0] else 0
            highest_recall = cumulative_recall[-1] if cumulative_recall.shape[0] else 0
            repeated_last_precision = np.tile(highest_precision, diff_to_largest_class)
            repeated_last_recall = np.tile(highest_recall, diff_to_largest_class)
            extended_precision = np.concatenate((cumulative_precision, repeated_last_precision))
            extended_recall = np.concatenate((cumulative_recall, repeated_last_recall))
            cumulative_precision_macro += extended_precision
            cumulative_recall_macro += extended_recall
        else:
            cumulative_precision_macro += cumulative_precision
            cumulative_recall_macro += cumulative_recall

        number_of_nonzero_classes += 1

    # calculate micro averaged precision and recall
    tp = cumulative_true_positives_micro
    fp = cumulative_false_positives_micro
    cumulative_precision_micro = np.where(tp + fp > 0, tp / (tp + fp), 0)
    cumulative_recall_micro = tp / total_number_gt
    
    # calculate macro averaged precision and recall
    cumulative_precision_macro /= number_of_nonzero_classes
    cumulative_recall_macro /= number_of_nonzero_classes

    return (cumulative_precisions, cumulative_recalls,
            cumulative_precision_micro, cumulative_recall_micro,
            cumulative_precision_macro, cumulative_recall_macro
    )


def get_f1_score(cumulative_precisions: List[np.ndarray],
                 cumulative_recalls: List[np.ndarray],
                 cumulative_precision_micro: np.ndarray,
                 cumulative_recall_micro: np.ndarray,
                 cumulative_precision_macro: np.ndarray,
                 cumulative_recall_macro: np.ndarray,
                 nr_classes: int) -> Tuple[List[np.ndarray],
                                           np.ndarray, np.ndarray]:
    """
    Computes the F1 score for every class.
    
    Args:
        cumulative_precisions: cumulative precisions for each class
        cumulative_recalls: cumulative recalls for each class
        cumulative_precision_micro: cumulative precision micro averaged
        cumulative_recall_micro: cumulative recall micro averaged
        cumulative_precision_macro: cumulative precision macro averaged
        cumulative_recall_macro: cumulative recall macro averaged
        nr_classes: number of classes

    Returns:
        cumulative F1 score per class,
        cumulative F1 score micro averaged, cumulative F1 score macro averaged
    """
    cumulative_f1_scores = [[]]
    
    # iterate over all classes
    for class_id in range(1, nr_classes + 1):
        cumulative_precision = cumulative_precisions[class_id]
        cumulative_recall = cumulative_recalls[class_id]
        if not np.count_nonzero(cumulative_precision + cumulative_recall):
            cumulative_f1_scores.append([])
            continue
        f1_score = 2 * ((cumulative_precision * cumulative_recall) / (cumulative_precision + cumulative_recall + 0.001))
        cumulative_f1_scores.append(f1_score)

    f1_score_micro = 2 * ((cumulative_precision_micro * cumulative_recall_micro) /
                          (cumulative_precision_micro + cumulative_recall_micro + 0.001))
    f1_score_macro = 2 * ((cumulative_precision_macro * cumulative_recall_macro) /
                          (cumulative_precision_macro + cumulative_recall_macro + 0.001))
    
    return cumulative_f1_scores, f1_score_micro, f1_score_macro


def get_mean_average_precisions(cumulative_precisions: List[np.ndarray],
                                cumulative_recalls: List[np.ndarray],
                                nr_classes: int) -> List[float]:
    """
    Computes the mean average precision for each class and returns them.
    
    Args:
        cumulative_precisions: cumulative precisions for each class
        cumulative_recalls: cumulative recalls for each class
        nr_classes: number of classes

    Returns:
        average precision per class
    """
    average_precisions = [0.0]

    # Iterate over all classes.
    for class_id in range(1, nr_classes + 1):
    
        cumulative_precision = cumulative_precisions[class_id]
        cumulative_recall = cumulative_recalls[class_id]
        
        # We will compute the precision at all unique recall values.
        unique_recalls, unique_recall_indices, unique_recall_counts = np.unique(cumulative_recall,
                                                                                return_index=True,
                                                                                return_counts=True)
    
        # Store the maximal precision for each recall value and the absolute difference
        # between any two unique recall values in the lists below. The products of these
        # two numbers constitute the rectangular areas whose sum will be our numerical
        # integral.
        maximal_precisions = np.zeros_like(unique_recalls)
        recall_deltas = np.zeros_like(unique_recalls)
    
        # Iterate over all unique recall values in reverse order. This saves a lot of computation:
        # For each unique recall value `r`, we want to get the maximal precision value obtained
        # for any recall value `r* >= r`. Once we know the maximal precision for the last `k` recall
        # values after a given iteration, then in the next iteration, in order compute the maximal
        # precisions for the last `l > k` recall values, we only need to compute the maximal precision
        # for `l - k` recall values and then take the maximum between that and the previously computed
        # maximum instead of computing the maximum over all `l` values.
        # We skip the very last recall value, since the precision after the last recall value
        # 1.0 is defined to be zero.
        for i in range(len(unique_recalls) - 2, -1, -1):
            begin = unique_recall_indices[i]
            end = unique_recall_indices[i + 1]
            # When computing the maximal precisions, use the maximum of the previous iteration to
            # avoid unnecessary repeated computation over the same precision values.
            # The maximal precisions are the heights of the rectangle areas of our integral under
            # the precision-recall curve.
            maximal_precisions[i] = np.maximum(np.amax(cumulative_precision[begin:end]),
                                               maximal_precisions[i + 1])
            # The differences between two adjacent recall values are the widths of our rectangle areas.
            recall_deltas[i] = unique_recalls[i + 1] - unique_recalls[i]
    
        average_precision = np.sum(maximal_precisions * recall_deltas)
        average_precisions.append(average_precision)
    
    return average_precisions


def get_mean_average_precision(average_precisions: List[float]) -> float:
    """
    Computes the mean average precision over all classes and returns it.
    
    Args:
        average_precisions: list of average precisions per class

    Returns:
        mean average precision over all classes
    """
    average_precisions = np.copy(average_precisions)
    average_precisions = average_precisions[average_precisions != 0.0]
    return np.average(average_precisions[:])
