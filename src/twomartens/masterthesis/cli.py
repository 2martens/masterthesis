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
Provides CLI actions.

Functions:
    config(...): handles the config component
    train(...): trains a network
    test(...): evaluates a network
    val(...): runs predictions on the validation data
    prepare(...): prepares the SceneNet ground truth data
"""
import argparse

import math

from twomartens.masterthesis import config as conf


def config(args: argparse.Namespace) -> None:
    _config_execute_action(args, conf.get_property, conf.set_property, conf.list_property_values)


def prepare(args: argparse.Namespace) -> None:
    import pickle
    
    from twomartens.masterthesis import data
    
    file_names_photos, file_names_instances, \
        instances = data.prepare_scenenet_data(conf.get_property("Paths.scenenet"),
                                               args.protobuf_path)
    with open(f"{conf.get_property('Paths.scenenet_gt')}/"
              f"{args.ground_truth_path}/photo_paths.bin", "wb") as file:
        pickle.dump(file_names_photos, file)
        
    with open(f"{conf.get_property('Paths.scenenet_gt')}/"
              f"{args.ground_truth_path}/instance_paths.bin", "wb") as file:
        pickle.dump(file_names_instances, file)
        
    with open(f"{conf.get_property('Paths.scenenet_gt')}/"
              f"{args.ground_truth_path}/instances.bin", "wb") as file:
        pickle.dump(instances, file)


def train(args: argparse.Namespace) -> None:
    if args.network == "ssd" or args.network == "bayesian_ssd":
        _ssd_train(args)
    elif args.network == "auto_encoder":
        _auto_encoder_train(args)


def test(args: argparse.Namespace) -> None:
    if args.network == "ssd" or args.network == "bayesian_ssd":
        _ssd_test(args)
    elif args.network == "auto_encoder":
        _auto_encoder_test(args)


def evaluate(args: argparse.Namespace) -> None:
    if args.network == "ssd":
        _ssd_evaluate(args)
    else:
        raise NotImplementedError


def visualise(args: argparse.Namespace) -> None:
    import pickle
    
    from matplotlib import pyplot
    import numpy as np
    from PIL import Image
    
    from twomartens.masterthesis.ssd_keras.eval_utils import coco_utils
    
    with open(f"{args.ground_truth_path}/photo_paths.bin", "rb") as file:
        file_names = pickle.load(file)
    with open(f"{args.ground_truth_path}/instances.bin", "rb") as file:
        instances = pickle.load(file)
    
    output_path = f"{args.output_path}/visualise/{args.trajectory}"
    annotation_file_train = f"{args.coco_path}/annotations/instances_train2014.json"
    cats_to_classes, _, cats_to_names, _ = coco_utils.get_coco_category_maps(annotation_file_train)
    
    colors = pyplot.cm.hsv(np.linspace(0, 1, 81)).tolist()
    
    i = 0
    nr_images = len(file_names[args.trajectory])
    nr_digits = math.ceil(math.log10(nr_images))
    for file_name, labels in zip(file_names[args.trajectory], instances[args.trajectory]):
        if not labels:
            continue
        
        # only loop through selected trajectory
        with Image.open(file_name) as image:
            figure = pyplot.figure(figsize=(20, 12))
            pyplot.imshow(image)
            
            current_axis = pyplot.gca()
            
            for instance in labels:
                bbox = instance['bbox']
                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = bbox[2]
                ymax = bbox[3]
                color = colors[cats_to_classes[int(instance['coco_id'])]]
                label = f"{cats_to_names[int(instance['coco_id'])]}: {instance['wordnet_class_name']}, " \
                    f"{instance['wordnet_id']}"
                current_axis.add_patch(
                    pyplot.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
                current_axis.text(xmin, ymin, label, size='x-large', color='white',
                                  bbox={'facecolor': color, 'alpha': 1.0})
            pyplot.savefig(f"{output_path}/{str(i).zfill(nr_digits)}")
            pyplot.close(figure)
        
        i += 1


def measure_mapping(args: argparse.Namespace) -> None:
    import pickle
    
    from twomartens.masterthesis.ssd_keras.eval_utils import coco_utils
    
    with open(f"{args.ground_truth_path}/instances.bin", "rb") as file:
        instances = pickle.load(file)
    
    output_path = f"{args.output_path}/measure/{args.tarball_id}"
    annotation_file_train = f"{args.coco_path}/annotations/instances_train2014.json"
    cats_to_classes, _, _, _ = coco_utils.get_coco_category_maps(annotation_file_train)
    
    for i, trajectory in enumerate(instances):
        counts = {cat_id: 0 for cat_id in cats_to_classes.keys()}
        for labels in trajectory:
            for instance in labels:
                counts[instance['coco_id']] += 1
        
        with open(f"{output_path}/{i}.bin", "wb") as file:
            pickle.dump(counts, file)


def _config_execute_action(args: argparse.Namespace, on_get: callable,
                           on_set: callable, on_list: callable) -> None:
    if args.action == "get":
        print(str(on_get(args.property)))
    elif args.action == "set":
        on_set(args.property, args.value)
    elif args.action == "list":
        on_list()


def _ssd_train(args: argparse.Namespace) -> None:
    import os
    import pickle
    
    import tensorflow as tf

    from twomartens.masterthesis import data
    from twomartens.masterthesis import ssd
    
    tf.enable_eager_execution()
    
    batch_size = conf.get_property("Parameters.batch_size")
    image_size = conf.get_property("Parameters.ssd_image_size")
    use_dropout = False if args.network == "ssd" else True

    summary_path = conf.get_property("Paths.summaries")
    summary_path = f"{summary_path}/{args.network}/train/{args.iteration}"
    os.makedirs(summary_path, exist_ok=True)
    weights_path = conf.get_property("Paths.weights")
    coco_path = conf.get_property("Paths.coco")
    pre_trained_weights_file = f"{weights_path}/{args.network}/VGG_coco_SSD_300x300_iter_400000.h5"
    weights_path = f"{weights_path}/{args.network}/train/"
    os.makedirs(weights_path, exist_ok=True)
    
    # load prepared ground truth
    train_gt_path = conf.get_property('Paths.scenenet_gt_train')
    val_gt_path = conf.get_property('Paths.scenenet_gt_val')
    with open(f"{train_gt_path}/photo_paths.bin", "rb") as file:
        file_names_train = pickle.load(file)
    with open(f"{train_gt_path}/instances.bin", "rb") as file:
        instances_train = pickle.load(file)
    with open(f"{val_gt_path}/photo_paths.bin", "rb") as file:
        file_names_val = pickle.load(file)
    with open(f"{val_gt_path}/instances.bin", "rb") as file:
        instances_val = pickle.load(file)

    # model
    if use_dropout:
        ssd_model = ssd.DropoutSSD(mode='training', weights_path=pre_trained_weights_file)
    else:
        ssd_model = ssd.SSD(mode='training', weights_path=pre_trained_weights_file)
    
    train_generator, train_length = \
        data.load_scenenet_data(file_names_train, instances_train, coco_path,
                                predictor_sizes=ssd_model.predictor_sizes,
                                batch_size=batch_size,
                                resized_shape=(image_size, image_size),
                                training=True, evaluation=False, augment=False,
                                nr_trajectories=1)
    val_generator, val_length = \
        data.load_scenenet_data(file_names_val, instances_val, coco_path,
                                predictor_sizes=ssd_model.predictor_sizes,
                                batch_size=batch_size,
                                resized_shape=(image_size, image_size),
                                training=False, evaluation=False, augment=False,
                                nr_trajectories=1)
    del file_names_train, instances_train, file_names_val, instances_val
    
    if args.debug and conf.get_property("Debug.train_images"):
        from twomartens.masterthesis import debug
        
        train_data = next(train_generator)
        train_length -= batch_size
        train_images = train_data[0]
        train_labels = train_data[1]

        debug.save_ssd_train_images(train_images, train_labels, summary_path)
    
    nr_batches_train = int(math.floor(train_length / batch_size))
    nr_batches_val = int(math.floor(val_length / batch_size))
    
    if args.debug and conf.get_property("Debug.summaries"):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=summary_path
        )
    else:
        tensorboard_callback = None
    
    history = ssd.train_keras(
        train_generator,
        nr_batches_train,
        val_generator,
        conf.get_property("Parameters.steps_per_val_epoch"),
        ssd_model,
        weights_path,
        args.iteration,
        initial_epoch=0,
        nr_epochs=args.num_epochs,
        lr=conf.get_property("Parameters.learning_rate"),
        tensorboard_callback=tensorboard_callback
    )
    
    with open(f"{summary_path}/history", "wb") as file:
        pickle.dump(history.history, file)


def _auto_encoder_train(args: argparse.Namespace) -> None:
    import os
    
    import tensorflow as tf
    from tensorflow.python.ops import summary_ops_v2
    
    from twomartens.masterthesis import data
    from twomartens.masterthesis.aae import train
    
    tf.enable_eager_execution()
    coco_path = args.coco_path
    category = args.category
    batch_size = 16
    image_size = 256
    coco_data = data.load_coco_train(coco_path, category, num_epochs=args.num_epochs, batch_size=batch_size,
                                     resized_shape=(image_size, image_size))
    summary_path = conf.get_property("Paths.summary")
    summary_path = f"{summary_path}/{args.network}/train/category-{category}/{args.iteration}"
    train_summary_writer = summary_ops_v2.create_file_writer(
        summary_path
    )
    os.makedirs(summary_path, exist_ok=True)
    
    weights_path = conf.get_property("Paths.weights")
    weights_path = f"{weights_path}/{args.network}/category-{category}"
    os.makedirs(weights_path, exist_ok=True)
    if args.debug:
        with train_summary_writer.as_default():
            train.train_simple(coco_data, iteration=args.iteration,
                               weights_prefix=weights_path,
                               zsize=16, lr=0.0001, verbose=args.verbose, image_size=image_size,
                               channels=3, train_epoch=args.num_epochs, batch_size=batch_size)
    else:
        train.train_simple(coco_data, iteration=args.iteration,
                           weights_prefix=weights_path,
                           zsize=16, lr=0.0001, verbose=args.verbose, image_size=image_size,
                           channels=3, train_epoch=args.num_epochs, batch_size=batch_size)


def _ssd_test(args: argparse.Namespace) -> None:
    import pickle
    import os
    
    import tensorflow as tf
    
    from twomartens.masterthesis import data
    from twomartens.masterthesis import ssd
    from twomartens.masterthesis.ssd_keras.keras_layers import keras_layer_AnchorBoxes
    from twomartens.masterthesis.ssd_keras.keras_layers import keras_layer_L2Normalization
    from twomartens.masterthesis.ssd_keras.keras_loss_function import keras_ssd_loss
    
    
    config = tf.ConfigProto()
    config.log_device_placement = False
    config.gpu_options.allow_growth = False
    tf.enable_eager_execution(config=config)
    
    batch_size = conf.get_property("Parameters.batch_size")
    image_size = conf.get_property("Parameters.ssd_image_size")
    forward_passes_per_image = conf.get_property("Parameters.ssd_forward_passes_per_image")
    use_dropout = False if args.network == "ssd" else True
    
    weights_path = conf.get_property("Paths.weights")
    output_path = conf.get_property("Paths.output")
    coco_path = conf.get_property("Paths.coco")
    checkpoint_path = f"{weights_path}/{args.network}/train/{args.train_iteration}"
    model_file = f"{checkpoint_path}/ssd300.h5"
    output_path = f"{output_path}/{args.network}/val/{args.iteration}/"
    os.makedirs(output_path, exist_ok=True)
    
    # load prepared ground truth
    ground_truth_path = conf.get_property("Paths.scenenet_gt_test")
    with open(f"{ground_truth_path}/photo_paths.bin", "rb") as file:
        file_names_photos = pickle.load(file)
    with open(f"{ground_truth_path}/instances.bin", "rb") as file:
        instances = pickle.load(file)

    # model
    ssd_model = tf.keras.models.load_model(model_file, custom_objects={
        "L2Normalization": keras_layer_L2Normalization.L2Normalization,
        "AnchorBoxes": keras_layer_AnchorBoxes.AnchorBoxes
    })
    # TODO finde clean solution rather than Copy & Paste
    learning_rate_var = tf.keras.backend.variable(conf.get_property("Parameters.learning_rate"))
    ssd_loss = keras_ssd_loss.SSDLoss()
    
    ssd_model.compile(
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate_var,
                                         beta1=0.9, beta2=0.999),
        loss=ssd_loss.compute_loss,
        metrics=[
            "categorical_accuracy"
        ]
    )
    
    test_generator, length_dataset = \
        data.load_scenenet_data(file_names_photos, instances, coco_path,
                                predictor_sizes=None,
                                batch_size=batch_size,
                                resized_shape=(image_size, image_size),
                                training=False, evaluation=True, augment=False,
                                nr_trajectories=1)
    del file_names_photos, instances

    nr_digits = math.ceil(math.log10(math.ceil(length_dataset / batch_size)))
    steps_per_epoch = int(math.ceil(length_dataset / batch_size))
    ssd.predict_keras(test_generator,
                      steps_per_epoch,
                      ssd_model,
                      use_dropout,
                      forward_passes_per_image,
                      (image_size, image_size),
                      output_path,
                      nr_digits)


def _auto_encoder_test(args: argparse.Namespace) -> None:
    import os
    
    import tensorflow as tf
    from tensorflow.python.ops import summary_ops_v2
    
    from twomartens.masterthesis import data
    from twomartens.masterthesis.aae import run
    
    tf.enable_eager_execution()
    coco_path = conf.get_property("Paths.coco")
    category = args.category
    category_trained = args.category_trained
    batch_size = 16
    image_size = 256
    coco_data = data.load_coco_val(coco_path, category, num_epochs=1,
                                   batch_size=batch_size, resized_shape=(image_size, image_size))

    summary_path = conf.get_property("Paths.summary")
    summary_path = f"{summary_path}/{args.network}/val/category-{category}/{args.iteration}"
    os.makedirs(summary_path, exist_ok=True)
    use_summary_writer = summary_ops_v2.create_file_writer(
        summary_path
    )
    
    weights_path = conf.get_property("Paths.weights")
    weights_path = f"{weights_path}/{args.network}/category-{category_trained}"
    os.makedirs(weights_path, exist_ok=True)
    if args.debug:
        with use_summary_writer.as_default():
            run.run_simple(coco_data, iteration=args.iteration_trained,
                           weights_prefix=weights_path,
                           zsize=16, verbose=args.verbose, channels=3, batch_size=batch_size,
                           image_size=image_size)
    else:
        run.run_simple(coco_data, iteration=args.iteration_trained,
                       weights_prefix=weights_path,
                       zsize=16, verbose=args.verbose, channels=3, batch_size=batch_size,
                       image_size=image_size)


def _ssd_evaluate(args: argparse.Namespace) -> None:
    import glob
    import os
    import pickle
    
    import numpy as np
    import tensorflow as tf
    
    from twomartens.masterthesis import evaluate
    from twomartens.masterthesis import ssd
    
    tf.enable_eager_execution()
    
    batch_size = 16
    use_dropout = False if args.network == "ssd" else True
    output_path = conf.get_property("Paths.output")
    evaluation_path = conf.get_property("Paths.evaluation")
    output_path = f"{output_path}/{args.network}/val/{args.iteration}"
    evaluation_path = f"{evaluation_path}/{args.network}"
    result_file = f"{evaluation_path}/results-{args.iteration}.bin"
    label_file = f"{output_path}/labels.bin"
    predictions_file = f"{output_path}/predictions.bin"
    predictions_per_class_file = f"{output_path}/predictions_class.bin"
    os.makedirs(evaluation_path, exist_ok=True)
    
    # retrieve labels and un-batch them
    files = glob.glob(f"{output_path}/*ssd_labels*")
    labels = []
    for filename in files:
        with open(filename, "rb") as file:
            # get labels per batch
            label_dict = pickle.load(file)
            labels.extend(label_dict['labels'])
    
    # store labels for later use
    with open(label_file, "wb") as file:
        pickle.dump(labels, file)
    
    number_gt_per_class = evaluate.get_number_gt_per_class(labels, ssd.N_CLASSES)
    
    # retrieve predictions and un-batch them
    files = glob.glob(f"{output_path}/*ssd_predictions*")
    predictions = []
    for filename in files:
        with open(filename, "rb") as file:
            # get predictions per batch
            _predictions = pickle.load(file)
            predictions.extend(_predictions)
    del _predictions
    
    # prepare predictions for further use
    with open(predictions_file, "wb") as file:
        pickle.dump(predictions, file)
    
    predictions_per_class = evaluate.prepare_predictions(predictions, ssd.N_CLASSES)
    del predictions
    
    with open(predictions_per_class_file, "wb") as file:
        pickle.dump(predictions_per_class, file)
    
    # compute matches between predictions and ground truth
    true_positives, false_positives, \
    cum_true_positives, cum_false_positives, open_set_error = evaluate.match_predictions(predictions_per_class,
                                                                                         labels,
                                                                                         ssd.N_CLASSES)
    del labels
    cum_precisions, cum_recalls = evaluate.get_precision_recall(number_gt_per_class,
                                                                cum_true_positives,
                                                                cum_false_positives,
                                                                ssd.N_CLASSES)
    f1_scores = evaluate.get_f1_score(cum_precisions, cum_recalls, ssd.N_CLASSES)
    average_precisions = evaluate.get_mean_average_precisions(cum_precisions, cum_recalls, ssd.N_CLASSES)
    mean_average_precision = evaluate.get_mean_average_precision(average_precisions)
    
    results = {
        "true_positives":             true_positives,
        "false_positives":            false_positives,
        "cumulative_true_positives":  cum_true_positives,
        "cumulative_false_positives": cum_false_positives,
        "cumulative_precisions":      cum_precisions,
        "cumulative_recalls":         cum_recalls,
        "f1_scores":                  f1_scores,
        "mean_average_precisions":    average_precisions,
        "mean_average_precision":     mean_average_precision,
        "open_set_error":             open_set_error
    }
    
    with open(result_file, "wb") as file:
        pickle.dump(results, file)
