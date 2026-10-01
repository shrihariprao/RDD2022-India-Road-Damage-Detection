"""
Custom SSD-style MobileNetV2 baseline for RDD2022 India
road-damage detection.

Classes:
    D00 - Longitudinal cracks
    D10 - Transverse cracks
    D20 - Alligator cracks
    D40 - Potholes

Architecture:
    MobileNetV2 ImageNet backbone
    Input: 320 x 320
    Feature maps:
        block_6_expand_relu  -> 40 x 40 x 192
        block_13_expand_relu -> 20 x 20 x 576
        out_relu             -> 10 x 10 x 1280

Each feature map is projected to 256 channels.

Detection heads:
    6 anchors per spatial location
    classification head
    bounding-box regression head

Preliminary experiment:
    Batch size      : 8
    Learning rate   : 1e-4
    Optimizer       : Adam
    Epochs          : 2
    Backbone        : frozen
    Classification : focal loss
    Regression      : Smooth-L1 / Huber

Important:
    This is a custom SSD-style implementation.
    It is NOT claimed to be the official TensorFlow
    Model Garden SSD implementation.

This file is a reproducibility/reference implementation
for the completed preliminary experiment.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


CLASS_NAMES = [
    "D00",
    "D10",
    "D20",
    "D40",
]

NUM_CLASSES = len(CLASS_NAMES) + 1  # background + 4 classes

IMAGE_SIZE = 320
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 2

NUM_ANCHORS = 6

ASPECT_RATIOS = [
    (1.0, 1.0),
    (1.0, 1.5),
    (1.5, 1.0),
    (0.7, 1.4),
    (1.4, 0.7),
    (1.0, 2.0),
]

SCALES = [
    0.08,
    0.20,
    0.40,
]

DATA_ROOT = Path(
    "/content/drive/MyDrive/"
    "Road_Damage_Detection_Project/"
    "RDD2022_India_4Class_Master"
)

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/"
    "Road_Damage_Detection_Project/"
    "results/SSD_MobileNetV2_preliminary"
)


def build_backbone():
    """
    Create the MobileNetV2 feature extractor.

    The ImageNet-trained backbone is frozen for the
    preliminary experiment.
    """

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
        include_top=False,
        weights="imagenet",
    )

    backbone.trainable = False

    feature_names = [
        "block_6_expand_relu",
        "block_13_expand_relu",
        "out_relu",
    ]

    feature_outputs = [
        backbone.get_layer(
            name
        ).output
        for name in feature_names
    ]

    return Model(
        inputs=backbone.input,
        outputs=feature_outputs,
        name="MobileNetV2_Backbone",
    )


def make_feature_head(
    feature_map,
):
    """
    Project one backbone feature map to 256 channels.
    """

    x = layers.Conv2D(
        256,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(feature_map)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    return x


def build_ssd_model():
    """
    Build the custom SSD-style detector.
    """

    inputs = layers.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
        name="image",
    )

    backbone = build_backbone()

    features = backbone(inputs)

    processed_features = [
        make_feature_head(
            feature
        )
        for feature in features
    ]

    class_outputs = []
    box_outputs = []

    for feature in processed_features:

        class_head = layers.Conv2D(
            NUM_ANCHORS * NUM_CLASSES,
            kernel_size=3,
            padding="same",
            name="class_head",
        )(feature)

        box_head = layers.Conv2D(
            NUM_ANCHORS * 4,
            kernel_size=3,
            padding="same",
            name="box_head",
        )(feature)

        class_outputs.append(
            class_head
        )

        box_outputs.append(
            box_head
        )

    return Model(
        inputs=inputs,
        outputs=[
            class_outputs,
            box_outputs,
        ],
        name="Custom_SSD_MobileNetV2",
    )


def generate_anchors():
    """
    Generate normalized anchor boxes.

    Anchor coordinates use:
        [xmin, ymin, xmax, ymax]

    with values clipped to [0, 1].
    """

    feature_sizes = [
        40,
        20,
        10,
    ]

    anchors = []

    for feature_size, scale in zip(
        feature_sizes,
        SCALES,
    ):

        for row in range(
            feature_size
        ):

            for col in range(
                feature_size
            ):

                cx = (
                    col + 0.5
                ) / feature_size

                cy = (
                    row + 0.5
                ) / feature_size

                for ar_width, ar_height in ASPECT_RATIOS:

                    width = (
                        scale
                        * ar_width
                    )

                    height = (
                        scale
                        * ar_height
                    )

                    xmin = max(
                        0.0,
                        cx - width / 2.0,
                    )

                    ymin = max(
                        0.0,
                        cy - height / 2.0,
                    )

                    xmax = min(
                        1.0,
                        cx + width / 2.0,
                    )

                    ymax = min(
                        1.0,
                        cy + height / 2.0,
                    )

                    anchors.append(
                        [
                            xmin,
                            ymin,
                            xmax,
                            ymax,
                        ]
                    )

    return np.asarray(
        anchors,
        dtype=np.float32,
    )


def encode_box(
    box,
    anchor,
):
    """
    Encode a normalized box against an anchor
    using SSD-style center/scale parameterization.
    """

    xmin, ymin, xmax, ymax = box

    axmin, aymin, axmax, aymax = anchor

    box_cx = (
        xmin + xmax
    ) / 2.0

    box_cy = (
        ymin + ymax
    ) / 2.0

    box_w = max(
        xmax - xmin,
        1e-8,
    )

    box_h = max(
        ymax - ymin,
        1e-8,
    )

    anchor_cx = (
        axmin + axmax
    ) / 2.0

    anchor_cy = (
        aymin + aymax
    ) / 2.0

    anchor_w = max(
        axmax - axmin,
        1e-8,
    )

    anchor_h = max(
        aymax - aymin,
        1e-8,
    )

    tx = (
        box_cx - anchor_cx
    ) / anchor_w

    ty = (
        box_cy - anchor_cy
    ) / anchor_h

    tw = np.log(
        box_w / anchor_w
    )

    th = np.log(
        box_h / anchor_h
    )

    return np.asarray(
        [
            tx,
            ty,
            tw,
            th,
        ],
        dtype=np.float32,
    )


def focal_loss(
    y_true,
    y_pred,
    alpha=0.25,
    gamma=2.0,
):
    """
    Binary focal loss used for the classification term.
    """

    y_pred = tf.clip_by_value(
        y_pred,
        1e-7,
        1.0 - 1e-7,
    )

    ce = -(
        y_true
        * tf.math.log(y_pred)
        + (
            1.0 - y_true
        )
        * tf.math.log(
            1.0 - y_pred
        )
    )

    p_t = (
        y_true * y_pred
        + (
            1.0 - y_true
        )
        * (
            1.0 - y_pred
        )
    )

    alpha_factor = (
        y_true * alpha
        + (
            1.0 - y_true
        )
        * (1.0 - alpha)
    )

    modulating_factor = tf.pow(
        1.0 - p_t,
        gamma,
    )

    return tf.reduce_mean(
        alpha_factor
        * modulating_factor
        * ce
    )


def build_optimizer():
    """
    Adam optimizer used in the preliminary experiment.
    """

    return tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )


def count_anchors():
    """
    Return the number of generated anchors.
    """

    return sum(
        (
            size * size
            * NUM_ANCHORS
        )
        for size in [
            40,
            20,
            10,
        ]
    )


def save_model(
    model,
    output_path,
):
    """
    Save model weights.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_weights(
        str(output_path)
    )


def main():
    """
    Build the detector and print the architecture summary.

    Training is intentionally not started automatically.
    """

    model = build_ssd_model()

    optimizer = build_optimizer()

    anchors = generate_anchors()

    print(
        "Model:",
        "Custom SSD-style MobileNetV2",
    )

    print(
        "Input size:",
        f"{IMAGE_SIZE}x{IMAGE_SIZE}",
    )

    print(
        "Classes:",
        CLASS_NAMES,
    )

    print(
        "Anchors:",
        len(anchors),
    )

    print(
        "Expected anchors:",
        count_anchors(),
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
    )

    print(
        "Epochs:",
        EPOCHS,
    )

    print(
        "Optimizer:",
        optimizer.__class__.__name__,
    )

    print(
        "Trainable parameters:",
        sum(
            np.prod(
                variable.shape
            )
            for variable
            in model.trainable_variables
        ),
    )

    model.summary()


if __name__ == "__main__":
    main()
