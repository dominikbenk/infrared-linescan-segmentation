import argparse
import datetime
import os
import re
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import keras.backend as K

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=4, type=int, help="Seed")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--threads", default=0, type=int, help="Threads")
parser.add_argument("--epochs", default=10, type=int,  help="Number of epochs")
parser.add_argument("--dev_size", default=500, type=int, help="Number of images kept for dev")
parser.add_argument("--input_size", default=256, type=int, help="Dimensions of input images")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout")
parser.add_argument("--bn", default=False, action='store_true', help="Include batchnorm")
parser.add_argument("--n_filters", default=16, type=int, help="Filters for UNet")
parser.add_argument("--rotate", default=0, type=int, help="Max rotation (left and right)")
parser.add_argument("--data_path", default='train_512_10_20', type=str, help="Data to load")
parser.add_argument("--augment", default=[0.0,0.0,0.2,20,0], nargs="+", type=float,  help="Augment parameters: [w_shift,h_shift,zoom,shear,rotation]")
parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing")
parser.add_argument("--cutout_size", default=32, type=int, help="cutout size")

# model inspired by https://arxiv.org/abs/1505.04597
def UNet(input_height, input_width, filters, dropout, batchnorm):

    inputs = tf.keras.layers.Input([None, None, 1])
    # contracting path
    def down_scale(layer, filters, dropout, batchnorm):
        conv = tf.keras.layers.Conv2D(filters, 3, 1, "same", activation=None, kernel_initializer="he_normal")(layer)
        if batchnorm:
          conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation("relu")(conv)

        conv = tf.keras.layers.Conv2D(filters, 3, 1, "same", activation=None, kernel_initializer="he_normal")(conv)
        if batchnorm:
          conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation("relu")(conv)

        pooled = tf.keras.layers.MaxPool2D(2, 2)(conv)
        pooled = tf.keras.layers.Dropout(dropout)(pooled)

        return conv, pooled

    # expansive path
    def up_scale(layer, layer_to_concat, filters):
        conv = tf.keras.layers.Conv2DTranspose(filters, 3, 2, padding='same')(layer)
        conv = tf.keras.layers.Concatenate()([conv, layer_to_concat])
        conv = tf.keras.layers.Dropout(dropout)(conv)

        conv = tf.keras.layers.Conv2D(filters, 3, 1, "same", activation=None, kernel_initializer="he_normal")(conv)
        if batchnorm:
          conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation("relu")(conv)

        conv = tf.keras.layers.Conv2D(filters, 3, 1, "same", activation=None, kernel_initializer="he_normal")(conv)
        if batchnorm:
          conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation("relu")(conv)

        return conv

    # number of filters doubles in each layer, while resolution is cut in half
    conv1, pooled1 = down_scale(inputs, filters, dropout=dropout, batchnorm=batchnorm)
    conv2, pooled2 = down_scale(pooled1, filters * 2, dropout=dropout, batchnorm=batchnorm)
    conv3, pooled3 = down_scale(pooled2, filters * 4, dropout=dropout, batchnorm=batchnorm)
    conv4, pooled4 = down_scale(pooled3, filters * 8, dropout=dropout, batchnorm=batchnorm)
    conv5, _ = down_scale(pooled4, filters * 16, dropout=dropout, batchnorm=batchnorm)

    up_conv1 = up_scale(conv5, conv4, filters * 8)
    up_conv2 = up_scale(up_conv1, conv3, filters * 4)
    up_conv3 = up_scale(up_conv2, conv2, filters * 2)
    up_conv4 = up_scale(up_conv3, conv1, filters)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(up_conv4)

    model = tf.keras.models.Model(inputs, outputs)
    return model
    

def main(args):
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # defining metrics
    class MeanIOU(tf.keras.metrics.MeanIoU):
        def update_state(self, y_true, y_pred, sample_weight=None):
            shift = args.input_size // 4
            # subsetting only center
            y_pred = tf.image.crop_to_bounding_box(y_pred, shift, shift, 2 * shift, 2 * shift)
            y_true = tf.image.crop_to_bounding_box(y_true, shift, shift, 2 * shift, 2 * shift)
            y_pred = tf.cast(y_pred > 0.5, tf.int32)
            return super().update_state(y_true, y_pred, sample_weight)

    def f1(y_true, y_pred):
      shift = args.input_size // 4
      # subsetting only center
      y_pred = tf.image.crop_to_bounding_box(y_pred, shift, shift, 2 * shift, 2 * shift)
      y_true = tf.image.crop_to_bounding_box(y_true, shift, shift, 2 * shift, 2 * shift)
      y_pred = K.round(y_pred)
      tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
      tn = K.sum(K.cast((1 - y_true)*(1 - y_pred), 'float'), axis=0)
      fp = K.sum(K.cast((1 - y_true)*y_pred, 'float'), axis=0)
      fn = K.sum(K.cast(y_true*(1 - y_pred), 'float'), axis=0)

      p = tp / (tp + fp + K.epsilon())
      r = tp / (tp + fn + K.epsilon())

      f1 = 2 * p * r / (p + r + K.epsilon())
      f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
      return K.mean(f1)

    # generator for random data augmentation
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
              width_shift_range=args.augment[0],
              height_shift_range=args.augment[1],
              zoom_range=args.augment[2],
              shear_range=int(args.augment[3]),
              rotation_range=int(args.augment[4]),
              horizontal_flip=True,
              vertical_flip=True,
              fill_mode="reflect")

    h = args.input_size
    w = args.input_size
    generator = tf.random.Generator.from_seed(args.seed)

    # augmenting training data
    def train_augment(image, mask):
      image = tf.expand_dims(image, 2)
      mask = tf.expand_dims(mask, 2)
      concat = tf.concat([image, mask], axis=2)
      concat = tf.ensure_shape(tf.numpy_function(image_generator.random_transform, [concat], tf.uint8), concat.shape)
      image, mask = tf.expand_dims(tf.cast(concat[:, :, 0], tf.float32) / 255.0, 2), tf.expand_dims(concat[:, :, 1], axis=2)
      image = tfa.image.random_cutout(tf.expand_dims(image, 0), mask_size = args.cutout_size, constant_values = 0)
      return image[0,:,:,:], mask

    # augmenting validation data
    def dev_augment(image, mask):
      image = tf.expand_dims(image, 2)
      mask = tf.expand_dims(mask, 2)
      image = tf.cast(image, tf.float32) / 255.0
      return image, mask


    # loading data
    train = tf.data.experimental.load(args.data_path,
                                      element_spec=(tf.TensorSpec(shape=(args.input_size, args.input_size), dtype=tf.uint8),
                                                    tf.TensorSpec(shape=(args.input_size, args.input_size), dtype=tf.uint8)))
   
    print(f'total observations: {len(train)}')

    # train-dev split
    dev = train.take(args.dev_size) 
    train = train.skip(args.dev_size)
    train = train.shuffle(20000)
   

    train = train.map(train_augment)
    dev = dev.map(dev_augment)

    train = train.batch(args.batch_size)
    dev = dev.batch(args.batch_size)

    # creating instance of the U-net
    model = UNet(input_height=args.input_size, 
                  input_width=args.input_size, 
                  filters = args.n_filters,
                  dropout = args.dropout,
                  batchnorm = args.bn)

    # defining cross-entropy dice loss
    def crossentropy_dice_loss(y_true, y_pred, shift=args.input_size//4):
        def dice_loss(y_true, y_pred):
          numerator = 2 * tf.reduce_sum(y_true * y_pred)
          denominator = tf.reduce_sum(y_true + y_pred)
          return 1 - numerator / denominator
 
        y_pred = tf.image.crop_to_bounding_box(y_pred, shift, shift, 2 * shift, 2 * shift)
        y_true = tf.image.crop_to_bounding_box(y_true, shift, shift, 2 * shift, 2 * shift)
        y_true = tf.cast(y_true, tf.float32)

        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)(y_true, y_pred) + dice_loss(y_true, y_pred)
        return tf.reduce_mean(loss)

    # setting up compilation parameters
    model.compile(
      optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.experimental.CosineDecay(
          initial_learning_rate=0.001,decay_steps=len(train)*args.epochs)),
      loss = crossentropy_dice_loss,
      metrics = [MeanIOU(2),f1])

    # creating directory for to save the training and validation progress
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # callbacks with the progress
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, write_graph=False, update_freq=100, profile_batch=0)

    # fitting the U-net
    logs = model.fit(
        train,
        epochs=args.epochs,
        validation_data=dev,
        callbacks=[tb_callback],
    )

    now = datetime.datetime.now()
    # saving the trained model
    model_path = 'model_'+now.strftime("%d-%m-%Y_%H-%M")+'.h5'
    model.save(model_path)
    print('Model has been saved as: '+model_path)
    

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
