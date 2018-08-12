import logging

from keras import losses, optimizers, backend as K
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.layers import Flatten, Dense

logger = logging.getLogger(__name__)


def iou_calc_first(y_true, y_pred):
    _y_true = y_true[:, :,:,0:1]
    _y_pred = y_pred[:, :,:,0:1]


    seg = K.cast(K.flatten(_y_true > .5), 'int32')
    pre = K.cast(K.flatten(_y_pred > .5), 'int32')
    inter = K.sum(seg * pre)

    return 2 * (inter + 1) / (K.sum(seg) + K.sum(pre) + 1)


def iou_calc_second(y_true, y_pred):
    _y_true = y_true[:, :,:,1:2]
    _y_pred = y_pred[:, :,:,1:2]


    seg = K.cast(K.flatten(_y_true > .5), 'int32')
    pre = K.cast(K.flatten(_y_pred > .5), 'int32')
    inter = K.sum(seg * pre)

    return 2 * (inter + 1) / (K.sum(seg) + K.sum(pre) + 1)


def iou_calc_third(y_true, y_pred):
    _y_true = y_true[:, :,:,2:3]
    _y_pred = y_pred[:, :,:,2:3]


    seg = K.cast(K.flatten(_y_true > .5), 'int32')
    pre = K.cast(K.flatten(_y_pred > .5), 'int32')
    inter = K.sum(seg * pre)

    return 2 * (inter + 1) / (K.sum(seg) + K.sum(pre) + 1)


class PretrainedDecoderRawEncoderUnet():
    def __init__(self, img_size, classification, skip_connections, mask_channels=1):
        self.img_size = img_size
        self.classification = classification
        self.skip_connections = skip_connections
        self.mask_channels = mask_channels
        self.neural_net = None
        self.encoder_layers = None

    def freeze_encoder_blocks(self, depth=None):
        sorted_shapes = sorted({attributes[1][1:3] for attributes in self.encoder_layers})
        for name, shape, index in self.encoder_layers:
            if shape[1:3] in sorted_shapes[depth:]:
                try:
                    self.neural_net.get_layer(name).trainable = False
                except ValueError:
                    logger.info('Layer {name} is not present in final neural net'.format(name=name))

    def compile(self, loss=losses.categorical_crossentropy,
                optimizer=optimizers.Adadelta(), metrics=None):
        self.neural_net.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, training_generator, validation_generator=None, epochs=None, callbacks='default', **kwargs):
        if callbacks is 'default':
            lower_lear = ReduceLROnPlateau(monitor='loss', factor=.33, patience=10, verbose=0, mode='auto', cooldown=10)
            callback_tb = TensorBoard()
            callbacks = [lower_lear, callback_tb]

        self.neural_net.fit_generator(generator=training_generator, validation_data=validation_generator,
                                      epochs=epochs, callbacks=callbacks, **kwargs)

    def predict(self, pred_generator):
        return self.neural_net.predict_generator(generator=pred_generator)

    def store_model(self, path):
        self.neural_net.save(path)

    def _add_classification_branch(self, encoder):
        classification = Flatten(name='flatten')(encoder.output)
        classification = Dense(64, activation='relu', name='fc1')(classification)
        res_classification = Dense(1, activation='sigmoid', name='classification')(classification)
        return res_classification
