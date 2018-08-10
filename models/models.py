import logging

from keras import losses, optimizers
from keras.layers import Flatten, Dense

logger = logging.getLogger(__name__)


class PretrainedDecoderRawEncoderUnet():
    def __init__(self, img_size, classification, skip_connections):
        self.img_size = img_size
        self.classification = classification
        self.skip_connections = skip_connections
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

    def compile(self, loss=losses.binary_crossentropy,
                optimizer=optimizers.Adadelta(), metrics=None):
        self.neural_net.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, training_generator, validation_generator=None, epochs=None, callbacks=None, **kwargs):
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
