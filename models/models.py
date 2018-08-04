from keras import losses, optimizers


class PretrainedDecoderRawEncoderUnet():
    def __init__(self, img_size):
        self.img_size = img_size
        self.neural_net = None
        self.decoder_layers = None

    def freeze_decoder_blocks(self, depth=None):
        sorted_shapes = sorted({attributes[1][1:3] for attributes in self.decoder_layers})
        for name, shape, index  in self.decoder_layers:
            if shape in sorted_shapes[depth:]:
                self.neural_net.layers[index].trainable = False
                if name:
                    assert name == self.neural_net.layers[index].name, 'name is not equal for same index'

    def compile(self, loss=losses.categorical_crossentropy,
                optimizer=optimizers.Adadelta(), metrics=None):
        self.neural_net.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, training_generator, validation_generator=None, epochs=None, callbacks=None, **kwargs):
        self.neural_net.fit_generator(generator=training_generator, validation_data=validation_generator,
                                      epochs=epochs, callbacks=callbacks, **kwargs)

    def predict(self, pred_generator):
        return self.neural_net.predict_generator(generator=pred_generator)

    def store_model(self, path):
        self.neural_net.save(path)


