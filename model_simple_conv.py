import logging
import operator

from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.pooling import MaxPooling1D
from keras.layers import Dense, Merge


class SimpleConvModelKeras(ModelBase):
    def _log_classes_info(self):
        for slot, vals in self.slot_classes.iteritems():
            logging.info('  %s:' % slot)
            for val, val_ndx in sorted(vals.iteritems(), key=lambda x: x[1]):
                logging.info('    - %s (%d)' % (val, val_ndx))

    def __init__(self, slots, slot_classes, vocab, config, in_save_path):
        self.vocab = vocab
        self.slots = slots
        self.slot_classes = slot_classes
        self.slot_classes_reversed = {
            slot_class: map(
                lambda x: x[0],
                sorted(self.slot_classes[slot_class].items(),
                       key=operator.itemgetter(1))
            )
            for slot_class in self.slot_classes
            }
        self.config = config
        self.save_path = in_save_path


        logging.info('We have the following classes:')
        self._log_classes_info()

    def init_model(self, in_X):
        input_layer = Embedding(
            input_dim=len(self.vocab) + 1,
            output_dim=self.config['emb_size']
        )
        max_pool = MaxPooling1D()
        slot_mlps = []
        for slot in self.slots:
            logging.info(
                'Building output classifier for the slot "{}"'.format(slot)
            )
            # for layer_id in range(self.oclf_n_layers):
            #     result.add(
            #         # TODO: add Gaussian random initialization to each layer
            #         Dense(
            #             input_dim=input_dim,
            #             output_dim=oclf_n_hidden,
            #             activation=oclf_activation
            #         )
            #    )
            n_classes = len(self.slot_classes[slot])
            slot_mlps.append(
                Dense(
                    output_dim=n_classes,
                    activation=self.config['oclf_activation']
                )(max_pool)
            )
        merge_layer = Merge(
            name='mlps_merged',
            mode='concat',
            concat_axis=1
        )(slot_mlps)

        self.model = Model(input=input_layer, output=merge_layer)
        self.model.compile(
            optimizer=self.config['opt_type'],
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def prepare_data_train(self, seqs, slots):
        return self._prepare_data(seqs, slots, with_labels=True)

    def prepare_data_predict(self, seqs, slots):
        return self._prepare_data(seqs, slots, with_labels=False)
