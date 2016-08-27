import logging
import operator

import numpy as np

from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Convolution1D
from keras.layers import Dense, Merge, Input, Reshape, Flatten

from model_base import ModelBase


class SimpleConvModelKeras(ModelBase):
    def __init__(self, slots, slot_classes, vocab, config, in_save_path):
        ModelBase.__init__(self)
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
        self.max_sequence_length = in_X.shape[1]
        input_layer = Input(name='input', shape=in_X[0].shape)
        emb = Embedding(
            len(self.vocab),
            self.config['emb_size'],
            name='embedding'
        )(input_layer)
        conv = Convolution1D(64, 3, name='conv')(emb)
        max_pool = MaxPooling1D(name='max_pool')(conv)
        flatten = Flatten()(max_pool)
        slot_mlps = []
        for slot in self.slots:
            logging.info(
                'Building output classifier for the slot "{}"'.format(slot)
            )
            n_classes = len(self.slot_classes[slot])
            slot_mlps.append(
                Dense(
                    name='dense_{}'.format(slot),
                    output_dim=n_classes,
                    activation=self.config['oclf_activation'],
                    init='normal'
                )(flatten)
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

    def prepare_data(self, seqs, slots, with_labels=True):
        x = []
        y_labels = []
        x_one_hot_length = len(self.vocab)

        for sequence in seqs:
            one_hot_sequence = []
            for one_hot in sequence['data']:
                # x_vec = np.zeros(x_one_hot_length)
                # x_vec[one_hot] = 1
                one_hot_sequence.append(one_hot)
            x.append(one_hot_sequence)
            labels = sequence['labels']

            one_hot_frame_sequence = self._build_frame_one_hot(
                labels[-1],
                slots
            )
            y_labels.append(one_hot_frame_sequence)

        assert len(x) == len(y_labels)

        max_sequence_length = self.max_sequence_length \
            if self.max_sequence_length \
            else max(map(len, x))
        x_zero_pad = np.zeros(1)
        x = self._pad_sequences(x, [x_zero_pad], max_sequence_length)

        data = [x]
        if with_labels:
            data += [np.array(y_labels)]

        return tuple(data)
