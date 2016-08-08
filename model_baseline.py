import logging

import numpy as np

from keras.models import Model
from keras.layers import Dense, Activation, Input, Merge
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint


class BaselineModelKeras(object):
    def _log_classes_info(self):
        for slot, vals in self.slot_classes.iteritems():
            logging.info('  %s:' % slot)
            for val, val_ndx in sorted(vals.iteritems(), key=lambda x: x[1]):
                logging.info('    - %s (%d)' % (val, val_ndx))

    def __init__(self, slots, slot_classes, vocab, config, in_save_path):
        self.vocab = vocab
        self.max_sequence_length = None
        self.slots = slots
        self.slot_classes = slot_classes
        self.config = config
        self.save_path = in_save_path

        logging.info('We have the following classes:')
        self._log_classes_info()

    def init_model(self, in_X):
        self.max_sequence_length = in_X.shape[1]
        input_layer = Input(name='input', shape=in_X[0].shape)
        logging.info(
            'Creating LSTM layer with %d neurons.' % (self.config['n_cells'])
        )
        main_lstm = LSTM(
            output_dim=self.config['n_cells'],
            name='main_lstm',
            dropout_W=self.config['p_drop']
        )(input_layer)

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
                )(main_lstm)
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

    def init_loaded(self):
        pass

    def train(self, X, y):
        self.init_model(X)
        checkpointer = ModelCheckpoint(
            filepath=self.save_path,
            verbose=True,
            save_best_only=True
        )
        self.model.fit(
            X,
            y,
            batch_size=self.config['mb_size'],
            nb_epoch=self.config['n_epochs'],
            verbose=True,
            shuffle=True,
            callbacks=[checkpointer],
            validation_split=0.2
        )

    def evaluate(self, X, y):
        return self.model.evaluate(
            X,
            y,
            batch_size=self.config['mb_size'],
            verbose=True
        )

    def save(self, in_file_name):
        self.model.save(in_file_name)

    def prepare_data_train(self, sequences, slots):
        X, y = self.prepare_data(sequences, slots)
        self.init_model(X)

    def prepare_data_test(self, sequences, slots):
        X, y = self.prepare_data(sequences, slots)

    def prepare_data(self, seqs, slots, with_labels=True):
        x = []
        y_labels = []
        x_one_hot_length = len(self.vocab)
        y_one_hot_length = sum(map(len, self.slot_classes.values()))

        for sequence in seqs:
            one_hot_sequence = []
            for one_hot in sequence['data']:
                x_vec = np.zeros(x_one_hot_length)
                x_vec[one_hot] = 1
                one_hot_sequence.append(x_vec)
            x.append(one_hot_sequence)
            labels = sequence['labels']

            one_hot_frame_sequence = \
                self._build_frame_one_hot(labels[-1], slots)
            y_labels.append(one_hot_frame_sequence)

        assert len(x) == len(y_labels)

        max_sequence_length = self.max_sequence_length \
            if self.max_sequence_length \
            else max(map(len, x))
        x_zero_pad = np.zeros(x_one_hot_length)
        x = self._pad_sequences(x, [x_zero_pad], max_sequence_length)

        data = [x]
        if with_labels:
            data += [np.array(y_labels)]

        return tuple(data)

    def _pad_sequences(self, in_sequences, in_pad_by, in_max_len):
        result = []
        for sequence in in_sequences:
            pad_length = max(0, in_max_len - len(sequence))
            padded_sequence = sequence[:in_max_len] + in_pad_by * pad_length
            result.append(padded_sequence)
        return np.asarray(result, dtype=np.int32)

    def _build_frame_one_hot(self, in_frame_label, in_slots):
        frame_map = {
            slot: np.zeros(len(self.slot_classes[slot]))
            for slot in in_slots
        }
        for index, slot in enumerate(in_slots):
            lbl_val = in_frame_label['slots'][slot]
            if not lbl_val:
                continue
            frame_map[slot][lbl_val] = 1
        one_hot_concatenated = reduce(
            lambda x, y: np.concatenate((x, y)),
            [frame_map[slot] for slot in in_slots],
            []
        )
        return one_hot_concatenated
