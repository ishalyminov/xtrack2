import codecs
import json
import logging
import operator

from keras.models import Model
from keras.layers import Dense, Input, Merge
from keras.layers.recurrent import LSTM
from keras.models import load_model

from model_base import ModelBase


class BaselineModelKeras(ModelBase):
    def __init__(self, slots, slot_classes, vocab, config, in_save_path):
        ModelBase.__init__(self)
        self.vocab = vocab
        self.slots = slots
        self.slot_classes = slot_classes
        self.slot_classes_reversed = {
            slot_class: map(
                lambda x: x[0],
                sorted(self.slot_classes[slot_class].items(), key=operator.itemgetter(1))
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
            n_classes = len(self.slot_classes[slot])
            slot_mlps.append(
                Dense(
                    output_dim=n_classes,
                    activation=self.config['oclf_activation'],
                    init='normal'
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

    @classmethod
    def load(self, in_file_name):
        model_params = BaselineModelKeras._load_model_params(
            in_file_name + '.model_params.json'
        )
        result = BaselineModelKeras(
            model_params['slots'],
            model_params['slot_classes'],
            model_params['vocab'],
            model_params['config'],
            model_params['save_path']
        )
        result.max_sequence_length = model_params['max_sequence_length']
        result.model = load_model(in_file_name)
        return result

    @classmethod
    def _load_model_params(self, in_file_name):
        with codecs.getreader('utf-8')(open(in_file_name)) as input:
            return json.load(input)
            self.vocab = params_json['vocab']
            self.max_sequence_length = params_json['max_sequence_length']
            self.slots = params_json['slots']
            self.slot_classes = params_json['slot_classes']
            self.config = params_json['sonfig']
            self.save_path = params_json['save_path']
