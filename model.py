import logging
import operator

from keras.engine import Input, Model
from keras.engine import merge
from keras.layers import Dense, Embedding, Dropout, LSTM, Reshape

from model_base import ModelBase


class ModelKeras(ModelBase):
    def __init__(self, slots, slot_classes, vocab, config, in_save_path):
        ModelBase.__init__(self)
        self.vocab = vocab
        self.slots = slots
        self.slot_classes = slot_classes
        self.slot_classes_reversed = {
            slot_class: map(
                lambda x: x[0],
                sorted(
                    self.slot_classes[slot_class].items(),
                    key=operator.itemgetter(1)
                )
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
        if self.config.get('init_emb_from', None):
            # input_token_layer.init_from(init_emb_from, vocab)
            logging.info(
                'Initializing token embeddings from: {}'.format(
                    self.config['init_emb_from']
                )
            )
        else:
            logging.info('Initializing token embedding randomly.')

        # not used for now
        # input_layers = [emb]
        # sum_layer = Merge(input_layers, mode='sum')(emb)

        last_layer = Reshape(
            (self.max_sequence_length, self.config['emb_size'])
        )(emb)
        if self.config.get('input_n_layers', None):
            last_layer = self._build_mlp(
                'input',
                [self.config['input_n_hidden']] * self.config['input_n_layers'],
                last_layer
            )

        if self.config.get('token_supervision', None):
            last_layer = self._build_mlp(
                'ts',
                [len(self.slots) * 2],
                ['sigmoid'],
                last_layer
            )

            # TODO: HOW'S THAT?
            # y_tokens_label = tt.itensor3()
            # token_supervision_loss_layer = TokenSupervisionLossLayer()
            # token_supervision_loss_layer.connect(
            #     slot_value_pred,
            #     y_tokens_label
            # )

        logging.info(
            'There are {} input layers'.format(self.config['input_n_layers'])
        )

        for i in range(self.config['rnn_n_layers']):
            # Forward LSTM layer
            logging.info(
                'Creating LSTM layer with {} neurons'.format(
                    self.config['n_cells']
                )
            )
            n_cells = self.config['n_cells']
            f_lstm_layer = LSTM(
                n_cells,
                name='flstm_{}'.format(i),
                dropout_W=self.config['p_drop'],
                dropout_U=self.config['p_drop'],
                # return_sequences=True
            )(last_layer)

            # Backward LSTM
            if self.config['lstm_bidi']:
                b_lstm_layer = LSTM(
                    n_cells,
                    name='blstm_{}'.format(i),
                    # return_sequences=True,
                    go_backwards=True,
                    dropout_W=self.config['p_drop'],
                    dropout_U=self.config['p_drop']
                )(last_layer)
                lstm_merge = merge([f_lstm_layer, b_lstm_layer], mode='sum')
                last_layer = lstm_merge
            else:
                last_layer = f_lstm_layer

        assert last_layer is not None

        costs = []
        predictions = []
        n_layers = self.config['n_layers']
        if n_layers:
            for slot in self.slots:
                logging.info('Building output classifier for %s.' % slot)
                slot_mlp = self._build_mlp(
                    'mlp_%s' % slot,
                    [len(self.slot_classes[slot])] * n_layers,
                    [self.config['oclf_activation']] * n_layers,
                    last_layer
                )
                predictions.append(slot_mlp)
            merge_layer = merge(predictions, mode='concat')

        self.model = Model(input=input_layer, output=merge_layer)
        self.model.compile(
            optimizer=self.config['opt_type'],
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def init_loaded(self):
        pass

    def init_word_embeddings(self, w):
        self.input_emb.set_value(w)

    def _build_mlp(self, in_name, in_layer_sizes, in_activations, in_input):
        last_layer = in_input
        for layer_index, layer_params in enumerate(zip(in_layer_sizes, in_activations)):
            layer_size, layer_activation = layer_params
            input_transform_i = Dense(
                layer_size,
                name='{}_{}'.format(in_name, layer_index),
                activation=layer_activation,
                init='normal'
            )(last_layer)
            last_layer = input_transform_i
            dropout = Dropout(self.config['p_drop'])(last_layer)
            last_layer = dropout
        return last_layer
