import logging

import operator
import theano.tensor as tt
from keras.engine import Input
from keras.engine import Merge

from model_base import ModelBase

from keras.layers import Dense, Embedding, Dropout


class ModelKeras(ModelBase):
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

        input_layers = [emb]
        sum_layer = Merge(input_layers, mode='sum')(emb)
        last_layer = sum_layer
        if self.config.get('input_n_layers', None):
            last_layer = self._build_mlp(
                [self.config['input_n_hidden']] * self.config['input_n_layers'],
                last_layer,
                'input'
            )

        if self.config.get('token_supervision', None):
            last_layer = self._build_mlp(
                [len(self.slots) * 2],
                ['sigmoid'],
                'ts',
                last_layer
            )

            # TODO: HOW'S THAT?
            # y_tokens_label = tt.itensor3()
            # token_supervision_loss_layer = TokenSupervisionLossLayer()
            # token_supervision_loss_layer.connect(
            #     slot_value_pred,
            #     y_tokens_label
            # )

        logging.info('There are %d input layers.' % self.config['input_n_layers'])

        h_t_layer = IdentityInput(None, n_cells)
        mlps = []
        mlp_params = []
        for slot in slots:
            n_classes = len(slot_classes[slot])
            slot_mlp = MLP(
                [oclf_n_hidden  ] * oclf_n_layers + [n_classes],
                [oclf_activation] * oclf_n_layers + ['softmax'],
                [0.0            ] * oclf_n_layers + [0.0      ],
                name="mlp_%s" % slot
            )
            slot_mlp.connect(h_t_layer)
            mlps.append(slot_mlp)
            mlp_params.extend(slot_mlp.get_params())


        for i in range(rnn_n_layers):
            # Forward LSTM layer.
            logging.info('Creating LSTM layer with %d neurons.' % (n_cells))
            if x_include_mlp:
                f_lstm_layer = LstmWithMLP(
                    name="flstm_%d" % i,
                    size=n_cells,
                    seq_output=True,
                    out_cells=False,
                    peepholes=lstm_peepholes,
                    p_drop=p_drop,
                    enable_branch_exp=enable_branch_exp,
                    mlps=mlps
                )
            else:
                f_lstm_layer = LstmRecurrent(
                    name="flstm_%d" % i,
                    size=n_cells,
                    seq_output=True,
                    out_cells=False,
                    peepholes=lstm_peepholes,
                    p_drop=p_drop,
                    enable_branch_exp=enable_branch_exp
                )
            f_lstm_layer.connect(prev_layer)

            if lstm_bidi:
                b_lstm_layer = LstmRecurrent(
                    name="blstm_%d" % i,
                    size=n_cells,
                    seq_output=True,
                    out_cells=False,
                    backward=True,
                    peepholes=lstm_peepholes,
                    p_drop=p_drop,
                    enable_branch_exp=enable_branch_exp
                )
                b_lstm_layer.connect(prev_layer)
                lstm_zip = ZipLayer(
                    concat_axis=2,
                    layers=[f_lstm_layer, b_lstm_layer]
                )
                prev_layer = lstm_zip
            else:
                prev_layer = f_lstm_layer

        assert prev_layer is not None

        y_seq_id = tt.ivector()
        y_time = tt.ivector()
        y_weight = tt.vector()
        y_label = {}
        for slot in slots:
            y_label[slot] = tt.ivector(name='y_label_%s' % slot)

        cpt = CherryPick()
        cpt.connect(prev_layer, y_time, y_seq_id)

        costs = []
        predictions = []
        for slot, slot_lstm_mlp in zip(slots, mlps):
            logging.info('Building output classifier for %s.' % slot)
            n_classes = len(slot_classes[slot])
            if oclf_n_layers > 0:
                slot_mlp = MLP(
                    [oclf_n_hidden  ] * oclf_n_layers,
                    [oclf_activation] * oclf_n_layers,
                    [p_drop         ] * oclf_n_layers,
                    name="mlp_%s" % slot
                )
                slot_mlp.connect(cpt)

            slot_softmax = BiasedSoftmax(
                name='softmax_%s' % slot,
                size=n_classes
            )
            if oclf_n_layers > 0:
                slot_softmax.connect(slot_mlp)
            else:
                slot_softmax.connect(cpt)

            predictions.append(slot_softmax.output(dropout_active=False))

            slot_objective = WeightedCrossEntropyObjective()
            slot_objective.connect(
                y_hat_layer=slot_softmax,
                y_true=y_label[slot],
                y_weights=y_weight
            )
            costs.append(slot_objective)
        if token_supervision:
            costs.append(token_supervision_loss_layer)

        cost = SumOut()
        cost.connect(*costs)  #, scale=1.0 / len(slots))
        self.params = params = list(cost.get_params())
        n_params = sum(p.get_value().size for p in params)
        logging.info('This model has %d parameters:' % n_params)
        for param in sorted(params, key=lambda x: x.name):
            logging.info(
                '  - %20s: %10d' % (param.name, param.get_value().size, )
            )
        cost_value = cost.output(dropout_active=True)

        lr = tt.scalar('lr')
        clipnorm = 0.5
        reg = updates.Regularizer(l1=l1, l2=l2)

        loss_args = list(input_args)
        loss_args += [y_seq_id, y_time]
        loss_args += [y_weight]
        loss_args += [y_label[slot] for slot in slots]
        if token_supervision:
            loss_args += [y_tokens_label]

    def init_loaded(self):
        pass

    def init_word_embeddings(self, w):
        self.input_emb.set_value(w)

    def _build_mlp(self, in_layer_sizes, in_activation, in_input, in_name):
        last_layer = in_input
        for layer_index, layer_size in enumerate(in_layer_sizes):
            input_transform_i = Dense(
                layer_size,
                name='{}_{}'.format(in_name, layer_index),
                activation=in_activation,
                init='normal'
            )(last_layer)
            last_layer = input_transform_i
            dropout = Dropout(self.config['p_drop'])(last_layer)
            last_layer = dropout
        return last_layer
