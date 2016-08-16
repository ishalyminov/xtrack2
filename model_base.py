import numpy as np


class ModelBase(object):
    def get_frame_label_from_prediction(
        self,
        in_predictions,
        string_values=True
    ):
        total_values_number = sum(map(len, self.slot_classes.values()))
        assert total_values_number == in_predictions.shape[1]

        result = []
        for prediction in in_predictions:
            frame_label = {}
            slot_offset = 0
            for slot in self.slot_classes_reversed:
                slot_values_number = len(self.slot_classes[slot])
                slot_value_index = np.argmax(
                    prediction[slot_offset:slot_offset + slot_values_number]
                )
                frame_label[slot] = \
                    self.slot_classes_reversed[slot][slot_value_index] \
                    if string_values \
                    else slot_value_index
                slot_offset += slot_values_number
            result.append(frame_label)
        return result

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

    def prepare_data_train(self, sequences, slots):
        X, y = self.prepare_data(sequences, slots, with_labels=True)
        self.init_model(X)
        return X, y

    def prepare_data_test(self, sequences, slots, with_labels=True):
        data = self.prepare_data(sequences, slots, with_labels)
        return data

    def prepare_data(self, seqs, slots, with_labels=True):
        x = []
        y_labels = []
        x_one_hot_length = len(self.vocab)

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