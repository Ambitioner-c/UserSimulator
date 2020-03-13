import json
import argparse
import pickle

from nlu import nlu
import chardet
import numpy as np
from nlu.utils import *


def fwd_pass(x_s, params, **kwargs):
    hidden_size = model_params['model']['Wd'].shape[0]
    output_size = model_params['model']['Wd'].shape[1]
    input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
    # print(input_size)
    # print(hidden_size)
    # print(output_size)

    model = {}
    # Recurrent weights: take x_t, h_{t-1}, and bias unit, and produce the 3 gates and the input to cell signal
    model['WLSTM'] = init_weights(input_size + hidden_size + 1, 4 * hidden_size)
    # print(model['WLSTM'].shape)
    model['bWLSTM'] = init_weights(input_size + hidden_size + 1, 4 * hidden_size)

    # Hidden-Output Connections
    model['Wd'] = init_weights(hidden_size, output_size) * 0.1
    model['bd'] = np.zeros((1, output_size))

    # Backward Hidden-Output Connections
    model['bWd'] = init_weights(hidden_size, output_size) * 0.1
    model['bbd'] = np.zeros((1, output_size))

    predict_mode = kwargs.get('predict_mode', False)

    Ws = x_s['word_vectors']

    WLSTM = model['WLSTM']
    bWLSTM = model['bWLSTM']

    n, xd = Ws.shape
    # print(n, xd)

    d = model['Wd'].shape[0]  # size of hidden layer
    # print(d)
    Hin = np.zeros((n, WLSTM.shape[0]))  # xt, ht-1, bias
    Hout = np.zeros((n, d))
    IFOG = np.zeros((n, 4 * d))
    IFOGf = np.zeros((n, 4 * d))  # after nonlinearity
    Cellin = np.zeros((n, d))
    Cellout = np.zeros((n, d))

    # backward
    bHin = np.zeros((n, WLSTM.shape[0]))  # xt, ht-1, bias
    bHout = np.zeros((n, d))
    bIFOG = np.zeros((n, 4 * d))
    bIFOGf = np.zeros((n, 4 * d))  # after nonlinearity
    bCellin = np.zeros((n, d))
    bCellout = np.zeros((n, d))

    for t in range(n):
        prev = np.zeros(d) if t == 0 else Hout[t - 1]
        Hin[t, 0] = 1  # bias
        Hin[t, 1:1 + xd] = Ws[t]
        Hin[t, 1 + xd:] = prev

        # compute all gate activations. dots:
        IFOG[t] = Hin[t].dot(WLSTM)

        IFOGf[t, :3 * d] = 1 / (1 + np.exp(-IFOG[t, :3 * d]))  # sigmoids; these are three gates
        IFOGf[t, 3 * d:] = np.tanh(IFOG[t, 3 * d:])  # tanh for input value

        Cellin[t] = IFOGf[t, :d] * IFOGf[t, 3 * d:]
        if t > 0:
            Cellin[t] += IFOGf[t, d:2 * d] * Cellin[t - 1]

        Cellout[t] = np.tanh(Cellin[t])
        Hout[t] = IFOGf[t, 2 * d:3 * d] * Cellout[t]

        # backward hidden layer
        b_t = n - 1 - t
        bprev = np.zeros(d) if t == 0 else bHout[b_t + 1]
        bHin[b_t, 0] = 1
        bHin[b_t, 1:1 + xd] = Ws[b_t]
        bHin[b_t, 1 + xd:] = bprev

        bIFOG[b_t] = bHin[b_t].dot(bWLSTM)
        bIFOGf[b_t, :3 * d] = 1 / (1 + np.exp(-bIFOG[b_t, :3 * d]))
        bIFOGf[b_t, 3 * d:] = np.tanh(bIFOG[b_t, 3 * d:])

        bCellin[b_t] = bIFOGf[b_t, :d] * bIFOGf[b_t, 3 * d:]
        if t > 0:
            bCellin[b_t] += bIFOGf[b_t, d:2 * d] * bCellin[b_t + 1]

        bCellout[b_t] = np.tanh(bCellin[b_t])
        bHout[b_t] = bIFOGf[b_t, 2 * d:3 * d] * bCellout[b_t]

    Wd = model['Wd']
    bd = model['bd']
    fY = Hout.dot(Wd) + bd

    bWd = model['bWd']
    bbd = model['bbd']
    bY = bHout.dot(bWd) + bbd

    Y = fY + bY

    cache = {}
    if not predict_mode:
        cache['WLSTM'] = WLSTM
        cache['Hout'] = Hout
        cache['Wd'] = Wd
        cache['IFOGf'] = IFOGf
        cache['IFOG'] = IFOG
        cache['Cellin'] = Cellin
        cache['Cellout'] = Cellout
        cache['Hin'] = Hin

        cache['bWLSTM'] = bWLSTM
        cache['bHout'] = bHout
        cache['bWd'] = bWd
        cache['bIFOGf'] = bIFOGf
        cache['bIFOG'] = bIFOG
        cache['bCellin'] = bCellin
        cache['bCellout'] = bCellout
        cache['bHin'] = bHin

        cache['Ws'] = Ws
    # print(cache)

    return Y, cache


def parse_nlu_to_diaact(nlu_vector, string):
        """ Parse BIO and Intent into Dia-Act """

        tmp = 'BOS ' + string + ' EOS'
        words = tmp.lower().split(' ')

        diaact = {}
        diaact['diaact'] = "inform"
        diaact['request_slots'] = {}
        diaact['inform_slots'] = {}

        intent = nlu_vector[-1]
        index = 1
        pre_tag = nlu_vector[0]
        pre_tag_index = 0

        slot_val_dict = {}

        while index < (len(nlu_vector) - 1):  # except last Intent tag
            cur_tag = nlu_vector[index]
            if cur_tag == 'O' and pre_tag.startswith('B-'):
                slot = pre_tag.split('-')[1]
                slot_val_str = ' '.join(words[pre_tag_index:index])
                slot_val_dict[slot] = slot_val_str
            elif cur_tag.startswith('B-') and pre_tag.startswith('B-'):
                slot = pre_tag.split('-')[1]
                slot_val_str = ' '.join(words[pre_tag_index:index])
                slot_val_dict[slot] = slot_val_str
            elif cur_tag.startswith('B-') and pre_tag.startswith('I-'):
                if cur_tag.split('-')[1] != pre_tag.split('-')[1]:
                    slot = pre_tag.split('-')[1]
                    slot_val_str = ' '.join(words[pre_tag_index:index])
                    slot_val_dict[slot] = slot_val_str
            elif cur_tag == 'O' and pre_tag.startswith('I-'):
                slot = pre_tag.split('-')[1]
                slot_val_str = ' '.join(words[pre_tag_index:index])
                slot_val_dict[slot] = slot_val_str

            if cur_tag.startswith('B-'):
                pre_tag_index = index

            pre_tag = cur_tag
            index += 1

        if cur_tag.startswith('B-') or cur_tag.startswith('I-'):
            slot = cur_tag.split('-')[1]
            slot_val_str = ' '.join(words[pre_tag_index:-1])
            slot_val_dict[slot] = slot_val_str

        if intent != 'null':
            arr = intent.split('+')
            diaact['diaact'] = arr[0]
            diaact['request_slots'] = {}
            for ele in arr[1:]:
                # request_slots.append(ele)
                diaact['request_slots'][ele] = 'UNK'

        diaact['inform_slots'] = slot_val_dict

        # add rule here
        for slot in diaact['inform_slots'].keys():
            slot_val = diaact['inform_slots'][slot]
            if slot_val.startswith('bos'):
                slot_val = slot_val.replace('bos', '', 1)
                diaact['inform_slots'][slot] = slot_val.strip(' ')

                # rule for taskcomplete
                if 'request_slots' in diaact.keys():
                    if 'taskcomplete' in diaact['request_slots'].keys():
                        del diaact['request_slots']['taskcomplete']
                        diaact['inform_slots']['taskcomplete'] = 'PLACEHOLDER'

                    # rule for request
                    if len(diaact['request_slots']) > 0:
                        diaact['diaact'] = 'request'

        return diaact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nlu_model_path', dest='nlu_model_path', type=str,
                        default='./deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p',
                        help='path to the NLU model file')
    args = parser.parse_args()
    params = vars(args)
    # print('Dialog Parameters: ')
    # print(json.dumps(params, indent=2))

    ################################################################################
    # load trained NLU model
    ################################################################################
    nlu_model_path = params['nlu_model_path']
    model_params = pickle.load(open(nlu_model_path, 'rb'), encoding='iso-8859-1')

    # # (80, 116)
    # hidden_size = model_params['model']['Wd'].shape[0]
    # output_size = model_params['model']['Wd'].shape[1]
    #
    # # print(model_params['params']['model'])
    #
    # input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
    # # print(model_params['word_dict'].keys())

    # nlu_model = nlu()
    # nlu_model.load_nlu_model(nlu_model_path)
    # rep = nlu_model.parse_str_to_vector("I'm the hero.")
    # print(rep)

    # string = "What are the drama films today?"
    string = "Watch Iron Man movie this afternoon."
    tmp = 'BOS ' + string + ' EOS'
    words = tmp.lower().split(' ')
    # print(words)

    # print(len(model_params['word_dict']))
    vecs = np.zeros((len(words), len(model_params['word_dict'])))
    for w_index, w in enumerate(words):
        # print(w_index)
        # print(w)
        if w.endswith(',') or w.endswith('.') or w.endswith('?'):
            w = w[0:-1]
            # print(w)
        if w in model_params['word_dict'].keys():
            vecs[w_index][model_params['word_dict'][w]] = 1
            # print('y')
        else:
            vecs[w_index][model_params['word_dict']['unk']] = 1
            # print('n')

    rep = {}
    rep['word_vectors'] = vecs
    rep['raw_seq'] = string

    Ys, cache = fwd_pass(rep, model_params, predict_model=True)

    maxes = np.amax(Ys, axis=1, keepdims=True)
    # print(maxes)

    e = np.exp(Ys - maxes)  # 为了使数值稳定，应移到良好的数值范围内
    # print(e)
    probs = e / np.sum(e, axis=1, keepdims=True)
    # print(probs)
    if np.all(np.isnan(probs)):
        probs = np.zeros(probs.shape)

    # print(model_params['tag_set'])
    inverse_tag_dict = {model_params['tag_set'][k]: k for k in model_params['tag_set'].keys()}
    # print(inverse_tag_dict)
    # 对意图标签的特殊处理
    for tag_id in inverse_tag_dict.keys():
        if inverse_tag_dict[tag_id].startswith('B-') or inverse_tag_dict[tag_id].startswith('I-') or \
                inverse_tag_dict[tag_id] == 'O':
            probs[-1][tag_id] = 0

    # print(probs)
    pred_words_indices = np.nanargmax(probs, axis=1)
    # print(pred_words_indices)

    pred_tags = [inverse_tag_dict[index] for index in pred_words_indices]
    print(pred_tags)

    my_diaact = parse_nlu_to_diaact(pred_tags, string)
    print(my_diaact)
