"""
Created on Jun 13, 2016
An LSTM decoder - add tanh after cell before output gate
@author: xiul
"""

from .seq_seq import SeqToSeq
from .utils import *


class LSTM(SeqToSeq):
    def __init__(self, input_size, hidden_size, output_size):
        self.model = {}
        # Recurrent weights: take x_t, h_{t-1}, and bias unit, and produce the 3 gates and the input to cell signal
        # (1526 + 80 + 1, 4 * 80) = (1607, 320)
        self.model['WLSTM'] = init_weights(input_size + hidden_size + 1, 4 * hidden_size)
        # Hidden-Output Connections
        # (80, 116)
        self.model['Wd'] = init_weights(hidden_size, output_size) * 0.1
        # (1, 116)
        self.model['bd'] = np.zeros((1, output_size))

        self.update = ['WLSTM', 'Wd', 'bd']
        self.regularize = ['WLSTM', 'Wd']

        self.step_cache = {}
        
    """ 激活函数: Sigmoid, or tanh, or ReLu """
    def fwd_pass(self, x_s, params, **kwargs):
        predict_mode = kwargs.get('predict_mode', False)
        
        Ws = x_s['word_vectors']
        
        WLSTM = self.model['WLSTM']
        # (5, 1526)
        n, xd = Ws.shape

        # d = 80
        d = self.model['Wd'].shape[0]           # size of hidden layer
        # 输入的隐藏状态(5, 1607)
        Hin = np.zeros((n, WLSTM.shape[0]))     # xt, ht-1, bias
        # 输出的隐藏状态(5, 80)
        Hout = np.zeros((n, d))
        # 遗忘门的值(5, 4*80) = (5, 320)
        IFOG = np.zeros((n, 4*d))
        IFOGf = np.zeros((n, 4*d))              # after nonlinearity
        # 输入的细胞状态(5, 80)
        Cellin = np.zeros((n, d))
        # 输出的细胞状态(5, 80)
        Cellout = np.zeros((n, d))
    
        for t in range(n):
            if t == 0:
                prev = np.zeros(d)
            else:
                Hout[t - 1]
            # prev = np.zeros(d) if t==0 else Hout[t-1]
            # [1, word_vectors, 80个0]
            Hin[t, 0] = 1   # bias
            Hin[t, 1:1+xd] = Ws[t]
            Hin[t, 1+xd:] = prev
            
            # compute all gate activations. dots:
            # 计算遗忘门的值
            # (1, 320) = (1, 1607)点积(1607, 320)
            IFOG[t] = Hin[t].dot(WLSTM)
            
            IFOGf[t, :3*d] = 1/(1+np.exp(-IFOG[t, :3*d]))       # sigmoids; these are three gates
            IFOGf[t, 3*d:] = np.tanh(IFOG[t, 3*d:])             # tanh for input value

            # 初始的输入细胞
            Cellin[t] = IFOGf[t, :d] * IFOGf[t, 3*d:]
            if t > 0:
                # 后续的输入细胞
                Cellin[t] += IFOGf[t, d:2*d]*Cellin[t-1]
            
            Cellout[t] = np.tanh(Cellin[t])
            
            Hout[t] = IFOGf[t, 2*d:3*d] * Cellout[t]

        Wd = self.model['Wd']
        bd = self.model['bd']
            
        Y = Hout.dot(Wd)+bd
            
        cache = {}
        if not predict_mode:
            cache['WLSTM'] = WLSTM
            cache['Hout'] = Hout
            cache['Wd'] = Wd
            cache['IFOGf'] = IFOGf
            cache['IFOG'] = IFOG
            cache['Cellin'] = Cellin
            cache['Cellout'] = Cellout
            cache['Ws'] = Ws
            cache['Hin'] = Hin
            
        return Y, cache
    
    """ Backward Pass """
    def bwd_pass(self, d_y, cache):
        Wd = cache['Wd']
        Hout = cache['Hout']
        IFOG = cache['IFOG']
        IFOGf = cache['IFOGf']
        Cellin = cache['Cellin']
        Cellout = cache['Cellout']
        Hin = cache['Hin']
        WLSTM = cache['WLSTM']
        Ws = cache['Ws']
        
        n, d = Hout.shape

        # backprop the hidden-output layer
        dWd = Hout.transpose().dot(d_y)
        dbd = np.sum(d_y, axis=0, keepdims=True)
        dHout = d_y.dot(Wd.transpose())

        # backprop the LSTM
        dIFOG = np.zeros(IFOG.shape)
        dIFOGf = np.zeros(IFOGf.shape)
        dWLSTM = np.zeros(WLSTM.shape)
        dHin = np.zeros(Hin.shape)
        dCellin = np.zeros(Cellin.shape)
        dCellout = np.zeros(Cellout.shape)
        
        for t in reversed(range(n)):
            dIFOGf[t, 2*d:3*d] = Cellout[t] * dHout[t]
            dCellout[t] = IFOGf[t, 2*d:3*d] * dHout[t]
            
            dCellin[t] += (1-Cellout[t]**2) * dCellout[t]
            
            if t > 0:
                dIFOGf[t, d:2*d] = Cellin[t-1] * dCellin[t]
                dCellin[t-1] += IFOGf[t, d:2*d] * dCellin[t]
            
            dIFOGf[t, :d] = IFOGf[t, 3*d:] * dCellin[t]
            dIFOGf[t, 3*d:] = IFOGf[t, :d] * dCellin[t]
            
            # backprop activation functions
            dIFOG[t, 3*d:] = (1-IFOGf[t, 3*d:]**2) * dIFOGf[t, 3*d:]
            y = IFOGf[t, :3*d]
            dIFOG[t, :3*d] = (y*(1-y)) * dIFOGf[t, :3*d]
            
            # backprop matrix multiply
            dWLSTM += np.outer(Hin[t], dIFOG[t])
            dHin[t] = dIFOG[t].dot(WLSTM.transpose())
      
            if t > 0:
                dHout[t-1] += dHin[t, 1+Ws.shape[1]:]
        
        # dXs = dXsh.dot(Wxh.transpose())
        return {'WLSTM': dWLSTM, 'Wd': dWd, 'bd': dbd}
