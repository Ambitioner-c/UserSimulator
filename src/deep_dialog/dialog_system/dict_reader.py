"""
Created on May 18, 2016
@author: xiul, t-zalipt
"""


def text_to_dict(path):
    """ 以字典形式读入文本文件，其中键是文本，值是索引（行号） """
    
    slot_set = {}
    with open(path, 'r') as f:
        index = 0
        for line in f.readlines():
            slot_set[line.strip('\n').strip('\r')] = index
            index += 1
    return slot_set
