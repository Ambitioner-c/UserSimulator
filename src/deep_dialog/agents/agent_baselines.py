"""
Created on May 25, 2016
@author: xiul, t-zalipt
"""

import copy
import random
import dialog_config
from .agent import Agent


class InformAgent(Agent):
    """
    一个测试系统的简单代理器.
    这个代理器会简单地通知所有的槽，然后发出：taskcomplete。
    """

    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = ''
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1
        self.current_slot_id = 0

    def state_to_action(self, state):
        """ 运行当前状态下的策略并产生一个活动 """
        
        self.state['turn'] += 2
        # if self.current_slot_id < len(self.slot_set.keys()):
        if self.current_slot_id < self.slot_cardinality:
            slot = self.slot_set.keys()[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {slot: "PLACEHOLDER"}
            act_slot_response['request_slots'] = {}
            act_slot_response['turn'] = self.state['turn']
        else:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}, 'turn': self.state['turn']}
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}


class RequestAllAgent(Agent):
    """一个测试系统的简单代理器。这个代理器会简单地请求所有的槽，然后发出：thanks()"""
    
    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = ''
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1
        self.current_slot_id = 0

    def state_to_action(self, state):
        """ 运行当前状态下的策略并产生一个活动 """
        
        self.state['turn'] += 2
        if self.current_slot_id < len(dialog_config.sys_request_slots):
            slot = dialog_config.sys_request_slots[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "PLACEHOLDER"}
            act_slot_response['turn'] = self.state['turn']
        else:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}, 'turn': self.state['turn']}
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}


class RandomAgent(Agent):
    """ 这是一个测试接口的简单代理器。这个代理器会随机地选择活动。 """

    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = ''
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1

    def state_to_action(self, state):
        """ 运行当前状态下的策略并产生一个活动 """
        
        self.state['turn'] += 2
        act_slot_response = copy.deepcopy(random.choice(dialog_config.feasible_actions))
        act_slot_response['turn'] = self.state['turn']
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}


class EchoAgent(Agent):
    """
    一个通知所有requested槽的简单代理器，
    当用户停止请求时发出通知(taskcomplete)。
    """

    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = ''
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1

    def state_to_action(self, state):
        """ 运行当前状态下的策略并产生一个活动 """
        user_action = state['user_action']
        
        self.state['turn'] += 2
        act_slot_response = {}
        act_slot_response['inform_slots'] = {}
        act_slot_response['request_slots'] = {}
        ########################################################################
        # 找出用户是否有任何请求
        # 如果有的话，通知它
        ########################################################################
        if user_action['diaact'] == 'request':
            requested_slot = user_action['request_slots'].keys()[0]

            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'][requested_slot] = "PLACEHOLDER"
        else:
            act_slot_response['diaact'] = "thanks"

        act_slot_response['turn'] = self.state['turn']
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}


class RequestBasicsAgent(Agent):
    """
    一个测试系统的简单代理器。
    这个代理器会简单地请求所有基本槽，然后发出：thacks()。
    """
    
    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = 'UNK'
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1
        self.current_slot_id = 0
        self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']
        self.phase = 0

    def state_to_action(self, state):
        """ 运行当前状态下的策略并产生一个活动 """
        
        self.state['turn'] += 2
        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
            act_slot_response['turn'] = self.state['turn']
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"}, 'request_slots': {}, 'turn': self.state['turn']}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}, 'turn': self.state['turn']}
        else:
            raise Exception("THIS SHOULD NOT BE POSSIBLE (AGENT CALLED IN UNANTICIPATED WAY)")
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}

