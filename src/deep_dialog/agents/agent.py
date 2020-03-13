"""
Created on May 17, 2016
@author: xiul, t-zalipt
"""


class Agent:
    """所有代理类的原型，定义了他们必须支持的接口"""
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        """ 代理类的构造函数

        参数:
        movie_dict      --  这个东西现在在这里但并不属于这里——代理器并不知道什么电影
        act_set         --  活动集  #### 这个集合不应该更抽象一点吗？我们不想让我们的代理器更广泛地使用吗？
        slot_set        --  可用的槽集
        """
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())
        
        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']

        self.current_action = None
        self.nlg_model = None
        self.nlu_model = None

    def initialize_episode(self):
        """ 初始化一个事件。每当一个新的事件运行时都会调用这个函数。 """
        self.current_action = {}                      # TODO 将此变量名改成“当前操作”
        self.current_action['diaact'] = None          # TODO 如果它发生了一个活动，把它称为一个状态还有意义吗？又是哪一个活动？ 最近的那个吗？
        self.current_action['inform_slots'] = {}
        self.current_action['request_slots'] = {}
        self.current_action['turn'] = 0

    def state_to_action(self, state, available_actions):
        """ 根据当前探索/开采策略，获取当前状态并返回一个活动

        我们定义了一个灵活的代理器，他们可以在act_slot表征或者act_slot_value表征上操作。
        我们也定义了一个灵活的响应，返回一个键为[act_slot_response, act_slot_value_response]的词典。这种方式的command-line代理器可以继续操作values。

        参数:
        state               --   一个(history, kb_results)元组，history是之前活动的序列，kb_results包含匹配到正确约束的结果数量信息。
        user_action         --   一个用来运行命令行代理器的legacy表征。我们应该移除这个ASAP，但是还没有。
        available_actions   --   一个当前状态允许的活动列表

        返回:
        act_slot_action         --   An action consisting of one act and >= 0 slots as well as which slots are informed vs requested.
        act_slot_value_action   --   An action consisting of acts slots and values in the legacy format. This can be used in the future for training agents that take value into account and interact directly with the database
        """
        act_slot_response = None
        act_slot_value_response = None
        return {"act_slot_response": act_slot_response, "act_slot_value_response": act_slot_value_response}

    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
        """  寄存来自环境中的反馈，存储为将来的训练数据

        参数:
        s_t                 --  上次采取活动的状态
        a_t                 --  上一个代理器活动
        reward              --  活动后立即得到的奖励
        s_tplus1            --  最新活动后的状态转变
        episode_over        --  表示这是不是最终活动的boolean值

        返回:
        None
        """
        pass

    def set_nlg_model(self, nlg_model):
        self.nlg_model = nlg_model  
    
    def set_nlu_model(self, nlu_model):
        self.nlu_model = nlu_model

    def add_nl_to_action(self, agent_action):
        """将NL添加到代理的会话活动中"""

        if agent_action['act_slot_response']:
            agent_action['act_slot_response']['nl'] = ""
            # self.nlg_model.translate_diaact(agent_action['act_slot_response']) # NLG
            user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(agent_action['act_slot_response'], 'agt')
            agent_action['act_slot_response']['nl'] = user_nlg_sentence
        elif agent_action['act_slot_value_response']:
            agent_action['act_slot_value_response']['nl'] = ""
            # self.nlg_model.translate_diaact(agent_action['act_slot_value_response']) # NLG
            user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(agent_action['act_slot_value_response'], 'agt')
            agent_action['act_slot_response']['nl'] = user_nlg_sentence
