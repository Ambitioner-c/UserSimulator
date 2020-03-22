import pickle

from dialog_system import text_to_dict
from usersims import RuleSimulator

import dialog_config

from nlu import nlu
from nlg import nlg


max_turn = 40
num_episodes = 150

agt = 5
usr = 1

dict_path = r'deep_dialog/data/dicts.v3.p'
goal_file_path = r'deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p'

# load the user goals from .p file
all_goal_set = pickle.load(open(goal_file_path, 'rb'), encoding='iso-8859-1')

# split goal set
split_fold = 5
goal_set = {'train': [], 'valid': [], 'test': [], 'all': []}
for u_goal_id, u_goal in enumerate(all_goal_set):
    if u_goal_id % split_fold == 1:
        goal_set['test'].append(u_goal)
    else:
        goal_set['train'].append(u_goal)
    goal_set['all'].append(u_goal)
# end split goal set

movie_kb_path = r'deep_dialog/data/movie_kb.1k.p'
movie_kb = pickle.load(open(movie_kb_path, 'rb'), encoding='iso-8859-1')

act_set = text_to_dict(r'deep_dialog/data/dia_acts.txt')
slot_set = text_to_dict(r'deep_dialog/data/slot_set.txt')

# a movie dictionary for user simulator - slot:possible values
movie_dictionary = pickle.load(open(dict_path, 'rb'), encoding='iso-8859-1')

dialog_config.run_mode = 1
dialog_config.auto_suggest = 0

# Parameters for User Simulators
usersim_params = {}
usersim_params['max_turn'] = max_turn
usersim_params['slot_err_probability'] = 0.00
usersim_params['slot_err_mode'] = 0
usersim_params['intent_err_probability'] = 0.00
usersim_params['simulator_run_mode'] = 1
usersim_params['simulator_act_level'] = 1
usersim_params['learning_phase'] = 'all'

user_sim = RuleSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)

# load trained NLG model
nlg_model_path = r'deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p'
diaact_nl_pairs = r'deep_dialog/data/dia_act_nl_pairs.v6.json'
nlg_model = nlg()
nlg_model.load_nlg_model(nlg_model_path)
nlg_model.load_predefine_act_nl_pairs(diaact_nl_pairs)

user_sim.set_nlg_model(nlg_model)


################################################################################
# load trained NLU model
################################################################################
nlu_model_path = r'deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p'
nlu_model = nlu()
nlu_model.load_nlu_model(nlu_model_path)

user_sim.set_nlu_model(nlu_model)

# print(dialog_config.start_dia_acts.keys())
user_action = user_sim.initialize_episode()
print(user_action)
