"""
Created on May 17, 2016
@author: xiul, t-zalipt
"""

sys_request_slots = ['moviename', 'theater', 'starttime', 'date',
                     'numberofpeople', 'genre', 'state', 'city',
                     'zip', 'critic_rating', 'mpaa_rating', 'distanceconstraints',
                     'video_format', 'theater_chain', 'price', 'actor',
                     'description', 'other', 'numberofkids']
sys_inform_slots = ['moviename', 'theater', 'starttime', 'date',
                    'genre', 'state', 'city', 'zip',
                    'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format',
                    'theater_chain', 'price', 'actor', 'description',
                    'other', 'numberofkids', 'taskcomplete', 'ticket']

start_dia_acts = {
    # 'greeting':[],
    'request': ['moviename', 'starttime', 'theater', 'city',
                'state', 'date', 'genre', 'ticket',
                'numberofpeople']
}

################################################################################
# 对话状态
################################################################################
FAILED_DIALOG = -1
SUCCESS_DIALOG = 1
NO_OUTCOME_YET = 0

################################################################################
# 奖励
################################################################################
SUCCESS_REWARD = 50
FAILURE_REWARD = 0
PER_TURN_REWARD = 0

################################################################################
#  特殊槽值
################################################################################
I_DO_NOT_CARE = "I do not care"
NO_VALUE_MATCH = "NO VALUE MATCHES!!!"
TICKET_AVAILABLE = 'Ticket Available'

################################################################################
#  约束检查
################################################################################
CONSTRAINT_CHECK_FAILURE = 0
CONSTRAINT_CHECK_SUCCESS = 1

################################################################################
#  NLG 波束搜索
################################################################################
nlg_beam_size = 10

################################################################################
#  run_mode: 0 for dia-act; 1 for NL; 2 for no output
################################################################################
run_mode = 0
auto_suggest = 0

################################################################################
#   RL代理器要使用的一组基本可行操作
################################################################################
feasible_actions = [
    ############################################################################
    # greeting活动
    ############################################################################
    # {'diaact': "greeting", 'inform_slots': {}, 'request_slots': {}},
    ############################################################################
    # confirm_question活动
    ############################################################################
    {'diaact': "confirm_question", 'inform_slots': {}, 'request_slots': {}},
    ############################################################################
    # confirm_answer活动
    ############################################################################
    {'diaact': "confirm_answer", 'inform_slots': {}, 'request_slots': {}},
    ############################################################################
    # thanks活动
    ############################################################################
    {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}},
    ############################################################################
    # deny活动
    ############################################################################
    {'diaact': "deny", 'inform_slots': {}, 'request_slots': {}},
]
############################################################################
# 添加inform活动
############################################################################
for slot in sys_inform_slots:
    feasible_actions.append({'diaact': 'inform', 'inform_slots': {slot: "PLACEHOLDER"}, 'request_slots': {}})

############################################################################
# 添加request活动
############################################################################
for slot in sys_request_slots:
    feasible_actions.append({'diaact': 'request', 'inform_slots': {}, 'request_slots': {slot: "UNK"}})
