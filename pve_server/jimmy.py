import rlcard
import itertools
import os
from collections import Counter, OrderedDict
from heapq import nlargest
from enum import Enum

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

env = rlcard.make('no-limit-holdem')

DouZeroCard2RLCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                      8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                      13: 'K', 14: 'A', 17: '2', 20: 'B', 30: 'R'}

RLCard2DouZeroCard = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                      '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
                      'K': 13, 'A': 14, '2': 17, 'B': 20, 'R': 30}

EnvCard2RealCard = {'3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
                    '8': '8', '9': '9', 'T': 'T', 'J': 'J', 'Q': 'Q',
                    'K': 'K', 'A': 'A', '2': '2', 'B': 'X', 'R': 'D'}

RealCard2EnvCard = {'3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
                    '8': '8', '9': '9', 'T': 'T', 'J': 'J', 'Q': 'Q',
                    'K': 'K', 'A': 'A', '2': '2', 'X': 'B', 'D': 'R'}


class Action(Enum):
    FOLD = 0
    CHECK_CALL = 1
    #CALL = 2
    # RAISE_3BB = 3
    RAISE_HALF_POT = 2
    RAISE_POT = 3
    # RAISE_2POT = 5
    ALL_IN = 4
    # SMALL_BLIND = 7
    # BIG_BLIND = 8


pretrained_dir = 'experiments/dmc_result/nolimitholdem'
device = torch.device('cpu')
players = []
for i in range(4):
    model_path = os.path.join(pretrained_dir, str(i)+'_0.pth')
    agent = torch.load(model_path, map_location=device)
    agent.set_device(device)
    players.append(agent)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if (request.headers.get('Content-Type') != 'application/json'):
            return 'Content-Type not supported!'

        print("/predict:", request.json)
        # Player postion
        player_position = request.json['player_position']
        if player_position not in ['0', '1', '2', '3']:
            return jsonify({'status': 1, 'message': 'player_position must be 0, 1, 2 or 3'})
        player_position = int(player_position)

        # Player hand cards
        player_hand_cards = request.json['hand']
        if len(player_hand_cards) != 2:
            return jsonify({'status': 2, 'message': 'the number of hand cards should be 2'})

        # public cards
        public_cards = request.json['public_cards']
        if len(public_cards) < 0 or len(public_cards) > 5:
            return jsonify({'status': 3, 'message': 'the number of landlord cards should be 0-5'})

        # All Chips that all players have put in(array)
        all_in_chips = request.json['all_in_chips']
        pot = np.sum([c for c in range(len(all_in_chips))])
        # The chips that this player has put in until now
        my_chips = request.json['my_chips']
        my_chips = int(my_chips)
        # The chips that the number of chips the player has remained
        all_remained = request.json['all_remained']
        all_remained = int(all_remained)
        # Game stage
        stage = request.json['stage']
        stage = int(stage)
        # All action that the player has been legal(array)
        legal_actions = [Action(c) for c in request.json['legal_actions']]

        # RLCard state
        state = {}
        state['hand'] = player_hand_cards
        state['public_cards'] = public_cards
        state['all_chips'] = all_in_chips
        state['my_chips'] = my_chips
        state['legal_actions'] = legal_actions
        state['stakes'] = all_remained
        state['current_player'] = player_position
        state['pot'] = pot
        state['stage'] = stage

        # Prediction
        state = env._extract_state(state)
        action, info = players[player_position].eval_step(state)
        ############## DEBUG ################
        if app.debug:
            print('--------------- DEBUG START --------------')
            print(action, info)
            print('--------------- DEBUG END --------------')
        ############## DEBUG ################
        return jsonify({'status': 0, 'message': 'success'})
    except:
        import traceback
        traceback.print_exc()
        return jsonify({'status': -1, 'message': 'unkown error'})


@app.route('/legal', methods=['POST'])
def legal():
    if request.method == 'POST':
        try:
            if (request.headers.get('Content-Type') != 'application/json'):
                return 'Content-Type not supported!'
            player_hand_cards = [RealCard2EnvCard[c]
                                 for c in request.json['player_hand_cards']]
            rival_move = [RealCard2EnvCard[c]
                          for c in request.json['rival_move']]
            if rival_move == '':
                rival_move = 'pass'
            player_hand_cards = [RLCard2DouZeroCard[c]
                                 for c in player_hand_cards]
            rival_move = [RLCard2DouZeroCard[c] for c in rival_move]
            legal_actions = _get_legal_card_play_actions(
                player_hand_cards, rival_move)
            legal_actions = [''.join([DouZeroCard2RLCard[c]
                                      for c in a]) for a in legal_actions]
            for i in range(len(legal_actions)):
                if legal_actions[i] == 'pass':
                    legal_actions[i] = ''
            legal_actions = ','.join(
                [''.join([EnvCard2RealCard[c] for c in action]) for action in legal_actions])
            return jsonify({'status': 0, 'message': 'success', 'legal_action': legal_actions})
        except:
            import traceback
            traceback.print_exc()
            return jsonify({'status': -1, 'message': 'unkown error'})


def _extract_state(state):
    current_hand = _cards2array(state['current_hand'])
    others_hand = _cards2array(state['others_hand'])

    last_action = ''
    if len(state['trace']) != 0:
        if state['trace'][-1][1] == 'pass':
            last_action = state['trace'][-2][1]
        else:
            last_action = state['trace'][-1][1]
    last_action = _cards2array(last_action)

    last_9_actions = _action_seq2array(_process_action_seq(state['trace']))

    if state['self'] == 0:  # landlord
        landlord_up_played_cards = _cards2array(state['played_cards'][2])
        landlord_down_played_cards = _cards2array(state['played_cards'][1])
        landlord_up_num_cards_left = _get_one_hot_array(
            state['num_cards_left'][2], 17)
        landlord_down_num_cards_left = _get_one_hot_array(
            state['num_cards_left'][1], 17)
        obs = np.concatenate((current_hand,
                              others_hand,
                              last_action,
                              last_9_actions,
                              landlord_up_played_cards,
                              landlord_down_played_cards,
                              landlord_up_num_cards_left,
                              landlord_down_num_cards_left))
    else:
        landlord_played_cards = _cards2array(state['played_cards'][0])
        for i, action in reversed(state['trace']):
            if i == 0:
                last_landlord_action = action
        last_landlord_action = _cards2array(last_landlord_action)
        landlord_num_cards_left = _get_one_hot_array(
            state['num_cards_left'][0], 20)

        teammate_id = 3 - state['self']
        teammate_played_cards = _cards2array(
            state['played_cards'][teammate_id])
        last_teammate_action = 'pass'
        for i, action in reversed(state['trace']):
            if i == teammate_id:
                last_teammate_action = action
        last_teammate_action = _cards2array(last_teammate_action)
        teammate_num_cards_left = _get_one_hot_array(
            state['num_cards_left'][teammate_id], 17)
        obs = np.concatenate((current_hand,
                              others_hand,
                              last_action,
                              last_9_actions,
                              landlord_played_cards,
                              teammate_played_cards,
                              last_landlord_action,
                              last_teammate_action,
                              landlord_num_cards_left,
                              teammate_num_cards_left))

    legal_actions = {env._ACTION_2_ID[action]: _cards2array(
        action) for action in state['actions']}
    extracted_state = OrderedDict({'obs': obs, 'legal_actions': legal_actions})
    extracted_state['raw_obs'] = state
    extracted_state['raw_legal_actions'] = [a for a in state['actions']]
    return extracted_state


Card2Column = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, 'T': 7,
               'J': 8, 'Q': 9, 'K': 10, 'A': 11, '2': 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}


def _cards2array(cards):
    if cards == 'pass':
        return np.zeros(54, dtype=np.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(cards)
    for card, num_times in counter.items():
        if card == 'B':
            jokers[0] = 1
        elif card == 'R':
            jokers[1] = 1
        else:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
    return np.concatenate((matrix.flatten('F'), jokers))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='nolimitholdem backend')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    app.run(host="localhost", port=6666, debug=args.debug)
