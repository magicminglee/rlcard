import rlcard
import os
from enum import Enum

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

env = rlcard.make('no-limit-holdem')


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='nolimitholdem backend')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    app.run(host="localhost", port=6666, debug=args.debug)
