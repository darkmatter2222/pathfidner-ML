import datetime
import json
import os
import random
import numpy as np
import scipy as sp
import tensorflow as tf
from tf_agents.environments import py_environment
import matplotlib
matplotlib.use('Agg')
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
tf.compat.v1.enable_v2_behavior()


class race_to_the_hot(py_environment.PyEnvironment):
    def __init__(self, window_name):
        self.score_history = []
        self.start_time = datetime.datetime.now()
        self.this_game_boards = []
        self.last_game_boards = []
        self.real_time_chart = []
        self.real_time_game = []
        self.board_width = 10
        self.board_height = 10
        self.master_step_counter = 0
        self.image_counter = 0
        self.gif_counter = 0
        self.uuid = window_name
        self.sigma_y = self.board_width / 2
        self.sigma_x = self.board_height / 2
        self.channels = 3
        self.frames = self.board_width * self.board_height
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.frames, self.board_height, self.board_width, self.channels), dtype=np.int32, minimum=0,
            maximum=1, name='observation')
        self._state = np.zeros([self.board_height, self.board_width, self.channels])
        # 0 = Reward Placement
        # 1 = Player Placement
        # 2 = Heat Map
        self._episode_ended = False
        self.this_turn = 0
        self.max_turns = self.frames
        self.total_reward = 0
        self.heatmap_reward = self.board_height * self.board_width
        self.player_location = {'y': 0, 'x': 0}
        self.set_goal()
        self.game_history = []
        self.state_history = [self._state] * self.frames
        self.save_image = False
        self.enable_render_image = True
        self.enable_render_chart = False
        self.image_render_counter = 0
        self.image_write_counter = 0
        self.image_history = []
        self.episode = 1
        # loading configuration...
        print('loading configuration...')
        _config = {}
        with open('config.json') as f:
            _config = json.load(f)
        self._images_dir = os.path.join(_config['files']['policy']['base_dir'],
                                        _config['files']['policy']['images']['stills']['dir'],
                                        _config['files']['policy']['images']['stills']['name'])

    def append_score(self, action, count):
        self.score_history.append(action)
        if len(self.score_history) > 10:
            del self.score_history[0]

    def set_goal(self):
        self.set_player()
        rand_y = 0
        rand_x = 0

        while True:
            if self.board_width > 1:
                rand_x = random.randrange(0, self.board_width)
            if self.board_height > 1:
                rand_y = random.randrange(0, self.board_height)
            if self.player_location['x'] != rand_x or self.player_location['y'] != rand_y:
                break

        self._state[rand_y, rand_x][0] = 1

        reward_heatmap = np.zeros([self.board_height, self.board_width])
        reward_heatmap[rand_y, rand_x] = self.board_height * self.board_width
        sigma_y = self.board_width / 2
        sigma_x = self.board_height / 2

        # Apply gaussian filter
        sigma = [sigma_y, sigma_x]
        reward_heatmap = sp.ndimage.filters.gaussian_filter(reward_heatmap, sigma, mode='constant')

        for height in range(self.board_height):
            for width in range(self.board_width):
                self._state[height][width][2] = reward_heatmap[height, width]

    def render_new_state(self):
        for height in range(self.board_height):
            for width in range(self.board_width):
                self._state[height, width][1] = 0

        if 0 <= self.player_location['y'] < self.board_width and \
                0 <= self.player_location['x'] < self.board_height:
            self._state[self.player_location['y'], self.player_location['x']][1] = 1

    def set_player(self):
        rand_y = 0
        rand_x = 0

        if self.board_width > 1:
            rand_x = random.randrange(0, self.board_width)
        if self.board_height > 1:
            rand_y = random.randrange(0, self.board_height)
        self.player_location['x'] = rand_x
        self.player_location['y'] = rand_y
        self._state[rand_y, rand_x][1] = 1

    def action_spec(self):
        return_object = self._action_spec
        return return_object

    def observation_spec(self):
        return_object = self._observation_spec
        return return_object

    def _reset(self):
        self._state = np.zeros([self.board_height, self.board_width, self.channels])
        self._episode_ended = False
        self.total_reward = 0
        self.player_location = {'y': 0, 'x': 0}
        self.this_turn = 0
        self.set_goal()
        self.game_history = []
        self.image_history = []
        self.state_history = [self._state] * self.frames
        self.last_game_boards = self.this_game_boards.copy()
        self.this_game_boards = []
        self.episode += 1

        return_object = ts.restart(np.array(self.state_history, dtype=np.int32))
        return return_object

    def _step(self, action):
        self.master_step_counter += 1
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return_object = self.reset()
            return return_object

        reward = 0
        info = ''
        map_edge_exist = True
        continue_reward = -1
        win_reward = 10
        loose_reward = -10
        # 0=N 1=E 2=S 3=W
        if action == 0:  # Move North
            if map_edge_exist:
                self.player_location['y'] = self.player_location['y'] - 1
            else:
                if self.player_location['y'] == 0:
                    self.player_location['y'] = self.board_height - 1
                else:
                    self.player_location['y'] = self.player_location['y'] - 1
        elif action == 1:  # Move East
            if map_edge_exist:
                self.player_location['x'] = self.player_location['x'] + 1
            else:
                if self.player_location['x'] == self.board_width - 1:
                    self.player_location['x'] = 0
                else:
                    self.player_location['x'] = self.player_location['x'] + 1
        elif action == 2:  # Move South
            if map_edge_exist:
                self.player_location['y'] = self.player_location['y'] + 1
            else:
                if self.player_location['y'] == self.board_height - 1:
                    self.player_location['y'] = 0
                else:
                    self.player_location['y'] = self.player_location['y'] + 1
        elif action == 3:  # Move West
            if map_edge_exist:
                self.player_location['x'] = self.player_location['x'] - 1
            else:
                if self.player_location['x'] == 0:
                    self.player_location['x'] = self.board_width - 1
                else:
                    self.player_location['x'] = self.player_location['x'] - 1
        else:
            raise ValueError

        # Max Tries?
        if self.this_turn == self.max_turns - 1:
            info = 'Max Tries'
            self._episode_ended = True
            self.append_score('timeout', 1)
            reward += loose_reward
        else:
            # Loose Fall Off Map?
            if self.player_location['y'] < 0 or self.player_location['x'] < 0 or \
                    self.player_location['x'] >= self.board_width or self.player_location['y'] >= self.board_height:
                info = 'Loose Fall Off Map'
                self.append_score('loss', 1)
                self._episode_ended = True
                reward += loose_reward
            elif self._state[self.player_location['y'], self.player_location['x']][0] == 1:
                info = 'Won Got the Goal'
                self.append_score('win', 1)
                self._episode_ended = True
                reward += win_reward
            elif self._state[self.player_location['y'], self.player_location['x']][2] != 0:
                info = 'Continue w/ reward'
                self._episode_ended = False
                reward += continue_reward + self._state[self.player_location['y'], self.player_location['x']][0]
            else:
                info = 'Continue'
                self._episode_ended = False
                reward += continue_reward

        # preference
        if (action == 0 or action == 2) and self.this_turn < 2:
            reward += 0.1

        self.render_new_state()

        self.game_history.append(info)
        self.total_reward += reward
        self.this_turn += 1

        self.state_history.append(self._state)

        del self.state_history[:1]
        if self._episode_ended:
            return_object = ts.termination(np.array(self.state_history, dtype=np.int32), reward)
            return return_object
        else:
            return_object = ts.transition(np.array(self.state_history, dtype=np.int32), reward=reward, discount=1.0)
            return return_object
