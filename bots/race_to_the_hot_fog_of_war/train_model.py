import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from environments.race_to_the_hot.env import race_to_the_hot
from tqdm import tqdm
import threading
import os
import json
import numpy as np
from tf_agents.policies import policy_saver
from multiprocessing import Process
from tf_agents.trajectories import time_step as ts
import time

# loading configuration...
print('loading configuration...')
_config = {}
with open('config.json') as f:
    _config = json.load(f)

tf.compat.v1.enable_v2_behavior()

tf.config.set_visible_devices([], 'GPU')


class master():
    def __init__(self):
        self.num_iterations = 20000000
        self.initial_collect_steps = 10
        self.collect_steps_per_iteration = 100
        self.replay_buffer_max_length = 10000
        self.batch_size = 64 * 10
        self.learning_rate = 0.000001
        self.train_steps = 1000
        self.num_eval_episodes = 10

        self.save_policy_dir = os.path.join(_config['files']['policy']['base_dir'],
                                        _config['files']['policy']['save_policy']['dir'],
                                        _config['files']['policy']['save_policy']['name'])

        self.checkpoint_policy_dir = os.path.join(_config['files']['policy']['base_dir'],
                                              _config['files']['policy']['checkpoint_policy']['dir'],
                                              _config['files']['policy']['checkpoint_policy']['name'])

        self.train_py_env = race_to_the_hot(window_name='Training')
        self.eval_py_env = race_to_the_hot(window_name='Testing')

        self.train_env = tf_py_environment.TFPyEnvironment(self.train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(self.eval_py_env)

        self.agent = None
        self.replay_buffer = None
        self.random_policy = None
        self.train_checkpointer = None
        self.tf_policy_saver = None
        self.dataset = None
        self.train_step_counter = None
        self.iterator = None

    def build_network(self):
        _fc_layer_params = (512,)

        _q_net = q_network.QNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=_fc_layer_params)

        _optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.train_step_counter = tf.Variable(0)

        self.agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=_q_net,
            optimizer=_optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.train_step_counter)

        self.agent.initialize()

        _eval_policy = self.agent.policy
        _collect_policy = self.agent.collect_policy

        self.random_policy = random_tf_policy.RandomTFPolicy(self.train_env.time_step_spec(),
                                                         self.train_env.action_spec())
        self.agent.train_step_counter.assign(0)

    def build_replay_buffer(self):
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buffer_max_length)

        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size,
            num_steps=2).prefetch(3)

        self.agent.train = common.function(self.agent.train)

        self.iterator = iter(rtth.dataset)

    def save_checkpoint_init(self):
        self.train_checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpoint_policy_dir,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=self.replay_buffer,
            global_step=self.train_step_counter
        )

        self.tf_policy_saver = policy_saver.PolicySaver(self.agent.policy)

    def compute_avg_return(self, environment, policy, num_episodes=1000):
        score = {'win': 0, 'loss': 0, 'timeout': 0}
        total_return = 0.0
        for _ in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return
            history = environment._env.envs[0].score_history
            final_step = history[len(history) - 1]
            if final_step == 'timeout':
                score['timeout'] += 1
            elif final_step == 'loss':
                score['loss'] += 1
            elif final_step == 'win':
                score['win'] += 1

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0], score


    def collect_step(self, environment, policy, buffer):
        time_step = environment.current_time_step()

        observation_root = time_step.observation.numpy()
        observation_root_padded = np.array([np.pad(observation_root[0], (2,), 'median')])
        player_matrix = observation_root_padded[:, 1, :, :, 3]
        player_location = np.unravel_index(np.argmax(player_matrix), np.array(player_matrix).shape)[1:3]
        new_observation = observation_root_padded[:, 2:102, player_location[0] - 1:player_location[0] + 2,
                          player_location[1] - 1:player_location[1] + 2, 2:5]

        new_ts = ts.TimeStep(time_step.step_type, time_step.reward, time_step.discount, new_observation)
        action_step = policy.action(new_ts)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(new_ts, action_step, next_time_step)
        buffer.add_batch(traj)

    def collect_step_threaded(self, environment, policy, buffer):
        p = Process(target=self.collect_step, args=(environment, policy, buffer,))
        p.start()

    def collect_data(self, env, policy, buffer, steps):
        for _ in range(steps):
            self.collect_step(env, policy, buffer)

    def perfom_initial_collect(self):
        self.compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)

    def perform_collection(self):
        while True:
            for _ in tqdm(range(rtth.train_steps)):
                for _ in range(rtth.collect_steps_per_iteration):
                    self.collect_step(self.train_env, self.agent.collect_policy, self.replay_buffer)

    def perform_training(self):
        while True:
            time.sleep(2)
            try:
                experience, unused_info = next(self.iterator)
                train_loss = self.agent.train(experience).loss
            except:
                lol =1

    def perform_testing(self):
        while True:
            time.sleep(5)
            try:
                avg_return, score = self.compute_avg_return(self.eval_env, self.agent.collect_policy)
                print('Average Return = {0:.2f}, score {1}'.format(avg_return, score))
            except:
                lol =1

    def perform_checkpoint_save(self):
        while True:
            time.sleep(300)
            try:
                self.train_checkpointer.save(self.train_step_counter)
                print('checkpointed')
            except Exception as ie:
                print('failed checkpointer')
                print(ie)
            try:
                self.tf_policy_saver.save(self.save_policy_dir)
                print('saved')
            except Exception as ie:
                print('failed saver')
                print(ie)


rtth = master()
rtth.build_network()
rtth.build_replay_buffer()
rtth.save_checkpoint_init()

restore_network = False
if restore_network:
    rtth.train_checkpointer.initialize_or_restore()
print('initial collect...')
rtth.perfom_initial_collect()

x = threading.Thread(target=rtth.perform_collection, args=())
x.start()

x = threading.Thread(target=rtth.perform_training, args=())
x.start()

x = threading.Thread(target=rtth.perform_checkpoint_save, args=())
x.start()

x = threading.Thread(target=rtth.perform_testing, args=())
x.start()


