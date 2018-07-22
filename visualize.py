import importlib

from args import parse_args
import time

from common.environment import create_env

import numpy as np
import Box2D

import torch
from torch.autograd import Variable

class Visualize():
    def __init__(self):
        self.show_actiongrid = False
        self.actiongrid_depth = 0
        self.show_stategrid = False
        self.paused = False

    def key_press(self, key, mod):
        # If 'a' pressed, toggle actiongrid: hidden -> both -> depth 0 -> depth 1
        if key == 97:
            if self.show_actiongrid:
                self.actiongrid_depth += 1
                if self.actiongrid_depth == 2:
                    self.actiongrid_depth = -1
                    self.show_actiongrid = False
            else:
                self.show_actiongrid = True
        # If 's' pressed, toggle stategrid
        elif key == 115:
            self.show_stategrid = not self.show_stategrid
        # If 'p' pressed, pause env
        elif key == 112:
            self.paused = not self.paused

    def main(self, args):
        if args['gpu_ids'][-1] != -1:
            torch.cuda.manual_seed(args['seed'])

        print(args)

        gpu_id = args['gpu_ids'][-1]

        start_time = time.time()

        self.env = create_env(args['env'], args)

        AC = importlib.import_module(args['model_name'])
        self.model = AC.ActorCritic(
            self.env.observation_space, self.env.action_space, args['stack_frames'], args)

        if args['load']:
            print('Loading model from: {0}{1}.dat'.format(
                args['load_model_dir'], args['env']))
            saved_state = torch.load('{0}{1}.dat'.format(
                args['load_model_dir'], args['env']), map_location=lambda storage, loc: storage)
            self.model.load_state_dict(saved_state)

        if gpu_id >= 0:
            torch.cuda.manual_seed(args['seed'])
        reward_sum = 0
        num_tests = 0
        reward_total_sum = 0

        # Have to call render() for the first time to build viewer
        self.env.render()
        self.env.unwrapped.viewer.window.on_key_press = self.key_press
        #env.unwrapped.viewer.window.on_key_release = key_release

        # start of player removal

        memory = None
        eps_len = 0
        done = True

        state, info = self.env.reset()
        state = torch.from_numpy(state).float()

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                self.model = model.cuda()
                state = state.cuda()
        self.model.eval()

        episode_count = 0
        while True:
            if done:
                episode_count += 1
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        self.model.load_state_dict(self.model.state_dict())
                else:
                    self.model.load_state_dict(self.model.state_dict())

            # action_test begin
            with torch.no_grad():
                if done:
                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):
                            memory = self.model.initialize_memory()
                    else:
                        memory = self.model.initialize_memory()
                else:
                    memory = self.model.reinitialize_memory(memory)
                state = state.unsqueeze(0)
                value, mu, sigma, memory = self.model(
                    (Variable(state), info, memory))
            mu = torch.clamp(mu.data, -1.0, 1.0)
            action = mu.cpu().numpy()[0]
            state, reward, done, info = self.env.step(action)

            self.env.render(model=self.model, show_stategrid=self.show_stategrid, show_actiongrid=self.show_actiongrid, actiongrid_depth=self.actiongrid_depth)

            state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    state = state.cuda()
            eps_len += 1
            done = done or eps_len >= args['max_episode_length']

            # action_test end
            reward_sum += reward

            if done:
                num_tests += 1
                reward_total_sum += reward_sum
                reward_mean = reward_total_sum / num_tests
                print("Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                    format(
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time)),
                        reward_sum, eps_len, reward_mean))


                reward_sum = 0
                eps_len = 0
                state, info = self.env.reset()
                #time.sleep(60)
                state = torch.from_numpy(state).float()
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state = state.cuda()

if __name__ == "__main__":
    visualize = Visualize()
    visualize.main(parse_args())
