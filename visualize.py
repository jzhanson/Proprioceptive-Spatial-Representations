import importlib

from args import parse_args
import time

from common.environment import create_env

import numpy as np
import Box2D

import torch
from torch.autograd import Variable

import sys

class Visualize():
    def __init__(self, args):

        self.env = create_env(args['env'], args)
        self.args = args

        self.show_actiongrid = False
        self.actiongrid_mode = 'gray'
        self.alpha = 0.5
        self.actiongrid_depth = -1
        self.actiongrid_clip = True
        self.show_stategrid = False

        self.manual_control = False
        #np.zeros(self.env.action_space.shape)
        self.manual_action = self.env.action_space.sample()
        self.manual_action_index = 0
        self.manual_increment_step = 16.
        self.manual_increments = (self.env.action_space.high -
            self.env.action_space.low) / self.manual_increment_step

        self.paused = False
        self.advance_step = False
        self.terminate_episode = False
        self.quit = False

    def key_press(self, key, mod):
        # 'a' : toggle actiongrid: hidden -> both -> depth 0 -> depth 1
        if key == 97:
            if self.show_actiongrid:
                self.actiongrid_depth += 1
                if self.actiongrid_depth == 2:
                    self.actiongrid_depth = -1
                    self.show_actiongrid = False
            else:
                self.show_actiongrid = True
            if self.show_actiongrid and self.actiongrid_depth == -1:
                print('showing actiongrid, both channels')
            elif self.show_actiongrid and self.actiongrid_depth == 0:
                print('showing actiongrid, channel 0')
            elif self.show_actiongrid and self.actiongrid_depth == 1:
                print('showing actiongrid, channel 1')
            else:
                print('hiding actiongrid')
        # 's' : toggle stategrid
        elif key == 115:
            self.show_stategrid = not self.show_stategrid
            print('showing stategrid')
        # 'p' : pause env
        elif key == 112:
            self.paused = not self.paused
        # spacebar : if env paused, advance one step. If env not paused, pause
        # env
        elif key == 32:
            if not self.paused:
                self.paused = True
            elif self.paused:
                self.advance_step = True
        # 'n' : terminate episode
        elif key == 110:
            self.terminate_episode = True
            print('terminating episode')
        # 'c' : switch actiongrid clipping values to 0 (default) to full
        elif key == 99:
            self.actiongrid_clip = not self.actiongrid_clip
            if self.actiongrid_clip:
                print('actiongrid clipping on')
            else:
                print('actiongrid clipping off')
        # 'h' : switch actiongrid mode from 'gray' (grayscale) to
        # 'heat' (heatmap) to 'rainbow' (rainbow)
        elif key == 104:
            if self.actiongrid_mode == 'gray':
                self.actiongrid_mode = 'heat'
                print('actiongrid changed to heatmap mode')
            elif self.actiongrid_mode == 'heat':
                self.actiongrid_mode = 'rainbow'
                print('actiongrid changed to rainbow mode')
            elif self.actiongrid_mode == 'rainbow':
                self.actiongrid_mode = 'gray'
                print('actiongrid changed to grayscale mode')
        # 't' : toggle state grid and action grid alpha from 0.5 -> 0.75 -> 1.0
        # -> 0.0 -> 0.25 -> 0.5 -> ...
        elif key == 116:
            self.alpha += 0.25
            if self.alpha > 1.0:
                self.alpha = 0.0
            print('state/action grid alpha is now ' + str(self.alpha))
        # 'q' : quit (close env and terminate main function)
        elif key == 113:
            print('quitting')
            self.quit = True
        # 'm' : begin/end manual control (independent of pausing or
        # single-stepping)
        elif key == 109:
            self.manual_control = not self.manual_control
            print('manual control enabled, printing action')
        # Up arrow : go up action index in manual control
        elif key == 65362:
            self.manual_action_index -= 1
            if self.manual_action_index == -1:
                self.manual_action_index = self.env.action_space.shape[0] - 1
        # Down arrow : go down action index in manual control
        elif key == 65364:
            self.manual_action_index += 1
            if self.manual_action_index == self.env.action_space.shape[0]:
                self.manual_action_index = 0
        # Left arrow : decrement action value by its appropriate increment
        elif key == 65361:
            self.manual_action[self.manual_action_index] \
                -= self.manual_increments[self.manual_action_index]
            if self.manual_action[self.manual_action_index]  \
                < self.env.action_space.low[self.manual_action_index]:
                self.manual_action[self.manual_action_index] \
                    = self.env.action_space.low[self.manual_action_index]
        # Right arrow : increment action value by its appropriate increment
        elif key == 65363:
            self.manual_action[self.manual_action_index] +=  \
                self.manual_increments[self.manual_action_index]
            if self.manual_action[self.manual_action_index]  \
                > self.env.action_space.high[self.manual_action_index]:
                self.manual_action[self.manual_action_index] \
                    = self.env.action_space.high[self.manual_action_index]
        # + : Increase fineness of manual control increments
        elif key == 43:
            self.manual_increment_step *= 2.
            self.manual_increments = (self.env.action_space.high -
                self.env.action_space.low) / self.manual_increment_step
            print('manual increment step fineness increased to '
                + str(self.manual_increment_step))
        # - : Decrease fineness of manual control increments
        elif key == 95:
            self.manual_increment_step /= 2.
            self.manual_increments = (self.env.action_space.high -
                self.env.action_space.low) / self.manual_increment_step
            print('manual increment step fineness decreased to '
                + str(self.manual_increment_step))
        else:
            print(key)


    def main(self):
        if self.args['gpu_ids'][-1] != -1:
            torch.cuda.manual_seed(self.args['seed'])

        gpu_id = self.args['gpu_ids'][-1]

        start_time = time.time()

        AC = importlib.import_module(self.args['model_name'])
        self.model = AC.ActorCritic(
            self.env.observation_space, self.env.action_space,
            self.args['stack_frames'], self.args)
        print('trainable parameters: ' + str(sum(p.numel() for p in
            self.model.parameters() if p.requires_grad)))

        if self.args['load_file'] != '':
            print('Loading model from: {0}'.format(self.args['load_file']))
            pthfile = torch.load('{0}'.format(self.args['load_file']),
                map_location=lambda storage, loc: storage.cpu())
            if self.args['load_best']:
                self.model.load_state_dict(pthfile['best_state_dict'])
            else:
                self.model.load_state_dict(pthfile['state_dict'])

        if gpu_id >= 0:
            torch.cuda.manual_seed(self.args['seed'])
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
            if self.quit:
                self.env.close()
                break

            if self.manual_control:
                # Couldn't get action printing in the viewer to work
                sys.stdout.write('\r' + str(self.manual_action))
                sys.stdout.flush()

            if self.paused:
                if self.advance_step:
                    self.advance_step = False
                else:
                    self.env.render(model=self.model,
                        show_stategrid=self.show_stategrid,
                        actiongrid_mode=self.actiongrid_mode if self.show_actiongrid else 'hide',
                        actiongrid_depth=self.actiongrid_depth,
                        actiongrid_clip=self.actiongrid_clip, alpha=self.alpha)
                    continue

            if done:
                episode_count += 1
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        self.model.load_state_dict(self.model.state_dict())
                else:
                    self.model.load_state_dict(self.model.state_dict())
                if self.terminate_episode:
                    self.terminate_episode = False

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
            if self.manual_control:
                action = self.manual_action
            state, reward, done, info = self.env.step(action)

            self.env.render(model=self.model,
                show_stategrid=self.show_stategrid,
                actiongrid_mode=self.actiongrid_mode if self.show_actiongrid else 'hide',
                actiongrid_depth=self.actiongrid_depth,
                actiongrid_clip=self.actiongrid_clip, alpha=self.alpha)

            state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    state = state.cuda()
            eps_len += 1
            done = done or eps_len >= self.args['max_episode_length'] or self.terminate_episode

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
    args = parse_args()
    print(args)

    visualize = Visualize(args)

    visualize.main()
