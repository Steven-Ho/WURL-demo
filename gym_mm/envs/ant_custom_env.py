import gym
from gym.envs.mujoco import ant, mujoco_env
import os
import re
from gym import utils
import sys
import numpy as np

GOALS = [(2, 2),
         (2, -2),
         (-2, -2),
         (-2, 2),
         (2, 2)]
GOAL_RADIUS = 1.0

class AntCustomEnv(ant.AntEnv):
    def __init__(self, gear_ratio=30):
        self._goal_index = 0
        print('WARNING: GEAR RATIO FOR ANT_CUSTOM_ENV MAY HAVE BEEN SET INCORRECTLY!!!')
        print('WARNING: HIDING XY COORDINATES!!!')
        assets_dir = os.path.join(os.path.dirname(os.path.realpath(gym.__file__)), 'envs', 'mujoco', 'assets')
        with open(os.path.join(assets_dir, 'ant.xml')) as f:
            xml = f.read()
        xml_custom_gear = re.sub('gear=\"\d+\"', 'gear=\"%d\"' % gear_ratio, xml)
        filename_custom_gear = os.path.join(assets_dir, 'ant_custom_gear.xml')
        with open(filename_custom_gear, 'w') as f:
            f.write(xml_custom_gear)
        mujoco_env.MujocoEnv.__init__(self, 'ant_custom_gear.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        '''Modified to not terminate when ant jumps and flips over.'''
        (obs, r, done, info) = super(AntCustomEnv, self).step(a)
        r, done = self._get_reward(obs)
        # done = False
        # TODO
        # obs = obs.copy()
        # obs[:2] = 0
        # end TODO
        return (obs, r, done, info)

    def _get_obs(self):
        '''Modified to include global x, y coordinates'''
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
            np.clip(self.data.cfrc_ext, -1, 1).flat,
        ])

    def _state(self):
        return self._get_obs()

    def _get_reward(self, obs):
        if self._goal_index == len(GOALS):
            return 0., True
        next_goal = GOALS[self._goal_index]
        pos = obs[:2]
        goal_dist = np.linalg.norm(np.array(next_goal) - pos)
        if goal_dist < GOAL_RADIUS:
            # print('reached goal!')
            self._goal_index += 1
            return 50.0, False
        return 0.0, False

    def reset(self):
        self._goal_index = 0
        return super(AntCustomEnv, self).reset()




if __name__ == '__main__':
    env = AntCustomEnv()
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
