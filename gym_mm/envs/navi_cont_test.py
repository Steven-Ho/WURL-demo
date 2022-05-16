from gym_mm.envs.navi_cont import Navi2DCont
import numpy as np

env = Navi2DCont()
done = False
t = 0
max_t = 100
while not done:
    action = np.random.normal(loc=[0.5, 0.5], scale=[1.0, 1.0], size=(2,1))
    state, r, done, _ = env.step(action)
    t += 1
    if t == max_t:
        done = True
env.render()