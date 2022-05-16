from gym.envs.registration import register

register(
    id='DiscreteNavigation-v0',
    entry_point='gym_mm.envs:Navi2D'
)

register(
    id='ContinuousNavigation-v0',
    entry_point='gym_mm.envs:Navi2DCont'
)

register(
    id='FreeRun-v0',
    entry_point='gym_mm.envs:FreeRun'
)

register(
    id='FreeRunD-v0',
    entry_point='gym_mm.envs:FreeRunD'
)

register(
    id='FreeRunSkill-v0',
    entry_point='gym_mm.envs:FreeRunSkill'
)

register(
    id='TreeMaze-v0',
    entry_point='gym_mm.envs:MazeCont'
)

register(
    id='WMaze-v0',
    entry_point='gym_mm.envs:MazeW'
)