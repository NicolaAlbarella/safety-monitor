from gym.envs.registration import register

register(id='LaneChange-v0', entry_point='envs.LaneChange:LaneChangeEnv')
register(id='IntersectionMasked-v0', entry_point='envs.Masked:IntersectionMaskedEnv')
register(id='Merge-v0', entry_point='envs.Merge:MergeEnv')
register(id='Intersection-v0', entry_point='envs.Intersection:IntersectionEnv')