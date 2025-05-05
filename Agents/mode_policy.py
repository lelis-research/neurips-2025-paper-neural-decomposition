def update_mode(mode, observation):
    x, y, ang, df, db = observation[0], observation[1], observation[2], observation[3], observation[4]

    if mode == 0:
        # going backward
        if db < 0.5:
            mode = 1

    elif mode == 1:
        # going forward an out
        if df < 0.5 and x > 0:
            mode = 0
        if y > 5 and x < -1:
            mode = 2
    return mode

def get_mode_action(mode):
    if mode == 0:
        action = -1, 0
        action_discrete = 0
    elif mode == 1:
        action = 1, 5
        action_discrete = 1
    elif mode == 2:
        action = 1, -3
        action_discrete = 2
    return action, action_discrete