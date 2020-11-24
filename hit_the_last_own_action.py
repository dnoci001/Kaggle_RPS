my_last_action = 0

def hit_the_last_own_action(observation, configuration):
    global my_last_action
    my_last_action = (my_last_action + 1) % 3
    
    return my_last_action