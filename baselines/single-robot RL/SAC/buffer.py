import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.speeds_memory = np.zeros((self.mem_size, 2))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_speeds_memory = np.zeros((self.mem_size, 2))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, speed, action, reward, state_, next_speed, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.speeds_memory[index] = speed
        self.new_state_memory[index] = state_
        self.new_speeds_memory[index] = next_speed
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        speeds = self.speeds_memory[batch]
        states_ = self.new_state_memory[batch]
        next_speeds = self.new_speeds_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, speeds, actions, rewards, states_, next_speeds, dones