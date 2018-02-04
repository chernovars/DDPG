from collections import deque
import random
import pickle

class ReplayBuffer(object):

    def __init__(self, buffer_size, env_name, load=False, save_folder=None):
        self.env_name = env_name
        self.max_size = buffer_size
        self.num_experiences = 0
        if save_folder is not None:
            path = save_folder + "/buffer_replay.txt"
        else:
            path = "./experiments/"+env_name+"/buffer_replay.txt"
        if not load:
            self.buffer = deque()
        else:
            try:
                with open(path, "rb") as fp:  # Unpickling
                    print("Loading buffer_replay from saved file")
                    self.buffer = pickle.load(fp)
            except Exception as error:
                print("Exception ", error, " happened during buffer_replay load. Creating new buffer.")
                self.buffer = deque()

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        return random.sample(self.buffer, batch_size)

    def size(self):
        return self.max_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.max_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

    def save_buffer(self, save_folder=None):
        path = ""
        if save_folder is None:
            path = "./experiments/"+self.env_name+"/buffer_replay.txt"
        else:
            path = save_folder + "/buffer_replay.txt"

        with open(path, "wb") as fp:  # Pickling
            print("save buffer_replay...")
            pickle.dump(self.buffer, fp)
