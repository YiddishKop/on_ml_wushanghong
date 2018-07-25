# import package needed
# %matplotlib inline
import matplotlib.pyplot as plt
import os
os.environ["SDL_VIDEODRIVER"] = "dummy" # make window not appear
import tensorflow as tf
import numpy as np
import skimage.color
import skimage.transform
from ple.games.flappybird import FlappyBird
from ple import PLE
game = FlappyBird()
env = PLE(game, fps=30, display_screen=False) # environment interface to game
env.reset_game()
env.act(0) # dummy input to get screen correct

# get rgb screen
screen = env.getScreenRGB()
plt.imshow(screen)
print(screen.shape)

# get grayscale screen
plt.figure()
screen = env.getScreenGrayscale()
plt.imshow(screen, cmap='gray')
print(screen.shape)

# C:\Users\conjugate-forever\Anaconda3\lib\site-packages\h5py\__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.


from ._conv import register_converters as _register_converters

# couldn't import doomish
# Couldn't import doom
# (288, 512, 3)
# (288, 512)

class YourAgent:   
    def select_action(self, input_screens, sess):
        """
        args:
            input_screens: list of frames preprocessed by preprocess function
            sess : tensorflow Session
        output:
            action: int (0 or 1)
        """
    def preprocess(self, screen):
        """
        this function preprocess screen that will be used in select_action function

        args:
            screen: screen to do some preprocessing
        output:
            preprocessed_screen 
        """

# E.g.
class Agent:
    def select_action(self, input_screens, sess):
        # epsilon-greedy
        if np.random.rand() < self.exploring_rate:
            action = np.random.choice(num_action) # Select a random action
        else:
            input_screens = np.array(input_screen[-1]).transpose([1,2,0]) # select current screen
            feed_dict = {
                self.input_screen: input_screen[None,:],
            }
            action = sess.run(self.pred, feed_dict=feed_dict)[0] # Select action with the highest q
        return action

    def preprocess(self, screen):
        screen = skimage.transform.resize(screen, [80, 80])
        return screen

def evaluate_step(agent, seed, sess):
    game = FlappyBird()
    env = PLE(game, fps=30, display_screen=False, rng=np.random.RandomState(seed))
    env.reset_game()
    env.act(0) # dummy input
    # grayscale input screen for this episode  
    input_screens = [agent.preprocess(env.getScreenGrayscale())]
    t = 0
    while not env.game_over():
        # feed four previous screen, select an action
        action = agent.select_action(input_screens, sess)
        # execute the action and get reward
        reward = env.act(env.getActionSet()[action])  # reward = +1 when pass a pipe, -5 when die       
        # observe the result
        screen_plum = env.getScreenGrayscale()  # get next screen
        # append grayscale screen for this episode
        input_screens.append(agent.preprocess(screen_plum))
        t+=1
        if t >= 1000: # maximum score to prevent run forever
            break
    return t
def evaluate(agent, sess):
    scores = []
    for seed in [...some hidden seed...]:
        score = evaluate_step(agent, seed, sess)
        scores.append(score)
    return scores

from evaluate import evaluate
agent = YourAgent() # init your agent, load checkpoint...
scores = evaluate(agent, sess) # evaluate

import pandas as pd
df = pd.DataFrame({
    'scores': scores
})
df.to_csv('./scores.csv')
