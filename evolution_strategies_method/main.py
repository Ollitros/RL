import joblib
from six import BytesIO
from evolution_strategies_method.pong import make_pong
import numpy as np
import matplotlib.pyplot as plt


def dumps(data):
    """converts whatever to string"""
    s = BytesIO()
    joblib.dump(data, s)
    return s.getvalue()


def loads(self, string):
    """converts string to whatever was dumps'ed in it"""
    return joblib.load(BytesIO(string))


env = make_pong()
print(env.action_space)

# get the initial state
s = env.reset()
print(s.shape)

# plot first observation. Only one frame
plt.imshow(s.swapaxes(1,2).reshape(-1,s.shape[-1]).T)
plt.show()


# after 10 frames
for _ in range(8):
    new_s,r,done, _ = env.step(env.action_space.sample())

plt.imshow(new_s.swapaxes(1,2).reshape(-1,s.shape[-1]).T,vmin=0)
plt.show()