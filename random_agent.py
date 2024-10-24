import gym, cv2
import numpy as np

from cv_show import frame

# Creating the environment
cliffEnv = gym.make("CliffWalking-v0")


# The same code from cv_show.py initializes and puts the agent
def initialize_frame():
    width, height = 600, 200
    img = np.ones(shape=(height, width, 3)) * 255.0
    margin_horizontal = 6
    margin_vertical = 2

    # Vertical Lines
    for i in range(13):
        img = cv2.line(img, (49 * i + margin_horizontal, margin_vertical),
                       (49 * i + margin_horizontal, 200 - margin_vertical), color=(0, 0, 0), thickness=1)

    # Horizontal Lines
    for i in range(5):
        img = cv2.line(img, (margin_horizontal, 49 * i + margin_vertical),
                       (600 - margin_horizontal, 49 * i + margin_vertical), color=(0, 0, 0), thickness=1)

    # Cliff Box
    img = cv2.rectangle(img, (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
                        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2), color=(255, 0, 255),
                        thickness=-1)
    img = cv2.putText(img, text="Cliff", org=(49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # Goal
    frame = cv2.putText(img, text="G", org=(49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return frame


# Puts the agent at a state
def put_agent(img, state):
    margin_horizontal = 6
    margin_vertical = 2

    # Ensure the state is an integer (extract it from the tuple if needed)
    state = state if isinstance(state, int) else state[0]

    # Now we can safely call np.unravel_index
    row, column = np.unravel_index(indices=state, shape=(4, 12))

    cv2.putText(img, text="A", org=(49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return img


done = False  # this will mark whether the episode is ended or not

frame = initialize_frame()

state = cliffEnv.reset()  # this resets the env and returns the state
state = state if isinstance(state, int) else state[0]  # Ensure state is an integer

while not done:
    frame2 = put_agent(frame.copy(), state)  # putting agent into the copy of the frame and passing the state
    cv2.imshow("Cliff Walking", frame2)
    cv2.waitKey(2)  # after the wait it will show the next frame

    action = int(np.random.randint(low=0, high=4,size=1))  # here we are taking a random action
    print(f"Action taken: {action}")

    next_step = cliffEnv.step(action)  # step() returns state, reward, done, info
    state, reward, done, info = next_step if len(next_step) == 4 else (next_step[0], next_step[1], next_step[2], None)

    state = state if isinstance(state, int) else state[0]  # Ensure state is an integer
    print(f"New state: {state}, Reward: {reward}, Done: {done}, Info: {info}")

cliffEnv.close()
