import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from past.utils import old_div

import gym, logging
from gym import spaces
from gym.utils import seeding

logging.basicConfig(filename='./ray_res.log', filemode='w')

class MazeEnv(gym.Env):
    """Configurable environment for maze. """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 maze_generator,
                 size,
                 pob_size=1,
                 action_type='VonNeumann',
                 obs_type='full',
                 live_display=False,
                 render_trace=False):
        """Initialize the maze. DType: list"""
        # Random seed with internal gym seeding
        self.seed()

        # Maze: 0: free space, 1: wall
        self.maze_generator = maze_generator
        self.maze = np.array(self.maze_generator.get_maze())
        self.maze_size = self.maze.shape
        self.free_spaces, self.init_state, self.goal_states = self.maze_generator.sample_state()
        self.size = size
        self.state = self.init_state

        self.render_trace = render_trace
        self.traces = []
        self.action_type = action_type
        self.obs_type = obs_type

        # If True, show the updated display each time render is called rather
        # than storing the frames and creating an animation at the end
        self.live_display = live_display

        # Action space: 0: Up, 1: Down, 2: Left, 3: Right
        if self.action_type == 'VonNeumann':  # Von Neumann neighborhood
            self.num_actions = 4
        elif action_type == 'Moore':  # Moore neighborhood
            self.num_actions = 8
        else:
            raise TypeError('Action type must be either \'VonNeumann\' or \'Moore\'')
        self.action_space = spaces.Discrete(self.num_actions)
        self.all_actions = list(range(self.action_space.n))

        # Size of the partial observable window
        self.pob_size = pob_size
        self.success_rate = 0

        # Observation space
        low_obs = 0  # Lowest integer in observation
        high_obs = 32  # Highest integer in observation
        if self.obs_type == 'full':
            self.observation_space = spaces.Box(low=low_obs,
                                                high=high_obs,
                                                shape=[6])
        elif self.obs_type == 'partial':
            self.observation_space = spaces.Box(low=low_obs,
                                                high=high_obs,
                                                shape=(self.pob_size * 2 + 1, self.pob_size * 2 + 1))
        else:
            raise TypeError('Observation type must be either \'full\' or \'partial\'')

        # Colormap: order of color is, free space, wall, agent, food, poison
        self.cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray'])
        self.bounds = [0, 1, 2, 3, 4, 5, 6]  # values for each color
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)

        self.ax_imgs = []  # For generating videos

        self.EMPTY = 0
        self.WALL = 1
        self.AGENT = 2
        self.GOAL = 3
        
        self.lookup_table = None
        
    def read_table(self, x, y):
        return self.lookup_table[x][y]

    def step(self, action):
        old_state = self.state
        # Update current state
        self.state = self._next_state(self.state, action)
        cur_x = self.state[0]
        cur_y = self.state[1]
        
        def dynamic_reward_function():
            return self.read_table(old_state[0], old_state[1]) - self.read_table(cur_x, cur_y)

        # Footprint: Record agent trajectory
        self.traces.append(self.state)
            
        if self._goal_test(self.state):  # Goal check
            reward = 100
            self.success_rate += 1
            logging.warning('Reached' + str(reward))
            done = True
        elif self.state == old_state:  # Hit wall
            reward = -1 * dynamic_reward_function()
            done = False
        else:  # Non-terminal state
            reward = -1 * dynamic_reward_function()
            done = False
        
        new_state, frame = self._get_obs()
        
        info = {
           "success_rate": self.success_rate,
            "frame": frame
        }

        return new_state, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def reset(self):
        # Reset maze
        self.maze = np.array(self.maze_generator.get_maze())

        # Set current state be initial state
        self.state = self.init_state

        # Clean the list of ax_imgs, the buffer for generating videos
        self.ax_imgs = []

        return self._get_obs()

    def render(self, mode='human', close=False):
        if close:
            plt.close()
            return

        partial_obs, obs = self._get_full_obs()

        # For rendering traces: Only for visualization, does not affect the observation data
        if self.render_trace:
            obs[list(zip(*self.traces[:-1]))] = self.VISITED

        # Create Figure for rendering
        if not hasattr(self, 'fig'):  # initialize figure and plotting axes
            self.fig, (self.ax_full) = plt.subplots(nrows=1, ncols=1)
        self.ax_full.axis('off')
        # self.ax_partial.axis('off')

        self.fig.show()
        if self.live_display:
            # Only create the image the first time
            if not hasattr(self, 'ax_full_img'):
                self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
            # if not hasattr(self, 'ax_partial_img'):
                # self.ax_partial_img = self.ax_partial.imshow(partial_obs, cmap=self.cmap, norm=self.norm, animated=True)
            # Update the image data for efficient live video
            self.ax_full_img.set_data(obs)
            # self.ax_partial_img.set_data(partial_obs)
        else:
            # Create a new image each time to allow an animation to be created
            self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
            # self.ax_partial_img = self.ax_partial.imshow(partial_obs, cmap=self.cmap, norm=self.norm, animated=True)

        plt.draw()

        if self.live_display:
            # Update the figure display immediately
            self.fig.canvas.draw()
        else:
            # Put in AxesImage buffer for video generation
            self.ax_imgs.append([self.ax_full_img])  # List of axes to update figure frame

            self.fig.set_dpi(100)

        plt.pause(.1)
        return self.fig

    def _goal_test(self, state):
        """Return True if current state is a goal state."""
        if type(self.goal_states[0]) == list:
            return list(state) in self.goal_states
        elif type(self.goal_states[0]) == tuple:
            return tuple(state) in self.goal_states

    def _next_state(self, state, action):
        """Return the next state from a given state by taking a given action."""

        # Transition table to define movement for each action
        if self.action_type == 'VonNeumann':
            transitions = {0: [-1, 0], 1: [+1, 0], 2: [0, -1], 3: [0, +1]}
        elif self.action_type == 'Moore':
            transitions = {0: [-1, 0], 1: [+1, 0], 2: [0, -1], 3: [0, +1],
                           4: [-1, +1], 5: [+1, +1], 6: [-1, -1], 7: [+1, -1]}

        new_state = [state[0] + transitions[action][0], state[1] + transitions[action][1]]
        if self.maze[new_state[0]][new_state[1]] == 1:  # Hit wall, stay there
            return state
        else:  # Valid move for 0, 2, 3, 4
            return new_state

    def _get_obs(self):
        if self.obs_type == 'full':
            return self._get_full_obs()

    def _get_full_obs(self):
        """Return a 2D array representation of maze."""
        frame_obs = np.array(self.maze)
        
        # Set goal positions
        for goal in self.goal_states:
            frame_obs[goal[0]][goal[1]] = 3  # 3: goal

        # Set current position
        # Come after painting goal positions, avoid invisible within multi-goal regions
        frame_obs[self.state[0]][self.state[1]] = 2  # 2: agent
        
        # xA, yA, Wn, Ws, We, Ww
        xA = self.state[0]
        yA = self.state[1]
        new_obs = np.array([xA, yA, 0, 0, 0, 0])
        
        if self.maze[xA][yA-1] == 1: new_obs[2] = 1 # north wall
        if self.maze[xA][yA+1] == 1: new_obs[3] = 1 # south wall
        if self.maze[xA+1][yA] == 1: new_obs[4] = 1 # east wall
        if self.maze[xA-1][yA] == 1: new_obs[5] = 1 # west wall
        
        # return obs
        return new_obs, frame_obs

    def _get_partial_obs(self, size=1):
        """Get partial observable window according to Moore neighborhood"""
        # Get maze with indicated location of current position and goal positions
        maze = self._get_full_obs()
        pos = np.array(self.state)

        under_offset = np.min(pos - size)
        over_offset = np.min(len(maze) - (pos + size + 1))
        offset = np.min([under_offset, over_offset])

        if offset < 0:  # Need padding
            maze = np.pad(maze, np.abs(offset), 'constant', constant_values=1)
            pos += np.abs(offset)

        return maze[pos[0] - size: pos[0] + size + 1, pos[1] - size: pos[1] + size + 1]

    def _get_video(self, interval=200, gif_path=None):
        if self.live_display:
            # TODO: Find a way to create animations without slowing down the live display
            print('Warning: Generating an Animation when live_display=True not yet supported.')
        anim = animation.ArtistAnimation(self.fig, self.ax_imgs, interval=interval)

        if gif_path is not None:
            anim.save(gif_path, writer='imagemagick', fps=10)
        return anim

    def render_learning(self, policy, qf, vmin, vmax):
        # adapted from https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/GridWorld.py
        obs = self._get_full_obs()
        ROWS, COLS = self.maze_size
        MIN_RETURN = None
        MAX_RETURN = None
        SHIFT = .1
        cmap_actions = colors.ListedColormap(['.5', 'k'], 'Actions')
        V = np.zeros((ROWS, COLS))

        if not hasattr(self, 'fig_value'):
            plt.figure("Value Function")
            self.fig_value = plt.imshow(np.array(self.maze), cmap=self.cmap, norm=self.norm, animated=False)
            self.im_value = plt.imshow(np.zeros_like(obs), cmap=plt.cm.RdYlGn, vmin=vmin, vmax=vmax, alpha=0.7)
            plt.colorbar()
            plt.xticks(np.arange(COLS), fontsize=12)
            plt.yticks(np.arange(ROWS), fontsize=12)

        if not hasattr(self, 'fig_policy'):
            plt.figure("Policy")
            self.fig_policy = plt.imshow(obs, cmap=self.cmap, norm=self.norm, animated=False)

            plt.xticks(np.arange(COLS), fontsize=12)
            plt.yticks(np.arange(ROWS), fontsize=12)
            # Create quivers for each action. 4 in total
            X = np.arange(ROWS) - SHIFT
            Y = np.arange(COLS)
            X, Y = np.meshgrid(X, Y)
            DX = DY = np.ones(X.shape)
            C = np.zeros(X.shape)
            C[0, 0] = 1  # Making sure C has both 0 and 1
            # length of arrow/width of bax. Less then 0.5 because each arrow is
            # offset, 0.4 s nice but could be better/auto generated
            arrow_ratio = 0.4
            Max_Ratio_ArrowHead_to_ArrowLength = 0.25
            ARROW_WIDTH = 0.5 * Max_Ratio_ArrowHead_to_ArrowLength / 5.0
            self.upArrows_fig = plt.quiver(
                Y, X, DY, DX, C,
                units='y', cmap=cmap_actions, width=-1 * ARROW_WIDTH,
                scale_units="height", scale=old_div(ROWS, arrow_ratio))
            self.upArrows_fig.set_clim(vmin=0, vmax=1)
            X = np.arange(ROWS) + SHIFT
            Y = np.arange(COLS)
            X, Y = np.meshgrid(X, Y)
            self.downArrows_fig = plt.quiver(
                Y, X, DY, DX, C,
                units='y', cmap=cmap_actions, width=-1 * ARROW_WIDTH,
                scale_units="height", scale=old_div(ROWS, arrow_ratio))
            self.downArrows_fig.set_clim(vmin=0, vmax=1)
            X = np.arange(ROWS)
            Y = np.arange(COLS) - SHIFT
            X, Y = np.meshgrid(X, Y)
            self.leftArrows_fig = plt.quiver(
                Y, X, DY, DX, C,
                units='x', cmap=cmap_actions, width=ARROW_WIDTH,
                scale_units="width", scale=old_div(COLS, arrow_ratio))
            self.leftArrows_fig.set_clim(vmin=0, vmax=1)
            X = np.arange(ROWS)
            Y = np.arange(COLS) + SHIFT
            X, Y = np.meshgrid(X, Y)
            self.rightArrows_fig = plt.quiver(
                Y, X, DY, DX, C,
                units='x', cmap=cmap_actions, width=ARROW_WIDTH,
                scale_units="width", scale=old_div(COLS, arrow_ratio))
            self.rightArrows_fig.set_clim(vmin=0, vmax=1)
            plt.show()

        plt.figure("Policy")
        # Boolean 3 dimensional array. The third array highlights the action.
        # Thie mask is used to see in which cells what actions should exist
        Mask = np.ones((COLS, ROWS, self.num_actions), dtype='bool')
        arrowSize = np.zeros((COLS, ROWS, self.num_actions), dtype='float')
        # 0 = suboptimal action, 1 = optimal action
        arrowColors = np.zeros((COLS, ROWS, self.num_actions), dtype='uint8')
        for r in range(ROWS):
            for c in range(COLS):
                if obs[r, c] != self.WALL:
                    _obs = np.array(self.maze)
                    for goal in self.goal_states: _obs[goal[0], goal[1]] = self.GOAL
                    _obs[r, c] = self.AGENT
                    act, _ = policy.get_action(_obs.reshape(1, -1))
                    Qs = [qf(ptu.Variable(ptu.from_numpy(_obs).view(1, -1)),
                             ptu.Variable(a.view(1, -1)))
                          for a in ptu.eye(self.num_actions)]
                    V[r, c] = ptu.get_numpy(torch.max(torch.cat(Qs)))
                    Mask[c, r, :] = False
                    arrowColors[c, r, act] = 1
                    arrowSize[c, r, :] = policy.get_actions(obs.reshape(1, -1))

        # Show Policy Up Arrows
        DX = arrowSize[:, :, 2]
        DY = np.zeros((ROWS, COLS))
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 2])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 2])
        C = np.ma.masked_array(arrowColors[:, :, 2], mask=Mask[:, :, 2])
        self.upArrows_fig.set_UVC(DY, DX, C)
        # Show Policy Down Arrows
        DX = -arrowSize[:, :, 3]
        DY = np.zeros((ROWS, COLS))
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 3])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 3])
        C = np.ma.masked_array(arrowColors[:, :, 3], mask=Mask[:, :, 3])
        self.downArrows_fig.set_UVC(DY, DX, C)
        # Show Policy Left Arrows
        DX = np.zeros((ROWS, COLS))
        DY = -arrowSize[:, :, 0]
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 0])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 0])
        C = np.ma.masked_array(arrowColors[:, :, 0], mask=Mask[:, :, 0])
        self.leftArrows_fig.set_UVC(DY, DX, C)
        # Show Policy Right Arrows
        DX = np.zeros((ROWS, COLS))
        DY = arrowSize[:, :, 1]
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 1])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 1])
        C = np.ma.masked_array(arrowColors[:, :, 1], mask=Mask[:, :, 1])
        self.rightArrows_fig.set_UVC(DY, DX, C)
        plt.draw()
        plt.pause(.1)

        plt.figure("Value Function")
        self.im_value.set_data(V)
        plt.draw()
        plt.pause(.1)