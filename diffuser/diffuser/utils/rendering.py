import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import gym
import mujoco_py as mjc
import warnings
import cv2 as cv

from .arrays import to_np
from .video import save_video, save_videos

from diffuser.datasets.d4rl import load_environment
from dlp_utils import get_recon_from_dlps

#-----------------------------------------------------------------------------#
#------------------------------- helper structs ------------------------------#
#-----------------------------------------------------------------------------#

def env_map(env_name):
    '''
        map D4RL dataset names to custom fully-observed
        variants for rendering
    '''
    if 'halfcheetah' in env_name:
        return 'HalfCheetahFullObs-v2'
    elif 'hopper' in env_name:
        return 'HopperFullObs-v2'
    elif 'walker2d' in env_name:
        return 'Walker2dFullObs-v2'
    else:
        return env_name

#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#
class ParticleRenderer:
    '''
        particle renderer that takes particles as input and decode them into images
    '''

    def __init__(self, env, particle_dim=10, single_view=False, latent_rep_model=None, **kwargs):
        self.env = env
        self.latent_rep_model=latent_rep_model
        self.particle_dim = particle_dim
        self.require_bg = True
        self.single_view = single_view
    
    def render(self, particles, front_bg, side_bg, ret_glimpse=False):
        if self.env is None:
            latent_rep_model = self.latent_rep_model
            device='cuda'
        else:
            latent_rep_model = self.env.latent_rep_model
            device = self.env.device
        particles = particles.reshape(1, -1, self.particle_dim)
        if self.single_view:
            particles = particles[:, :particles.shape[1], :]
        else:
            front_particles = particles[:, :particles.shape[1]//2, :]
            side_particles = particles[:, particles.shape[1]//2:, :]
        
        if ret_glimpse:
            front_image, glimpse_image = get_recon_from_dlps(front_particles, front_bg, latent_rep_model, device, ret_glimpse=ret_glimpse)
            final_image = np.concatenate([front_image, glimpse_image], axis=0)
        else:
            if self.single_view:
                front_image = get_recon_from_dlps(particles, front_bg, latent_rep_model, device)
                final_image = front_image
            else:
                front_image = get_recon_from_dlps(front_particles, front_bg, latent_rep_model, device)
                side_image = get_recon_from_dlps(side_particles, side_bg, latent_rep_model, device)
                final_image = np.concatenate([front_image, side_image], axis=0)
        return final_image
    
    def renders(self, samples, front_bg, side_bg, **kwargs):
        sample_images = []
        for sample in samples:
            sample_images.append(self.render(sample, front_bg, side_bg, **kwargs))
        return np.concatenate(sample_images, axis=1)

    def composite(self, savepath, paths, front_bg, side_bg, **kwargs):
        images = []
        for path in paths:
            img = self.renders(to_np(path), front_bg, side_bg, **kwargs)
            images.append(img)
        images = np.concatenate(images, axis=0)

        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images

class MatplotlibRenderer:
    '''
        default matplotlib renderer
    '''

    def __init__(self, env, num_entity=1, **kwargs):
        self.num_entity = num_entity

    def render(self, observation, dim=256, dpi=100, is_goal=False):
        if type(dim) == int:
            dim = (dim, dim)
        
        # Calculate figure size in inches to match desired output dimensions
        figsize = (dim[0] / dpi, dim[1] / dpi)

        state = observation
        object_states = state.reshape(self.num_entity+1, -1)[:,:2]
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow'][:self.num_entity+1]
        
        # Create a figure with the calculated size
        fig = Figure(figsize=figsize, dpi=dpi)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)  # Add a subplot to the figure

        # Plot the first marker as a circle
        if is_goal:
            marker = 'x'
        else:
            marker = 'o'
        ax.scatter(object_states[0, 0], object_states[0, 1], c=colors[0], s=50, marker=marker)

        # Plot the rest of the markers as squares
        if len(object_states) > 1:
            if is_goal:
                marker = 'x'
            else:
                marker = 's'
            ax.scatter(object_states[1:, 0], object_states[1:, 1], c=colors[1:], s=50, marker=marker)
            
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')

        # Set fixed axis limits
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        # ax.grid(True)

        # Render the plot to a buffer
        canvas.draw()

        # Convert the buffer to a numpy array
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Return the RGB image
        return image

    def _renders_og(self, observations, dim, **kwargs):
        images = []
        for t, observation in enumerate(observations):
            img = self.render(observation, dim, **kwargs)
            img = img.copy()
            cv.putText(img, f'Timestep: {t}', (10, 10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            images.append(img)
        return np.stack(images, axis=0)

    def _renders(self, observations, dim, gc=True, **kwargs):
         # Initialize a base image to blend onto; start with a blank canvas
        base_image = np.ones((dim[1], dim[0], 3), dtype=np.uint8) * 255
        
        # Calculate the step to decrease alpha for each observation
        alpha_step = 1.0 / len(observations)
        alpha = 1.0  # Initial alpha value for the first observation
        
        for i, observation in enumerate(observations):
            # Render the observation to an image
            is_goal = gc and i == len(observations) - 1
            img = self.render(observation, dim=dim, is_goal=is_goal, **kwargs)
            if is_goal:
                alpha = 1.0  # Reset alpha for the goal image
            
            # Apply alpha blending
            # Assuming non-white pixels are the foreground
            foreground_mask = np.any(img < 255, axis=-1, keepdims=True)
            base_image = (
                base_image * (1 - alpha * foreground_mask) + img * (alpha * foreground_mask)
            ).astype(np.uint8)
            
            # Decrease alpha for the next image
            alpha -= alpha_step
        return base_image

    def renders(self, samples, dim, gc=True, **kwargs):

        sample_images = self._renders(samples, dim, gc=gc, **kwargs)

        # composite = np.ones_like(sample_images[0]) * 255
        # for img in sample_images:
        #     mask = get_image_mask(img)
        #     composite[mask] = img[mask]

        return sample_images

    def composite(self, savepath, paths, dim=(256, 256), gc=True, **kwargs):
        print(paths.shape)
        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            img = self.renders(to_np(path), dim=dim, gc=gc, **kwargs)
            images.append(img)
        images = np.concatenate(images, axis=0)

        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images

    def render_rollout(self, savepath, states, goal, **video_kwargs):
        if type(states) is list: states = np.array(states)
        images = self._renders_og(states, (256,256))
        if len(goal.shape) == 2:
            goal_images = self._renders_og(goal, (256,256))
            full_images = np.concatenate([images, goal_images], axis=-2)
        else:
            goal_image = self.render(goal, (256,256), is_goal=True)
            full_images = np.concatenate([images, np.tile(goal_image[None], (len(images), 1, 1, 1))], axis=-2)
        save_video(savepath, full_images, **video_kwargs)
    
    def render_rollout_joint(self, savepath, states1, goal1, states2, goal2, **video_kwargs):
        if type(states1) is list: states1 = np.array(states1)
        if type(states2) is list: states2 = np.array(states2)
        images1 = self._renders_og(states1, (256,256))
        images2 = self._renders_og(states2, (256,256))
        goal_image1 = self.render(goal1, (256,256), is_goal=True)
        goal_image2 = self.render(goal2, (256,256), is_goal=True)
        full_images = np.concatenate([images1, np.tile(goal_image1[None], (len(images1), 1, 1, 1)), images2, np.tile(goal_image2[None], (len(images2), 1, 1, 1))], axis=-2)
        save_video(savepath, full_images, **video_kwargs)

class MuJoCoRenderer:
    '''
        default mujoco renderer
    '''

    def __init__(self, env):
        if type(env) is str:
            env = env_map(env)
            self.env = gym.make(env)
        else:
            self.env = env
        ## - 1 because the envs in renderer are fully-observed
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)
        try:
            self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)
        except:
            print('[ utils/rendering ] Warning: could not initialize offscreen renderer')
            self.viewer = None

    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def pad_observations(self, observations):
        qpos_dim = self.env.sim.data.qpos.size
        ## xpos is hidden
        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
        xpos = np.cumsum(xvel) * self.env.dt
        states = np.concatenate([
            xpos[:,None],
            observations,
        ], axis=-1)
        return states

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None):

        if type(dim) == int:
            dim = (dim, dim)

        if self.viewer is None:
            return np.zeros((*dim, 3), np.uint8)

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [xpos, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        sample_images = self._renders(samples, partial=partial, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        return composite

    def composite(self, savepath, paths, dim=(1024, 256), **kwargs):

        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [5, 2, 0.5],
            'elevation': 0
        }
        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            img = self.renders(to_np(path), dim=dim, partial=True, qvel=True, render_kwargs=render_kwargs, **kwargs)
            images.append(img)
        images = np.concatenate(images, axis=0)

        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images

    def render_rollout(self, savepath, states, **video_kwargs):
        if type(states) is list: states = np.array(states)
        images = self._renders(states, partial=True)
        save_video(savepath, images, **video_kwargs)

    def render_plan(self, savepath, actions, observations_pred, state, fps=15):
        ## [ batch_size x horizon x observation_dim ]
        observations_real = rollouts_from_state(self.env, state, actions)

        ## there will be one more state in `observations_real`
        ## than in `observations_pred` because the last action
        ## does not have an associated next_state in the sampled trajectory
        observations_real = observations_real[:,:-1]

        images_pred = np.stack([
            self._renders(obs_pred, partial=True)
            for obs_pred in observations_pred
        ])

        images_real = np.stack([
            self._renders(obs_real, partial=False)
            for obs_real in observations_real
        ])

        ## [ batch_size x horizon x H x W x C ]
        images = np.concatenate([images_pred, images_real], axis=-2)
        save_videos(savepath, *images)

    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        '''
            diffusion_path : [ n_diffusion_steps x batch_size x 1 x horizon x joined_dim ]
        '''
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [10, 2, 0.5],
            'elevation': 0,
        }

        diffusion_path = to_np(diffusion_path)

        n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape

        frames = []
        for t in reversed(range(n_diffusion_steps)):
            print(f'[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}')

            ## [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[:, :, :self.observation_dim]

            frame = []
            for states in states_l:
                img = self.composite(None, states, dim=(1024, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
                frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        save_video(savepath, frames, **video_kwargs)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

#-----------------------------------------------------------------------------#
#---------------------------------- rollouts ---------------------------------#
#-----------------------------------------------------------------------------#

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
            f'but got state of size {state.size}')
        state = state[:qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def rollouts_from_state(env, state, actions_l):
    rollouts = np.stack([
        rollout_from_state(env, state, actions)
        for actions in actions_l
    ])
    return rollouts

def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        ## if terminated early, pad with zeros
        observations.append( np.zeros(obs.size) )
    return np.stack(observations)
