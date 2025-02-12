import numpy as np
import scipy.interpolate as interpolate
import pdb

POINTMASS_KEYS = ['observations', 'actions', 'next_observations', 'deltas']

#-----------------------------------------------------------------------------#
#--------------------------- multi-field normalizer --------------------------#
#-----------------------------------------------------------------------------#

class DatasetNormalizer:

    def __init__(self, dataset, normalizer, particle_normalizer=None, path_lengths=None):
        self.observation_dim = dataset['observations'].shape[-1]
        self.action_dim = dataset['actions'].shape[-1]

        if type(normalizer) == str:
            normalizer = eval(normalizer)
        if particle_normalizer is not None and type(particle_normalizer) == str:
            particle_normalizer = eval(particle_normalizer)
            if 'goals' in dataset._dict:
                goal_X = dataset['goals'][:, 0:1, :, :]
                goal_X = goal_X.reshape(-1, *goal_X.shape[2:])
                print("goals X shape: ", goal_X.shape)
            else:
                goal_X = None

        dataset = flatten(dataset, path_lengths)
        self.normalizers = {}
        for key, val in dataset.items():
            try:
                if key == 'observations' and particle_normalizer is not None:
                    ### concatenate goals to observations for normalizing
                    if goal_X is not None:
                        obs_goal = np.concatenate([val, goal_X], axis=0)
                    else:
                        obs_goal = val
                    self.normalizers[key] = particle_normalizer(obs_goal)
                elif key == 'goals' and particle_normalizer is not None:
                    continue
                else:
                    self.normalizers[key] = normalizer(val)
            except Exception as e:
                print(e)
                print(f'[ utils/normalization ] Skipping {key} | {normalizer}')

    def __repr__(self):
        string = ''
        for key, normalizer in self.normalizers.items():
            string += f'{key}: {normalizer}]\n'
        return string

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)

    def get_field_normalizers(self):
        return self.normalizers

def flatten(dataset, path_lengths):
    '''
        flattens dataset of { key: [ n_episodes x max_path_lenth x dim ] }
            to { key : [ (n_episodes * sum(path_lengths)) x dim ]}
    '''
    flattened = {}
    for key, xs in dataset.items():
        if len(xs) == 0:
            continue
        assert len(xs) == len(path_lengths)
        if 'info' not in key:
            flattened[key] = np.concatenate([
                x[:length]
                for x, length in zip(xs, path_lengths)
            ], axis=0)
    return flattened

#-----------------------------------------------------------------------------#
#-------------------------- single-field normalizers -------------------------#
#-----------------------------------------------------------------------------#

class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class DebugNormalizer(Normalizer):
    '''
        identity function
    '''

    def normalize(self, x, *args, **kwargs):
        return x

    def unnormalize(self, x, *args, **kwargs):
        return x


class GaussianNormalizer(Normalizer):
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = self.X.mean(axis=0)
        self.stds = self.X.std(axis=0)
        self.z = 1

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    '''
            f'''means: {np.round(self.means, 2)}\n    '''
            f'''stds: {np.round(self.z * self.stds, 2)}\n'''
        )

    def normalize(self, x):
        return (x - self.means) / (self.stds + 1e-6)

    def unnormalize(self, x):
        return x * self.stds + self.means


class LimitsNormalizer(Normalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''

    def normalize(self, x):
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins

class SafeLimitsNormalizer(LimitsNormalizer):
    '''
        functions like LimitsNormalizer, but can handle data for which a dimension is constant
    '''

    def __init__(self, *args, eps=1, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.mins)):
            if self.mins[i] == self.maxs[i]:
                print(f'''
                    [ utils/normalization ] Constant data in dimension {i} | '''
                    f'''max = min = {self.maxs[i]}'''
                )
                self.mins -= eps
                self.maxs += eps

#-----------------------------------------------------------------------------#
#-------------------------- Particle normalizers -----------------------------#
#-----------------------------------------------------------------------------#

class ParticleNormalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X):
        assert len(X.shape) == 3
        self.x_dim = X.shape[-1]
        self.X = X.astype(np.float32)
        self.mins = X.reshape(-1, X.shape[-1]).min(axis=0)
        self.maxs = X.reshape(-1, X.shape[-1]).max(axis=0)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()

class ParticleGaussianNormalizer(ParticleNormalizer):
    '''
        normalizes to zero mean and unit variance, input is N x number of particles x dim
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = self.X.reshape(-1, self.X.shape[-1]).mean(axis=0)
        self.stds = self.X.reshape(-1, self.X.shape[-1]).std(axis=0)
        self.z = 1

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    '''
            f'''means: {np.round(self.means, 2)}\n    '''
            f'''stds: {np.round(self.z * self.stds, 2)}\n'''
        )

    def normalize(self, x):
        x_unflat = x.reshape(-1, self.x_dim)
        ret_unflat = (x_unflat - self.means) / self.stds
        return ret_unflat.reshape(x.shape)

    def unnormalize(self, x):
        x_unflat = x.reshape(-1, self.x_dim)
        ret_unflat = x_unflat * self.stds + self.means
        return ret_unflat.reshape(x.shape)

class ParticleLimitsNormalizer(ParticleNormalizer):
    '''
        normalizes to zero mean and unit variance, input is N x number of particles x dim
    '''

    def __init__(self, X, eps=1):
        super().__init__(X)
        for i in range(len(self.mins)):
            if self.mins[i] == self.maxs[i]:
                print(f'''
                    [ utils/normalization ] Constant data in dimension {i} | '''
                    f'''max = min = {self.maxs[i]}'''
                )
                self.mins[i] -= eps
                self.maxs[i] += eps

    def normalize(self, x):
        x_unflat = x.reshape(-1, self.x_dim)
        ret_unflat = (x_unflat - self.mins) / (self.maxs - self.mins) ## [0, 1]
        ret_unflat = 2 * ret_unflat - 1 # [-1, 1]
        return ret_unflat.reshape(x.shape)

    def unnormalize(self, x, eps=1e-4):
        x_unflat = x.reshape(-1, self.x_dim)
        if x_unflat.max() > 1 + eps or x_unflat.min() < -1 - eps:
            print(f'[ datasets/dlp ] Warning: sample out of range | ({x_unflat.min():.4f}, {x_unflat.max():.4f})')
            x_unflat = np.clip(x_unflat, -1, 1)
        
        ## [ -1, 1 ] --> [ 0, 1 ]
        x_unflat = (x_unflat + 1) / 2.
        ret_unflat = x_unflat * (self.maxs - self.mins) + self.mins
        return ret_unflat.reshape(x.shape)

class ParticleDoNothingNormalizer(ParticleNormalizer):
    '''
        normalizes to zero mean and unit variance, input is N x number of particles x dim
    '''

    def __init__(self, X, eps=1):
        super().__init__(X)

    def normalize(self, x):
        return x

    def unnormalize(self, x, eps=1e-4):
        return x
#-----------------------------------------------------------------------------#
#------------------------------- CDF normalizer ------------------------------#
#-----------------------------------------------------------------------------#

class CDFNormalizer(Normalizer):
    '''
        makes training data uniform (over each dimension) by transforming it with marginal CDFs
    '''

    def __init__(self, X):
        super().__init__(atleast_2d(X))
        self.dim = self.X.shape[1]
        self.cdfs = [
            CDFNormalizer1d(self.X[:, i])
            for i in range(self.dim)
        ]

    def __repr__(self):
        return f'[ CDFNormalizer ] dim: {self.mins.size}\n' + '    |    '.join(
            f'{i:3d}: {cdf}' for i, cdf in enumerate(self.cdfs)
        )

    def wrap(self, fn_name, x):
        shape = x.shape
        ## reshape to 2d
        x = x.reshape(-1, self.dim)
        out = np.zeros_like(x)
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out[:, i] = fn(x[:, i])
        return out.reshape(shape)

    def normalize(self, x):
        return self.wrap('normalize', x)

    def unnormalize(self, x):
        return self.wrap('unnormalize', x)

class CDFNormalizer1d:
    '''
        CDF normalizer for a single dimension
    '''

    def __init__(self, X):
        assert X.ndim == 1
        self.X = X.astype(np.float32)
        quantiles, cumprob = empirical_cdf(self.X)
        self.fn = interpolate.interp1d(quantiles, cumprob)
        self.inv = interpolate.interp1d(cumprob, quantiles)

        self.xmin, self.xmax = quantiles.min(), quantiles.max()
        self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def __repr__(self):
        return (
            f'[{np.round(self.xmin, 2):.4f}, {np.round(self.xmax, 2):.4f}'
        )

    def normalize(self, x):
        x = np.clip(x, self.xmin, self.xmax)
        ## [ 0, 1 ]
        y = self.fn(x)
        ## [ -1, 1 ]
        y = 2 * y - 1
        return y

    def unnormalize(self, x, eps=1e-4):
        '''
            X : [ -1, 1 ]
        '''
        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            print(
                f'''[ dataset/normalization ] Warning: out of range in unnormalize: '''
                f'''[{x.min()}, {x.max()}] | '''
                f'''x : [{self.xmin}, {self.xmax}] | '''
                f'''y: [{self.ymin}, {self.ymax}]'''
            )

        x = np.clip(x, self.ymin, self.ymax)

        y = self.inv(x)
        return y

def empirical_cdf(sample):
    ## https://stackoverflow.com/a/33346366

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def atleast_2d(x):
    if x.ndim < 2:
        x = x[:,None]
    return x

