import numpy as np
import scipy.ndimage.filters
import scipy.ndimage
import scipy.stats
import torch

class NullSensor:
    def observe(self, s):
        return s

class OffsetSensor:
    def __init__(self, offset):
        self.offset = offset

    def observe(self, s):
        return s + self.offset

class NoisySensor:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def observe(self, s):
        if s.ndim == 0:
            x = s + self.sigma * np.random.randn()
        elif s.ndim == 1:
            x = s + self.sigma * np.random.randn(len(s))
        else:
            x = s + self.sigma * np.random.randn(s.shape[0], *s.shape[1:])
        return x

class ImageSensor:
    def __init__(self, range, pixel_density=1):
        assert isinstance(range, (tuple, list, np.ndarray))
        self.range = range
        self.size = (pixel_density*range[0][-1],pixel_density*range[1][-1])

    def observe(self, s):
        assert s.ndim > 0 and s.shape[-1] == 2
        if s.ndim==1:
            s = np.expand_dims(s, axis=0)
        n_samples = s.shape[0]
        digitized = scipy.stats.binned_statistic_2d(s[:,0],s[:,1],np.arange(n_samples), statistic='count', bins=self.size, range=self.range, expand_binnumbers=True)
        digitized = digitized[-1].transpose()
        x = np.zeros([n_samples,self.size[0],self.size[1]])
        for i in range(n_samples):
            x[i,digitized[i,0]-1,digitized[i,1]-1] = 1
        if n_samples==1:
            x = x[0,:,:]
        return x

class ResampleSensor:
    def __init__(self, scale, order=0):
        self.scale = scale
        self.order = order

    def observe(self, s):
        return scipy.ndimage.zoom(s, zoom=self.scale, order=self.order)

class BlurSensor:
    def __init__(self, sigma=0.6, truncate=1.0):
        self.sigma = sigma
        self.truncate = truncate

    def observe(self, s):
        return scipy.ndimage.filters.gaussian_filter(s, sigma=self.sigma, truncate=self.truncate, mode='nearest')

class TorchSensor:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
    def observe(self, s):
        return torch.as_tensor(s, dtype=self.dtype)

class SensorChain:
    def __init__(self, sensors):
        self.sensors = sensors

    def observe(self, s):
        for sensor in self.sensors:
            s = sensor.observe(s)
        return s
