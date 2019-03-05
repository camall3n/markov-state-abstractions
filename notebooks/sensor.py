import numpy as np
import scipy.ndimage.filters
import scipy.stats

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
    def __init__(self, size):
        assert isinstance(size, (tuple, list, np.ndarray))
        self.size = size

    def observe(self, s):
        assert s.ndim == 2 and s.shape[-1] == 2
        n_samples = s.shape[0]
        digitized = scipy.stats.binned_statistic_2d(s[:,0],s[:,1],np.arange(n_samples), bins=self.size, expand_binnumbers=True)[-1].transpose()
        x = np.zeros([n_samples,self.size[0],self.size[1]])
        for i in range(n_samples):
            x[i,digitized[i,0]-1,digitized[i,1]-1] = 1
        return x

class BlurSensor:
    def __init__(self, sigma=0.6, truncate=1.0):
        self.sigma = sigma
        self.truncate = truncate

    def observe(self, s):
        return scipy.ndimage.filters.gaussian_filter(s, sigma=self.sigma, truncate=self.truncate, mode='nearest')

class SensorChain:
    def __init__(self, sensors):
        self.sensors = sensors

    def observe(self, s):
        for sensor in self.sensors:
            s = sensor.observe(s)
        return s
