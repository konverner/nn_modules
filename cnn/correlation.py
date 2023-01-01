import numpy as np


class Corr1d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel = np.random.uniform(0, 1, size=(out_channels, in_channels, kernel_size))

    def set_kernel(self, kernel):
        self.kernel = kernel

    def __call__(self, g):
        """
        g : (in_channels, in_length)
        """
        result = []
        in_length = len(g[0])
        for c_out in range(self.out_channels):
            channel = []
            for n in range(0, in_length - self.kernel_size + 1, self.stride):
                s = 0
                for c_in in range(self.in_channels):
                    s += self._step(self.kernel[c_out][c_in], g[c_in], n)
                channel.append(s)
            result.append(channel)
        return np.array(result)

    def _step(self, f, g, n):
        """
        f : (kernel_size,)
        g : (in_length,)
        n : argument
        """
        s = 0
        for k in range(len(f)):
            if k + n < len(g):
                s += f[k] * g[k + n]
        return s
