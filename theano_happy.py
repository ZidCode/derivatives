import matplotlib.colors as colors
import numpy as np
import theano
import theano.tensor as T
from theano import pp
import matplotlib.pyplot as plt


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def main():
    tmp_x = np.arange(0, 10, .1)
    tmp_a = np.arange(0.2, 0.5, 0.01)
    tmp_y = np.zeros((len(tmp_x), len(tmp_a)))
    for ida, a in enumerate(tmp_a):
        tmp_y[:,ida] = np.sin(tmp_x * a) 
    plot2d(tmp_a, tmp_x, tmp_y)

    x = T.dscalar('x')
    a = T.dscalar('a')
    y = T.sin(x * a)
    ga = T.grad(y, a)
    pp(ga)
    f = theano.function([x, a], ga)
    a_series = np.arange(0.2, 0.5, .01)
    x_series = np.arange(0, 10, .1)
    y_series = np.zeros((len(x_series), len(a_series)))
    for idx, x_ in enumerate(x_series):
        for ida, a_ in enumerate(a_series):
            y_series[idx, ida] = f(x_, a_)
    plot2d(a_series, x_series, y_series)
    plot3d(a_series, x_series, y_series)


def plot2d(X, Y, Z):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    plt.figure()
    plt.pcolormesh(X, Y, Z, norm=MidpointNormalize(midpoint=0.), cmap='RdBu_r')
    plt.gca().set_aspect("auto")
    cb = plt.colorbar()
    plt.xlabel('a')
    plt.ylabel('x')
    plt.show()
   

def plot3d(X, Y, Z): 
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
    cset = ax.contour(X, Y, Z, zdir='z', offset=-11, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='x', offset=0, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='y', offset=12, cmap=cm.coolwarm)
    
    ax.set_xlabel('X')
    ax.set_xlim(0, 0.8)
    ax.set_ylabel('Y')
    ax.set_ylim(-1, 12)
    ax.set_zlabel('Z')
    ax.set_zlim(-11, 4)
    
    plt.show()    


if __name__ == "__main__":
    main()
