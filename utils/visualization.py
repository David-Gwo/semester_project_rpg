import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation


class Dynamic3DTrajectory:
    def __init__(self, data, sparsing_factor):

        self.data = np.array(data)

        self.data_len = len(data[0])
        self.n_lines = len(data)

        if sparsing_factor == 0:
            sparse_data = np.arange(0, self.data_len)
        else:
            sparse_data = np.arange(0, self.data_len, sparsing_factor)

        self.sparsed_data = self.data[:, sparse_data, :]
        self.colors = plt.cm.jet(np.linspace(0, 1, self.n_lines))

        self.max_buffer_size = 300

        self.data_len = len(sparse_data)

        x_data = np.concatenate(tuple([dat[sparse_data, 0] for dat in data]))
        y_data = np.concatenate(tuple([dat[sparse_data, 1] for dat in data]))
        z_data = np.concatenate(tuple([dat[sparse_data, 2] for dat in data]))

        self.max_x = np.max(x_data)
        self.min_x = np.min(x_data)
        self.max_x += (self.max_x - self.min_x) * 0.2
        self.min_x -= (self.max_x - self.min_x) * 0.2

        self.max_y = np.max(y_data)
        self.min_y = np.min(y_data)
        self.max_y += (self.max_y - self.min_y) * 0.2
        self.min_y -= (self.max_y - self.min_y) * 0.2

        self.max_z = np.max(z_data)
        self.min_z = np.min(z_data)
        self.max_z += (self.max_z - self.min_z) * 0.2
        self.min_z -= (self.max_z - self.min_z) * 0.2

        self.figure = None
        self.ax = None
        self.lines = None
        self.projection_lines = None

    def on_launch(self):
        self.figure = plt.figure()
        self.ax = axes3d.Axes3D(self.figure)

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

        # Set up plot
        self.ax.set_zlim3d([self.min_z, self.max_z])
        self.ax.set_ylim3d([self.min_y, self.max_y])
        self.ax.set_xlim3d([self.min_x, self.max_x])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.lines = sum([self.ax.plot([], [], [], '-', c=c) for c in self.colors], [])
        self.projection_lines = sum([self.ax.plot([], [], [], '-', c=c, alpha=0.2) for c in self.colors], [])
        self.projection_lines += sum([self.ax.plot([], [], [], '-', c=c, alpha=0.2) for c in self.colors], [])

        self.lines += self.projection_lines

        self.ax.set_title('3D Test')

    def on_init(self):
        for line in self.lines:
            line.set_data([], [])
            line.set_3d_properties([])

        return self.lines

    def animate(self, i):
        i = (2 * i) % self.data.shape[1]
        for j, (line, xi) in enumerate(zip(self.lines[:self.n_lines], self.sparsed_data)):
            x, y, z = xi[:i].T

            if len(x) > self.max_buffer_size:
                x = x[len(x) - self.max_buffer_size:]
                y = y[len(y) - self.max_buffer_size:]
                z = z[len(z) - self.max_buffer_size:]

            line.set_data(x, y)
            line.set_3d_properties(z)

            self.lines[j + self.n_lines].set_data(x, self.max_y)
            self.lines[j + self.n_lines].set_3d_properties(z)

            self.lines[j + 2 * self.n_lines].set_data(np.ones(len(y)) * self.min_x, y)
            self.lines[j + 2 * self.n_lines].set_3d_properties(z)

        return self.lines

    def __call__(self):
        self.on_launch()

        ani = animation.FuncAnimation(self.figure, self.animate, init_func=self.on_init, frames=self.data_len,
                                      interval=5, blit=True, repeat=False)

        plt.show()

        return self.figure
