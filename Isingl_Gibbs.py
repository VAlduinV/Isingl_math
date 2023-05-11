import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk


class IsingModel:
    def __init__(self, size, beta):
        self.size = size
        self.beta = beta
        self.spin_grid = np.random.choice([-1, 1], size=(size, size))

    def __repr__(self):
        return f"IsingModel(size={self.size}, beta={self.beta})"

    def total_energy(self):
        return -np.sum(self.spin_grid * (np.roll(self.spin_grid, 1, axis=0) + np.roll(self.spin_grid, 1, axis=1)))

    def delta_energy(self, i, j):
        s = self.spin_grid[i, j]
        neighbours = self.spin_grid[(i + 1) % self.size, j] + self.spin_grid[i - 1, j] + self.spin_grid[
            i, (j + 1) % self.size] + self.spin_grid[i, j - 1]
        return 2 * s * neighbours

    def gibbs_step(self):
        for i in range(self.size):
            for j in range(self.size):
                delta_E = self.delta_energy(i, j)
                prob = 1 / (1 + np.exp(self.beta * delta_E))
                if np.random.rand() < prob:
                    self.spin_grid[i, j] *= -1

    def simulate(self, steps):
        for step in range(steps):
            self.gibbs_step()
        return self.spin_grid


if __name__ == '__main__':
    # Створюємо об'єкт IsingModel та запускаємо симуляції для різних значень beta
    betas = [0.2, 0.5, 1.0]
    plt.style.use("cyberpunk")

    for beta in betas:
        ising = IsingModel(20, beta)
        spin_config = ising.simulate(1000)
        print(f'beta = {beta}:\n Matrix:\n{spin_config}')
        plt.figure()
        plt.imshow(spin_config, cmap='gray')
        mplcyberpunk.add_glow_effects()
        plt.title(f'Final Spin Configuration, beta={beta}')
        plt.show()
