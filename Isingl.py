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

    def flip_spin(self, i, j):
        self.spin_grid[i, j] *= -1

    def delta_energy(self, i, j):
        s = self.spin_grid[i, j]
        neighbours = self.spin_grid[(i + 1) % self.size, j] + self.spin_grid[i - 1, j] + self.spin_grid[
            i, (j + 1) % self.size] + self.spin_grid[i, j - 1]
        return 2 * s * neighbours

    def metropolis_step(self):
        i, j = np.random.randint(0, self.size, 2)
        delta_E = self.delta_energy(i, j)
        if delta_E < 0 or np.random.rand() < np.exp(-self.beta * delta_E):
            self.flip_spin(i, j)

    def simulate(self, steps):
        for step in range(steps):
            self.metropolis_step()
        return self.spin_grid


if __name__ == '__main__':
    plt.style.use("cyberpunk")
    # Створюємо об'єкт IsingModel та запускаємо симуляцію
    "Експериментуйте вводячи значення від 20-600 визначаючи розмір"
    ising = IsingModel(20, 0.5)
    """
        Цей код виводить графік фінальної конфігурації спінів після симуляції. 
        Зауважте, що ви можете змінювати параметри симуляції (розмір сітки, β, кількість кроків), 
        щоб бачити, як вони впливають на результати.
        Цей код створює модель Ізінга, розміром 20x20 із випадковим початковим розподілом спінів. 
        За допомогою алгоритму Метрополіса-Гастінгса проводяться кроки симуляції, 
        кожен з яких або перевертає спін в випадковому місці (якщо це зменшує енергію), 
        або перевертає його з деякою ймовірністю, якщо це збільшує енергію.
    """
    print(ising)  # Тепер це виведе "IsingModel(size=20, beta=0.5)"

    spin_config = ising.simulate(10000)
    print(spin_config)

    # Plot the final spin configuration
    plt.imshow(spin_config, cmap='jet')
    mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.5)
    mplcyberpunk.add_glow_effects()
    plt.title('Final Spin Configuration')
    plt.show()
