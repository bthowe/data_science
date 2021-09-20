import numpy as np
import matplotlib.pyplot as plt

def p_9():
    problem_65_9 = lambda x: np.cos(x) - np.sqrt(1 - np.cos(x) ** 2)
    to_radian = lambda x: x * np.pi / 180
    print(problem_65_9(to_radian(315)))

def p_11():
    x = np.linspace(-np.pi, np.pi, 1000)
    y = np.log2(x)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y)
    plt.show()

def p_12():
    print(np.arctan(3/5))
    print(np.arctan(5/3))

def main():
    # p_9()
    # p_11()
    p_12()

if __name__ == '__main__':
    main()
