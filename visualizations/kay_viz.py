import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def algebra1_114_b1():
    f = lambda x: 3 ** x
    x = np.linspace(-3, 3, 1_000)
    y = f(x)
    
    sns.set_theme(style='whitegrid')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y)
    plt.savefig('/Users/travis/Downloads/algebra1_114_b1.png')
    plt.show()


def algebra1_114_b2():
    f = lambda x: (1 / 3) ** x
    x = np.linspace(-3, 3, 1_000)
    y = f(x)
    
    sns.set_theme(style='whitegrid')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y)
    # plt.savefig('/Users/travis/Downloads/algebra1_114_b1.png')
    plt.show()


def algebra1_114_b():
    f1 = lambda x: 3 ** x
    x = np.linspace(-3, 3, 1_000)
    y1 = f1(x)
    
    f2 = lambda x: (1 / 3) ** x
    x = np.linspace(-3, 3, 1_000)
    y2 = f2(x)
    
    sns.set_theme(style='whitegrid')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='pink')
    ax.plot(x, y2, color='cornflowerblue')
    plt.savefig('/Users/travis/Downloads/algebra1_114_b.png')
    # plt.show()


def algebra1_114_4():
    """
    x   |   y
    ___________
    -3  |   1 / 27
    -2  |   1 / 9
    -1  |   1 / 3
    0   |   1
    1   |   3
    2   |   9
    3   |   27
    4   |   81
    """
    f1 = lambda x: 2 ** x
    x = np.linspace(-3, 3, 1_000)
    y1 = f1(x)
    
    f2 = lambda x: 3 ** x
    x = np.linspace(-3, 3, 1_000)
    y2 = f2(x)
    
    f3 = lambda x: 4 ** x
    x = np.linspace(-3, 3, 1_000)
    y3 = f3(x)
    
    f4 = lambda x: 5 ** x
    x = np.linspace(-3, 3, 1_000)
    y4 = f4(x)
    
    sns.set_theme(style='whitegrid')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='pink')
    ax.plot(x, y2, color='cornflowerblue')
    ax.plot(x, y3, color='purple')
    ax.plot(x, y4, color='green')
    plt.savefig('/Users/travis/Downloads/algebra1_114_4.png')
    # plt.show()


def algebra1_114_4():
    f1 = lambda x: 2 ** x
    x = np.linspace(-3, 3, 1_000)
    y1 = f1(x)
    
    f2 = lambda x: 3 ** x
    x = np.linspace(-3, 3, 1_000)
    y2 = f2(x)
    
    f3 = lambda x: 4 ** x
    x = np.linspace(-3, 3, 1_000)
    y3 = f3(x)
    
    f4 = lambda x: 5 ** x
    x = np.linspace(-3, 3, 1_000)
    y4 = f4(x)
    
    sns.set_theme(style='whitegrid')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='pink')
    ax.plot(x, y2, color='cornflowerblue')
    ax.plot(x, y3, color='purple')
    ax.plot(x, y4, color='green')
    plt.savefig('/Users/travis/Downloads/algebra1_114_4.png')
    # plt.show()




def main():
    # algebra1_114_b1()
    # algebra1_114_b2()
    # algebra1_114_b()
    algebra1_114_4()


if __name__ == '__main__':
    main()
