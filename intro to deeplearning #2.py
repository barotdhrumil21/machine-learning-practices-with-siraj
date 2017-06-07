

#code is below

from numpy import *


def compute_error(b, m, points):
    totalerror = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]

        totalerror += (y - (m * x + b)) ** 2
        return totalerror / float(len(points))


def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        b_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def gradient_runner(points, starting_b, starting_m, learning_rate, iterations):
    b = starting_b
    m = starting_m

    for i in range(iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]


def run():
    points = genfromtxt('data.csv', delimiter=',')
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    [b, m] = gradient_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print(b)
    print(m)


if __name__ == '__main__':
    run()

""if you are reading this siraj please give me some work with java or
python  i'll do it for freeee""
