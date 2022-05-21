import matplotlib.pyplot as plt


def visualize_solution(t, y, y_name, steps_number, title, file_path):
    plt.plot(t, y)
    plt.title(''.join([title, '\n', ' (Number of steps = ', str(steps_number), ')']))
    plt.xlabel('t')
    plt.ylabel(y_name)

    plt.savefig(file_path + title + '.png')
    plt.show()
