import matplotlib.pyplot as plt


def plot_graphics(x_axis, y_axis, x_label, y_label, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_axis, y_axis, 'o', color="#A60628")
    plt.axvline(x=2.269185, label=r'$T_c$', color='yellow')
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)
    ax.grid()
    ax.set_title(title)
    ax.legend(loc=2)
    plt.show()


def plot_graphics_with_error_bar(x_list, y_list, x_label, y_label, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x_list, y_list, fmt='.k')
    plt.axvline(x=2.269185, label=r'$T_c$', color='yellow')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()
    ax.set_title(title)
    plt.show()
