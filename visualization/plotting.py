import matplotlib.pyplot as plt

def plot_histogram(variable, title='', add_intercept=True, intercept=0):
    plt.hist(variable)

    if add_intercept:
        plt.axhline(y=intercept, xmin=0.0, xmax=1.0, color='r')
    plt.title(title)
    plt.show()

    return None