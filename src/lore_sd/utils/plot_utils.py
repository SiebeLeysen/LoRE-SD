def clean_axis(ax):
    """
    Remove all tick labels, ticks and spines from the axis to clean it up. Mainly used to show images with plt.imshow().
    :param ax: matplotlib.pyplot axis
    :return: None
    """
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)