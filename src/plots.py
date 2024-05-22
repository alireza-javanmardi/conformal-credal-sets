import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mpltern
from mpltern.datasets import get_dirichlet_pdfs

credal_color = (57/255, 172/255, 115/255) #greenish
gt_color = (255/255,153/255,0/255) #orangish


def violinplot_cvg_len_data(cvg_data, len_data, alphas, methods):
    """violin plots of the coverage and lengths in one figure

    Args:
        cvg_data(np.array): coverage matrix where rows represents the seeds, each len(alphas) columns belong to a CP method (shape: (len(seeds), len(alphas)*len(methods)))
        len_data(np.array): length matrix where rows represents the seeds, each len(alphas) columns belong to a CP method (shape: (len(seeds), len(alphas)*len(methods)))
        alphas(list): list of alphas
        methods(list): list of CP methods
    """

    labels = [] #labels for legend
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 40
    fig, ax1 = plt.subplots(figsize=(40, 10))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    ax1.set_ylabel('Average Coverage')
    violin_parts1 = ax1.violinplot(cvg_data, showmeans=True)
    violinplot_set_color(violin_parts1, 'tab:blue')

    ax1.set_xticks(np.arange(1, len(alphas)*len(methods)+1), labels=alphas*len(methods))
    ax1.set_xlim(0.25, len(alphas)*len(methods) + 0.75)
    ax1.set_xlabel(r'Miscoverage rate ($\alpha$)')

    #vertical lines for separating methods from each other
    for i in range(len(methods)-1):
        ax1.axvline(x=len(alphas)*(i+1)+ 0.5, color='k')
    
    #Horizontal lines for nominal coverage 
    for a in alphas:
        ax1.axhline(y=1-a, color='k', linestyle=":")
    ax1.set_ylim([0,1])
    labels.append((mpatches.Patch(color=violin_parts1['bodies'][0].get_facecolor()), 'Average Coverage'))



    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Average Efficiency')  
    violin_parts2 = ax2.violinplot(len_data, showmeans=True)
    violinplot_set_color(violin_parts2, 'r')
    ax2.set_ylim([0,1])
    secax = ax1.secondary_xaxis('top')
    secax.set_xticks(np.arange((len(alphas)+1)/2, len(alphas)*len(methods), len(alphas)), labels=methods)
    labels.append((mpatches.Patch(color=violin_parts2['bodies'][0].get_facecolor()), 'Average Efficiency'))
    plt.legend(*zip(*labels), loc="lower left")
    
def violinplot_set_color(violin_parts, color):
    """set the color of the violin plot

    Args:
        violin_parts: return values of the violinplot
        color (str): the color
    """
    for pc in violin_parts['bodies']:
        pc.set_color(color)
        pc.set_alpha(0.3)
    violin_parts['cbars'].set_color('k')
    violin_parts['cmaxes'].set_color('k')
    violin_parts['cmins'].set_color('k')
    violin_parts['cmeans'].set_color('k')


def plot_interval(a, b, p=None, ph=None, ax=None):
    """plot uncertainty interval 

    Args:
        a (float): between 0 and 1, the lower entropy of the credal set
        b (float): between 0 and 1, the upper entropy of the credal set
        p (float): between 0 and 1, the entropy of the ground-truth distribution
        ph (float): between 0 and 1, the entropy of the predicted distribution
        ax (axes, optional): In case we want the plot to be in a specific given ax. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,1))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 30
    
    # Plot the line from 0 to 1
    ax.plot([0, 1], [0, 0], color='black', linewidth=4)
    ax.fill_betweenx([0.1, 0], a, b, color=credal_color, alpha=1, zorder=3, linewidth=4)
    ax.scatter(0,0, marker=3,color='black', linewidth=4)
    ax.scatter(1,0, marker=3, color='black', linewidth=4)
    ax.scatter(0.5,0, marker=3, color='black', linewidth=4)
    if p is not None:
        ax.scatter(p,0.05, marker="X", color=gt_color, edgecolors="w",  zorder=3, s=200)
    if ph is not None:
        ax.scatter(ph,0.05, marker="X", color='black', edgecolors="w",  zorder=3, s=200)
    
    
    # Remove axis lines and ticks
    ax.axis('off')
    ax.set_ylim([-0.2,0.2])
    # Set ticks at 0, 0.5, and 1 on the line segment
    # ax.text(0.006, -0.07, '0 \n(Lowest AU)', ha='center', va='top')
    # ax.text(0.5, -0.07, '0.5', ha='center', va='top')
    # ax.text(1.006, -0.07, '1 \n(Highest AU)', ha='center', va='top')
    ax.text(0.006, -0.07, '0 ', ha='center', va='top')
    ax.text(0.5, -0.07, '0.5', ha='center', va='top')
    ax.text(1.006, -0.07, '1 ', ha='center', va='top')

def plot_cifar_dist(p):
    """bar plot cifar10 distributions 

    Args:
        p (array): a distribution over 10 classes 
    """
    labels = ["plane",  "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    fig, ax = plt.subplots(figsize=(10, 7))
    bar_container = ax.bar(np.arange(0,10),p, width=0.8)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 55
    # plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.ylim([0,1])
    # plt.title("Distribution of human annotations")
    plt.xticks(np.arange(0,10), labels=labels, rotation='vertical')
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        # labelbottom=False,
        labelleft=False
        )
    
def plot_cifar_img(img):
    """plot cifar10 images

    Args:
        img (array): an image from cifar10 
    """
    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.axis('off')
    plt.margins(x=0)

def my_ternery(credal_set, p, labels, ph=None, alpha=None, ax=None):
    if ax is None:
        plt.figure(figsize=(10,10))
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 70
        ax = plt.subplot(projection="ternary")
    ax.set_tlabel(labels[0])
    ax.set_llabel(labels[1])
    ax.set_rlabel(labels[2])
    ax.tick_params(tick1On=False, tick2On=False, label1On=False, label2On=False)
    ax.scatter(credal_set[:,0], credal_set[:,1], credal_set[:,2], color=credal_color,  rasterized=True)
    ax.scatter(p[0], p[1], p[2], s=500, marker="s", color=gt_color, edgecolors="w", zorder=3)
    if alpha is not None:
        t, l, r, v = get_dirichlet_pdfs(n=100, alpha=alpha)
        ax.tricontour(t, l, r, v, colors="k", linewidths=1.5, zorder=3, label="Predcited second_order distribution")
    if ph is not None: 
        ax.scatter(ph[0], ph[1], ph[2], s=500, c='k', edgecolors="w", zorder=3, label="Predcited distribution")
    