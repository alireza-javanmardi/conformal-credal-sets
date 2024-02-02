import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def prob_projector(p, axis): 
    """distribute the probability mass of given axis equally between the other two 

    Args:
        p (array): the input 3-d categorical probability distribution
        axis (int): 0, 1, or 2. 

    Returns:
        array: the resulting distribution
    """
    new_p = p.copy()
    for i in range(len(p)): 
        new_p[i] = new_p[i] + p[axis]/2
    new_p[axis] = 0
    return new_p 




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