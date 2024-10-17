import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import datetime

def corr_dashboard(C, d, long_term_plot = True, 
                   process="MRC process",  years = [],
                   reflection_visual = False, reflection=[]):
    
    """
    Generate a the correlation figures (including eigenvalues and determinant-
                                        plots).
    
    Parameters:
    C (df): containing rho, lambda and determinant paths
    d (int): the dimension of the correlation matrix.
    long_term_plot (binary): include long-term mean plot?
    process (string): name of the process.
    Years (list): list containing periods that need to be marked.
    reflection_visual (binary): visualise places where reflection is used.
    reflection (list): list with time points where reflection is used.
    """
    
    sns.set_style("whitegrid")
    nr_of_unique_elements = int( d * (d-1) / 2 )
    
    # Choosing colors for the plots
    cmap = plt.get_cmap("tab20c")
    norm = plt.Normalize(0, np.max([nr_of_unique_elements, d]))
    colors = cmap(norm(range(np.max([nr_of_unique_elements, d]))))
    
    # Initializing the plot
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(2, 1, 1)  # initialize the top Axes
    ax2 = fig.add_subplot(2, 2, 3)  # initialize the bottom left Axes
    ax3 = fig.add_subplot(2, 2, 4)  # initialize the bottom right Axes
    
    titles = {
        ax1: f'Path of the off-diagonal elements of the {process}',
        ax2: 'Eigenvalues',
        ax3: 'Determinant'}
    
    ylims = {ax1: [-1, 1], ax2: [0, d], ax3: [0, 1]}
    
    color_counter = 0  # Dummy variable for coloring
    
    if reflection_visual:
        for ax in (ax1, ax2, ax3):
            for i in range(0,len(reflection)):
                if reflection[i]>0:
                    ax.plot([i,i],ylims[ax], color= 'lightgray')
        fig.legend(['Activation scaled spectral method'], loc='upper center', 
                       ncol=9, bbox_to_anchor=(0.5, 0))
    
    # Plot correlation paths
    for col1 in range(d):
        for col2 in range(col1+1,d):
            
            if long_term_plot:
                key = f'bar(ρ({col1 + 1},{col2 + 1}))'
                ax1.plot(C[key], color=colors[color_counter], 
                         linestyle = (0, (5,5)), alpha = 0.75)
            
            
            key = f'ρ({col1 + 1},{col2 + 1})'
            ax1.plot(C[key], color='white', linewidth=3)
            ax1.plot(C[key], label= f'$\\rho_{{{col1 + 1},{col2 + 1}}}$', color=colors[color_counter])
            
            color_counter += 1
    ax1.legend(loc = 'lower left')
    
    color_counter = 0  # Dummy variable for coloring
    
    # Plot eigenvalue paths
    for col1 in range(d):
        
        if long_term_plot:
            key = f'bar(λ)_{col1+1}'
            ax2.plot(C[key], color=colors[color_counter], 
                     linestyle = (0, (5,5)), alpha = 0.75)
        
        key = f'λ({col1+1})'
        ax2.plot(C[key], color='white', linewidth=3)
        ax2.plot(C[key], color=colors[color_counter], linewidth=1,
                 label= f'$\\lambda_{{{col1 + 1}}}$' )
        color_counter += 1
    ax2.legend(loc = 'upper left')
    
    # Plot determinant path
    if long_term_plot:
        key = 'bar(determinant)'
        ax3.plot(C[key], color='black', 
                 linestyle = (0, (5,5)), alpha = 0.75)
    
    
    ax3.plot(C['determinant'], color='white', linewidth=3)
    ax3.plot(C['determinant'], color='black', linewidth=1, label="Determinant")
    

    #plot marked periods
    if len(years) > 0:
        
        cmap = plt.get_cmap("magma")
        norm = plt.Normalize(0, len(years))
        rect_colors = cmap(norm(range(len(years))))
        rect_colors[:, 3] = rect_colors[:, 3] * 0.25
        
        # Initialize lists to store handles and labels for the rectangles
        rect_handles = []
        rect_labels = []
        
        # Standard visualization tasks for ax2 and ax3
        for ax in (ax1, ax2, ax3):
            counter = 0
            for year in years:
                bottom = ylims[ax][0]
                height = ylims[ax][1] - ylims[ax][0]
                
                left_date   = year[0]
                right_date  = year[1]
                
                # Create the rectangle using the bottom-left corner (left_date, bottom) and width, height
                # Note: Using `datetime.timedelta` to handle the width
                width_rect = (right_date - left_date).days
                square = patches.Rectangle((left_date, bottom), datetime.timedelta(days=width_rect), height, 
                                           linewidth=1, facecolor=rect_colors[counter], 
                                           label=year[2])
                
                # Add the rectangle to the plot
                ax.add_patch(square)
                
                # Store the handle and label
                if year[2] not in rect_labels:
                    rect_handles.append(square)
                    rect_labels.append(year[2])
                
                counter += 1
                
            # Create a combined legend for the rectangles
            fig.legend(handles=rect_handles, labels=rect_labels, loc='upper center', 
                           ncol=9, bbox_to_anchor=(0.5, 0))
                
    for ax in (ax1, ax2, ax3):
        ax.set_title(titles[ax])
        ax.set_xlabel('time')
        ax.grid(True)
        ax.set_xlim([C.index[0], C.index[-1]])
        ax.set_ylim(ylims[ax])
        
    plt.tight_layout()
    
    fig.savefig(f"simulation_{process}.png", bbox_inches='tight')
    fig.savefig(f"simulation_{process}.pdf", bbox_inches='tight')
    
    plt.show()
    
def plot_with_rectangles(fig, ax, df, years, ylims, ncol=10):
    """
    Plots data with shaded rectangles for given years.
    """
    cmap = plt.get_cmap("magma")
    norm = plt.Normalize(0, len(years))
    rect_colors = cmap(norm(range(len(years))))
    rect_colors[:, 3] = rect_colors[:, 3] * 0.25
    
    rect_handles = []
    rect_labels = []
    
    for counter, year in enumerate(years):
        bottom = ylims[0]
        height = ylims[1] - ylims[0]
        
        left_date = year[0]
        right_date = year[1]
        
        width = (right_date - left_date).days
        square = patches.Rectangle((left_date, bottom), datetime.timedelta(days=width), height,
                                   linewidth=1, facecolor=rect_colors[counter], label=year[2])
        
        ax.add_patch(square)
        
        if year[2] not in rect_labels:
            rect_handles.append(square)
            rect_labels.append(year[2])
    
    if len(years) > 0:
        fig.legend(handles=rect_handles, labels=rect_labels, 
                   loc='upper center', ncol = ncol, bbox_to_anchor=(0.5, 0))

def setup_axis(ax, title, log, xlims, ylims):
    """
    Sets up the axis with title, x and y limits, and optionally a logarithmic scale.
    """
    ax.set_title(title)
    ax.set_xlabel('time (years)')
    ax.set_xlim(xlims)
    
    if log:
        ax.set_yscale('log')
        ax.set_ylim(ylims[1])
    else:
        ax.set_ylim(ylims[0])
    ax.grid(True)

def kAC_dashboard(df, width, history, means_df=None, years=[], 
                       log=False, ncol=10,
                       actual_df = None):
    """
    Generate a the kappa A figures.
    
    Parameters:
    df (df): df of kappa_1, ... , a_1, ...
    width (string): window width used for correlation estimation.
    means_df (binary): 
    Years (list): list containing periods that need to be marked.
    log (binary): do we want log y-axis?
    ncol (int): nr of columns used for legend visualisation.
    actual_df (df): df containing real kappa and a values each period. If no 
                    actual data available, then use "None".
    """    
    
    sns.set_style("whitegrid")
    
    # Determine number of κ and γ plots
    d = len([col for col in df.columns if col.startswith('κ(')])
    
    # Setup figures and axes for κ, γ, and C_bar plots
    fig1, axes1 = plt.subplots(d, 2, figsize=(10, d * 2))
    plt.suptitle(f'Daily estimates of $κ$ and $A$ ($w = {width}, h = {history}$)', fontsize=16)
    
    # Plot κ and γ
    for col1 in range(d):
        ax_kappa, ax_a = axes1[col1]
        
        # Plot κ
        ax_kappa.plot(df[f'κ({col1+1})'], color='white', linewidth=3)
        ax_kappa.plot(df[f'κ({col1+1})'], label=f'$\\kappa_{{{col1 + 1}}}$', color='black', linewidth=1)
        
        # Plot γ
        ax_a.plot(df[f'γ({col1+1})'], color='white', linewidth=3)
        ax_a.plot(df[f'γ({col1+1})'], label=f'$a_{{{col1 + 1}}}$', color='black', linewidth=1)
        
        # Define limits
        xlims = [df.index[0], df.index[-1]]
        ylims_st = {
            ax_kappa: [-.01, 0.2],
            ax_a: [-.01, 0.2]
        }
        ylims_log = {
            ax_kappa: [1e-8, np.max(df[f'κ({col1+1})'])],
            ax_a: [1e-8, np.max(df[f'γ({col1+1})'])]
        }
        
        ylims = ylims_log if log else ylims_st
        
        ax_kappa.set_ylim(ylims[ax_kappa])
        ax_a.set_ylim(ylims[ax_a])
        
        # Add horizontal lines and rectangles
        if means_df is not None:
            for period_index, year in enumerate(years):
                start_date, end_date, _ = year
                ax_kappa.hlines(means_df[f'κ({col1+1})'].iloc[period_index], start_date, end_date, colors='red', linestyles='--', linewidth=1)
                ax_a.hlines(means_df[f'γ({col1+1})'].iloc[period_index], start_date, end_date, colors='red', linestyles='--', linewidth=1)
                
                if actual_df is not None:
                    ax_kappa.hlines(actual_df[f'κ({col1+1})'].iloc[period_index], start_date, end_date, colors='blue', linestyles='--', linewidth=1)
                    ax_a.hlines(actual_df[f'γ({col1+1})'].iloc[period_index], start_date, end_date, colors='blue', linestyles='--', linewidth=1)
            
                
        # Setup axes and plot rectangles
        setup_axis(ax_kappa, f'$\\kappa_{{{col1 + 1}}}$', log, xlims, ylims[ax_kappa])
        setup_axis(ax_a, f'$a_{{{col1 + 1}}}$', log, xlims, ylims[ax_a])
        
        plot_with_rectangles(fig1,ax_kappa, df, years, ylims[ax_kappa], ncol)
        plot_with_rectangles(fig1,ax_a, df, years, ylims[ax_a], ncol)
    
    plt.tight_layout()
    plt.show()
    
    # Plot C_bar and execution times
    kappa_columns = [col for col in df.columns if col.startswith('κ(')]
    d = len(kappa_columns)
    
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    
    # Plot bar_ρ
    cmap = plt.get_cmap("tab20c")
    norm = plt.Normalize(0, d * (d - 1) // 2)
    colors = cmap(norm(range(d * (d - 1) // 2)))
    
    color_counter = 0
    for col1 in range(d):
        for col2 in range(col1 + 1, d):
            key = f'bar_ρ({col1 + 1},{col2 + 1})'
            ax1.plot(df[key], color='white', linewidth=3)
            ax1.plot(df[key], 
                     label = f'$\\bar{{C}}_{{{col1 + 1},{col2 + 1}}}$', 
                     color=colors[color_counter])
            color_counter += 1
    
    ax1.set_title(f'Estimates of $\\bar{{C}}$ each period ($w = {width}, h = {history}$)')
    ax1.set_xlabel('time (years)')
    ax1.set_xlim([df.index[0], df.index[-1]])
    ax1.set_ylim([-1, 1])
    ax1.grid(True)
    ax1.legend(loc='lower left', ncol = 1)
    
    # Plot execution times
    ax2.plot(df['execution_times'], color='white', linewidth=3)
    ax2.plot(df['execution_times'], color='black', linewidth=0.5)
    ax2.set_title(f'Execution times of the estimation procedure ($w = {width}, h = {history}$)')
    ax2.set_xlabel('time (years)')
    ax2.set_ylabel('time (s)')
    ax2.set_xlim([df.index[0], df.index[-1]])
    ax2.set_ylim([0, np.max(df['execution_times'])])
    ax2.grid(True)
    
    # Add rectangles for both plots
    plot_with_rectangles(fig2, ax1, df, years, [-1, 1], ncol)
    plot_with_rectangles(fig2, ax2, df, years, 
                         [0, np.max(df['execution_times'])], ncol)
    
    plt.tight_layout()
    
    fig1.savefig(f"kA_w{width}_h{history}.png", bbox_inches='tight')
    fig1.savefig(f"kA_w{width}_h{history}.pdf", bbox_inches='tight')
    fig2.savefig(f"C_bar_exctime_w{width}_h{history}.png", bbox_inches='tight')
    fig2.savefig(f"C_bar_exctime_w{width}_h{history}.pdf", bbox_inches='tight')
    
    plt.show()
    

def corr_dashboard_ax1(C, d, long_term_plot = True, 
                   process="MRC process",  years = [],
                   reflection_visual = False, reflection=[]):
    """
    Generate a the correlation figures (excluding eigenvalues and determinant-
                                        plots).
    
    Parameters:
    C (df): containing rho, lambda and determinant paths
    d (int): the dimension of the correlation matrix.
    long_term_plot (binary): include long-term mean plot?
    process (string): name of the process.
    Years (list): list containing periods that need to be marked.
    reflection_visual (binary): visualise places where reflection is used.
    reflection (list): list with time points where reflection is used.
    """
    
    sns.set_style("whitegrid")
    nr_of_unique_elements = int( d * (d-1) / 2 )
    
    # Choosing colors for the plots
    cmap = plt.get_cmap("tab20c")
    norm = plt.Normalize(0, np.max([nr_of_unique_elements, d]))
    colors = cmap(norm(range(np.max([nr_of_unique_elements, d]))))
    
    # Initializing the plot
    fig, ax = plt.subplots(figsize=(10,6))
    
    titles = {
        ax: f'Path of the off-diagonal elements of the {process}'}
    
    # Example dummy y-limits
    ylims = {ax: [-1, 1]}  # adjust as necessary
    
    color_counter = 0  # Dummy variable for coloring
    if reflection_visual:

        for i in range(0,len(reflection)):
            if reflection[i]>0:
                ax.plot([i,i],ylims[ax], color= 'lightgray')
        fig.legend(['Activation scaled spectral method'], loc='upper center', 
                   ncol=9, bbox_to_anchor=(0.5, 0))
    
    # Plot correlation paths
    for col1 in range(d):
        for col2 in range(col1+1,d):
            
            if long_term_plot:
                key = f'bar(ρ({col1 + 1},{col2 + 1}))'
                ax.plot(C[key], color=colors[color_counter], 
                         linestyle = (0, (5,5)), alpha = 0.75)
            
            
            key = f'ρ({col1 + 1},{col2 + 1})'
            ax.plot(C[key], color='white', linewidth=3)
            ax.plot(C[key], label= f'$\\rho_{{{col1 + 1},{col2 + 1}}}$', color=colors[color_counter])
            
            color_counter += 1
    ax.legend(loc = 'lower left')
    
    color_counter = 0  # Dummy variable for coloring
    
    if len(years) > 0:
        
        cmap = plt.get_cmap("magma")
        norm = plt.Normalize(0, len(years))
        rect_colors = cmap(norm(range(len(years))))
        rect_colors[:, 3] = rect_colors[:, 3] * 0.25
        
        # Initialize lists to store handles and labels for the rectangles
        rect_handles = []
        rect_labels = []
        
        # Standard visualization tasks for ax2 and ax3

        counter = 0
        for year in years:
            bottom = ylims[ax][0]
            height = ylims[ax][1] - ylims[ax][0]
            
            left_date   = year[0]
            right_date  = year[1]
            
            # Create the rectangle using the bottom-left corner (left_date, bottom) and width, height
            # Note: Using `datetime.timedelta` to handle the width
            width_rect = (right_date - left_date).days
            square = patches.Rectangle((left_date, bottom), datetime.timedelta(days=width_rect), height, 
                                       linewidth=1, facecolor=rect_colors[counter], 
                                       label=year[2])
            
            # Add the rectangle to the plot
            ax.add_patch(square)
            
            # Store the handle and label
            if year[2] not in rect_labels:
                rect_handles.append(square)
                rect_labels.append(year[2])
            
            counter += 1
            
        # Create a combined legend for the rectangles
        fig.legend(handles=rect_handles, labels=rect_labels, loc='upper center', 
                       ncol=7, bbox_to_anchor=(0.5, 0))
                

    ax.set_title(titles[ax])
    ax.set_xlabel('time')
    ax.grid(True)
    ax.set_xlim([C.index[0], C.index[-1]])
    ax.set_ylim(ylims[ax])
        
    plt.tight_layout()
    
    fig.savefig(f"simulation_{process}_ax1.png", bbox_inches='tight')
    fig.savefig(f"simulation_{process}_ax1.pdf", bbox_inches='tight')
    
    plt.show()

def TD_plot(C, Cbar, TD, TDbar, nu, d, title="MRC process"):
    
    """
    Generate a the correlation and TD figures.
    
    Parameters:
    C (df): df containing rho
    Cbar (matrix array): \bar{C}
    TD (df): df of tail dependence
    TDbar (matrix array): tail dependence of \bar{C}
    nu (value): degrees of freedom
    d (int): the dimension of the correlation matrix.
    title (string): name of the process.
    """
    
    sns.set_style("whitegrid")
    nr_of_unique_elements = int(d * (d - 1) / 2)
    
    # Choosing colors for the plots
    cmap = plt.get_cmap("tab20c")
    norm = plt.Normalize(0, np.max([nr_of_unique_elements, d]))
    colors = cmap(norm(range(np.max([nr_of_unique_elements, d]))))
    
    # Initializing the plot
    fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(2, 12))
    
    titles = {ax3: 'PDF: MRC', ax4: f'PDF: $\\lambda^L$, $\\nu = {nu}$'}
    
    # Example dummy y-limits
    ylims = {ax3: [-1, 1],
             ax4: [0, 1]}
    
    color_counter = 0  # Dummy variable for coloring
    
    # Plot correlation paths
    for col1 in range(d):
        for col2 in range(col1 + 1, d):
            if (col1,col2) == (0,2) or (col1,col2) == (1,3):
                ax3.hist(C[:, col1, col2], bins=20, alpha=1, density=True,
                         color=colors[color_counter],
                         orientation='horizontal',
                         label= f'$\\rho_{{{col1 + 1},{col2 + 1}}}$')
                
                ax4.hist(TD[:, col1, col2], bins=20, alpha=0.8, density=True,
                         color=colors[color_counter],
                         orientation='horizontal',
                         label= f'$\\lambda^L_{{{col1 + 1},{col2 + 1}}}$')
                                
                if col1 == d - 2 and col2 == d - 1:
                    ax3.axhline(Cbar[col1, col2], color=colors[color_counter], linestyle='dashed', linewidth=1.5,
                                label='$\\bar{C}$-values')
                    ax4.axhline(TDbar[col1, col2], color=colors[color_counter], linestyle='dashed', linewidth=1.5,
                                label='$\\lambda^L$ for $\\bar{C}$')
                    
                else:
                    ax3.axhline(Cbar[col1, col2], color=colors[color_counter], linestyle='dashed', linewidth=1.5)
                    ax4.axhline(TDbar[col1, col2], color=colors[color_counter], linestyle='dashed', linewidth=1.5)
                    
            color_counter += 1
    
    for ax in [ax3, ax4]:

        ax.set_title(titles[ax])
        ax.grid(True)
        ax.set_ylim(ylims[ax])
        ax.set_xlim([0,10])
        
        # remove y-axis
        ax.set_yticks([])
        ax.set_ylabel('')
        
        # ax.legend()
        
    plt.tight_layout()
    fig.savefig(f"simulation_{title}.png", bbox_inches='tight')
    fig.savefig(f"simulation_{title}.pdf", bbox_inches='tight')
    plt.show()