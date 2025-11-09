import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from .._common import Colors

@staticmethod
def Superviolins(
        df: pd.DataFrame,
        metric: str,
        units: str = None,
        use_stock_palette: bool = True,
        palette: str = "Accent",
        centre_val: str = "Mean", # reannotate to cond_vals
        middle_vals: str = "Mean", #reannotate to rep vals
        error_bars: str = "SEM",
        outline_lw: float = 1,
        sep_lw: float = 0,
        errorbar_lw: float = 1,
        bullet_lw: float = 0.75,
        bullet_size: int = 5,
        title: str = None,
        show_legend: bool = True,
        total_width: float = 0.8, # reannotate to violin_width
        plot_height: int = 5,
        plot_width: int = 12,
        bw: bool = None,
):

    x = "Condition"
    y = metric
    rep_col = "Replicate"
    bullet_size = bullet_size*10
    _on_legend = []

        
    qualitative = ["Pastel1", "Pastel2", "Paired", "Accent", "Dark2",
                    "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b",
                    "tab20c"]
    
    cols = [x, y, rep_col]
    missing_cols = [col for col in cols if col not in df.columns]
    if len(missing_cols) != 0:
        if len(missing_cols) == 1:
            raise AttributeError("Variable not found: " + missing_cols[0])
        else:
            add_on = "Missing variables: " + ", ".join(missing_cols)
            raise AttributeError(add_on)


    def get_kde_data(bw=None):
        """
        Fit kernel density estimators to the replicate of each condition,
        generate list of x and y co-ordinates of the histogram,
        stack them, and add to the subgroup_dict attribute

        Parameters
        ----------
        bw : float
            The percent smoothing to apply to the stripe outlines.
            Values should be between 0 and 1. The default value will result
            in an "optimal" value being used to smoothen the stripes. This
            value is calculated using Scott's Rule

        Returns
        -------
        None.

        """
        for group in subgroups:
            px = []
            norm_wy = []
            min_cuts = []
            max_cuts = []
            
            # get limits for fitting the kde 
            for r in unique_reps:
                sub = df[(df[rep_col] == r) & (df[x] == group)][y]
                min_cuts.append(sub.min())
                max_cuts.append(sub.max())
            min_cuts = sorted(min_cuts)
            max_cuts = sorted(max_cuts)
            
            # make linespace of points from highest_min_cut to lowest_max_cut
            points1 = list(np.linspace(np.nanmin(min_cuts),
                                    np.nanmax(max_cuts),
                                    num = 128))
            points = sorted(list(set(min_cuts + points1 + max_cuts)))
            
            for r in unique_reps:
                
                # first point to catch an empty list
                # caused by uneven rep numbers
                try:
                    sub = df[(df[rep_col] == r) & (df[x] == group)][y]
                    arr = np.array(sub)
                    
                    # remove nan or inf values which 
                    # could cause a kde ValueError
                    arr = arr[~(np.isnan(arr))]
                    kde = gaussian_kde(arr, bw_method=bw)
                    kde_points = kde.evaluate(points)
                    # kde_points = kde_points - np.nanmin(kde_points) #why?
                    
                    # use min and max to make the kde_points
                    # outside that dataset = 0
                    idx_min = min_cuts.index(arr.min())
                    idx_max = max_cuts.index(arr.max())
                    if idx_min > 0:
                        for p in range(idx_min):
                            kde_points[p] = 0
                    for idx in range(idx_max - len(max_cuts), 0):
                        if idx_max - len(max_cuts) != -1:
                            kde_points[idx+1] = 0
                    
                    # remove nan prior to combining arrays into dictionary
                    kde_wo_nan = _interpolate_nan(kde_points)
                    points_wo_nan = _interpolate_nan(points)
                    
                    # normalize each stripe's area to a constant
                    # so that they all have the same area when plotted
                    area = 30
                    kde_wo_nan = (area / sum(kde_wo_nan)) * kde_wo_nan
                    
                    norm_wy.append(kde_wo_nan)
                    px.append(points_wo_nan)
                except ValueError:
                    norm_wy.append([])
                    px.append(_interpolate_nan(points))
            px = np.array(px)
            
            # print Scott's factor to the console to help users
            # choose alternative values for bw
            try:
                if bw == None:
                    scott = kde.scotts_factor()
                    print(f"Fitting KDE with Scott's Factor: {scott:.3f}")
            except UnboundLocalError:
                pass
            
            # catch the error when there is an empty list added to the dictionary
            length = max([len(e) for e in norm_wy])
            
            # rescale norm_wy for display purposes
            norm_wy = [a if len(a) > 0 else np.zeros(length) for a in norm_wy]
            norm_wy = np.array(norm_wy)
            norm_wy = np.cumsum(norm_wy, axis = 0)
            try:
                norm_wy = norm_wy / np.nanmax(norm_wy) # [0,1]
            except ValueError:
                print("Failed to normalize y values")
            
            # update the dictionary with the normalized data
            # and corresponding x points
            subgroup_dict[group]["norm_wy"] = norm_wy
            subgroup_dict[group]["px"] = px
    
    def _interpolate_nan(arr):
        """
        Interpolate NaN values in numpy array of x co-ordinates of each fitted
        kernel density estimator to prevent gaps in the stripes of each Violin
        SuperPlot

        Parameters
        ----------
        arr : numpy array
            array of x co-ordinates of a fitted kde

        Returns
        -------
        arr : TYPE
            array of x co-ordinates with interpolation for each NaN value
            if present

        """
        diffs = np.diff(arr, axis=0)
        median_val = np.nanmedian(diffs)
        nan_idx = np.where(np.isnan(arr))
        if nan_idx[0].size != 0:
            arr[nan_idx[0][0]] = arr[nan_idx[0][0] + 1] - median_val
        return arr
    
    def _single_subgroup_plot(group, axis_point, mid_df,
                            total_width, colors=None):
        """
        Plot a Violin SuperPlot for the given condition

        Parameters
        ----------
        group : string
            Categorical condition to be plotted on the x axis
        axis_point : integer
            The point on the x-axis over which the Violin SuperPlot will be
            centred
        mid_df : Pandas DataFrame
            Dataframe containing the middle values of each replicate
        total_width : float
            Half the width of each Violin SuperPlot
        linewidth : float
            Width value for the outlines of each Violin SuperPlot and the 
            summary statistics skeleton plot

        Returns
        -------
        None.

        """
        
        # select scatter size based on number of replicates
        
        norm_wy = subgroup_dict[group]["norm_wy"]
        px = subgroup_dict[group]["px"]
        right_sides = np.array([norm_wy[-1]*-1 + i*2 for i in norm_wy])
        
        # create array of 3 lines which denote the 3 replicates on the plot
        new_wy = []
        for i in range(len(px)):
            if i == 0:
                newline = np.append(norm_wy[-1]*-1,
                                    np.flipud(right_sides[i]))
            else:
                newline = np.append(right_sides[i-1],
                                    np.flipud(right_sides[i]))
            new_wy.append(newline)
        new_wy = np.array(new_wy)
        
        # use last array to plot the outline
        outline_y = np.append(px[-1], np.flipud(px[-1]))
        append_param = np.flipud(norm_wy[-1]) * -1
        outline_x = np.append(norm_wy[-1],
                            append_param)*total_width + axis_point
        
        # Temporary fix; find original source of the
        # bug and correct when time allows
        if outline_x[0] != outline_x[-1]:
            xval = round(outline_x[0], 4)
            yval = outline_y[0]
            outline_x = np.insert(outline_x, 0, xval)
            outline_x = np.insert(outline_x, outline_x.size, xval)
            outline_y = np.insert(outline_y, 0, yval)
            outline_y = np.insert(outline_y, outline_y.size, yval)
        
        for i,a in enumerate(unique_reps):
            reshaped_x = np.append(px[i-1], np.flipud(px[i-1]))
            mid_val = mid_df[mid_df[rep_col] == a][y].values
            reshaped_y = new_wy[i] * total_width  + axis_point
            
            # check if _on_legend contains the rep
            if a in _on_legend:
                lbl = ""
            else:
                lbl = a
                _on_legend.append(a)
            
            # plot separating lines and stripes
            plt.plot(reshaped_y, reshaped_x, c="k",
                    linewidth=sep_lw)
            plt.fill(reshaped_y, reshaped_x, color=colors[i],
                    label=lbl, linewidth=sep_lw)
            
            # get the mid_val each replicate and find it in reshaped_x
            # then get corresponding point in reshaped_x to plot the points
            if mid_val.size > 0: # account for empty mid_val
                rx = pd.to_numeric(np.asarray(reshaped_x).reshape(-1), errors='coerce')
                arr = rx[np.isfinite(rx)]
                if arr.size == 0:
                    continue
                nearest = _find_nearest(arr, float(mid_val[0]))
                
                # find the indices of nearest in new_wy
                # there will be two because new_wy is a loop
                idx = np.where(rx == nearest)
                if idx[0].size < 2:
                    j = int(np.argmin(np.abs(rx - nearest)))
                    idx = (np.array([max(j-1, 0), j]),)
                x_vals = reshaped_y[idx]
                x_val = x_vals[0] + ((x_vals[1] - x_vals[0]) / 2)
                plt.scatter(x_val, mid_val[0], facecolors=colors[i],
                            edgecolors="Black", linewidth=bullet_lw,
                            zorder=10, marker="o", s=bullet_size)
        plt.plot(outline_x, outline_y, color="Black", linewidth=outline_lw)
        
    def plot_subgroups(centre_val="Mean", middle_vals="Mean", error_bars="SEM",
                    total_width=0.8, errorbar_lw=1, show_legend=True, colors=None):
        """
        Plot all subgroups of the df attribute

        Parameters
        ----------
        centre_val : string
            Central measure used for the skeleton plot. Either mean or median
        middle_vals : string
            Central measure of each replicate. Either mean, median, or robust
            mean
        error_bars : string
            Method for displaying error bars in the skeleton plot. Either SEM
            for standard error of the mean, SD for standard deviation,
            or CI for 95% confidence intervals
        total_width : float
            Half the width of each Violin SuperPlot
        linewidth : float
            Width value for the outlines of each Violin SuperPlot and the 
            summary statistics skeleton plot
        show_stats : string
            Either "yes" or "no" to overlay the statistics on the plot
        show_legend : bool
            True to show a legend on the generated plot

        Returns
        -------
        None.

        """
        plt.figure(figsize=(plot_width, plot_height))
        ticks = []
        lbls = []
        
        # width of the bars
        median_width = 0.4
        for i,a in enumerate(subgroups):
            ticks.append(i*2)
            lbls.append(a)            
            
            # calculate the mean/median value for
            # all replicates of the variable
            sub = df[df[x] == a]
            
            # robust mean calculates mean using data
            # between the 2.5 and 97.5 percentiles
            if middle_vals == "Mean":
                middle_vals = "mean"
            if middle_vals == "Median":
                middle_vals = "median"
            if middle_vals == "robust":
                
                # loop through replicates in sub
                subs = []
                for r in sub[rep_col].unique():
                    
                    # drop rows containing NaN values as
                    # they mess up the subsetting
                    s = sub[sub[rep_col] == r].dropna()
                    lower = np.percentile(s[y], 2.5)
                    upper = np.percentile(s[y], 97.5)
                    s = s.query(f"{lower} <= {y} <= {upper}")
                    subs.append(s)
                sub = pd.concat(subs)
                middle_vals = "mean"
            
            # calculate mean from remaining data
            means = sub.groupby(rep_col,
                                as_index=False).agg({y : middle_vals})
            _single_subgroup_plot(a, i*2, mid_df=means,
                                    total_width=total_width, colors=colors)
            
            # get mean or median line of the skeleton plot
            if centre_val == "Mean":
                mid_val = means[y].mean()
            else:
                mid_val = means[y].median()
            
            # get error bars for the skeleton plot
            if error_bars == "SEM":
                sem = means[y].sem()
                upper = mid_val + sem
                lower = mid_val - sem
            elif error_bars == "SD":
                upper = mid_val + means[y].std()
                lower = mid_val - means[y].std()
            else:
                lower, upper = norm.interval(0.95, loc=means[y].mean(),
                                    scale=means[y].std())
            
            # plot horizontal lines across the column, centered on the tick
            plt.plot([i*2 - median_width / 1.5, i*2 + median_width / 1.5],
                        [mid_val, mid_val], lw=errorbar_lw, color="k",
                        zorder=20)
            for b in [upper, lower]:
                plt.plot([i*2 - median_width / 4.5, i*2 + median_width / 4.5],
                        [b, b], lw=errorbar_lw, color="k", zorder=20)
            
            # plot vertical lines connecting the limits
            plt.plot([i*2, i*2], [lower, upper], lw=errorbar_lw, color="k",
                    zorder=20)
        
        # add legend
        if show_legend:
            plt.legend(loc=1, bbox_to_anchor=(1.1,1.1), frameon=True, fontsize=plot_height * 1.5, edgecolor='black', fancybox=False, framealpha=0.65)
        if not show_legend:
            plt.legend().set_visible(False)
        plt.title(title)

        axes = plt.gca() #Getting the current axis

        axes.spines['top'].set_visible(False) #It works
        axes.spines['right'].set_visible(False) #It works

        plt.xticks(ticks, lbls, rotation=345)
        plt.ylabel(f"{metric} {units}")
        plt.xlabel("Condition")

        return plt.gcf()
        
    def _find_nearest(array, value):
        """
        Helper function to find the value of an array nearest to the input
        value argument

        Parameters
        ----------
        array : numpy array
            array of x co-ordinates from the fitted kde
        value : float
            the middle value of 

        Returns
        -------
        float
            nearest value in array to the input value argument

        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]


    # force Condition and Replicate to string types
    df[x] = df[x].astype(str)
    df[rep_col] = df[rep_col].astype(str)
    
    # organize subgroups
    subgroups = tuple(sorted(df[x].unique().tolist()))
    
    # dictionary of arrays for subgroup data
    # loop through the keys and add an empty list
    # when the replicate numbers don"t match
    subgroup_dict = dict(
        zip(subgroups,
            [{"norm_wy" : [], "px" : []} for i in subgroups])
        )

    unique_reps = tuple(df[rep_col].unique())
    
    # make sure there"s enough colors for 
    # each subgroup when instantiating

    if use_stock_palette:
        try:
            if palette not in qualitative: raise AttributeError(f"Palette must be one of {qualitative}")
            cm = plt.get_cmap(palette)
            colors = [cm(i / cm.N) for i in range(len(unique_reps))]
        except Exception as e:
            print(e)
            print("Random colors were created for each unique replicate.")
            # colors = {rep: Colors.GenerateRandomColor() for rep in unique_reps}
            # colors = [colors[rep] for rep in unique_reps]
            colors = {r: Colors.GenerateRandomColor() for r in unique_reps}
            colors = [colors[r] for r in unique_reps]
    else: 
        try:
            try:
                cmap = Colors.noticequeue(df, tag='Replicate')
                cmap = cmap
                # colors = [cmap[rep] for rep in unique_reps]
                colors = [cmap[r] for r in unique_reps]
            except Exception:
                if palette not in qualitative: raise AttributeError(f"Palette must be one of {qualitative}")
                print("Could not find assigned replicate colors -> using stock palette instead.")
                cm = plt.get_cmap(palette)
                colors = [cm(i / cm.N) for i in range(len(unique_reps))]
        except Exception as e:
            print(e)
            print("Random colors were created for each unique replicate.")
            # colors = {rep: Colors.GenerateRandomColor() for rep in unique_reps}
            # colors = [colors[rep] for rep in unique_reps]
            colors = {r: Colors.GenerateRandomColor() for r in unique_reps}
            colors = [colors[r] for r in unique_reps]

    # get kde data for each subgroup
    get_kde_data(bw=bw) 
    # generate the plot
    fig = plot_subgroups(
            centre_val=centre_val,
            middle_vals=middle_vals,
            error_bars=error_bars,
            errorbar_lw=errorbar_lw,
            total_width=total_width,
            show_legend=show_legend,
            colors=colors,
        )

    return fig

