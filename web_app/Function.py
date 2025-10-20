import numpy as np
import pandas as pd
from math import floor, ceil
import os.path as op
from typing import List, Any
from scipy.stats import gaussian_kde, norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import seaborn as sns
from itertools import chain
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pandas.api.types import is_object_dtype


def _get_cmap(c_mode):
    """
    Get a colormap according to the selected color mode.

    """

    if c_mode == 'greyscale LUT':
        return plt.cm.gist_gray
    elif c_mode == 'reverse grayscale LUT':
        return plt.cm.gist_yarg
    elif c_mode == 'jet LUT':
        return plt.cm.jet
    elif c_mode == 'brg LUT':
        return plt.cm.brg
    elif c_mode == 'cool LUT':
        return plt.cm.cool
    elif c_mode == 'hot LUT':
        return plt.cm.hot
    elif c_mode == 'inferno LUT':
        return plt.cm.inferno
    elif c_mode == 'plasma LUT':
        return plt.cm.plasma
    elif c_mode == 'CMR-map LUT':
        return plt.cm.CMRmap
    elif c_mode == 'gist-stern LUT':
        return plt.cm.gist_stern
    elif c_mode == 'gnuplot LUT':
        return plt.cm.gnuplot
    elif c_mode == 'viridis LUT':
        return plt.cm.viridis
    elif c_mode == 'cividis LUT':
        return plt.cm.cividis
    elif c_mode == 'rainbow LUT':
        return plt.cm.rainbow
    elif c_mode == 'turbo LUT':
        return plt.cm.turbo
    elif c_mode == 'nipy-spectral LUT':
        return plt.cm.nipy_spectral
    elif c_mode == 'gist-ncar LUT':
        return plt.cm.gist_ncar
    elif c_mode == 'twilight LUT':
        return plt.cm.twilight
    elif c_mode == 'seismic LUT':
        return plt.cm.seismic
    else:
        return None







def SetUnits(t: str) -> dict:
    return {
        "Track length": "(µm)",
        "Track displacement": "(µm)",
        "Confinement ratio": "",
        "Track points": "",
        "Speed mean": f"(µm·{t}⁻¹)",
        "Speed median": f"(µm·{t}⁻¹)",
        "Speed max": f"(µm·{t}⁻¹)",
        "Speed min": f"(µm·{t}⁻¹)",
        "Speed std": f"(µm·{t}⁻¹)",
        "Direction mean (deg)": "",
        "Direction mean (rad)": "",
        "Direction std (deg)": "",
        "Direction std (rad)": "",
    }




class Process:

    @staticmethod
    def TryConvertNumeric(x: Any) -> Any:
        """
        Try to convert a string to an int or float, otherwise return the original value.
        """
        try:
            if isinstance(x, str):
                x_stripped = x.strip()
                num = float(x_stripped)
                if num.is_integer():
                    return int(num)
                else:
                    return num
            else:
                return x
        except ValueError:
            return x
        
    @staticmethod
    

    @staticmethod
    def MergeDFs(dataframes: List[pd.DataFrame], on: List[str]) -> pd.DataFrame:
        """
        Merges a list of DataFrames on the specified columns using an outer join.
        All values are coerced to string before merging, then converted back to numeric where possible.

        Parameters:
            dataframes: List of DataFrames to merge.
            on: List of column names to merge on.

        Returns:
            pd.DataFrame: The merged DataFrame with numerics restored where possible.
        """
        if not dataframes:
            raise ValueError("No dataframes provided for merging.")

        # Initialize the first DataFrame as the base for merging (all values as string)
        merged_df = dataframes[0].map(str)

        for df in dataframes[1:]:
            df = df.map(str)
            # Ensure all key columns are present for merging
            merge_columns = [col for col in df.columns if col not in merged_df.columns or col in on]
            merged_df = pd.merge(
                merged_df,
                df[merge_columns],
                on=on,
                how='outer'
            )

        # Use the static method for numeric conversion
        merged_df = merged_df.applymap(Process.TryConvertNumeric)
        return merged_df


    def Round(value, step, round_method="nearest"):
        """
        Rounds value to the nearest multiple of step.
        """
        if round_method == "nearest":
            return round(value)
        elif round_method == "floor":
            return floor(value)
        elif round_method == "ceil":
            return ceil(value)
        else:
            raise ValueError(f"Unknown round method: {round_method}")



class Threshold:

    @staticmethod
    def Normalize_01(df, col) -> pd.Series:
        """
        Normalize a column to the [0, 1] range.
        """
        # s = pd.to_numeric(df[col], errors='coerce')
        try:
            s = pd.Series(Process.TryFloat(df[col]), dtype=float)
            if _has_strings(s):
                normalized = pd.Series(0.0, index=s.index, name=col)
            lo, hi = s.min(), s.max()
            if lo == hi:
                normalized = pd.Series(0.0, index=s.index, name=col)
            else:
                normalized = pd.Series((s - lo) / (hi - lo), index=s.index, name=col)

        except Exception:
            normalized = pd.Series(0.0, index=df.index, name=col)

        return normalized  # <-- keeps index

    @staticmethod
    def JoinByIndex(a: pd.Series, b: pd.Series) -> pd.DataFrame:
        """
        Join two Series of potentially different lengths into a DataFrame.
        """

        if b.index.is_unique and not a.index.is_unique:
            df = a.rename(a.name).to_frame().set_index(a.index)
            df[b.name] = b.reindex(df.index)
        else:
            df = b.rename(b.name).to_frame().set_index(b.index)
            df[a.name] = a.reindex(df.index)

        return df

    



class Plot:

    
    
        class SuperviolinPlot:
            def __init__(self, metric="value", filename="", data_format="tidy", units: dict = None,
                        centre_val="mean", middle_vals="mean", error_bars="SEM",
                        paired_data="no", stats_on_plot="no",
                        total_width=0.8, outline_lw=1, dataframe=False,
                        sep_lw=0, bullet_lw=0.75, errorbar_lw=1, bullet_size=50, palette="Accent", use_my_colors=True,
                        bw="None", show_legend=True, plot_width=12, plot_height=5):
                self.errors = []
                self.df = dataframe
                self.x = "Condition"
                self.y = metric
                self.rep = "Replicate"
                self.units = units
                self.use_my_colors = use_my_colors
                self.palette = palette
                self.outline_lw = outline_lw
                self.sep_linewidth = sep_lw
                self.bullet_linewidth = bullet_lw
                self.errorbar_linewidth = errorbar_lw
                self.bullet_size = bullet_size
                self.middle_vals = middle_vals
                self.centre_val = centre_val
                self.paired = paired_data
                self.stats_on_plot = stats_on_plot
                self.total_width = total_width
                self.error_bars = error_bars
                self.show_legend = show_legend
                self._on_legend = []
                self.plot_width = plot_width
                self.plot_height = plot_height

                if bw != "None":
                    self.bw = bw
                else:
                    self.bw = None
                    
                qualitative = ["Pastel1", "Pastel2", "Paired", "Accent", "Dark2",
                            "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b",
                            "tab20c"]
                
                # ensure dataframe is loaded
                if self._check_df(filename, data_format):
                    
                    # ensure columns are all present in the dataframe
                    if self._cols_in_df():
                        
                        # force Condition and Replicate to string types
                        self.df[self.x] = self.df[self.x].astype(str)
                        self.df[self.rep] = self.df[self.rep].astype(str)
                        
                        # organize subgroups
                        self.subgroups = tuple(sorted(self.df[self.x].unique().tolist()))
                        
                        # dictionary of arrays for subgroup data
                        # loop through the keys and add an empty list
                        # when the replicate numbers don"t match
                        self.subgroup_dict = dict(
                            zip(self.subgroups,
                                [{"norm_wy" : [], "px" : []} for i in self.subgroups])
                            )
            
                        self.unique_reps = tuple(self.df[self.rep].unique())
                        
                        # make sure there"s enough colours for 
                        # each subgroup when instantiating
                        try:    
                            if self.use_my_colors:
                                cmap = _build_replicate_palette(self.df, palette_fallback=self.palette)
                                self.cmap = cmap
                                self.colours = [cmap[rep] for rep in self.unique_reps]
                            else:
                                self.cm = plt.get_cmap(self.palette)
                                self.colours = [self.cm(i / self.cm.N) for i in range(len(self.unique_reps))]
                        except Exception as e:
                            print(e)
                            print("Random colours were created for each unique replicate.")
                            # colours = {rep: _generate_random_color() for rep in self.unique_reps}
                            colours = {rep: _generate_random_color() for rep in self.unique_reps}
                            self.colours = [colours[rep] for rep in self.unique_reps]


                self.check_errors()
            
            def check_errors(self):
                num_errors = len(self.errors)
                if num_errors == 1:
                    print("Caught 1 error")
                    return True
                elif num_errors > 1:
                    print(f"Caught {len(self.errors)} errors")
                    for i,e in enumerate(self.errors, 1):
                        print(f"\t{i}. {e}")
                    return True
                else:
                    return False
                
                            
            def generate_plot(self):
                """
                Generate Violin SuperPlot by calling get_kde_data, plot_subgroups,
                and get_statistics if the errors list attribute is empty.

                Returns
                -------
                None.

                """
                # if no errors exist, create the superplot. Otherwise, report errors
                errors = self.check_errors()
                if not errors:
                    self.get_kde_data(self.bw)
                    self.plot_subgroups(self.centre_val, self.middle_vals,
                                        self.error_bars,
                                        self.total_width, self.outline_lw,
                                        self.stats_on_plot, self.show_legend)
                
            def _check_df(self, filename, data_format):
                """
                Read dataframe if file extension is valid. 

                Parameters
                ----------
                filename : string
                    name of the file containing the data for the Violin SuperPlot.
                    Must be CSV or an Excel workbook
                data_format : string
                    either "tidy" or "untidy" based on format of the data

                Returns
                -------
                bool
                    True, if a Pandas DataFrame object was created using the filename,
                    else False

                """
                if "bool" in str(type(self.df)):
                    if filename.endswith("csv"):
                        self.df = pd.read_csv(filename)
                        return True
                    elif ".xl" in filename:
                        if data_format == "tidy":
                            self.df = pd.read_excel(filename)
                        else:
                            self._make_tidy(filename)
                        return True
                    else:
                        self.errors.append("Incorrect filename or unsupported filetype")
                        return False
                else:
                    return True
            
            def _cols_in_df(self):
                """
                Check if all column names specified by the user are present in the 
                self.df DataFrame

                Returns
                -------
                bool
                    True if all supplied column names are present in the df attribute

                """
                cols = [self.x, self.y, self.rep]
                missing_cols = [col for col in cols if col not in self.df.columns]
                if len(missing_cols) != 0:
                    if len(missing_cols) == 1:
                        self.errors.append("Variable not found: " + missing_cols[0])
                    else:
                        add_on = "Missing variables: " + ", ".join(missing_cols)
                        self.errors.append(add_on)
                    return False
                else:
                    return True

            def get_kde_data(self, bw=None):
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
                for group in self.subgroups:
                    px = []
                    norm_wy = []
                    min_cuts = []
                    max_cuts = []
                    
                    # get limits for fitting the kde 
                    for rep in self.unique_reps:
                        sub = self.df[(self.df[self.rep] == rep) &
                                    (self.df[self.x] == group)][self.y]
                        min_cuts.append(sub.min())
                        max_cuts.append(sub.max())
                    min_cuts = sorted(min_cuts)
                    max_cuts = sorted(max_cuts)
                    
                    # make linespace of points from highest_min_cut to lowest_max_cut
                    points1 = list(np.linspace(np.nanmin(min_cuts),
                                            np.nanmax(max_cuts),
                                            num = 128))
                    points = sorted(list(set(min_cuts + points1 + max_cuts)))
                    
                    for rep in self.unique_reps:
                        
                        # first point to catch an empty list
                        # caused by uneven rep numbers
                        try:
                            sub = self.df[(self.df[self.rep] == rep) &
                                        (self.df[self.x] == group)][self.y]
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
                            kde_wo_nan = self._interpolate_nan(kde_points)
                            points_wo_nan = self._interpolate_nan(points)
                            
                            # normalize each stripe's area to a constant
                            # so that they all have the same area when plotted
                            area = 30
                            kde_wo_nan = (area / sum(kde_wo_nan)) * kde_wo_nan
                            
                            norm_wy.append(kde_wo_nan)
                            px.append(points_wo_nan)
                        except ValueError:
                            norm_wy.append([])
                            px.append(self._interpolate_nan(points))
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
                    self.subgroup_dict[group]["norm_wy"] = norm_wy
                    self.subgroup_dict[group]["px"] = px
            
            @staticmethod
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
            
            def _single_subgroup_plot(self, group, axis_point, mid_df,
                                    total_width, linewidth):
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
                
                norm_wy = self.subgroup_dict[group]["norm_wy"]
                px = self.subgroup_dict[group]["px"]
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
                
                for i,a in enumerate(self.unique_reps):
                    reshaped_x = np.append(px[i-1], np.flipud(px[i-1]))
                    mid_val = mid_df[mid_df[self.rep] == a][self.y].values
                    reshaped_y = new_wy[i] * total_width  + axis_point
                    
                    # check if _on_legend contains the rep
                    if a in self._on_legend:
                        lbl = ""
                    else:
                        lbl = a
                        self._on_legend.append(a)
                    
                    # plot separating lines and stripes
                    plt.plot(reshaped_y, reshaped_x, c="k",
                            linewidth=self.sep_linewidth)
                    plt.fill(reshaped_y, reshaped_x, color=self.colours[i],
                            label=lbl, linewidth=self.sep_linewidth)
                    
                    # get the mid_val each replicate and find it in reshaped_x
                    # then get corresponding point in reshaped_x to plot the points
                    if mid_val.size > 0: # account for empty mid_val
                        arr = reshaped_x[np.logical_not(np.isnan(reshaped_x))]
                        nearest = self._find_nearest(arr, mid_val[0])
                        
                        # find the indices of nearest in new_wy
                        # there will be two because new_wy is a loop
                        idx = np.where(reshaped_x == nearest)
                        x_vals = reshaped_y[idx]
                        x_val = x_vals[0] + ((x_vals[1] - x_vals[0]) / 2)
                        plt.scatter(x_val, mid_val[0], facecolors=self.colours[i],
                                    edgecolors="Black", linewidth=self.bullet_linewidth,
                                    zorder=10, marker="o", s=self.bullet_size)
                plt.plot(outline_x, outline_y, color="Black", linewidth=self.outline_lw)
                
            def plot_subgroups(self, centre_val="mean", middle_vals="mean", error_bars="SEM",
                            total_width=0.8, linewidth=1,
                            show_stats="no", show_legend=True):
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
                plt.figure(figsize=(self.plot_width, self.plot_height))
                ticks = []
                lbls = []
                
                # width of the bars
                median_width = 0.4
                for i,a in enumerate(self.subgroups):
                    ticks.append(i*2)
                    lbls.append(a)            
                    
                    # calculate the mean/median value for
                    # all replicates of the variable
                    sub = self.df[self.df[self.x] == a]
                    
                    # robust mean calculates mean using data
                    # between the 2.5 and 97.5 percentiles
                    if middle_vals == "robust":
                        
                        # loop through replicates in sub
                        subs = []
                        for rep in sub[self.rep].unique():
                            
                            # drop rows containing NaN values as
                            # they mess up the subsetting
                            s = sub[sub[self.rep] == rep].dropna()
                            lower = np.percentile(s[self.y], 2.5)
                            upper = np.percentile(s[self.y], 97.5)
                            s = s.query(f"{lower} <= {self.y} <= {upper}")
                            subs.append(s)
                        sub = pd.concat(subs)
                        middle_vals = "mean"
                    
                    # calculate mean from remaining data
                    means = sub.groupby(self.rep,
                                        as_index=False).agg({self.y : middle_vals})
                    self._single_subgroup_plot(a, i*2, mid_df=means,
                                            total_width=total_width,
                                            linewidth=self.sep_linewidth)
                    
                    # get mean or median line of the skeleton plot
                    if centre_val == "mean":
                        mid_val = means[self.y].mean()
                    else:
                        mid_val = means[self.y].median()
                    
                    # get error bars for the skeleton plot
                    if error_bars == "SEM":
                        sem = means[self.y].sem()
                        upper = mid_val + sem
                        lower = mid_val - sem
                    elif error_bars == "SD":
                        upper = mid_val + means[self.y].std()
                        lower = mid_val - means[self.y].std()
                    else:
                        lower, upper = norm.interval(0.95, loc=means[self.y].mean(),
                                            scale=means[self.y].std())
                    
                    # plot horizontal lines across the column, centered on the tick
                    plt.plot([i*2 - median_width / 1.5, i*2 + median_width / 1.5],
                                [mid_val, mid_val], lw=self.errorbar_linewidth, color="k",
                                zorder=20)
                    for b in [upper, lower]:
                        plt.plot([i*2 - median_width / 4.5, i*2 + median_width / 4.5],
                                [b, b], lw=self.errorbar_linewidth, color="k", zorder=20)
                    
                    # plot vertical lines connecting the limits
                    plt.plot([i*2, i*2], [lower, upper], lw=self.errorbar_linewidth, color="k",
                            zorder=20)
                
                # add legend
                if show_legend:
                    plt.legend(loc=1, bbox_to_anchor=(1.1,1.1), frameon=True, fontsize=self.plot_height * 1.5, edgecolor='black', fancybox=False, framealpha=0.65)
                    
                
                plt.xticks(ticks, lbls, rotation=345)
                plt.ylabel(f"{self.y} {self.units.get(self.y) if self.units else ''}")
                plt.xlabel("Condition")

                return plt.gcf()
                
            @staticmethod
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


    class Tracks:
        
        