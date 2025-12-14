import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from itertools import chain
from scipy.stats import gaussian_kde, norm
from .._common import Colors



class BeyondSwarms:

        @staticmethod
        def SwarmPlot(
            df: pd.DataFrame,
            metric: str,
            *args,
            title: str = '',
            palette: str = 'tab10',
            use_stock_palette: bool = True,

            show_swarm: bool = True,
            swarm_size: int = 2,
            swarm_outline_color: str = 'black',
            swarm_alpha: float = 0.75,

            show_violin: bool = True,
            violin_fill_color: str = 'whitesmoke',
            violin_edge_color: str = 'lightgrey',
            violin_alpha: float = 0.5,
            violin_outline_width: float = 1,

            show_mean: bool = True,
            mean_span: float = 0.16,
            mean_color: str = 'black',
            mean_ls: str = '-',
            show_median: bool = True,
            median_span: float = 0.12,
            median_color: str = 'black',
            median_ls: str = '--',
            line_width: float = 2,

            show_error_bars: bool = True,
            errorbar_capsize: int = 4,
            errorbar_color: str = 'black',
            errorbar_lw: int = 2,
            errorbar_alpha: float = 0.8,

            show_mean_balls: bool = True,
            mean_ball_size: int = 90,
            mean_ball_outline_color: str = 'black',
            mean_ball_outline_width: float = 0.75,
            mean_ball_alpha: int = 1,
            show_median_balls: bool = False,
            median_ball_size: int = 70,
            median_ball_outline_color: str = 'black',
            median_ball_outline_width: float = 0.75,
            median_ball_alpha: int = 1,

            show_kde: bool = False,
            kde_inset_width: float = 0.5,
            kde_outline: float = 1,
            kde_alpha: float = 0.5,
            kde_fill: bool = False,

            units: str = '',
            show_legend: bool = True,
            show_grid: bool = False,
            open_spine: bool = True,

            plot_width: int = 15,
            plot_height: int = 9,
        ):
            """
            **Swarmplot plotting function.**

            ## Parameters:
                **df**:
                Track DataFrame;
                **metric**:
                Column name of the desired metric;
                **palette**:
                Qualitative color palette differentiating replicates (default: 'tab10');
                **show_swarm**:
                Show individual tracks as swarm points (default: True);
                **swarm_size**:
                Size of the swarm points (default: 5); *Swarm point size is automatically adjusted if the points are overcrowded*;
                **swarm_outline_color**:
                (default: 'black');
                **swarm_alpha**:
                Swarm points transparency (default: 0.5);
                **show_violin**:
                (default: True);
                **violin_fill_color**:
                (default: 'whitesmoke');
                **violin_edge_color**:
                (default: 'lightgrey');
                **violin_alpha**:
                Violins transparency (default: 0.5);
                **violin_outline_width**:
                (default: 1);
                **show_mean**:
                Show condition mean as a line (default: True);
                **mean_span**:
                Span length of the mean line (default: 0.12);
                **mean_color**:
                (default: 'black');
                **mean_ls**:
                Condition mean line style (default: '-');
                **show_median**:
                Show condition median as a line (default: True);
                **median_span**:
                Span length of the median line (default: 0.08);
                **median_color**:
                (default: 'black');
                **median_ls**:
                Condition median line style (default: '--');
                **line_width**:
                Line width of mean and median lines (default: 1);
                **set_main_line**:
                Set whether to show mean or median as a full line, while showing the other as a dashed line (default: 'mean');
                **show_error_bars**:
                Show standard deviation error bars around the mean (default: True);
                **errorbar_capsize**:
                Span length of the errorbar caps (default: 4);
                **errorbar_color**:
                (default: 'black');
                **errorbar_lw**:
                Line width of the error bars (default: 1);
                **errorbar_alpha**:
                Transparency of the error bars (default: 0.5);
                **show_mean_balls**:
                Show replicate means (default: True);
                **mean_ball_size**:
                (default: 5);
                **mean_ball_outline_color**:
                (default: 'black');
                **mean_ball_outline_width**:
                (default: 0.75);
                **mean_ball_alpha**:
                (default: 1);
                **show_median_balls**:
                Show replicate medians (default: False);
                **median_ball_size**:
                (default: 5);
                **median_ball_outline_color**:
                (default: 'black');
                **median_ball_outline_width**:
                (default: 0.75);
                **median_ball_alpha**:
                (default: 1);
                **show_kde**:
                Show inset KDE plotted next to each condition for each replicate (default: False);
                **kde_inset_width**:
                Height of the inset KDE (default: 0.5);
                **kde_outline**:
                Line width of the KDE outline (default: 1);
                **kde_alpha**:
                Transparency of the KDE (default: 0.5);
                **kde_fill**:
                Fill the KDE plots (default: False);
                **show_legend**:
                Show legend (default: True);
                **show_grid**:
                Show grid (default: False);
                **open_spine**:
                Don't show the top and right axes spines (default: True);
            """

            plt.figure(figsize=(plot_width, plot_height))
            ax = plt.gca()

            if df is None or df.empty:
                return 

            _df = df.copy()

            # === condition order & optional spacers (for KDE layout) ===
            conditions = _df['Condition'].unique().tolist()

            if show_kde:
                spaced_conditions = ["spacer_0"] + list(
                    chain.from_iterable(
                        (cond, f"spacer_{i+1}") if i < len(conditions) - 1 else (cond,)
                        for i, cond in enumerate(conditions)
                    )
                )
                # Categorical with unobserved categories retained (order matters for aligning x)
                _df['Condition'] = pd.Categorical(_df['Condition'],
                                                categories=spaced_conditions,
                                                ordered=True)
                categories_for_stats = spaced_conditions
            else:
                _df['Condition'] = pd.Categorical(_df['Condition'],
                                                categories=conditions,
                                                ordered=True)
                categories_for_stats = conditions

            if use_stock_palette:
                cyc = sns.color_palette(palette, n_colors=len(df['Replicate'].unique()))
                _palette = {r: cyc[i] for i, r in enumerate(df['Replicate'].unique())}

            else:
                _palette = Colors.BuildQualPalette(_df, tag='Replicate')

            # === stats (single pass) ===
            # keep all categories (observed=False) so spacers appear with NaNs
            _cond_stats = (
                _df.groupby('Condition', observed=False)[metric]
                .agg(mean='mean', median='median', std='std', count='count')
                .reindex(categories_for_stats)        # align to category order
                .reset_index()
            )
            _rep_stats = (
                _df.groupby(['Condition', 'Replicate'], observed=False)[metric]
                .agg(mean='mean', median='median')
                .reset_index()
            )

            # === base layers (seaborn handles packing/jitter efficiently) ===
            if show_swarm:
                sns.swarmplot(
                    data=_df,
                    x="Condition",
                    y=metric,
                    hue='Replicate',
                    palette=_palette,
                    size=swarm_size,
                    edgecolor=swarm_outline_color,
                    dodge=False,
                    alpha=swarm_alpha,
                    legend=False,
                    zorder=1,
                    ax=ax,
                )

            if show_violin:
                sns.violinplot(
                    data=_df,
                    x='Condition',
                    y=metric,
                    color=violin_fill_color,
                    edgecolor=violin_edge_color if violin_edge_color else None,
                    linewidth=violin_outline_width,   # (seaborn uses 'linewidth')
                    inner=None,
                    gap=0.1,
                    alpha=violin_alpha,
                    zorder=0,
                    ax=ax,
                )

            # === replicate mean/median markers (single scatter each) ===
            if show_mean_balls:
                sns.scatterplot(
                    data=_rep_stats,
                    x='Condition', y='mean',
                    hue='Replicate',
                    palette=_palette,
                    edgecolor=mean_ball_outline_color,
                    s=mean_ball_size,
                    legend=False,
                    alpha=mean_ball_alpha,
                    linewidth=mean_ball_outline_width,
                    zorder=4,
                    ax=ax,
                )

            if show_median_balls:
                sns.scatterplot(
                    data=_rep_stats,
                    x='Condition', y='median',
                    hue='Replicate',
                    palette=_palette,
                    edgecolor=median_ball_outline_color,
                    s=median_ball_size,
                    legend=False,
                    alpha=median_ball_alpha,
                    linewidth=median_ball_outline_width,
                    zorder=4,
                    ax=ax,
                )

            # === vectorized lines & error bars (big win) ===
            # Build x centers as 0..N-1 in category order (seaborn does the same internally)
            n = len(categories_for_stats)
            x_centers = np.arange(n)

            # Mask out spacers so we don't draw stats on them
            is_spacer = _cond_stats['Condition'].astype(str).str.startswith('spacer')
            valid = ~is_spacer

            y_mean = _cond_stats.loc[valid, 'mean'].to_numpy()
            y_median = _cond_stats.loc[valid, 'median'].to_numpy()
            y_std = _cond_stats.loc[valid, 'std'].to_numpy()
            x_valid = x_centers[valid.to_numpy()]

            # Mean & median short spans using vectorized hlines
            if show_mean and y_mean.size:
                xmin = x_valid - mean_span
                xmax = x_valid + mean_span
                ax.hlines(y_mean, xmin, xmax,
                        colors=mean_color, linestyles=mean_ls,
                        linewidths=line_width, zorder=3, label='Mean')

            if show_median and y_median.size:
                xmin_m = x_valid - median_span
                xmax_m = x_valid + median_span
                # If both present, only label once (matplotlib de-dupes identical labels later)
                ax.hlines(y_median, xmin_m, xmax_m,
                        colors=median_color, linestyles=median_ls,
                        linewidths=line_width, zorder=3,
                        label=('Median' if not show_mean else None))

            if show_error_bars and y_mean.size:
                ax.errorbar(
                    x_valid, y_mean, yerr=y_std,
                    fmt='none',
                    color=errorbar_color,
                    alpha=errorbar_alpha,
                    linewidth=errorbar_lw,
                    capsize=errorbar_capsize,
                    zorder=3,
                    label=('Mean ± SD' if (show_mean or show_median) else 'SD')
                )

            # === KDE insets (loop only once per actual condition) ===
            if show_kde:
                # After base layers, use established y-limits to size insets
                y_ax_min, y_ax_max = ax.get_ylim()

                # Iterate over actual conditions (even positions in spaced layout)
                for i, cond in enumerate(conditions):
                    group_df = _df[_df['Condition'] == cond]
                    if group_df.empty:
                        continue

                    # inset geometry in data coords
                    x_pos = 2 * i   # even positions: 0,2,4...
                    offset_x = 0.25
                    inset_height = y_ax_max - (y_ax_max - group_df[metric].max()) + abs(y_ax_min * 2)

                    inset_ax = ax.inset_axes(
                        [x_pos - offset_x, y_ax_min, kde_inset_width, inset_height],
                        transform=ax.transData, zorder=0, clip_on=True
                    )
                    sns.kdeplot(
                        data=group_df, y=metric, hue='Replicate',
                        fill=kde_fill, alpha=kde_alpha, lw=kde_outline,
                        palette=_palette, ax=inset_ax, legend=False,
                        zorder=0, clip=(y_ax_min, y_ax_max),
                    )
                    inset_ax.invert_xaxis()
                    inset_ax.set_xticks([]); inset_ax.set_yticks([])
                    inset_ax.set_xlabel(''); inset_ax.set_ylabel('')
                    sns.despine(ax=inset_ax, left=True, bottom=True, top=True, right=True)

                # Ticks: show only real conditions
                ticks = [i for i, lbl in enumerate(categories_for_stats) if not str(lbl).startswith('spacer')]
                labels = [categories_for_stats[i] for i in ticks]
                plt.xticks(ticks=ticks, labels=labels)
                plt.xlim(-0.75, len(categories_for_stats))

            # === axes cosmetics (unchanged) ===
            plt.title(title)
            plt.xlabel("Condition")
            plt.ylabel(f"{metric} {units}")

            if show_legend:
                handles, labels = [], []

                # Replicate entries
                for r in _df['Replicate'].astype(str).unique().tolist():
                    c = _palette.get(r, 'grey')
                    handles.append(mlines.Line2D([], [], linestyle='None',
                                                marker='o', markersize=8,
                                                markerfacecolor=c,
                                                markeredgecolor='black',
                                                label=(str(r) + " median")))
                    labels.append(str(r) + " median")

                # Stats entries (mirror your original logic)
                if show_mean and not show_error_bars:
                    handles.append(mlines.Line2D([], [], color=mean_color,
                                                linestyle=mean_ls, linewidth=line_width,
                                                label='Mean'))
                    labels.append('Mean')
                elif show_error_bars and not show_mean:
                    handles.append(mlines.Line2D([], [], color=errorbar_color,
                                                linestyle='-', linewidth=errorbar_lw,
                                                marker='_', markersize=10,
                                                label='SD'))
                    labels.append('SD')
                elif show_mean and show_error_bars:
                    handles.append(mlines.Line2D([], [], color=errorbar_color,
                                                linestyle='-', linewidth=errorbar_lw,
                                                marker='_', markersize=10,
                                                label='Mean ± SD'))
                    labels.append('Mean ± SD')

                if show_median:
                    handles.append(mlines.Line2D([], [], color=median_color,
                                                linestyle=median_ls, linewidth=line_width,
                                                label='Median'))
                    labels.append('Median')

                leg = ax.legend(handles, labels, title='Legend',
                                title_fontsize=12, fontsize=10,
                                loc='upper right', bbox_to_anchor=(1.15, 1),
                                frameon=True)
                try:
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1))
                except Exception:
                    pass
            else:
                try:
                    plt.legend().remove()
                except Exception:
                    pass

            sns.despine(top=open_spine, right=open_spine, bottom=False, left=False)
            plt.tick_params(axis='y', which='major', length=7, width=1.5, direction='out', color='black')
            plt.tick_params(axis='x', which='major', length=5, width=1.5, direction='out', color='black', rotation=345)
            if show_grid:
                plt.grid(show_grid, axis='y', color='lightgrey', linewidth=1.5, alpha=0.2)
            else:
                plt.grid(False)

            # Keep your legend move safeguard
            try:
                if plt.gca().get_legend() is not None:
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1), fontsize=plot_height * 1.5)
            except Exception:
                try:
                    ax = plt.gca()
                    if plt.gca().get_legend() is not None:
                        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1), fontsize=plot_height * 1.5)
                except Exception:
                    pass


            return plt.gcf()



class SuperViolins:
    def __init__(self, metric="value", filename="", data_format="tidy", units: str = None,
                 centre_val="Mean", middle_vals="Mean", error_bars="SEM",
                 paired_data="no", stats_on_plot="no",
                 total_width=0.8, outline_lw=1, dataframe=False,
                 sep_lw=0, bullet_lw=0.75, errorbar_lw=1, bullet_size=5, palette="Accent", use_stock_palette=True,
                 bw="None", show_legend=True, plot_width=12, plot_height=5, title=""):
        self.errors = []
        self.df = dataframe
        self.x = "Condition"
        self.metric = metric
        self.y = metric
        self.rep = "Replicate"
        self.units = units
        self.use_stock_palette = use_stock_palette
        self.palette = palette
        self.outline_lw = outline_lw
        self.sep_linewidth = sep_lw
        self.bullet_linewidth = bullet_lw
        self.errorbar_linewidth = errorbar_lw
        self.bullet_size = bullet_size*10
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
        self.title = title

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

                if self.use_stock_palette:
                    try:
                        if self.palette not in qualitative: raise AttributeError(f"Palette must be one of {qualitative}")
                        self.cm = plt.get_cmap(self.palette)
                        self.colours = [self.cm(i / self.cm.N) for i in range(len(self.unique_reps))]
                    except Exception as e:
                        print(e)
                        print("Random colours were created for each unique replicate.")
                        colours = {rep: Colors.GenerateRandomColor() for rep in self.unique_reps}
                        self.colours = [colours[rep] for rep in self.unique_reps]
                else: 
                    try:
                        try:
                            cmap = Colors.BuildQualPalette(self.df, tag='Replicate')
                            self.cmap = cmap
                            self.colours = [cmap[rep] for rep in self.unique_reps]
                        except Exception:
                            if self.palette not in qualitative: raise AttributeError(f"Palette must be one of {qualitative}")
                            print("Could not find assigned replicate colours -> using stock palette instead.")
                            self.cm = plt.get_cmap(self.palette)
                            self.colours = [self.cm(i / self.cm.N) for i in range(len(self.unique_reps))]
                    except Exception as e:
                        print(e)
                        print("Random colours were created for each unique replicate.")
                        colours = {rep: Colors.GenerateRandomColor() for rep in self.unique_reps}
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
                rx = pd.to_numeric(np.asarray(reshaped_x).reshape(-1), errors='coerce')
                arr = rx[np.isfinite(rx)]
                if arr.size == 0:
                    continue
                nearest = self._find_nearest(arr, float(mid_val[0]))
                
                # find the indices of nearest in new_wy
                # there will be two because new_wy is a loop
                idx = np.where(rx == nearest)
                if idx[0].size < 2:
                    j = int(np.argmin(np.abs(rx - nearest)))
                    idx = (np.array([max(j-1, 0), j]),)
                x_vals = reshaped_y[idx]
                x_val = x_vals[0] + ((x_vals[1] - x_vals[0]) / 2)
                plt.scatter(x_val, mid_val[0], facecolors=self.colours[i],
                            edgecolors="Black", linewidth=self.bullet_linewidth,
                            zorder=10, marker="o", s=self.bullet_size)
        plt.plot(outline_x, outline_y, color="Black", linewidth=self.outline_lw)
        
    def plot_subgroups(self, centre_val="Mean", middle_vals="Mean", error_bars="SEM",
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
                middle_vals = "Mean"
            
            # calculate mean from remaining data
            means = sub.groupby(self.rep,
                                as_index=False).agg({self.y : middle_vals})
            self._single_subgroup_plot(a, i*2, mid_df=means,
                                       total_width=total_width,
                                       linewidth=self.sep_linewidth)
            
            # get mean or median line of the skeleton plot
            if centre_val == "Mean":
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
            
        plt.title(self.title)
        
        plt.xticks(ticks, lbls, rotation=345)
        plt.ylabel(self.metric + ' ' + self.units)
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





def funcSuperViolins(
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

