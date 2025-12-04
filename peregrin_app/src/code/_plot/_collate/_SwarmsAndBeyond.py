import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from itertools import chain
from .._common import Colors


@staticmethod
def SwarmsAndBeyond(
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
    **kwargs
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
    noticequeue = kwargs.get('noticequeue') if 'noticequeue' in kwargs else lambda: None

    if df is None or df.empty:
        return

    plt.figure(figsize=(plot_width, plot_height))
    ax = plt.gca()

     

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
        _palette = Colors.BuildQualPalette(_df, tag='Replicate', noticequeue=noticequeue)

    print(_palette)

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

