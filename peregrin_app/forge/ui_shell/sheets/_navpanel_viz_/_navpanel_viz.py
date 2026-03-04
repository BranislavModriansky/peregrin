from shiny import ui

from ._sub_tracks import subpanel_tracks
from ._sub_dirdist import subpanel_dirdist
from ._sub_msd import subpanel_msd
from ._sub_turnangs import subpanel_turnangs
from ._sub_series import subpanel_series
from ._sub_superplots import subpanel_superplots
from ._sub_flow import subpanel_flow

navpanel_viz = ui.nav_panel(
    "Visualisation",
    ui.navset_pill_list(

        subpanel_tracks,
        subpanel_dirdist,
        subpanel_msd,
        subpanel_turnangs,
        subpanel_series,  
        subpanel_superplots,

        widths = (2, 10)

    )
)
