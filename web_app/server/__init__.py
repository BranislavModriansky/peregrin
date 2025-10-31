from shiny import Inputs, Outputs, Session
from .state import build_state
from .data.input import mount_data_input
from .data.display import mount_data_display
from .data.labeling import mount_data_labeling
from .filters.threshold1d.build_thresholds import mount_thresholds_build
from .filters.threshold1d.calc_thresholds import mount_thresholds_calc
from .filters.threshold1d.export_info import mount_thresholds_info_export
# from .gates.gating2d import *
# from .viz.tracks import mount_tracks
from .viz.tracks import MountTracks
from .viz.superplots import mount_superplots
# from .viz.timeseries import mount_timeseries
from .jobs.loaders import mount_loaders
from .jobs.background_tasks import mount_tasks


def Server(input: Inputs, output: Outputs, session: Session):
    S = build_state()
    mount_data_input(input, output, session, S)
    mount_data_display(input, output, session, S)
    mount_data_labeling(input, output, session, S)
    mount_thresholds_build(input, output, session, S)
    mount_thresholds_calc(input, output, session, S)
    mount_thresholds_info_export(input, output, session, S)
    # mount_gating2d(input, output, session, S)   
    # mount_tracks(input, output, session, S)
    MountTracks.realistic_reconstruction(input, output, session, S)
    MountTracks.polar_reconstruction(input, output, session, S)
    MountTracks.animated_reconstruction(input, output, session, S)
    MountTracks.lut_map(input, output, session, S)
    mount_superplots(input, output, session, S)
    # mount_timeseries(input, output, session, S)
    mount_loaders(input, output, session, S)
    mount_tasks(S)
