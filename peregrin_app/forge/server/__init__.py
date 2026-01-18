from shiny import Inputs, Outputs, Session

from src.code import NoticeQueue

from .state import build_state
from .data.input import mount_data_input
from .data.display import mount_data_display
from .data.labeling import mount_data_labeling
from .filters.threshold1d.build_thresholds import mount_thresholds_build
from .filters.threshold1d.calc_thresholds import mount_thresholds_calc
from .filters.threshold1d.export_info import mount_thresholds_info_export

from .viz.tracks import MountTracks
from .viz.distributions import MountDistributions
from .viz.superplots import mount_superplots
from .viz.msd import mount_plot_msd

from .bg_jobs.loaders import mount_loaders
from .bg_jobs.buttons import mount_buttons
from .bg_jobs.notify import mount_notifier
from .bg_jobs.theme import set_theme
from .bg_jobs.statcols import mount_statcols

    




def Server(input: Inputs, output: Outputs, session: Session):

    
    
    S = build_state()
    noticequeue = NoticeQueue()
    
    args = (input, output, session, S)
    kwargs = {'noticequeue': noticequeue}

    args1 = args + (noticequeue,)
    
    mount_notifier(noticequeue)

    set_theme(*args)

    mount_data_input(*args, **kwargs)
    mount_data_display(*args)
    mount_data_labeling(*args, **kwargs)
    mount_thresholds_build(*args)
    mount_thresholds_calc(*args, **kwargs)
    mount_thresholds_info_export(*args)


    # MountTracks.realistic_reconstruction(*args, **kwargs)
    # MountTracks.polar_reconstruction(*args, **kwargs)
    # MountTracks.animated_reconstruction(*args, **kwargs)
    # MountTracks.lut_map(*args)
    # MountTracks()(*args1)
    MountTracks(*args1)
    MountDistributions(*args1)
    mount_superplots(*args, **kwargs)
    mount_plot_msd(*args, **kwargs)


    mount_loaders(*args)
    mount_buttons(*args, **kwargs)
    mount_statcols(*args, **kwargs)


    
if __name__ != "__main__":
    pass