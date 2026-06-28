from ..data import naive_ctr, naive_cxcl12, naive_mu
from ..src.compute.stats import stats


spots, tracks = stats.spots(naive_ctr), stats.tracks(stats.spots(naive_ctr))