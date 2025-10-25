
@staticmethod
def frame_interval_ms(fps: float) -> float:
    """Return the interval between frames in milliseconds."""
    if fps <= 0:
        raise ValueError("Frame rate must be positive.")
    return 1000.0 / fps