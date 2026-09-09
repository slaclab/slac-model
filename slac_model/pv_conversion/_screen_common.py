"""Screen-image helpers shared by the particle-tracking backends (Bmad, Impact)."""

from beamphysics import ParticleGroup
import numpy as np


def histogram_screen_image(
    beam: ParticleGroup,
    shape: tuple[int, int],
    pixel_size: float,
    normalize: bool = False
):

    """Histogram a particle `beam`'s (x, y) 
    coordinates into a `shape`-sized image.
    
    Parameters
    ----------
    beam: ParticleGroup
        Beam at screen.
    shape: tuple[int, int]
        Image shape.
    pixel_size: float
        Pixel size in meters.
    normalize: bool
        Flag to determine if screen image should be normalized to [0,1] scale.
    
    
    """
    half_width = (shape[0] * pixel_size / 2, shape[1] * pixel_size / 2)
    hist, _ = beam.histogramdd(
        "x",
        "y",
        bins=shape,
        range=((-half_width[0], half_width[0]), (-half_width[1], half_width[1])),
    )
    if normalize:
        max_value = np.max(hist)
        hist = hist / max_value if max_value > 0 else hist
    return hist


def get_screen_image_array_size(simulator, element_name: str, shape, index: int) -> int:
    return shape[index]


def get_screen_resolution(simulator, element_name: str, pixel_size: float) -> float:
    """ return the screen resolution in um"""
    return pixel_size * 1e6
