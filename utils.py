import numpy as np

def mask_annulus(shape, center, inner_radius, outer_radius):
    cy, cx = center

    ys, xs = np.indices(shape)
    return ((ys - cy )**2 + (xs - cx )**2 <= (outer_radius)**2) &\
           ((ys - cy )**2 + (xs - cx )**2 >= inner_radius**2)

def pixels_in_annulus(shape, center, inner_radius, outer_radius):
    ys, xs = np.indices(shape)
    mask = mask_annulus(shape, center, inner_radius, outer_radius)
    return ys[mask], xs[mask]