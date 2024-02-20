import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


def show_diagram(
        theta1=-np.pi / 12,  # -15 deg
        theta2= np.pi / 3,   #  60 deg
        theta3=-np.pi / 4,   # -45 deg
        l=20,
        r_l=2,
        r_m=1.5,
        ):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim([-2.0 * l, 2.0 * l])
    ax.set_ylim([-2.5 * l, 2.0 * l])

    xy_base_bottom       = [                                              0,                                          -2 * l]
    xy_base_bottom_left  = [                                             -l,                                          -2 * l]
    xy_base_bottom_right = [                                              l,                                          -2 * l]
    xy_pendula_1_top     = [                       - l * np.sin(theta1) / 2,                          l * np.cos(theta1) / 2]
    xy_pendula_1_bottom  = [                         l * np.sin(theta1) / 2,                        - l * np.cos(theta1) / 2]
    xy_pendula_2_top     = [xy_pendula_1_bottom[0] - l * np.sin(theta2) / 2, xy_pendula_1_bottom[1] + l * np.cos(theta2) / 2]
    xy_pendula_2_bottom  = [xy_pendula_1_bottom[0] + l * np.sin(theta2) / 2, xy_pendula_1_bottom[1] - l * np.cos(theta2) / 2]
    xy_pendula_3_top     = [xy_pendula_2_bottom[0] - l * np.sin(theta3) / 2, xy_pendula_2_bottom[1] + l * np.cos(theta3) / 2]
    xy_pendula_3_bottom  = [xy_pendula_2_bottom[0] + l * np.sin(theta3) / 2, xy_pendula_2_bottom[1] - l * np.cos(theta3) / 2]

    # Support
    ax.vlines(xy_base_bottom[0], xy_base_bottom[0], xy_base_bottom[1], lw=8, colors='black', zorder=1)
    ax.plot([xy_base_bottom_left[0], xy_base_bottom_right[0]], [xy_base_bottom_left[1], xy_base_bottom_right[1]], lw=16, color='black', zorder=1)

    # Rods
    ax.plot([xy_pendula_1_top[0], xy_pendula_1_bottom[0]], [xy_pendula_1_top[1], xy_pendula_1_bottom[1]], lw=2 * r_l, color='blue',  zorder=2)
    ax.plot([xy_pendula_2_top[0], xy_pendula_2_bottom[0]], [xy_pendula_2_top[1], xy_pendula_2_bottom[1]], lw=2 * r_l, color='red',   zorder=3)
    ax.plot([xy_pendula_3_top[0], xy_pendula_3_bottom[0]], [xy_pendula_3_top[1], xy_pendula_3_bottom[1]], lw=2 * r_l, color='green', zorder=4)

    # Masses
    ax.add_patch(Circle(xy_pendula_1_top,    radius=r_m, color='blue',  zorder=2))
    ax.add_patch(Circle(xy_pendula_1_bottom, radius=r_m, color='blue',  zorder=2))
    ax.add_patch(Circle(xy_pendula_2_top,    radius=r_m, color='red',   zorder=3))
    ax.add_patch(Circle(xy_pendula_2_bottom, radius=r_m, color='red',   zorder=3))
    ax.add_patch(Circle(xy_pendula_3_top,    radius=r_m, color='green', zorder=4))
    ax.add_patch(Circle(xy_pendula_3_bottom, radius=r_m, color='green', zorder=4))

    # Pins
    ax.add_patch(Circle(             [0, 0], radius=r_m / 2, color='blue',  zorder=2))
    ax.add_patch(Circle(xy_pendula_1_bottom, radius=r_m / 2, color='red',   zorder=3))
    ax.add_patch(Circle(xy_pendula_2_bottom, radius=r_m / 2, color='green', zorder=4))

    return fig