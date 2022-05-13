import math
from typing import NamedTuple

import cv2
import numpy as np


class Rect(NamedTuple):
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def center(self):
        return (
            self.left + int(self.width / 2),
            self.top + int(self.height / 2),
        )


def draw_gradient_circle(img, center, size, color, bgcolor):
    """ Draws a circle that fades from color (at the center)
        to bgcolor (at the boundaries)
    """
    for i in range(1, size):
        a = 1 - i / size
        c = np.add(np.multiply(color[0:3], a), np.multiply(bgcolor[0:3], 1 - a))
        cv2.circle(img, center, i, c, 2)


def draw_gradient_wedge(img, center, size, color, bgcolor, start_angle, delta_angle):
    """ Draws a wedge that fades from color (at the center)
        to bgcolor (at the boundaries)
    """
    for i in range(1, size):
        a = 1 - i / size
        c = np.add(np.multiply(color, a), np.multiply(bgcolor, 1 - a))
        cv2.ellipse(
            img, center, (i, i), start_angle, -delta_angle / 2, delta_angle / 2, c, 2,
        )


def draw_goal_radar(
    pointgoal,
    overlay_target,
    r: Rect,
    start_angle=0,
    fov=0,
    goalcolor=(184, 0, 50, 255),
    wincolor=(0, 0, 0, 0),
    maskcolor=(85, 75, 70, 255),
    bgcolor=(255, 255, 255, 255),
    gradientcolor=(174, 112, 80, 255),
):
    """ Draws a radar that shows the goal as a dot
    """
    mag = pointgoal[0]  # magnitude (>=0)
    nm = mag / (mag + 1)  # normalized magnitude (0 to 1)
    xy = pointgoal[1:]
    size = int(round(0.45 * min(r.width, r.height)))
    center = r.center
    target = (
        int(round(center[0] + xy[1] * size * nm)),
        int(round(center[1] - xy[0] * size * nm)),
    )
    img = np.zeros((r.height, r.width, 4), np.uint8)
    if wincolor is not None:
        cv2.rectangle(
            img, (r.left, r.top), (r.right, r.bottom), wincolor, cv2.LINE_AA
        )  # Fill with window color
    cv2.circle(img, center, size, bgcolor, -1)  # Circle with background color
    if fov > 0:
        masked = 360 - fov
        cv2.ellipse(
            img,
            center,
            (size, size),
            start_angle + 90,
            -masked / 2,
            masked / 2,
            maskcolor,
            -1,
        )
    if gradientcolor is not None:
        if fov > 0:
            draw_gradient_wedge(
                img, center, size, gradientcolor, bgcolor, start_angle - 90, fov,
            )
        else:
            draw_gradient_circle(img, center, size, gradientcolor, bgcolor)

    cv2.circle(img, target, 8, goalcolor, -1)

    if overlay_target is not None:
        bottom = overlay_target.shape[0]
        top = bottom - r.height
        left = overlay_target.shape[1] // 2 - r.width // 2
        right = left + r.width
        alpha = 0.5 * img[..., 3] / 255
        rgb = img[..., 0:3]
        overlay = np.add(
            np.multiply(
                overlay_target[top:bottom, left:right],
                np.expand_dims(1 - alpha, axis=2),
            ),
            np.multiply(rgb, np.expand_dims(alpha, axis=2)),
        )
        overlay_target[top:bottom, left:right] = overlay
