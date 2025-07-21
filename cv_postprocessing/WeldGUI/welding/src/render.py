import numpy as np
import cv2

def render_mask(im, mask):
    bitwise_mask = cv2.bitwise_and(im, im, mask=mask)
    rendered_img = im - bitwise_mask
    bitwise_mask = cv2.cvtColor(bitwise_mask, cv2.COLOR_RGB2GRAY)
    highlighted_mask = 0.9 * np.stack([bitwise_mask] * 3, axis=-1) - 0.1 * np.stack([np.zeros_like(mask), mask, np.zeros_like(mask)], axis=-1)
    rendered_img = (rendered_img + highlighted_mask).astype(np.uint8())
    return rendered_img


def render_properties(im, properties):
    pass    

def render_sides_and_projections(im, side):
    im = cv2.line(im, *side['line'], color=(0, 0, 255), thickness=5)
    im = cv2.arrowedLine(im, *side['projection'], color=(0, 0, 255))
    im = cv2.arrowedLine(im = cv2.line(im, *side['projection'], color=(0, 0, 255))
)
    
def render_line(im, line, length_per_pex=1, unit='px', color=(0,0,255)):
    d = np.linalg.norm(line[0] - line[1]) * length_per_pex
    im = cv2.arrowedLine(im, *line, color=color, thickness=5)
    im = cv2.arrowedLine(im, *(line[::-1]), color=color, thickness=5)
    s = f'%.2f' % d + unit
    center = (line[0] + line[1]) / 2
    center[0] += 20
    center = center.astype(np.int32)
    im = cv2.putText(im, s, center, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
    return im


def draw_line(im , p1, p2, color, thikness):
    im = cv2.line(im, p1, p2, color, thikness)
    im = cv2.circle(im, p1, thikness * 2, color, thikness)
    im = cv2.circle(im, p2, thikness * 2, color, thikness)
    return im

