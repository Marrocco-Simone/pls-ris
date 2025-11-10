import numpy as np
from typing import Tuple, List
from heatmap_situations import Point

class Actor:
    """Represents a transmitter or receiver actor in the scene."""
    def __init__(self, name: str, position: Tuple[float, float, float],
                 orientation: Tuple[float, float, float], rows: int, cols: int):
        self.name = name
        self.position = position
        self.orientation = orientation
        self.rows = rows
        self.cols = cols

def calculate_orientation(from_point: Point, to_point: Point) -> Tuple[float, float, float]:
    """Calculate orientation (roll, yaw, pitch) from one point toward another."""
    vec = (to_point['x'] - from_point['x'], to_point['y'] - from_point['y'])
    angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
    return (0.0, angle, 0.0)

def calculate_ris_orientation(ris_point: Point, incident_point: Point,
                              reflected_points: List[Point]) -> Tuple[float, float, float]:
    """Calculate RIS orientation to reflect from incident to reflected points."""
    vec_incident = (incident_point['x'] - ris_point['x'], incident_point['y'] - ris_point['y'])
    incident_angle = float(np.degrees(np.arctan2(vec_incident[1], vec_incident[0])))

    if len(reflected_points) == 0:
        return (0.0, incident_angle + 90, 0.0)

    reflected_angles = []
    for rp in reflected_points:
        vec = (rp['x'] - ris_point['x'], rp['y'] - ris_point['y'])
        reflected_angles.append(float(np.degrees(np.arctan2(vec[1], vec[0]))))

    avg_reflected = sum(reflected_angles) / len(reflected_angles)
    incident_from_opposite = incident_angle + 180
    bisector = (incident_from_opposite + avg_reflected) / 2

    return (0.0, bisector - 90, 0.0)
