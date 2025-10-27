from typing import List, TypedDict

class Building(TypedDict):
    x: int
    y: int
    width: int
    height: int
class Point(TypedDict):
    x: float
    y: float

class Situation(TypedDict):
    simulation_name: str
    calculate: bool
    force_recompute: bool
    width: int
    height: int
    resolution: float
    buildings: List[Building]
    transmitter: Point
    ris_points: List[Point]
    receivers: List[Point]

situations: List[Situation] = [
    {
        "simulation_name": "Single Reflection",
        "calculate": True,
        "force_recompute": False,
        "width": 20,
        "height": 20,
        "resolution": 0.5,
        "buildings": [
            {'x': 0, 'y': 10, 'width': 7, 'height': 10},
            {'x': 8, 'y': 0, 'width': 12, 'height': 8},
        ],
        "transmitter": {'x': 3,'y': 3},
        "ris_points": [
            {'x': 7,'y': 9}
        ],
        "receivers": [
            {'x': 16,'y': 11},
            {'x': 10,'y': 18}
        ]
    },
    {
        "simulation_name": "Single Reflection BIG",
        "calculate": False,
        "force_recompute": False,
        "width": 2000,
        "height": 2000,
        "resolution": 50,
        "buildings": [
            {'x': 0, 'y': 1000, 'width': 700, 'height': 1000},
            {'x': 800, 'y': 0, 'width': 1200, 'height': 800},
        ],
        "transmitter": {'x': 300, 'y': 300},
        "ris_points": [
            {'x': 700, 'y': 900}
        ],
        "receivers": [
            {'x': 1600, 'y': 1100},
            {'x': 1000, 'y': 1800}
        ]
    },
    {
        "simulation_name": "RISs in series, only final",
        "calculate": False,
        "force_recompute": False,
        "width": 20,
        "height": 20,
        "resolution": 0.5,
        "buildings": [
            {'x': 0, 'y': 10, 'width': 10, 'height': 10},
            {'x': 2, 'y': 4, 'width': 7, 'height': 1},
        ],
        "transmitter": {'x': 1, 'y': 1},
        "ris_points": [
            {'x': 0, 'y': 9}, 
            {'x': 10, 'y': 9}
        ],
        "receivers": [
            {'x': 16, 'y': 14}, 
            {'x': 12, 'y': 18}
        ]
    },
    {
        "simulation_name": "RISs in series",
        "calculate": False,
        "force_recompute": False,
        "width": 20,
        "height": 20,
        "resolution": 0.5,
        "buildings": [
            {'x': 0, 'y': 10, 'width': 10, 'height': 10},
            {'x': 3, 'y': 4, 'width': 7, 'height': 1},
            {'x': 15, 'y': 10, 'width': 5, 'height': 1},
            {'x': 9, 'y': 0, 'width': 1, 'height': 8},
            {'x': 5, 'y': 7, 'width': 7, 'height': 1},
        ],
        "transmitter": {'x': 1, 'y': 1},
        "ris_points": [
            {'x': 0, 'y': 9},
            {'x': 10, 'y': 9},
            {'x': 18, 'y': 6},
        ],
        "receivers": [
            {'x': 4, 'y': 5},
            {'x': 14, 'y': 16},
            {'x': 12, 'y': 18},
            {'x': 11, 'y': 3},
            {'x': 15, 'y': 1},
        ]
    },
    {
        "simulation_name": "RISs in parallel",
        "calculate": False,
        "force_recompute": False,
        "width": 20,
        "height": 20,
        "resolution": 0.5,
        "buildings": [
            { 'x': 8, 'y': 0, 'width': 1, 'height': 7 },
            { 'x': 12, 'y': 0, 'width': 1, 'height': 7 },
            { 'x': 8, 'y': 11, 'width': 5, 'height': 1 },
            { 'x': 8, 'y': 15, 'width': 5, 'height': 5 },
        ],
        "transmitter": {'x': 10, 'y': 1},
        "ris_points": [
            {'x': 10, 'y': 10},
            {'x': 2, 'y': 2},
            {'x': 18, 'y': 18},
            {'x': 2, 'y': 18},
            {'x': 18, 'y': 2},
        ],
        "receivers": [
            {'x': 10, 'y': 12},
            {'x': 5, 'y': 13},
            {'x': 15, 'y': 7},
        ]
    },
    {
        "simulation_name": "Long Corridor",
        "calculate": False,
        "force_recompute": False,
        "width": 100 * 1000,
        "height": 20 * 1000,
        "resolution": 1000,
        "buildings": [],
        "transmitter": {'x': 95 * 1000, 'y': 10 * 1000},
        "ris_points": [],
        "receivers": [
            {'x': 5 * 1000, 'y': 15 * 1000},
            {'x': 25 * 1000, 'y': 5 * 1000},
            {'x': 45 * 1000, 'y': 15 * 1000},
            {'x': 65 * 1000, 'y': 5 * 1000},
            {'x': 85 * 1000, 'y': 15 * 1000},
            {'x': 90 * 1000, 'y': 5 * 1000},
        ]
    },
]