"""Custom observation functions for the onelast intersection."""

import numpy as np
from gymnasium import spaces
from sumo_rl.environment.observations import ObservationFunction

# Incoming approach edges in SUMO (see onelast.rou.xml depart edges).
# Order: North, South, East, West — matches route origins:
#   North = -E3, South = E2, East = -E1, West = E0
# E1 and E3 are OUTGOING from the intersection; they stay empty at the approach.
APPROACH_EDGES = ("-E3", "E2", "-E1", "E0")
APPROACH_LABELS = ("North", "South", "East", "West")
MAX_QUEUE = 20.0


class FourApproachQueueObservation(ObservationFunction):
    """Normalized halting (queue) count per approach edge."""

    def __call__(self) -> np.ndarray:
        queues = []
        for edge in APPROACH_EDGES:
            halting = self.ts.sumo.edge.getLastStepHaltingNumber(edge)
            queues.append(min(halting / MAX_QUEUE, 1.0))
        return np.array(queues, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=np.zeros(len(APPROACH_EDGES), dtype=np.float32),
            high=np.ones(len(APPROACH_EDGES), dtype=np.float32),
        )
