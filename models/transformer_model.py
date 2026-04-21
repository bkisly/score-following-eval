from typing import Dict, Any

import numpy as np

from models.score_follower import ScoreFollower


class TransformerModel(ScoreFollower):
    def load_reference(self, reference_path: str) -> None:
        pass

    def process_frame(self, audio_frame: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        pass

    def reset(self) -> None:
        pass