import dataclasses

from evaluation.metrics import EvaluationMetrics


@dataclasses.dataclass
class PieceMetadata:
    title: str
    composer: str

    def to_dict(self):
        return {
            'title': self.title,
            'composer': self.composer,
        }

@dataclasses.dataclass
class Piece:
    midi_path: str
    audio_path: str
    metadata: PieceMetadata

@dataclasses.dataclass
class ExperimentVariation:
    factor: float
    result: EvaluationMetrics

@dataclasses.dataclass
class EvaluationResult:
    piece_metadata: PieceMetadata
    result: EvaluationMetrics
