import importlib.util
from pathlib import Path

_DEPLOY_PATH = Path(__file__).parents[1] / "deploy" / "modal_fast_scene.py"
_SPEC = importlib.util.spec_from_file_location("bookforge_modal_fast_scene", _DEPLOY_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_box_intersection_fraction = _MODULE._box_intersection_fraction
_collapse_nested_detections = _MODULE._collapse_nested_detections


def test_nested_detector_boxes_collapse_without_merging_distinct_actors() -> None:
    nested = _collapse_nested_detections(
        [
            {"score": 0.73, "box": (255.0, 182.0, 376.0, 314.0)},
            {"score": 0.35, "box": (225.0, 182.0, 812.0, 376.0)},
        ]
    )
    distinct = _collapse_nested_detections(
        [
            {"score": 0.76, "box": (134.0, 135.0, 308.0, 321.0)},
            {"score": 0.75, "box": (257.0, 222.0, 423.0, 320.0)},
        ]
    )

    assert len(nested) == 1
    assert len(distinct) == 2


def test_subject_object_overlap_uses_subject_area() -> None:
    subject = (255.0, 182.0, 376.0, 314.0)
    containing_boat = (224.0, 271.0, 663.0, 375.0)
    distant_boat = (478.0, 241.0, 727.0, 350.0)

    assert _box_intersection_fraction(subject, containing_boat) > 0.05
    assert _box_intersection_fraction(subject, distant_boat) == 0
