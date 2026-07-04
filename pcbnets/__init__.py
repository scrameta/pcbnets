"""pcbnets — interactive PCB net explorer from rasterised Gerber layers."""

from .audit import (
    PolarityVerdict,
    AlignmentVerdict,
    PostMergeCheck,
    audit_alignment,
    check_merged_nets,
    detect_polarity,
    detect_offset,
    make_audit_overlay,
    score_alignment,
)
from .gerber import (
    GerbvMissingError,
    detect_layers,
    load_or_create_layers_json,
    rasterise,
    read_layers_json,
    write_layers_json,
)
from .masks import MASK_LAYERS, MASK_POSITION, SILK_LAYERS, SILK_POSITION, load_masks, threshold_mask
from .nets import extract_nets, merge_nets, merge_nets_debug, explain_merge_path, UnionFind
from .prepare import (
    DEFAULT_OUTER_LAYERS,
    LayerCorrection,
    PreparationReport,
    default_outer_for,
    prepare_masks,
)
from .render import build_grid_and_idmap, labels_to_rgb

__version__ = "0.3.2"

__all__ = [
    # masks
    "load_masks", "threshold_mask", "SILK_LAYERS", "SILK_POSITION", "MASK_LAYERS", "MASK_POSITION",
    # nets
    "extract_nets", "merge_nets", "merge_nets_debug", "explain_merge_path", "UnionFind",
    # prepare
    "prepare_masks", "PreparationReport", "LayerCorrection",
    "DEFAULT_OUTER_LAYERS", "default_outer_for",
    # audit
    "detect_polarity", "detect_offset", "score_alignment",
    "audit_alignment", "check_merged_nets", "make_audit_overlay",
    "PolarityVerdict", "AlignmentVerdict", "PostMergeCheck",
    # gerber
    "detect_layers", "rasterise", "read_layers_json",
    "write_layers_json", "load_or_create_layers_json", "GerbvMissingError",
    # render
    "build_grid_and_idmap", "labels_to_rgb",
    "__version__",
]
