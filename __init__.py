from .videohelpersuite.nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

from .videohelpersuite.frame_processor_node import (
    FrameByFrameVideoProcessor,
    NODE_CLASS_MAPPINGS as FBF_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as FBF_DISPLAY_MAPPINGS,
)

from .videohelpersuite.video_merger_node import (
    VideoMerger,
    NODE_CLASS_MAPPINGS as MERGER_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MERGER_DISPLAY_MAPPINGS,
)

import folder_paths
from .videohelpersuite.server import server
from .videohelpersuite import documentation
from .videohelpersuite import latent_preview

WEB_DIRECTORY = "./web"

# merge mappings
NODE_CLASS_MAPPINGS.update(FBF_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(FBF_DISPLAY_MAPPINGS)

NODE_CLASS_MAPPINGS.update(MERGER_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MERGER_DISPLAY_MAPPINGS)

documentation.format_descriptions(NODE_CLASS_MAPPINGS)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
