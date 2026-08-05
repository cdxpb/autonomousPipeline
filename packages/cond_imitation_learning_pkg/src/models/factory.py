
from .pilotnet_models import (
    PilotNetBasic,
    PilotNetV3,
    PilotNetClassNHead,
    PilotNetRegNHead,
    PilotNetRegNHeadV2,
    PilotNetRegSmaller,
    PilotNetRegNCILTemporal,
    PilotNetFiLM
)

def get_pilotnet_model(approach: int, model_set: str = "full", skip_segmentation: bool = False):
    if approach == 3:
        return PilotNetRegNHead()
    elif approach == 4:
        return PilotNetRegNCILTemporal(num_frames=3)
    elif approach == 5:
        if model_set == "smaller":
            return PilotNetRegSmaller()
        else:
            return PilotNetRegNHeadV2()
    elif approach == 6:
        return PilotNetClassNHead()
    elif approach == 7:
        if model_set == "smaller":
            return PilotNetRegSmaller() # Assuming approach 7 smaller uses same architecture as 5 smaller
        else:
            return PilotNetFiLM()
    else:
        # Default fallback
        return PilotNetBasic(in_channels=3 if skip_segmentation else 1)
