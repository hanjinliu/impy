import numpy as np

def drag_translation(layer, event):
    """
    Manually translate image layer in xy-plane while pushing Alt.
    """    
    if "Alt" in event.modifiers:
        last_event_position = event.position
        yield
        while event.type == "mouse_move":
            dpos = np.array(last_event_position) - np.array(event.position)
            last_event_position = event.position
            layer.translate -= dpos
            yield
