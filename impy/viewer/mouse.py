import numpy as np
from .utils import iter_layer

mouse_drag_callbacks = ["drag_translation"]
mouse_wheel_callbacks = ["wheel_resize"]
mouse_move_callbacks = ["on_move"]

def trace_mouse_drag(viewer, event, func=None):
    if func is None:
        return None
    last_event_position = event.position
    yield
    
    if event.type in ("mouse_release", "mouse_move"):
        clicked_layer = None
        for layer in reversed(list(iter_layer(viewer, "Image"))):
            if layer.visible:
                clicked_pos = layer.world_to_data(viewer.cursor.position)
                if all(0 <= pos and pos < s 
                       for pos, s in zip(clicked_pos, layer.data.shape)):
                    clicked_layer = layer
                    break
                
        if clicked_layer is None:
            viewer.layers.selection = set()
        elif len(viewer.layers.selection) > 1 and event.type == "mouse_move":
            pass
        elif "Control" in event.modifiers and event.type == "mouse_release":
            viewer.layers.selection.add(clicked_layer)
        else:
            viewer.layers.selection = {clicked_layer}
                
    while event.type == "mouse_move":
        dpos = np.array(last_event_position) - np.array(event.position)
        [func(layer, dpos) for layer in viewer.layers.selection]
        last_event_position = event.position
        yield

def drag_translation(viewer, event):
    if viewer.dims.ndisplay == 3:
        # forbid translation in 3D mode
        return None
    if "Alt" in event.modifiers and "Shift" not in event.modifiers:
        """
        Manually translate image layer in xy-plane while pushing "Alt".
        """ 
        if event.button == 1:
            def func(layer, dpos):
                layer.translate -= dpos[-layer.translate.size:]
                
        else:
            func = None
        
    elif "Shift" in event.modifiers:
        """
        Manually translate image layer in x/y-direction.
        """ 
        if event.button == 1:
            def func(layer, dpos):
                dpos[-2] = 0.0
                layer.translate -= dpos[-layer.translate.size:]
                return None
            
        elif event.button == 2:
            def func(layer, dpos):
                dpos[-1] = 0.0
                layer.translate -= dpos[-layer.translate.size:]
                return None
        
        else:
            func = None
    else:
        return None
    
    return trace_mouse_drag(viewer, event, func)

def wheel_resize(viewer, event):
    """
    Manually resize image layer in xy-plane while pushing "Alt".
    """ 
    if "Alt" in event.modifiers:
        scale_texts = []
        delta = event.delta[0]
        if delta > 0:
            factor = 1 + event.delta[0]*0.1
        else:
            factor = 1/(1 - event.delta[0]*0.1)
        for layer in viewer.layers.selection:
            scale = layer.scale.copy()
            scale[-2:] *= factor
            layer.scale = scale
            scale_texts.append(f"{int(scale[-1]*100)}%")
        viewer.text_overlay.visible = True    
        viewer.text_overlay.text = ", ".join(scale_texts)

def on_move(viewer, event):
    viewer.text_overlay.visible = False
