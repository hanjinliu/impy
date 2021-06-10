import numpy as np

def trace_mouse_drag(viewer, event, func=None):
    if func is None:
        return None
    last_event_position = event.position
    selected_visible_layers = list(filter(lambda x: x.visible, viewer.layers.selection))
    yield
    while event.type == "mouse_move":
        dpos = np.array(last_event_position) - np.array(event.position)
        [func(layer, dpos) for layer in selected_visible_layers]
        last_event_position = event.position
        yield
    
def drag_translation(viewer, event):
    if viewer.dims.ndisplay == 3:
        # forbid translation in 3D mode
        return None
    if ("Alt",) == event.modifiers:
        """
        Manually translate image layer in xy-plane while pushing "Alt".
        """ 
        if event.button == 1:
            def func(layer, dpos):
                layer.translate -= dpos[-layer.translate.size:]
                return None
        else:
            func = None
        
    elif ("Shift", "Alt") == event.modifiers:
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
        selected_visible_layers = list(filter(lambda x: x.visible, viewer.layers.selection))
        factor = 1 + event.delta[0]*0.1
        for layer in selected_visible_layers:
            scale = layer.scale.copy()
            scale[-2:] *= factor
            layer.scale = scale

