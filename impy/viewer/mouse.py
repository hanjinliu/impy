import numpy as np

def drag_translation(layer, event):
    
    # TODO: other modifiers or combinations of modifiers
    # - x, y restricted
    
    if ("Alt",) == event.modifiers:
        """
        Manually translate image layer in xy-plane while pushing "Alt".
        """ 
        if event.button == 1:
            last_event_position = event.position
            yield
            while event.type == "mouse_move":
                dpos = np.array(last_event_position) - np.array(event.position)
                last_event_position = event.position
                layer.translate -= dpos
                yield
        elif event.button == 2:
            # something here?
            pass
    
    elif ("Control", "Alt") == event.modifiers:
        """
        Manually crop image.
        """ 
        if event.button == 1:
            start = (event.position[-2:] - layer.translate[-2:]).astype("int64")
            while event.type == "mouse_move":
                # draw rectangle
                yield
            end = (event.position[-2:] - layer.translate[-2:]).astype("int64")
            
            sl0 = slice(*sorted([start[0], end[0]]))
            sl1 = slice(*sorted([start[1], end[1]]))
            
            
            layer.data = layer.data[..., sl0, sl1]
                    
            return None
            

def wheel_resize(layer, event):
    """
    Manually resize image layer in xy-plane while pushing "Alt".
    """ 
    if "Alt" in event.modifiers:
        factor = 1 + event.delta[0]*0.1
        scale = layer.scale.copy()
        scale[-2:] *= factor
        layer.scale = scale