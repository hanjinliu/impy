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
        # This is not the best practice because multiple images cannot be cropped at the
        # same time
        if event.button == 1:
            pos0 = event.position
            start = ((event.position - layer.translate)/layer.scale).astype("int64")
            while event.type != "mouse_release":
                # draw rectangle
                yield
            end = ((event.position - layer.translate)/layer.scale).astype("int64")
            sl = []
            for i in range(layer.ndim):
                sl0 = sorted([start[i], end[i]])
                sl0[1] += 1
                sl.append(slice(*sl0))
            
            layer.data = layer.data[tuple(sl)]
            layer.translate += np.round(pos0/layer.scale)*layer.scale
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