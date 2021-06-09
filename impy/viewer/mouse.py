import numpy as np

def drag_translation(layer, event):
    
    # TODO: other modifiers or combinations of modifiers
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
                layer.translate -= dpos[-layer.translate.size:]
                yield
        elif event.button == 2:
            # something here?
            pass
    
    elif ("Shift", "Alt") == event.modifiers:
        
        """
        Manually translate image layer in x/y-direction.
        """ 
        if event.button == 1:
            last_event_position = event.position
            yield
            while event.type == "mouse_move":
                dpos = np.array(last_event_position) - np.array(event.position)
                dpos[-2] = 0.
                last_event_position = event.position
                layer.translate -= dpos[-layer.translate.size:]
                yield
        elif event.button == 2:
            last_event_position = event.position
            yield
            while event.type == "mouse_move":
                dpos = np.array(last_event_position) - np.array(event.position)
                dpos[-1] = 0.
                last_event_position = event.position
                layer.translate -= dpos[-layer.translate.size:]
                yield
            

def wheel_resize(layer, event):
    """
    Manually resize image layer in xy-plane while pushing "Alt".
    """ 
    if "Alt" in event.modifiers:
        factor = 1 + event.delta[0]*0.1
        scale = layer.scale.copy()
        scale[-2:] *= factor
        layer.scale = scale