import numpy as np

"""
event attrs:
button: 1
buttons: []
delta: [0. 0.]
handled: False
is_dragging: False
modifiers: (<Key 'Alt'>,)
pos: [139 335]
position: (0.0, 7.485511127471, 3.7939119824528014)
time: 1623204750.00107
type: mouse_press
"""
def drag_translation(layer, event):
    """
    Manually translate image layer in xy-plane while pushing "Alt".
    """ 
    # TODO: other modifiers or combinations of modifiers
    if "Alt" in event.modifiers:
        if event.button == 2 and event.type == "mouse_press":
            layer.translate -= layer.translate
            yield
        else:
            last_event_position = event.position
            yield
            while event.type == "mouse_move":
                dpos = np.array(last_event_position) - np.array(event.position)
                last_event_position = event.position
                layer.translate -= dpos
                yield
