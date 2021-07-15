import numpy as np
import napari

mouse_drag_callbacks = ["drag_translation", "profile_shape"]
mouse_wheel_callbacks = ["wheel_resize"]
mouse_move_callbacks = ["on_move"]

def trace_mouse_drag(viewer:napari.Viewer, event, func=None):
    if func is None:
        return None
    last_event_position = event.position
    yield
    
    if event.type in ("mouse_release", "mouse_move"):
        clicked_layer = None
        for layer in reversed(viewer.layers):
            if not isinstance(layer, napari.layers.Image):
                continue
            if layer.visible:
                clicked_pos = layer.world_to_data(viewer.cursor.position)
                if all(0 <= pos and pos < s 
                       for pos, s in zip(clicked_pos, layer.data.shape)):
                    clicked_layer = layer
                    break
                
        if clicked_layer is None:
            viewer.layers.selection = set()
        else:    
            viewer.text_overlay.text = clicked_layer.name
            viewer.text_overlay.font_size = 5
            viewer.text_overlay.color = "white"
            if len(viewer.layers.selection) > 1 and event.type == "mouse_move":
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

def drag_translation(viewer:napari.Viewer, event):
    if viewer.dims.ndisplay == 3:
        # forbid translation in 3D mode
        return None
    if "Alt" in event.modifiers and "Shift" not in event.modifiers:
        """
        Manually translate image layer in xy-plane while pushing "Alt".
        """ 
        if event.button == 1:
            def func(layer, dpos):
                if not isinstance(layer, napari.layers.Shapes):
                    layer.translate -= dpos[-layer.translate.size:]
                
        else:
            func = None
        
    elif "Shift" in event.modifiers:
        """
        Manually translate image layer in x/y-direction.
        """ 
        if event.button == 1:
            def func(layer, dpos):
                if not isinstance(layer, napari.layers.Shapes):
                    dpos[-2] = 0.0
                    layer.translate -= dpos[-layer.translate.size:]
            
        elif event.button == 2:
            def func(layer, dpos):
                if not isinstance(layer, napari.layers.Shapes):
                    dpos[-1] = 0.0
                    layer.translate -= dpos[-layer.translate.size:]
        
        else:
            func = None
    else:
        return None
    
    return trace_mouse_drag(viewer, event, func)

def wheel_resize(viewer:napari.Viewer, event):
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
            ratio = scale[-1]/layer.metadata["init_scale"][-1]
            scale_texts.append(f"{round(ratio*100)}%")
        viewer.text_overlay.text = ", ".join(scale_texts)
        viewer.text_overlay.font_size = 10
        viewer.text_overlay.color = "white"

def on_move(viewer:napari.Viewer, event):
    viewer.text_overlay.text = ""

def profile_shape(viewer:napari.Viewer, event):
    active_layer = viewer.layers.selection.active
    active_plane = list(viewer.dims.order[-2:])
    if not isinstance(active_layer, napari.layers.Shapes):
        return None
            
    if event.button == 1:
        dy, dx = active_layer.scale[-2:]
        yield
        while event.type in ("mouse_move", "mouse_release"):
            unit = viewer.scale_bar.unit
            
            # get the current selected shape
            selected = active_layer.selected_data
            if len(selected) != 1:
                return None
            i = next(iter(selected)) # a little bit faster than selected.copy().pop()
            s = active_layer.data[i][:, active_plane] # selected shape
            s_type = active_layer.shape_type[i]
                
            if s_type == "rectangle":
                v1 = s[1] - s[0]
                v2 = s[1] - s[2]
                x = np.hypot(*v1)
                y = np.hypot(*v2)
                rad = np.arctan2(*v1)
                deg = np.rad2deg(rad)
                
                # prepare text overlay
                if np.abs(np.sin(rad)) < 1e-4:
                    text = f"{y:.1f} ({y*dy:.3g} {unit}) x {x:.1f} ({x*dx:.3g} {unit})"
                else:
                    if dy == dx:
                        text = f"{y:.1f} ({y*dy:.3g} {unit}) x {x:.1f} ({x*dx:.3g} {unit})\nangle = {deg:.1f} deg"
                    else:
                        degreal = np.rad2deg(np.arctan2(y*dy, x*dx))
                        text = f"{y:.1f} ({y*dy:.3g} {unit}) x {x:.1f} ({x*dx:.3g} {unit})\nangle = {deg:.1f} ({degreal:.1f}) deg"
                
                # update text overlay
                viewer.text_overlay.font_size = 5
                viewer.text_overlay.color = active_layer.current_edge_color
                viewer.text_overlay.text = text
                
            elif s_type == "line":
                v = s[0] - s[1]
                y, x = np.abs(v)
                deg = np.rad2deg(np.arctan2(*v))
                
                # prepare text overlay
                if dy == dx:
                    text = f"L = {np.hypot(y, x):.1f} ({np.hypot(y*dy, x*dx):.3g} {unit})\nangle = {deg:.1f} deg"
                else:
                    deg = np.rad2deg(np.arctan2(y/dy, x/dx))
                    text = f"L = {np.hypot(y, x):.1f} ({np.hypot(y*dy, x*dx):.3g} {unit}) angle = {deg:.1f} ({degreal:.1f}) deg"
                
                # update text overlay
                viewer.text_overlay.font_size = 5
                viewer.text_overlay.color = active_layer.current_edge_color
                viewer.text_overlay.text = text
            yield
            
    elif event.button == 2:
        last_mode = active_layer.mode
        active_layer.mode = "select"
        yield
        while event.type == "mouse_move":
            yield
        active_layer.mode = last_mode
