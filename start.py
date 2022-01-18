import impy as ip
import napari
import sys

if __name__ == "__main__":
    ip.gui.start()
    ip.gui.viewer.update_console({"ip": ip})
    sys.exit(napari.run(gui_exceptions=True))
