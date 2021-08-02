import impy as ip
import napari

if __name__ == "__main__":
    ip.gui.start()
    ip.gui.viewer.update_console({"ip":ip})
    napari.run()
    input()