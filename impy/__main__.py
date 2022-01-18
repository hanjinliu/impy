import argparse
import impy as ip
import sys

def _open_ipython(path=None):
    import IPython as ipy
    user_ns = {"ip": ip}
    if path is not None:
        img = ip.imread(path)
        user_ns.update({"img": img})
    ipy.start_ipython(user_ns=user_ns)


def _open_napari(path=None):
    import napari
    ip.gui.start()
    ip.gui.viewer.update_console({"ip": ip})
    if path is not None:
        img = ip.imread(path)
        ip.gui.viewer.update_console({"img": img})
    sys.exit(napari.run(gui_exceptions=True))


def _apply_function(path: str = None, save_path: str = None, fname: str = None, **kwargs):
    if path is None or save_path is None or fname is None:
        raise TypeError
    img = ip.imread(path)
    out: ip.ImgArray = getattr(img, fname)(**kwargs)
    out.imsave(save_path)
    

def main():
    parser = argparse.ArgumentParser(description="Command line interface of impy.")

    parser.add_argument("-I", "--input", help="Path to input image.")
    parser.add_argument("-O", "--output", help="Path to output image (don't have to exist).")
    parser.add_argument("-i", "--ipython", help="Open IPython with namespace 'ip' and 'img'.", action="store_true")
    parser.add_argument("-f", "--function", help="Method that will be applied to the input image.")
    parser.add_argument("-n", "--napari", help="Open a napari viewer with namespace 'ip' and 'img'.", action="store_true")
    parser.add_argument("-a", "--args", help="Arguments passed to the function.")
    parser.add_argument("-V", "--version", action="version", version=f"impy version {ip.__version__}")
    args = parser.parse_args()

    args, unknown = parser.parse_known_args()
    print(args, unknown)
    if args.ipython:
        _open_ipython(args.I)
    elif args.napari:
        _open_napari(args.I)
    elif args.function is not None:
        _apply_function(args.I, args.output, args.function, unknown)
        
        

if __name__ == "__main__":
    main()