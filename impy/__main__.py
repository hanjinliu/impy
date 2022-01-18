from __future__ import annotations
import argparse
import inspect
from typing import Any
from pathlib import Path
import ast
import sys

import impy as ip


def _open_ipython(path, unknown):
    import IPython as ipy
    user_ns = {"ip": ip}
    if path is None and unknown:
        # the first argument is --input
        path, *unknown = unknown
        
    if path is not None:
        img = ip.imread(path)
        user_ns["img"] = img
    
    # sys.argv should be hidden from Ipython, otherwise it will raise unnecessary error.
    sys.argv = sys.argv[:1]
    ipy.start_ipython(user_ns=user_ns)


def _open_napari(path, unknown):
    import napari
    
    if path is None and unknown:
        # the first argument is --input
        path, *unknown = unknown
    sys.argv = sys.argv[:1]
    user_ns = {"ip": ip}
    if path is not None:
        img = ip.imread(path)
        user_ns["img"] = img
        
    ip.gui.start()
    if path is not None:
        ip.gui.add(img)
    ip.gui.viewer.window._qt_viewer.console.push(user_ns)
    sys.exit(napari.run(gui_exceptions=True))


def _eval_arg(key: str, value: str, sig: inspect.Signature):
    try:
        annot = sig.parameters[key].annotation
    except KeyError:
        _args = [f"--{k}" for k in sig.parameters.keys()]
        raise TypeError(
            f"Method got an unexpected keyword argument {key}. Allowed arguments are:\n"
            f"{', '.join(_args)}"
            )
    if annot in (str, "str"):
        return value
    else:
        return ast.literal_eval(value)


def _apply_function(path: str = None, 
                    save_path: str = None, 
                    fname: str = None, 
                    unknown: list[str] = []):
    unknown = unknown.copy()
    cls_method = getattr(ip.ImgArray, fname)
    sig = inspect.signature(cls_method)
    
    if unknown and path is None:
        # the first argument is --input
        path, *unknown = unknown
    if unknown and save_path is None:
        save_path, *unknown = unknown
    path = Path(path).resolve()
    save_path = Path(save_path).resolve()
    _vars = zip(["input path", "output path", "method name"], [path, save_path, fname])
    missing = set([s_ for s_, a in _vars if a is None])
    if missing:
        raise TypeError(
            f"Input path, output path and method name must be given but {', '.join(missing)} is missing.\n"
            "Basic Usage:\n"
            " $ impy some/input/path.tif some/output/path.tif -f method_name\n"
            " $ impy -I some/input/path.tif -O some/output/path.tif -f method_name\n"
            " $ impy --input some/input/path.tif --output some/output/path.tif --method method_name\n"
            " $ impy some/input/path.tif some/output/path.tif --method gaussian_filter --sigma 2.0\n"
            )
    
    # process unknown arguments
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
        
    i = 0
    length = len(unknown)
    while i < length:
        a = unknown[i]
        if not a.startswith("-"):
            if kwargs:
                raise TypeError("keyword arguments came after positional arguments.")
            args.append(ast.literal_eval(a))
        else:
            i += 1
            key = a.lstrip("-")
            v = _eval_arg(key, unknown[i], sig)
            kwargs[key] = v
        i += 1
    
    s = ", ".join([f"{a!r}" for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])
    expr = (
        " >>> import impy as ip\n"
        f" >>> img = ip.imread({str(path)!r})\n"
        f" >>> out = img.{fname}({s})\n"
        f" >>> out.imsave({str(save_path)!r})\n"
        )
    print(f"\nRunning following code:\n\n{expr}")
    img = ip.imread(path)
    out: ip.ImgArray = getattr(img, fname)(*args, **kwargs)
    out.imsave(save_path)


def main():
    parser = argparse.ArgumentParser(description="Command line interface of impy.")

    parser.add_argument("--input", help="Path to input image.")
    parser.add_argument("--output", help="Path to output image (don't have to exist).")
    parser.add_argument("-i", "--ipython", help="Open IPython with namespace 'ip' and 'img'.", action="store_true")
    parser.add_argument("-m", "--method", help="Method that will be applied to the input image.")
    parser.add_argument("-n", "--napari", help="Open a napari viewer with namespace 'ip' and 'img'.", action="store_true")
    parser.add_argument("-v", "--version", action="version", version=f"impy version {ip.__version__}")
    
    args, unknown = parser.parse_known_args()
    
    if args.ipython:
        _open_ipython(args.input, unknown)
    elif args.napari:
        _open_napari(args.input, unknown)
    elif args.method is not None:
        _apply_function(args.input, args.output, args.method, unknown)
    else:
        raise RuntimeError

if __name__ == "__main__":
    main()