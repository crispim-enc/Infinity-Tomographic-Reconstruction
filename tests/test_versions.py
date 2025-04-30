#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#
# get a list of all packages in toor
import importlib
import inspect
import os

# List of your custom packages


def get_packages(src_dir="toor"):
    """
    Get all Python packages inside the given directory.

    A package is defined as a directory containing an __init__.py file.

    :param src_dir: Path to the source directory.
    :return: Generator with package names as strings.
    """
    for root, dirs, files in os.walk(src_dir):
        if "__init__.py" in files:
            # Convert the directory path to a Python package name
            package = os.path.relpath(root, src_dir).replace(os.sep, ".")
            yield package


def get_package_info(package_name):
    try:
        # Dynamically import the package
        package = importlib.import_module(package_name)

        # Get package version
        version = getattr(package, "__version__", "Unknown version")
 
        # List all functions in the package
        functions = [
            func_name for func_name, func_obj in inspect.getmembers(package)
            if inspect.isfunction(func_obj) or inspect.isbuiltin(func_obj)
        ]
        classes = [
            class_name for class_name, class_obj in inspect.getmembers(package)
            if inspect.isclass(class_obj)
        ]
        submodules = [
            submodule_name for submodule_name, submodule_obj in inspect.getmembers(package)
            if inspect.ismodule(submodule_obj)
        ]
        modules = [
            module_name for module_name, module_obj in inspect.getmembers(package)
            if inspect.ismodule(module_obj)
        ]


        return version, classes
    except ModuleNotFoundError:
        return None, None


def main():
    src_path = "toor"  # Path to your source directory
    custom_packages = list(get_packages(src_path))

    custom_packages = custom_packages[1:]
    #add toor. to the package names
    custom_packages = [f"toor.{package}" for package in custom_packages]
    for package in custom_packages:
        version, classes = get_package_info(package)
        # if packedge is a submodule of package the text suffers an indentation
        path_length = len(package.split('.')) - 2

        if version is not None:
            print(f"    |"*path_length+f"Package: {package.split('.')[-1]}")
            print(f"    |"*path_length+f"Version: {version}")
            if path_length == 0:
                print(f"    |"*path_length+f"Objects: {', '.join(classes) if classes else 'None'}")
            else:
                print(f"    |"*path_length+f"Classes: {', '.join(classes) if classes else 'None'}")

            # print(f"Functions: {', '.join(functions) if functions else 'None'}")
            print("-" * 40)
        else:
            print(f"Package {package} not found.")
            print("-" * 40)


if __name__ == "__main__":
    main()