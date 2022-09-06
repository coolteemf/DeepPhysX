from os import listdir, symlink, unlink, mkdir
from os.path import join, islink, abspath, pardir, isdir
from pathlib import Path
from sys import argv
from site import USER_SITE, getsitepackages
from shutil import rmtree

from config import check_repositories


# Check user entry
if len(argv) != 2 or argv[1] not in ['set', 'del']:
    print("\nInvalid script option."
          "\nRun 'python3 dev.py set' to link DPX to your site package."
          "\nRun 'python3 dev.py del' to remove DPX links in your site package.")
    quit()

# Check repositories names
check_repositories()

# Init DeepPhysX packages and dependencies to install
PROJECT = 'DeepPhysX'
packages = ['Core']
available = ['Torch', 'Sofa']
root = abspath(join(Path(__file__).parent.absolute(), pardir))
site_pkg_repo = join(getsitepackages()[0], PROJECT) if len(getsitepackages()) == 1 else join(USER_SITE, PROJECT)

# Option 1: create the symbolic links
if argv[1] == 'set':

    # Create main repository in site-packages
    if not isdir(site_pkg_repo):
        mkdir(site_pkg_repo)

    # Link to every existing packages
    for package_name in listdir(root):
        if package_name in available:
            packages.append(package_name)

    # Create symbolic links in site-packages
    for package_name in packages:
        if not islink(join(site_pkg_repo, package_name)):
            symlink(src=join(root, package_name, 'src'), dst=join(site_pkg_repo, package_name))
            print(f"Linked {join(site_pkg_repo, package_name)} -> {join(root, package_name, 'src')}")

# Option 2: remove the symbolic links
else:
    if isdir(site_pkg_repo):
        for package_name in listdir(site_pkg_repo):
            unlink(join(site_pkg_repo, package_name))
            print(f"Unlinked {join(site_pkg_repo, package_name)} -> {join(root, package_name, 'src')}")
        rmtree(site_pkg_repo)
