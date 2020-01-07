

import subprocess
import sys
from dephell_versioning import bump_file, bump_version
from pathlib import Path
import arus
import os


def add_news_dev_version(new_version):
    new_lines = []
    with open('news.md', 'r') as f:
        lines = f.readlines()
        new_lines = ['# ' + new_version + '\n\n'] + lines
    with open('news.md', 'w') as f:
        f.writelines(new_lines)


if sys.argv[1] == 'major' or sys.argv[1] == 'minor' or sys.argv[1] == 'patch':
    new_version = bump_version(
        version=arus.__version__, rule=sys.argv[1], scheme='semver')
else:
    new_version = sys.argv[1]

print('new version is: ' + new_version)
confirm = input("Confirm to continue [y/n]?")

if confirm.lower() == 'y':
    print('Bump package version')
    subprocess.run("poetry version " + new_version, shell=True)

    print('Modify package version file')
    bump_file(path=Path('arus', '__init__.py'),
              old=arus.__version__, new=new_version)

    print('Modify news page version')
    add_news_dev_version(new_version)
else:
    exit(0)
