

import subprocess
import sys
from dephell_versioning import bump_file, bump_version
from pathlib import Path
import arus


def bump_news_version(new_version):
    new_lines = []
    with open('news.md', 'r') as f:
        lines = f.readlines()
        lines[0] = '# ' + new_version + '\n'
        new_lines = lines
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
    bump_news_version(new_version)

    print('Commit current state')
    subprocess.run('git add .')
    subprocess.run('git commit -m "Bump version to ' + new_version + '"')

    print('Tag version')
    if len(sys.argv) == 3:
        subprocess.run('git tag -a v' + new_version +
                       ' -m ' + '"' + sys.argv[2] + '"')
    else:
        subprocess.run('git tag -a v' + new_version)

    print('Push tag')
    subprocess.run('git push')
    subprocess.run('git push origin v' + new_version)
else:
    exit(0)
