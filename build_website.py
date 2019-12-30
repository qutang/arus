import subprocess
import shutil
import os

print('remove old')
shutil.rmtree('./docs/build', ignore_errors=True)
shutil.rmtree('./docs/source/generated', ignore_errors=True)

print('building documentations')
# build docs
cwd = os.path.join(os.getcwd(), 'docs')
subprocess.run("make html", shell=True, cwd=cwd)
