import subprocess
import shutil
import os

print('build arus demo app')
shutil.rmtree('./apps/arus_demo/build', ignore_errors=True)
shutil.rmtree('./apps/arus_demo/dist', ignore_errors=True)

cwd = os.path.join(os.getcwd(), 'apps', 'arus_demo')
subprocess.run("pyinstaller main.spec", shell=True, cwd=cwd)
