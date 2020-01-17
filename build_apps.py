import subprocess
import shutil
import os
import arus

print('build arus demo app')
shutil.rmtree('./apps/arus_demo/build', ignore_errors=True)
shutil.rmtree('./apps/arus_demo/dist', ignore_errors=True)

cwd = os.path.join(os.getcwd(), 'apps', 'arus_demo')
subprocess.run("pyinstaller main.spec", shell=True, cwd=cwd)

app_name = os.path.join(os.getcwd(), 'apps', 'arus_demo', 'releases',
                        'arus_demo_' + arus.__version__ + '.zip')

os.makedirs(os.path.dirname(app_name), exist_ok=True)
cwd = os.path.join(os.getcwd(), 'apps', 'arus_demo', 'dist', 'arus_demo')

subprocess.run("zip -r " + app_name +
               " ./", shell=True, cwd=cwd)

shutil.rmtree('./apps/arus_demo/build', ignore_errors=True)
