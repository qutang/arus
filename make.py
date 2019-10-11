import subprocess
import shutil

print('building documentations')
# build docs
subprocess.run("pdoc --html --html-dir docs .",
               capture_output=True, shell=True)

# move directory
subprocess.run('mv ./docs/arus/* ./docs/', shell=True, capture_output=True)

# clean up
subprocess.run('rmdir ./docs/arus', shell=True, capture_output=True)

print('packaging...')

subprocess.run("python setup.py sdist bdist_wheel",
               shell=True, capture_output=True)
