"""
Utilities for developers. 

1. Zip and release datasets to github.
2. Bump package versions.
3. Build packages and apps.
4. Control logging handlers.

Author: Qu Tang
Date: 01/30/2020
License: GNU v3
"""
import tarfile
import os
import logging

import dephell_versioning as deph
import subprocess
import pathlib
import shutil
import importlib


def compress_dataset(source_dir, out_dir, out_name):
    os.makedirs(out_dir, exist_ok=True)
    logging.info('Use Python tar module to compress dataset...')

    def exclude(tarinfo):
        if 'DerivedCrossParticipants' in tarinfo.name:
            return None
        else:
            return tarinfo
    with tarfile.open(os.path.join(out_dir, out_name), "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(
            source_dir), filter=exclude)

    logging.info('Compression is completed.')


def command_is_available(cmd):
    status, output = subprocess.getstatusoutput(cmd)
    if status == 0:
        return True
    else:
        return False


def _find_current_version(root, name):
    package_module = importlib.import_module(name)
    if hasattr(package_module, '__version__'):
        return package_module.__version__
    if command_is_available('poetry') and os.path.exists(os.path.join(root, 'pyproject.toml')):
        return subprocess.getoutput('poetry version').split(' ')[1]


def _bump_news_version(new_version, dev_version=False):
    new_lines = []
    with open('news.md', 'r') as f:
        lines = f.readlines()
        if dev_version:
            new_lines = ['# ' + new_version + '\n\n'] + lines
        else:
            lines[0] = '# ' + new_version + '\n'
            new_lines = lines
    with open('news.md', 'w') as f:
        f.writelines(new_lines)


def bump_package_version(root, name, nver, dev=False):
    """Bump package version

    Args:
        root (str): The root path of the package
        name (str): The name of the package
        nver (str): The new version of the package
        dev (bool, optional): Whether it is a development version. Defaults to False.

    Returns:
        str or None: New version or None
    """
    cver = _find_current_version(root, name)
    if nver in ['major', 'minor', 'patch']:
        nver = deph.bump_version(
            version=cver, rule=nver, scheme='semver')
    if dev:
        nver = nver + '+9000'
    logging.info('new version is ' + nver)
    confirm = input("Confirm to continue [y/n]?")
    if confirm.lower() == 'y':
        if command_is_available('poetry') and os.path.exists(os.path.join(root, 'pyproject.toml')):
            logging.info('Bump poetry package version')
            subprocess.run("poetry version " + nver, shell=True, cwd=root)

        if os.path.exists(os.path.join(root, name, '__init__.py')):
            logging.info('Bump package version file')
            deph.bump_file(path=pathlib.Path('arus', '__init__.py'),
                           old=cver, new=nver)

        if os.path.exists(os.path.join(root, 'news.md')):
            logging.info('Update news')
            _bump_news_version(nver, dev_version=dev)
        return nver
    else:
        return None


def commit_repo(message, repo_root=None):
    assert command_is_available('git')
    repo_root = repo_root or os.getcwd()
    logging.info('Commit with message: ' + message)
    subprocess.run('git add .', cwd=repo_root, shell=True)
    subprocess.run('git commit -m "{}"'.format(message),
                   cwd=repo_root, shell=True)


def tag_repo(version, message=None, repo_root=None):
    assert command_is_available('git')
    repo_root = repo_root or os.getcwd()
    message = message or "Tag with version {}".format(version)
    logging.info(
        'Tag repo with version {} and message: {}'.format(version, message))
    subprocess.run('git tag -a v{} -m {}'.format(version,
                                                 message), cwd=repo_root, shell=True)


def push_repo(branch='master', repo_root=None):
    assert command_is_available('git')
    repo_root = repo_root or os.getcwd()
    subprocess.run('git push origin {}'.format(
        branch), cwd=repo_root, shell=True)


def push_tag(version, repo_root=None):
    assert command_is_available('git')
    repo_root = repo_root or os.getcwd()
    logging.info('Push tag v{}'.format(version))
    subprocess.run('git push origin v' + version, cwd=repo_root, shell=True)


def make_sphinx_website(repo_root, docs_folder):
    logging.info('Clean up old documentation website')
    shutil.rmtree(os.path.join(repo_root, docs_folder,
                               'build'), ignore_errors=True)
    shutil.rmtree(os.path.join(repo_root, docs_folder,
                               'generated'), ignore_errors=True)
    shutil.rmtree(os.path.join(repo_root, docs_folder,
                               'source', 'examples'), ignore_errors=True)

    logging.info('Making sphinx documentation website')
    # build docs
    cwd = os.path.join(repo_root, docs_folder)
    subprocess.run("make html", shell=True, cwd=cwd)


def build_arus_app(root, app_name, version):
    logging.info('Build {}'.format(app_name))
    shutil.rmtree('./apps/{}/build'.format(app_name), ignore_errors=True)
    shutil.rmtree('./apps/{}/dist'.format(app_name), ignore_errors=True)

    cwd = os.path.join(root, 'apps', app_name)
    subprocess.run("pyinstaller main.spec", shell=True, cwd=cwd)

    app_path = os.path.join(os.getcwd(), 'apps', app_name, 'releases',
                            app_name + '_' + version + '.zip')

    os.makedirs(os.path.dirname(app_path), exist_ok=True)
    cwd = os.path.join(os.getcwd(), 'apps', app_name, 'dist', app_name)

    subprocess.run("zip -r " + app_path +
                   " ./", shell=True, cwd=cwd)
    shutil.rmtree('./apps/{}/build'.format(app_name), ignore_errors=True)


def set_default_logging():
    logging.basicConfig(
        level=logging.INFO, format='[%(levelname)s]%(asctime)s <P%(process)d-%(threadName)s> %(message)s')
