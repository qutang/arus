"""
Utilities for developers.

1. Zip and release datasets to github.
2. Bump package versions.
3. Build packages and apps.
4. Control logger handlers.

Author: Qu Tang
Date: 01/30/2020
License: GNU v3
"""
import tarfile
import os
from loguru import logger
import sys
import subprocess
import pathlib
import shutil
import importlib
import pprint
import re

from . import mhealth_format as mh


def _print_extra_dep_warning(e):
    msg = (
        "Arus developer module requirements are not installed. Some functionality may not be working.\n\n"
        "To use the full functionality, please install the metawear extra packages as follows:\n\n"
        "  pip install arus[dev]\n\n"
    )
    print(str(e) + "\n\n" + msg)


def compress_dataset(source_dir, out_dir, out_name):
    os.makedirs(out_dir, exist_ok=True)
    if command_is_available('tar --version'):
        logger.info('Use tar to compress dataset...')
        output_path = os.path.join(out_dir, out_name)
        subprocess.run(
            f'tar --exclude=".git" --exclude="DerivedCrossParticipants" --exclude=".gitignore" -zcvf {output_path} -C {source_dir} *', shell=True)
    else:
        logger.info('Use Python tar module to compress dataset...')

        def exclude(tarinfo):
            if 'DerivedCrossParticipants' in tarinfo.name or '.git' in tarinfo.name or '.gitignore' in tarinfo.name:
                return None
            else:
                return tarinfo
        with tarfile.open(os.path.join(out_dir, out_name), "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(
                source_dir), filter=exclude)

        logger.info('Compression is completed.')


def command_is_available(cmd):
    status, output = subprocess.getstatusoutput(cmd + " --version")
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
    try:
        deph = importlib.import_module('dephell_versioning')
    except ImportError as e:
        _print_extra_dep_warning(e)
        raise ImportError(e)
    cver = _find_current_version(root, name)
    if nver in ['major', 'minor', 'patch']:
        nver = deph.bump_version(
            version=cver, rule=nver, scheme='semver')
    if dev:
        nver = nver + '+9000'
    logger.info('new version is ' + nver)
    confirm = input("Confirm to continue [y/n]?")
    if confirm.lower() == 'y':
        if command_is_available('poetry') and os.path.exists(os.path.join(root, 'pyproject.toml')):
            logger.info('Bump poetry package version')
            subprocess.run("poetry version " + nver, shell=True, cwd=root)

        if os.path.exists(os.path.join(root, name, '__init__.py')):
            logger.info('Bump package version file')
            deph.bump_file(path=pathlib.Path('arus', '__init__.py'),
                           old=cver, new=nver)
        return nver
    else:
        return None


def commit_repo(message, repo_root=None):
    assert command_is_available('git')
    repo_root = repo_root or os.getcwd()
    logger.info('Commit with message: ' + message)
    subprocess.run('git add .', cwd=repo_root, shell=True)
    subprocess.run('git commit -m "{}"'.format(message),
                   cwd=repo_root, shell=True)


def tag_repo(version, message=None, repo_root=None):
    assert command_is_available('git')
    repo_root = repo_root or os.getcwd()
    message = message or "Tag with version {}".format(version)
    logger.info(
        'Tag repo with version {} and message: {}'.format(version, message))
    subprocess.run(
        f'git tag -a v{version} -m "{message}"', cwd=repo_root, shell=True)


def push_repo(branch='master', repo_root=None):
    assert command_is_available('git')
    repo_root = repo_root or os.getcwd()
    subprocess.run('git push origin {}'.format(
        branch), cwd=repo_root, shell=True)


def push_tag(version, repo_root=None):
    assert command_is_available('git')
    repo_root = repo_root or os.getcwd()
    logger.info('Push tag v{}'.format(version))
    subprocess.run('git push origin v' + version, cwd=repo_root, shell=True)


def build_arus_app(root, app_name, version):
    logger.info('Build {}'.format(app_name))
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


def _copy_files_for_website():
    src_files = [
        "README.md",
        "LICENSE.md",
        "CODE_OF_CONDUCT.md"
    ]
    dest_files = list(map(lambda f: os.path.join('docs', f), src_files))
    for src_file, dest_file in zip(src_files, dest_files):
        src_file = os.path.join('.', src_file)
        dest_file = os.path.join('.', dest_file)
        if os.path.exists(dest_file):
            os.remove(dest_file)
        logger.info(f'Copy {os.path.basename(src_file)} to docs folder')
        shutil.copyfile(src_file, dest_file)


def dev_website():
    _copy_files_for_website()
    generate_changelogs()
    subprocess.run("mkdocs serve -v", shell=True)


def build_website():
    _copy_files_for_website()
    generate_changelogs()
    subprocess.run("mkdocs gh-deploy -v", shell=True)


def get_git_tags():
    try:
        semver = importlib.import_module('semver')
    except ImportError as e:
        _print_extra_dep_warning(e)
        raise ImportError(e)
    tags = subprocess.check_output(
        " ".join(['git', 'tag']), shell=True, encoding='utf-8', text=True, universal_newlines=True)
    tags = tags.split('\n')
    tags = list(map(lambda tag: tag.strip('v'), filter(
        lambda tag: tag.startswith('v'), tags)))
    tags = sorted(tags, key=semver.VersionInfo.parse, reverse=True)
    return tags


def generate_changelogs(by_version=True):
    tags = get_git_tags()
    stop_tag = 'HEAD'
    if by_version:
        for start_tag in tags:
            start_tag = f'v{start_tag}'
            changelogs = subprocess.check_output(
                " ".join(['git', 'log', f'{start_tag}...{stop_tag}', '--pretty=format:%s-%h']), shell=True, encoding='utf-8',
                text=True, universal_newlines=True).split('\n')
            parsed = parse_changelogs(changelogs)
            write_changelog_to_file(parsed, start_tag, stop_tag)
            stop_tag = start_tag
    write_changelog_index_to_file(tags)
    return True


def parse_changelogs(changelogs):
    parsed = {}
    category_map = {
        'api': "API changes",
        'feat': "Features",
        'refactor': "Refactors",
        'fix': "Bug fixes",
        'test': "Test cases"
    }
    for changelog in changelogs:
        tokens = changelog.split('-')
        if len(tokens) < 2:
            logger.warning(f'Commit {changelog} is invalid, ignore it')
            continue
        msg = tokens[0]
        commit = tokens[1]
        category, scope, title = parse_conventional_commit(msg)
        category = category.lower()
        if category == "" or category not in category_map:
            logger.warning(
                f'Commit {commit} is not a conventional commit, ignore it')
            continue
        category = category_map[category]
        if category in parsed:
            parsed[category].append(
                {'scope': scope, 'title': title, 'commit': commit})
        else:
            parsed[category] = [
                {'scope': scope, 'title': title, 'commit': commit}]
    return parsed


def write_changelog_index_to_file(tags):
    item_template = "* [{tag}]({tag}/)\n"
    with open(os.path.join('.', 'docs', 'changelogs', 'index.md'), 'w', encoding='utf-8') as f:
        f.write(f'# Changelogs (Latest stable version: {tags[0]})\n\n')
        f.write(item_template.format(tag='dev'))
        for tag in tags:
            tag = f'v{tag}'
            f.write(item_template.format(tag=tag))


def write_changelog_to_file(parsed_changelogs, start_tag, stop_tag):
    template = "## {category}"

    item_template = "* {title} [{commit}](https://github.com/qutang/arus/commit/{commit}) (**{scope}**)"
    item_template2 = "* {title} [{commit}](https://github.com/qutang/arus/commit/{commit})"

    category_order = ['API changes',
                      'Features',
                      'Bug fixes',
                      'Refactors',
                      'Test cases']

    if stop_tag == 'HEAD':
        stop_tag = 'dev'

    os.makedirs(os.path.join('.', 'docs', 'changelogs'), exist_ok=True)
    with open(os.path.join('.', 'docs', 'changelogs', f'{stop_tag}.md'), 'w', encoding='utf-8') as f:
        f.write(f'# {stop_tag.upper()}\n')
        for category in category_order:
            if category not in parsed_changelogs:
                continue
            f.write(template.format(category=category))
            f.write('\n')
            for item in parsed_changelogs[category]:
                if item['scope'] == '':
                    f.write(item_template2.format(
                        title=item['title'], commit=item['commit']))
                else:
                    f.write(item_template.format(
                        scope=item['scope'], title=item['title'], commit=item['commit']))
                f.write('\n')


def parse_conventional_commit(msg):
    tokens = msg.split(':')
    if len(tokens) == 1:
        return "", "", ""
    category_scope = tokens[0]
    title = tokens[1].strip(' ')
    sub_tokens = re.findall(r'\w+', category_scope)
    category = sub_tokens[0]
    if len(sub_tokens) > 1:
        scope = sub_tokens[1]
    else:
        scope = ""
    return category, scope, title


def logging_dict(data):
    info = pprint.pformat(data, width=1)
    logger.info(info)


def logging_st_and_et(st, et):
    st_str = st.strftime('%Y-%m-%d %H:%M:%S.%f')
    et_str = et.strftime('%Y-%m-%d %H:%M:%S.%f')
    logger.info("{} - {}".format(st_str, et_str))
