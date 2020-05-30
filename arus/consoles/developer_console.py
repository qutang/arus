import click
import subprocess
import os
from .. import developer


@click.group()
def main():
    pass


@main.command(short_help='bump package version')
@click.argument('root')
@click.argument('name')
@click.argument('nver')
@click.option('-d', '--dev', is_flag=True, default=False, help='If it is a development version')
@click.option('-r', '--release', is_flag=True, default=False, help='If tag and release the version to github')
def bump(root, name, nver, dev=False, release=False):
    new_version = developer.bump_package_version(root, name, nver, dev)
    if new_version is not None and developer.command_is_available('git') and release:
        developer.commit_repo("Bump version to {}" % new_version)
        developer.tag_repo(new_version)
        developer.push_repo()
        developer.push_tag(new_version)


@main.command(short_help='make documentation website')
@click.argument('root')
@click.option('-f', '--folder', default='docs', help='The folder storing sphinx docs')
def website(root, folder):
    developer.make_sphinx_website(root, folder)


@main.group(short_help='build or run arus apps')
@click.pass_context
def app(ctx):
    ctx.ensure_object(dict)


@app.command(short_help='build app')
@click.argument('root')
@click.argument('name')
@click.argument('version')
@click.pass_context
def build(ctx, root, name, version):
    developer.build_arus_app(root, name, version)


@app.command(short_help='run app')
@click.argument('root')
@click.argument('name')
@click.pass_context
def run(ctx, root, name):
    subprocess.run(['python', os.path.join(
        root, 'apps', name, 'main.py')], shell=True)


@main.group(short_help="manipulate dataset")
@click.argument('name')
@click.pass_context
def dataset(ctx, name):
    ctx.ensure_object(dict)
    ctx.obj['name'] = name


@dataset.command(short_help='compress dataset')
@click.argument('src')
@click.argument('dest')
@click.pass_context
def compress(ctx, src, dest):
    developer.compress_dataset(src, dest, ctx.obj['name'] + '.tar.gz')
