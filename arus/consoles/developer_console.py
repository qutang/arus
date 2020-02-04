import click
from .. import developer


@click.group()
def main():
    developer.set_default_logging()
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


@main.command(short_help='build arus apps')
@click.argument('root')
@click.argument('name')
@click.argument('version')
def app(root, name, version):
    developer.build_arus_app(root, name, version)


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
