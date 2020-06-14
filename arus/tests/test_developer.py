from .. import developer
from loguru import logger


class TestGit:
    def test_get_tags(self):
        tags = developer.get_git_tags()
        logger.debug(tags)
        assert tags[-1] == '0.4.0'

    def test_generate_changelogs(self):
        a = developer.generate_changelogs()
        assert a
