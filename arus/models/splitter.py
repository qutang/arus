from sklearn.model_selection import TimeSeriesSplit


class TimeSeriesEpisodeSplit:
    def __init__(self, fcs_df, fs_names, task_name, n_split=5):
        self.fs_names = fs_names
        self.task_name = task_name
        self.episodes = fcs_df.groupby(task_name)
        self.spliters = {}
        self.n_split = n_split
        for name, episode_df in self.episodes:
            X = episode_df.index
            spliter = TimeSeriesSplit(n_splits=self.n_split)
            self.spliters[name] = spliter.split(X)

    def split(self, X=None, y=None, groups=None):
        for i in range(self.n_split):
            train_idx = []
            test_idx = []
            for name, episode_df in self.episodes:
                train, test = next(self.spliters[name])
                train = episode_df.index[train]
                test = episode_df.index[test]
                train_idx += list(train)
                test_idx += list(test)
            train_idx = sorted(train_idx)
            test_idx = sorted(test_idx)
            yield train_idx, test_idx
