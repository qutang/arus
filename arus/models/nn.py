from .. import mhealth_format as mh
from .. import extensions as ext
from .. import moment
import pandas as pd


def preprocess_processor(chunk_list, **kwargs):
    import pandas as pd
    from .. import dataset
    from .nn import preprocess
    results = []
    for df, st, et, prev_st, prev_et, name in chunk_list:
        if name in mh.SENSOR_PLACEMENTS:  # process sensors
            if df.empty:
                continue
            else:
                result = preprocess(
                    df, name=name, sr=kwargs[name]['sr'], st=st, et=et)
                if result is not None:
                    results.append(result)
    if len(results) == 0:
        combined_df = pd.DataFrame()
    elif len(results) < len(chunk_list):
        combined_df = pd.DataFrame()
    else:
        combined_df = pd.concat(results, axis=1, sort=False, )
    return combined_df


def preprocess(df, name, sr, st, et):
    beginning_delay = moment.Moment.get_duration(st, df.iloc[0, 0], unit='s')
    end_delay = moment.Moment.get_duration(df.iloc[-1, 0], et, unit='s')
    if beginning_delay > 1 or end_delay > 1:
        return None
    col_names = ['TS', name + '_0', name + '_1', name + '_2']
    ts = moment.Moment.seq_to_unix_timestamp(df['HEADER_TIME_STAMP'].values)
    X = df.iloc[:, 1:].values
    st = moment.Moment(st).to_unix_timestamp()
    et = moment.Moment(et).to_unix_timestamp()
    new_ts, new_X = ext.numpy.regularize_sr(ts, X, sr, st=st, et=et)
    new_df = pd.DataFrame(index=new_ts, data=new_X).reset_index(drop=False)
    new_df.columns = col_names
    new_df = new_df.set_index('TS')
    return new_df
