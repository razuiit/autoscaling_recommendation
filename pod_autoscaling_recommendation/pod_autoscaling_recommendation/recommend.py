import math
import warnings
from typing import Tuple

import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
test_accuracy = tf.keras.metrics.Accuracy()
from keras.models import load_model
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(0)
tf.random.set_seed(0)


class Config:
    # 定时任务最长长度（分钟）
    # max length of scheduled job
    MAX_SCHED_LEN = 240

    #Time steps for LSTM: Number of timestamps to observe to predict next classification
    TIMESTEPS = 5

    # 相对负载（CPU使用率）变化，相对变化 = （当前负载 – 前负载）/ 前负载
    # relative load change threshold
    RELATIVE_CHANGE = 0.2

    # 绝对负载（CPU使用率）变化，绝对变化 = 当前负载 – 前负载
    # absolute load change threshold
    ABSOLUTE_CHANGE = 0.03

    # 负载（CPU使用率）变化权重，动态CPU变化阈值 = CPU均值 * 负载（CPU使用率）变化权重
    # the quantile to select the spike threshold from the distribution of 1-slot change from left to right
    SPIKE_THRESHOLD_QUANTILE = 0.3

    # 定时任务频率，定时任务次数 = 天数 * 定时任务频率
    # regular job detection threshold: how many times a week should a spike occur to be considered a sched_job
    REG_JOB_DET_THRESH = 0.5

    # 每天数据量（目前为1440分钟）
    # granularity: how many samples in a day; 1440 means 1 minute/sample -> should be per service?
    NUM_DATA_POINTS = 1440

    # 输入数据中最少天数
    # minimum days for scheduled job detection
    MIN_DAYS = 2

    # 定时任务前后位移（目前为前后2分钟）
    # a window of slots to make a spike-flag fat over win samples before
    SHIFT = 2

    # 定时任务合并跨度（目前为30分钟）
    # loads between `JOB_MERGE_SPAN` will be merged into one job
    JOB_MERGE_SPAN = 30

    # 日均突增容忍次数
    # average daily burst tolerance
    BURSTS_PER_DAY = 3

    LOAD_COL = 'avg_cpu_util'


def plot_loads(sched_load_df: pd.DataFrame,
               norm_df: pd.DataFrame,
               title: str = "") -> None:
    # convert to scheduled job dict of {time_index: load}
    sched_jobs = sched_load_df['sched_load'].to_dict()

    total_load = pd.Series([load + sched_jobs.get(ts.hour * 60 + ts.minute, 0)
                            for i, (ts, load) in norm_df[['timestamp', Config.LOAD_COL]].iterrows()],
                           dtype=np.float)
    df = pd.concat([total_load, norm_df[Config.LOAD_COL]], axis=1)

    df.columns = ['sched_load', 'norm_load']

    f, ax = plt.subplots()

    ax.plot(df.index, df["sched_load"], color="red")
    ax.plot(df.index, df["norm_load"], color="blue")

    ax.set_title(f"{title}")
    #plt.show()

    # if True:
    #     img_buf = io.BytesIO()
    #     plt.savefig(img_buf, format='png')
    #     img_tag = await self.get_image_html_tag(img_buf)
    #     return img_tag


def __preprocess_dataset(df: pd.DataFrame,
                         num_data_points: int,
                         num_shift: int) -> Tuple[pd.Series, int]:
    """
    Preporcess df
    - fill more rows into `df` as full days;
    - shift by `offset`
    """
    # compute number of padding points
    num_padding = (num_data_points - len(df) % num_data_points) % num_data_points
    # use first day's data to pad
    if num_padding:
        print(f"append {num_padding} data points.")
        df = pd.concat([df, df.iloc[num_data_points - num_padding:num_data_points]])

    # shift df by `num_shift`
    df = pd.concat([df.iloc[num_shift:], df.iloc[:num_shift]]).reset_index(drop=True)

    return df, num_padding


def __detect_spikes(load: pd.Series,
                    relative_change: float,
                    absolute_change: float,
                    spike_threshold_quantile: int) -> np.ndarray:
    """
    Detect spikes in `load`, whose change is larger than `spike_threshold_quantile`.
    """
    # compute load threshold
    load_thresh = min(absolute_change, load.mean() * spike_threshold_quantile)

    # initialize spikes array with zeros
    spikes = np.zeros((len(load)), dtype=float)
    start_load = None
    for i in range(len(load)):
        # print(f"{i}    {str(start_load):<12}    {load.iloc[i]:<12}")

        # compute change between current load and
        #   - `start_load` if it is set;
        #   - or previous load if `start_load` is not set;
        prev = start_load or load.iloc[i - 1]
        change = load.iloc[i] - prev
        # enter a spike
        if change / prev >= relative_change and change >= load_thresh:
            # only set at the beginning of spike
            if start_load is None:
                # print(f"spike starts at {i}")
                # set `start_load` as the load just before the spike
                start_load = load.iloc[i - 1]
            # add load change
            spikes[i] = change

        # exit from a spike
        else:
            # print(f"spike ends at {i}")
            start_load = None

    return spikes


def __compute_masks(spikes: np.ndarray,
                    shift: int,
                    num_data_points: int,
                    freq_thresh=int) -> np.ndarray:
    """
    Compute scheduled job masks in a cycle of `num_data_points`.
    """
    # allow spikes within the `shift` to extend on both sides
    masks1 = pd.Series(np.concatenate([spikes[-shift * 2:], spikes]) > 0).rolling(window=shift * 2 + 1).max().dropna()
    # convert 1D to 2D
    masks1 = masks1.values.reshape(-1, num_data_points)
    # compute 1D masks by spike frequencies
    masks1 = masks1.sum(axis=0) >= freq_thresh

    # 1D masks without the `shift` extent on either side
    masks2 = (spikes > 0).reshape(-1, num_data_points).sum(axis=0) > 0

    # final `masks` meet both `masks1` and `masks2` conditions
    masks = np.logical_and(masks1, masks2)

    return masks


def __refine_masks(masks: np.ndarray,
                   max_sched_len: int) -> np.ndarray:
    """
    Remove long spikes, which is longer than `max_sched_len`.

    If `max_sched_len` is not given (None or 0), then return original masks.
    """
    if max_sched_len is None or max_sched_len == 0:
        return masks

    start = None
    for i, flag in enumerate(masks):
        if start is None and flag:
            start = i
        elif start is not None and not flag:
            if i - start > max_sched_len:
                masks[start:i + 1] = False
            start = None

    # remove last spike if it is long.
    if (start is not None) and (i - start > max_sched_len):
        masks[start:i + 1] = False

    return masks


def __split_norm_sched(df: pd.DataFrame,
                       masks: np.ndarray,
                       days: int,
                       num_data_points: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    - compute scheduled load;
    - compute normal df
        - split `load` into `norm_load` and `sched_load`;
    """
    load = df[Config.LOAD_COL].copy()
    # initialize `norm_load`
    norm_load = load.copy()
    # set load to nan when it is in spikes
    norm_load[np.tile(masks, days)] = np.nan
    # set linear values for normal load when it is in spikes
    norm_load = norm_load.interpolate(method='linear').fillna(method='bfill')
    # correct `norm_load` when it is larger than original `load`
    df[Config.LOAD_COL] = np.minimum(norm_load, load)

    # compute scheduled job load, and convert to 2D
    sched_load = (load - df[Config.LOAD_COL]).replace(0, np.nan).values.reshape(-1, num_data_points)
    # compute average of non-nan loads as daily scheduled job's load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sched_load = np.nanmean(sched_load, axis=0)
    print(f"{sum(masks)} scheduled job time-points are detected.")

    # convert to pd.Series
    sched_load = pd.Series(sched_load)
    sched_load = sched_load[masks].rename('sched_load')

    # total load at scheduled jobs
    total_load = pd.Series(load.values.reshape(-1, num_data_points).max(axis=0))
    total_load = total_load[masks].rename('load')

    sched_load_df = pd.concat([sched_load, total_load], axis=1)

    return sched_load_df, df

def nn_recommendation(df: pd.DataFrame, workloads: list[str]):
    #convert timestamp to float
    df['timestamp'] = (df['timestamp'] - df['timestamp'].iloc[0])/pd.to_timedelta('1Min')
    df["timestamp"] = pd.to_numeric(df["timestamp"], downcast="float")

    # split data
    data_test_static, data_valid_static, data_train_static = split_train_test_data(df['static'])
    data_test, data_valid, data_train = split_train_test_data(df.iloc[:,0:3])
    x_test_static = data_test_static[5:,:]
    x_valid_static = data_valid_static[5:,:]
    x_train_static = data_train_static[5:,:]


    num_init = 1
    k_mean_acc = []
    k_list = [Config.TIMESTEPS]

    # General Code for Testing Different TimeSteps and different models. Currently only one TimeStep and only one model
    for k in tqdm(k_list):
        X_train, y_train = create_lstm_data(data_train, k)
        X_valid, y_valid = create_lstm_data(data_valid, k)
        X_test, y_test = create_lstm_data(data_test, k)


        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_valid = scaler.transform(X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        # init_accs = []
        # for init in range(num_init):
        #     history = LSTM_model(X_train, y_train, X_valid, y_valid, k)
        #
        #     model = keras.models.load_model('best_model.h5')
        #     preds = model.predict(X_test)
        #     truePositives = np.argmax(preds, axis=1)
        #     truePositives = truePositives.flatten()
        #     truePositives = np.array_split(truePositives, len(workloads))
        #     for tp in truePositives:
        #         print(np.count_nonzero(tp))
        #
        #     cur_acc = np.round(np.mean(np.equal(np.argmax(preds, axis=1), y_test)), 3)
        #     # cur_acc=test_accuracy(preds, y_test)
        #
        #     print("Test set accuracy: {:.3%}".format(cur_acc))
        #
        #     #     init_accs.append(cur_acc)
        #     # k_mean_acc.append(np.mean(init_accs))
        #     #
        #     # plt.plot(k_list, k_mean_acc)
        #     # plt.title('LSTM Accuracy Depending on K-Length Sequence')
        #     # k_list[np.argmax(k_mean_acc)], np.round(np.max(k_mean_acc), 3)
        #     # # #plt.show()

        ########################################################################################################################

        # model = build_model(df, x_train_static, X_train)
        #
        # history = model.fit(
        #     [np.asarray(X_train).astype('float32'), np.asarray(x_train_static).astype('float32')],
        #
        #     y_train, epochs=200, verbose=0, validation_data=(
        #     [np.asarray(X_valid).astype('float32'), np.asarray(x_valid_static).astype('float32')], y_valid))
        #
        # # summarize history for accuracy
        #
        # # save model to single file
        # model.save('lstm_model.h6')


        # load model from single file
        model = load_model('lstm_model.h6') # change h5 to h6 if trained again
        # make predictions
        input = [np.asarray(X_test).astype('float32'),np.asarray(x_test_static).astype('float32')]
        preds = model.predict(input, verbose=0)
        preds = (preds > .5).astype(int)
        truePositives = np.array_split(preds, len(workloads))
        for tp in truePositives:
            print(np.count_nonzero(tp))


#######################################################################################################################



    # evaluate model

    # loss, accuracy, f1_score, precision, recall = model.evaluate(
    #     [np.asarray(x_train.values.reshape(-1, 1440, 2)).astype('float32'), np.asarray(x_test_static).astype('float32')], y_test,
    #     batch_size=Config.NUM_DATA_POINTS, verbose=0)
    #
    # # print output
    #
    # print("Accuracy:{} , Precision:{}, Recall:{}".format(accuracy, precision, recall))


def create_lstm_data(data, k):
    '''
    input:
        data - the numpy matrix of (n, p+1) shape, where n is the number of rows,
               p+1 is the number of predictors + 1 target column
        k    - the length of the sequence, namely, the number of previous rows
               (including current) we want to use to predict the target.
    output:
        X_data - the predictors numpy matrix of (n-k, k, p) shape
        y_data - the target numpy array of (n-k, 1) shape
    '''
    # initialize zero matrix of (n-k, k, p) shape to store the n-k number
    # of sequences of k-length and zero array of (n-k, 1) to store targets
    X_data = np.zeros([data.shape[0] - k, k, data.shape[1] - 1])
    y_data = []

    # run loop to slice k-number of previous rows as 1 sequence to predict
    # 1 target and save them to X_data matrix and y_data list
    for i in range(k, data.shape[0]):
        cur_sequence = data[i - k: i, :-1]
        cur_target = data[i - 1, -1]

        X_data[i - k, :, :] = cur_sequence.reshape(1, k, X_data.shape[2])
        y_data.append(cur_target)

    return X_data, np.asarray(y_data)

def LSTM_model(X_train, y_train, X_valid, y_valid, k):
    # Define and compile LSTM model
    model = Sequential()
    model.add(LSTM(100, input_shape=(k, X_train.shape[2])))
    model.add(Dense(2, activation='sigmoid'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='binary_crossentropy',  metrics=['accuracy'])
    # Early stopping and best model checkpoint parameters
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)
    # Train the model
    history = model.fit(X_train, tf.one_hot(y_train, depth=2),
                        validation_data=(X_valid, tf.one_hot(y_valid, depth=2)),
                        epochs=200, verbose=0, callbacks=[es, mc])
    return history

def build_model(df, x_train_static, x_train) -> tf.keras.Model:

    recurrent_input = tf.keras.Input(shape=(x_train.shape[1], x_train.shape[2]), name="TIMESERIES_INPUT")
    static_input = tf.keras.Input(shape=(x_train_static.shape[1],), name="STATIC_INPUT")
    # RNN Layers
    # layer - 1
    rec_layer_one = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, kernel_regularizer=tf.keras.regularizers.l2(0.01),
                             recurrent_regularizer=tf.keras.regularizers.l2(0.01), return_sequences=True),
        name="BIDIRECTIONAL_LAYER_1")(recurrent_input)
    rec_layer_one = tf.keras.layers.Dropout(0.1, name="DROPOUT_LAYER_1")(rec_layer_one)
    # layer - 2
    rec_layer_two = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, kernel_regularizer=tf.keras.regularizers.l2(0.01),
                             recurrent_regularizer=tf.keras.regularizers.l2(0.01)), name="BIDIRECTIONAL_LAYER_2")(
        rec_layer_one)
    rec_layer_two = tf.keras.layers.Dropout(0.1, name="DROPOUT_LAYER_2")(rec_layer_two)
    # SLP Layers
    static_layer_one = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu',
                                             name="DENSE_LAYER_1")(static_input)
    # Combine layers - RNN + SLP
    combined = tf.keras.layers.Concatenate(axis=1, name="CONCATENATED_TIMESERIES_STATIC")(
        [rec_layer_two, static_layer_one])
    combined_dense_two = tf.keras.layers.Dense(64, activation='relu', name="DENSE_LAYER_2")(combined)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name="OUTPUT_LAYER")(combined_dense_two)
    # Compile Model
    model = tf.keras.Model(inputs=[recurrent_input, static_input], outputs=[output])
    # binary cross entropy loss
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # # focal loss
    #
    # def focal_loss_custom(alpha, gamma):
    #
    # def binary_focal_loss(y_true, y_pred):
    #
    # fl = tf.losses.SigmoidFocalCrossEntropy(alpha=alpha, gamma=gamma)
    #
    # y_true_K = K.ones_like(y_true)
    #
    # focal_loss = fl(y_true, y_pred)
    #
    # return focal_loss
    #
    # return binary_focal_loss
    #
    # model.compile(loss=focal_loss_custom(alpha=0.2, gamma=2.0), optimizer='adam',
    #               metrics=['accuracy', f1_m, precision_m, recall_m])
    print(model.summary())
    return model


def split_train_test_data(df):
    test_set = pd.DataFrame([])
    train_set = pd.DataFrame([])
    valid_set = pd.DataFrame([])
    insert = 0
    df_days = df.groupby(df.index // Config.NUM_DATA_POINTS)
    for num_day, df_day in df_days:
        insert = insert + 1
        if (insert == 5):
            valid_set = pd.concat([valid_set, df_day], ignore_index=True, axis=0)
        elif (insert == 6 or insert == 7):
            test_set = pd.concat([test_set, df_day], ignore_index=True, axis=0)
            if insert == 7:
                insert = 0
        else:
            train_set = pd.concat([train_set, df_day], ignore_index=True, axis=0)
    return test_set.to_numpy(), valid_set.to_numpy(), train_set.to_numpy()


def __merge_jobs(sched_load_df: pd.DataFrame) -> pd.DataFrame:
    start, end, sched_loads, loads = None, None, [], []
    jobs = []
    for i, (sched_load, load) in sched_load_df.iterrows():
        if start is None:
            start = i
            end = i
            sched_loads.append(sched_load)
            loads.append(load)
        elif i - end <= Config.JOB_MERGE_SPAN:
            end = i
            sched_loads.append(sched_load)
            loads.append(load)
        else:
            jobs.append((start, end, max(sched_loads), max(loads)))
            start = i
            end = i
            sched_loads = [sched_load]
            loads = [load]
    if start:
        jobs.append((start, end, max(sched_loads), max(loads)))

    sched_jobs_df = pd.DataFrame(jobs, columns=['start', 'end', 'sched_load', 'load'])

    return sched_jobs_df


def __postprocess(norm_df: pd.DataFrame,
                  sched_load_df: pd.DataFrame,
                  num_shift: int,
                  num_padding: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Post-precess:
    - convert to scheduled job dict of {time_index: load};
    - shift back as input dataset by `num_shift`, and remove padding rows.
    """
    sched_jobs_df = __merge_jobs(sched_load_df)

    # shift back as input dataset by `num_shift`
    norm_df = pd.concat([norm_df.iloc[-num_shift:], norm_df.iloc[:-num_shift]]).reset_index(drop=True)
    # remove padding rows
    if num_padding:
        norm_df = norm_df.iloc[:-num_padding]

    return sched_jobs_df, norm_df


def __detect_scheduled_jobs(df: pd.DataFrame,
                            max_sched_len: int = Config.MAX_SCHED_LEN,
                            relative_change: float = Config.RELATIVE_CHANGE,
                            absolute_change: float = Config.ABSOLUTE_CHANGE,
                            spike_threshold_quantile: int = Config.SPIKE_THRESHOLD_QUANTILE,
                            reg_job_det_thresh: int = Config.REG_JOB_DET_THRESH,
                            num_data_points: int = Config.NUM_DATA_POINTS,
                            min_days: int = Config.MIN_DAYS,
                            shift: int = Config.SHIFT) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # no enough data to detect scheduled jobs
    if len(df) < num_data_points * min_days:
        print(f"no enough data to detect scheduled jobs!")
        return {}, df

    # compute `offset`
    start_ts = df['timestamp'].min()
    offset = start_ts.hour * 60 + start_ts.minute

    # compute number of point to shift based on `offset`
    num_shift = (num_data_points - offset) % num_data_points

    # number of days
    days = math.ceil(len(df) / num_data_points)

    # compute spike frequency threshold
    freq_thresh = round(reg_job_det_thresh * days)

    df, num_padding = __preprocess_dataset(df, num_data_points, num_shift)

    spikes = __detect_spikes(df[Config.LOAD_COL], relative_change, absolute_change, spike_threshold_quantile)

    masks = __compute_masks(spikes, shift, num_data_points, freq_thresh)

    masks = __refine_masks(masks, max_sched_len)

    sched_load_df, norm_df = __split_norm_sched(df, masks, days, num_data_points)

    sched_jobs_df, norm_df = __postprocess(norm_df, sched_load_df, num_shift, num_padding)

    return sched_jobs_df, norm_df, sched_load_df


def __elastic_recommend(df: pd.DataFrame) -> bool:

    # count small spikes per each 2 hours where spike size is larger than 0.02
    diff = df.groupby(df.index // 5).agg({Config.LOAD_COL: np.ptp})
    diff = pd.DataFrame(diff.values.reshape(-1, 84, order='F'))
    small_spikes = (diff[diff > 0.02].count(axis=0) >= 3).sum()

    # compute difference in mean of load per day
    mean = df.groupby(df.index // Config.NUM_DATA_POINTS).agg({Config.LOAD_COL: 'mean'})
    mean_change_count = 0
    i = 0
    for prev, curr in zip(mean[Config.LOAD_COL].iloc[:-1], mean[Config.LOAD_COL].iloc[1:]):
        change = abs(curr - prev)
        if change / prev >= Config.RELATIVE_CHANGE:
            mean_change_count += 1
        i += 1


    # diff =  (df[Config.LOAD_COL].rolling(5).min() - df[Config.LOAD_COL].rolling(5).max()).abs()
    # diff = pd.DataFrame(diff.values.reshape(-1, 120, order='F'))
    ## compute small spikes
    # small_spikes = (diff[diff > 0.02].count(axis=0) >= 5).sum()

    thresh = len(df) / Config.NUM_DATA_POINTS * Config.BURSTS_PER_DAY

    # return  large_change_count < thresh
    return  ((small_spikes * mean_change_count) < thresh)




def __preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={'时间': 'timestamp', '平均CPU使用率': 'avg_cpu_util', '最大CPU使用率': 'max_cpu_util'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S')  # .tz_convert('Asia/Shanghai')
    df = df.astype({'avg_cpu_util': float, 'max_cpu_util': float})
    df = df.loc[:, ['timestamp', Config.LOAD_COL]]

    # fill missing values
    df[Config.LOAD_COL] = df[Config.LOAD_COL].interpolate(method='linear')

    return df


def recommend(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    df = __preprocess(df)
    sched_jobs_df, norm_df, sched_load_df = __detect_scheduled_jobs(df)
    elastic_decision = __elastic_recommend(norm_df)
    return sched_jobs_df, norm_df, sched_load_df, elastic_decision


if __name__ == '__main__':
    from pathlib import Path

    # DATA_DIR = Path(str(Path.cwd()) + '/filestore/data/json_data')
    DATA_DIR = Path('../filestore/data/json_data')

    workloads = [
        'w-wisecloudstoragedistservice-contenthota1',
        'w-wisecloudnspobsservice-nsp2bturbo',
        'w-wisecloudabtestdispatchservice-runtime',
        'h-kidwatchchildcloudserverservice-product',
        'w-wisecloudcdnschedulerworkerservice-cnhispace',
        's-v2smarthomemessagecenter-turboproduct',
        's-scenariomanage2cservice-liven',
    ]

    full_df = pd.DataFrame([])
    index = 1
    for workload in workloads:
        print(workload)

        _df = pd.read_json(DATA_DIR / f"{workload}.json", convert_dates=['时间'])

        _sched_jobs_df, _norm_df, _sched_load_df, _elastic_decision = recommend(_df)

        plot_loads(_sched_load_df, _norm_df, f"Workload: {workload}, 适用于弹性伸缩：{_elastic_decision}")
        print(_sched_jobs_df)
        print(_elastic_decision)
        if index == 7: _elastic_decision = False
        _norm_df['labels'] = int(_elastic_decision == True)
        _norm_df['static'] = index
        index= index + 1
        full_df = full_df.append(_norm_df, ignore_index=True)

    nn_recommendation(full_df, workloads)
    print('done')
