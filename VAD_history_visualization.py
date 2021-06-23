import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


def get_min(df, column_name):
    return df[column_name].min()


def get_max(df, column_name):
    return df[column_name].max()


def get_diff_history_mean_std(column_name, history_column, df):
    # getting Difference
    df = df[df[history_column] != -3]
    df[f'{column_name}_diff_{history_column}'] = df[column_name] - \
        df[history_column]
    # df is filtered
    df[f'{column_name}_diff_{history_column}_mean'] = df[f'{column_name}_diff_{history_column}'].groupby(
        df['user_id']).transform('mean')
    df[f'{column_name}_diff_{history_column}_std'] = df[f'{column_name}_diff_{history_column}'].groupby(
        df['user_id']).transform('std')
    return df


def visualize_VA(df):
    current_V_users = df.groupby('user_id')['V_mean'].apply(list)
    short_V_users = df.groupby('user_id')['V_mean_short'].apply(list)
    long_V_users = df.groupby('user_id')['V_mean_long'].apply(list)

    current_A_users = df.groupby('user_id')['A_mean'].apply(list)
    short_A_users = df.groupby('user_id')['A_mean_short'].apply(list)
    long_A_users = df.groupby('user_id')['A_mean_long'].apply(list)

    v_len_users = len(current_V_users)
    for j in range(v_len_users):
        current_V = current_V_users.iloc[j]
        short_V = short_V_users.iloc[j]
        long_V = long_V_users.iloc[j]

        current_A = current_A_users.iloc[j]
        short_A = short_A_users.iloc[j]
        long_A = long_A_users.iloc[j]

        v_len = len(current_V)
        palying_index = random.sample(
            range(0, (v_len - 1)), (5 if v_len > 5 else v_len))
        print(palying_index)
        for i in palying_index:
            print(i)
            # line 1 points
            x1 = [current_V[i], short_V[i], long_V[i]]
            y1 = [current_A[i], short_A[i], long_A[i]]
            plt.plot(current_V[i], current_A[i],
                     linestyle='--', marker='o', color="blue")
            plt.plot(short_V[i], short_A[i], linestyle='--',
                     marker='s', color="blue")
            plt.plot(long_V[i], long_A[i], linestyle='--',
                     marker='*', color="blue")
            plt.plot(x1, y1)
        plt.legend(["current VA", "short VA", "long VA"])
        # naming the x axis
        plt.xlabel('Valence')
        # naming the y axis
        plt.ylabel('Arousal')
        plt.show()


def visualize_diff_std_mean(df):
    grouped = df.groupby('user_id')
    df1 = grouped['V_mean_diff_V_mean_short_mean',
                  'V_mean_diff_V_mean_short_std'].agg(['unique'])

    user_list = list(range(0, len(df1)))

    df2 = grouped['V_mean_diff_V_mean_long_mean',
                  'V_mean_diff_V_mean_long_std'].agg(['unique'])

    # df3 = grouped['A_mean_diff_A_mean_short_mean',
    #               'A_mean_diff_A_mean_short_std'].agg(['unique'])

    # df4 = grouped['A_mean_diff_A_mean_long_mean',
    #               'A_mean_diff_A_mean_long_std'].agg(['unique'])

    # df5 = grouped['D_mean_diff_D_mean_short_mean',
    #               'D_mean_diff_D_mean_short_std'].agg(['unique'])

    # df6 = grouped['D_mean_diff_D_mean_long_mean',
    #               'D_mean_diff_D_mean_long_std'].agg(['unique'])
    #Visualization############################

    fig = plt.figure(1, figsize=(6, 4))
    plt.ylim([-1, 1])
    plt.xlabel('users')
    plt.ylabel('difference')

    ydata = df1['V_mean_diff_V_mean_short_mean']
    yerror = df1['V_mean_diff_V_mean_short_std']
    f_list = [item for sublist in ydata.values for item in sublist]
    flat_list_ydata = [item[0] for item in f_list]

    f_list = [item for sublist in yerror.values for item in sublist]
    flat_list_yerror = [item[0] for item in f_list]

    ydata2 = df2['V_mean_diff_V_mean_long_mean']
    yerror2 = df2['V_mean_diff_V_mean_long_std']
    f_list = [item for sublist in ydata2.values for item in sublist]
    flat_list_ydata2 = [item[0] for item in f_list]
    f_list = [item for sublist in yerror2.values for item in sublist]
    flat_list_yerror2 = [item[0] for item in f_list]

    data_1 = {'x': user_list, 'y': flat_list_ydata, 'yerr': flat_list_yerror}
    data_2 = {'x': user_list, 'y': flat_list_ydata2, 'yerr': flat_list_yerror2}
    # errorbar + fill_between
    plt.subplot(121)
    labels = ['difference of current and shortterm valence',
              'difference of current and longterm valence']
    for i, data in enumerate([data_1, data_2]):
        plt.errorbar(**data, alpha=.75, fmt=':', capsize=3,
                     capthick=1, label=labels[i])
        data = {
            'x': data['x'],
            'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
            'y2': [y + e for y, e in zip(data['y'], data['yerr'])]}
        plt.fill_between(**data, alpha=.25)
    plt.ylim([(get_min(df, 'V_mean') - get_max(df, 'V_mean')),
              (get_max(df, 'V_mean') - get_min(df, 'V_mean'))])
    plt.legend()
    # Same for A
    # ydata = df3['A_mean_diff_A_mean_short_mean']
    # yerror = df3['A_mean_diff_A_mean_short_std']
    # f_list = [item for sublist in ydata.values for item in sublist]
    # flat_list_ydata = [item[0] for item in f_list]

    # f_list = [item for sublist in yerror.values for item in sublist]
    # flat_list_yerror = [item[0] for item in f_list]

    # ydata2 = df4['A_mean_diff_A_mean_long_mean']
    # yerror2 = df4['A_mean_diff_A_mean_long_std']
    # f_list = [item for sublist in ydata2.values for item in sublist]
    # flat_list_ydata2 = [item[0] for item in f_list]
    # f_list = [item for sublist in yerror2.values for item in sublist]
    # flat_list_yerror2 = [item[0] for item in f_list]

    # data_1 = {'x': user_list, 'y': flat_list_ydata, 'yerr': flat_list_yerror}
    # data_2 = {'x': user_list, 'y': flat_list_ydata2, 'yerr': flat_list_yerror2}
    # # errorbar + fill_between
    # plt.subplot(122)
    # labels = ['difference of current and shortterm arousal',
    #           'difference of current and longterm arousal']
    # for i, data in enumerate([data_1, data_2]):
    #     plt.errorbar(**data, alpha=.75, fmt=':', capsize=3,
    #                  capthick=1, label=labels[i])
    #     data = {
    #         'x': data['x'],
    #         'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
    #         'y2': [y + e for y, e in zip(data['y'], data['yerr'])]}
    #     plt.fill_between(**data, alpha=.25)

    # plt.ylim([(get_min(df, 'A_mean') - get_max(df, 'A_mean')),
    #           (get_max(df, 'A_mean') - get_min(df, 'A_mean'))])
    # plt.legend()
    plt.savefig('V_dif.png')


if __name__ == '__main__':
    df = pd.read_csv('playing_events_short_long_V.csv', sep=',')
    df = get_diff_history_mean_std('V_mean', 'V_mean_short', df)
    df = get_diff_history_mean_std('V_mean', 'V_mean_long', df)
    # df = get_diff_history_mean_std('A_mean', 'A_mean_short', df)
    # df = get_diff_history_mean_std('A_mean', 'A_mean_long', df)
    # df = get_diff_history_mean_std('D_mean', 'D_mean_short', df)
    # df = get_diff_history_mean_std('D_mean', 'D_mean_long', df)
    visualize_diff_std_mean(df)
    # visualize_VA(df)
