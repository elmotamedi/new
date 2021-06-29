import os
import pandas as pd
import pickle as pkl
import datetime
from statistics import mean


def create_playingevents_VAD():
    track_VAD_df = pd.read_csv('VAD_functionals.csv', sep='\t')
    track_user_time_df = pd.read_csv(
        'LFM-2b_2019_target_users.tsv', delimiter="\t")

    track_user_time_df.columns = [
        'user_id', 'country', 'age', 'gender', 'track', 'artist', 'timestamp']

    df = pd.merge(track_user_time_df, track_VAD_df, on=['track', 'artist'])

    return df


def groupby_user_id_sortby_timestamp(playing_events_VAD_df):
    grouped_playing_events_VAD_df = playing_events_VAD_df.groupby("user_id")
    sorted_playing_events_VAD_df = grouped_playing_events_VAD_df.apply(
        lambda x: x.sort_values(['timestamp'], ascending=False))

    sorted_playing_events_VAD_df = sorted_playing_events_VAD_df.droplevel(
        0).reset_index()  # reset multi_indexing
    return sorted_playing_events_VAD_df


def generate_shortterm_emotion_tags(df, column_name, m_param, file_name):
    timehistory = []
    column_value_history = []
    boolflag = True
    userhistory = df['user_id'].iloc[0]
    for i in range(len(df)):
        print(f'short_{column_name}_User{i}_from_{len(df)}')
        boolflag = True
        # first timestamp of user
        date_time_obj_0 = datetime.datetime.strptime(
            df.loc[i, 'timestamp'], '%Y-%m-%d %H:%M:%S')
        userhistory = df['user_id'].iloc[i]

        j = i
        while(boolflag == True):
            # if the user does not have enough history
            if j == len(df):
                timehistory.clear()
                column_value_history.clear()
                # for user i you should save -3 because there is not enough history
                rowIndex = df.index[i]
                df.loc[rowIndex, f'{column_name}_short'] = -3
                boolflag = False  # go to next user
                break
            if (df.loc[j, 'user_id'] != userhistory):
                timehistory.clear()
                column_value_history.clear()
                # for user i you should save -3 because there is not enough history
                rowIndex = df.index[i]
                df.loc[rowIndex, f'{column_name}_short'] = -3
                boolflag = False  # go to next user
                break
            cur_date_time_obj = datetime.datetime.strptime(
                df.loc[j, 'timestamp'], '%Y-%m-%d %H:%M:%S')
            diff = date_time_obj_0 - cur_date_time_obj
            diff_seconds = diff.total_seconds()
            diff_min = divmod(diff_seconds, 60)[0]
            if (diff_min <= m_param):
                timehistory.append(df.loc[j, 'timestamp'])
                column_value_history.append(df.loc[j, column_name])
            elif(diff_min > m_param):
                rowIndex = df.index[i]
                df.loc[rowIndex, f'{column_name}_short'] = mean(
                    set(column_value_history))
                boolflag = False
                timehistory.clear()
                column_value_history.clear()
            j = j + 1
        df2 = df.iloc[[i]]
        df2.to_csv(file_name, mode='a', header=True)
    return df


def generate_longterm_emotion_tags(df, column_name, h_param, file_name):
    timehistory = []
    column_value_history = []
    boolflag = True
    userhistory = df['user_id'].iloc[0]
    for i in range(len(df)):
        print(f'long_{column_name}_User{i}_from_{len(df)}')
        boolflag = True
        # first timestamp of user
        date_time_obj_0 = datetime.datetime.strptime(
            df.loc[i, 'timestamp'], '%Y-%m-%d %H:%M:%S')
        userhistory = df['user_id'].iloc[i]

        j = i
        while(boolflag == True):
            # if the user does not have enough history
            if j == len(df):
                timehistory.clear()
                column_value_history.clear()
                # for user i you should save -3 because there is not enough history
                rowIndex = df.index[i]
                df.loc[rowIndex, f'{column_name}_long'] = -3
                boolflag = False  # go to next user
                break
            if (df.loc[j, 'user_id'] != userhistory):
                timehistory.clear()
                column_value_history.clear()
                # for user i you should save -3 because there is not enough history
                rowIndex = df.index[i]
                df.loc[rowIndex, f'{column_name}_long'] = -3
                boolflag = False  # go to next user
                break
            cur_date_time_obj = datetime.datetime.strptime(
                df.loc[j, 'timestamp'], '%Y-%m-%d %H:%M:%S')
            diff = date_time_obj_0 - cur_date_time_obj
            diff_seconds = diff.total_seconds()
            diff_hours = divmod(diff_seconds, 3600)[0]
            if (diff_hours <= h_param):
                timehistory.append(df.loc[j, 'timestamp'])
                column_value_history.append(df.loc[j, column_name])
            elif(diff_hours > h_param):
                rowIndex = df.index[i]
                df.loc[rowIndex, f'{column_name}_long'] = mean(
                    set(column_value_history))
                boolflag = False
                timehistory.clear()
                column_value_history.clear()
            j = j + 1
        df2 = df.loc(i)[:]
        df2.to_csv(file_name, mode='a', header=True)
    return df


if __name__ == '__main__':
    playing_events_VAD_df = create_playingevents_VAD()

    sorted_playing_events_VAD_df = groupby_user_id_sortby_timestamp(
        playing_events_VAD_df)

    df_V = generate_shortterm_emotion_tags(
        sorted_playing_events_VAD_df, 'V_mean', 15, 'playing_events_short_history_V.csv')
    df_V.to_csv('playing_events_short_history_V_2.csv',
                index=False, header=True)
    df_V_l = generate_longterm_emotion_tags(
        df_V, 'V_mean', 24, 'playing_events_short_long_V.csv')
    df_V_l.to_csv('playing_events_short_long_V_2.csv',
                  index=False, header=True)

    df_VA = generate_shortterm_emotion_tags(
        df_V_l, 'A_mean', 15, 'playing_events_short_long_V_short_A.csv')
    df_VA.to_csv('playing_events_short_long_V_short_A_2.csv',
                 index=False, header=True)
    df_VA_l = generate_longterm_emotion_tags(
        df_VA, 'A_mean', 24, 'playing_events_short_long_VA.csv')
    df_VA_l.to_csv('playing_events_short_long_VA_2.csv',
                   index=False, header=True)

    df_VAD = generate_shortterm_emotion_tags(
        df_VA_l, 'D_mean', 15, 'playing_events_short_long_VA_short_D.csv')
    df_VAD.to_csv('playing_events_short_long_VA_short_D_2.csv',
                  index=False, header=True)
    df_VAD_l = generate_longterm_emotion_tags(
        df_VAD, 'D_mean', 24, 'playing_events_short_long_VAD.csv')
    df_VAD_l.to_csv('playing_events_short_long_VAD_2.csv',
                    index=False, header=True)
