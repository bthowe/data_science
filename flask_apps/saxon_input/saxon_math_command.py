import os
import sys
import json
import joblib
import datetime
import webbrowser
import numpy as np
import pandas as pd
import saxon_math_helpers
from pymongo import MongoClient
from collections import defaultdict
from flask import Flask, request, render_template, jsonify, redirect

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


app = Flask(__name__)

client = MongoClient()
db_number = client['math_book_info']
db_origin = client['math_exercise_origins']
db_performance = client['math_performance']
db_vocab = client['vocab']
db_script = client['scripture_commentary']
db_forms = client['forms']

lesson_lst = list(range(4, 13)) + list(range(14, 75))
book_lst = ['Math_5_4', 'Math_6_5', 'Math_7_6', 'Math_8_7', 'Algebra_1_2', 'Algebra_1', 'Algebra_2']


@app.route("/login")
def login():
    return render_template('login.html')


@app.route("/user_name", methods=['POST'])
def user_name():
    name = json.loads(request.data.decode('utf-8'))['person']
    print(name)
    joblib.dump(name, 'user_name.pkl')
    return ''


@app.route("/main_menu")
def main_menu():
    name = joblib.load('user_name.pkl')
    return render_template('main_menu.html', user_name=name)


@app.route("/dashboards")
def dashboards_new():
    return render_template('dashboards.html')

def the_big_one(book, df_number, df_origin, df_performance):
    # todo: maybe just remove the offending documents from the database
    if 'chapter' in df_performance.columns.tolist():
        df_performance.drop('chapter', 1, inplace=True)
    if 'miss_list' in df_performance.columns.tolist():
        df_performance.drop('miss_list', 1, inplace=True)

    df_performance = df_performance. \
        query('date == date'). \
        assign(date=pd.to_datetime(df_performance['date'])). \
        sort_values(['date', 'start_chapter', 'start_problem'])

    df_performance['end_chapter'] = df_performance['end_chapter'].astype(str)
    df_performance_test = df_performance.loc[df_performance['end_chapter'].str.contains('test', na=False)]
    df_performance_ass = df_performance.loc[~df_performance['end_chapter'].str.contains('test', na=False)]

    # these columns have different types across the various collections, which makes for a bit of a headache
    df_performance_ass['start_chapter'] = df_performance_ass['start_chapter'].astype(float).astype(int)
    df_performance_ass['end_chapter'] = df_performance_ass['end_chapter'].astype(float).astype(int)

    # assignments
    start_chapter_ass = df_performance_ass['start_chapter'].iloc[0]
    start_problem_ass = df_performance_ass['start_problem'].iloc[0]

    end_chapter_ass = df_performance_ass['end_chapter'].iloc[-1]
    end_problem_ass = df_performance_ass['end_problem'].iloc[-1]

    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    df_grande_ass = pd.DataFrame()
    for chapter in range(int(float(start_chapter_ass)), int(float(end_chapter_ass)) + 1):
        df_temp = pd.DataFrame()
        lesson_probs = df_number.query('chapter == {}'.format(chapter)).iloc[0]['num_lesson_probs']
        mixed_probs = int(df_number.query('chapter == {}'.format(chapter)).iloc[0]['num_mixed_probs'])
        origin_probs = df_origin.query('chapter == {}'.format(chapter)).iloc[0]['origin_list']
        missed_probs = []
        for dic in df_performance_ass.query('start_chapter == {}'.format(chapter))['miss_lst'].values.tolist() + df_performance_ass.query('end_chapter == {}'.format(chapter))['miss_lst'].values.tolist():
            try:
                missed_probs += dic[str(chapter)]
            except:
                pass
        missed_probs = list(set(missed_probs))

        if start_chapter_ass == end_chapter_ass:
            if start_problem_ass.isdigit():
                problem_lst = range(int(start_problem_ass), int(end_problem_ass) + 1)
                origin_lst = origin_probs[int(start_problem_ass): int(end_problem_ass) + 1]

            else:
                # I'm assuming the end_problem would not also be a letter
                start_ind = alphabet.find(start_problem_ass)
                end_ind = alphabet.find(lesson_probs)
                problem_lst = list(alphabet[start_ind: end_ind + 1]) + list(range(1, int(end_problem_ass) + 1))
                origin_lst = (end_ind - start_ind + 1) * [np.nan] + origin_probs[: int(end_problem_ass)]

        else:
            if chapter == start_chapter_ass:
                if start_problem_ass.isdigit():
                    problem_lst = list(range(int(start_problem_ass), mixed_probs + 1))
                    origin_lst = origin_probs[int(start_problem_ass) - 1:]

                else:
                    start_ind = alphabet.find(start_problem_ass)
                    end_ind = alphabet.find(lesson_probs)
                    problem_lst = list(alphabet[start_ind: end_ind + 1]) + list(range(1, mixed_probs + 1))
                    origin_lst = (end_ind - start_ind + 1) * [np.nan] + origin_probs

            elif chapter == end_chapter_ass:
                if end_problem_ass.isdigit():
                    start_ind = 0
                    end_ind = alphabet.find(lesson_probs)
                    problem_lst = list(alphabet[start_ind: end_ind + 1]) + list(range(1, int(end_problem_ass) + 1))
                    origin_lst = (end_ind - start_ind + 1) * [np.nan] + origin_probs[: int(end_problem_ass)]

                else:
                    start_ind = 0
                    end_ind = alphabet.find(end_problem_ass)
                    problem_lst = list(alphabet[start_ind: end_ind + 1])
                    origin_lst = (end_ind - start_ind + 1) * [np.nan]

            else:
                start_ind = 0
                end_ind = alphabet.find(lesson_probs)
                problem_lst = list(alphabet[start_ind: end_ind + 1]) + list(range(1, mixed_probs + 1))
                origin_lst = (end_ind - start_ind + 1) * [np.nan] + origin_probs

        df_temp['problem'] = problem_lst
        df_temp['origin'] = origin_lst
        df_temp['book'] = book
        df_temp['chapter'] = chapter
        df_temp['correct'] = df_temp.apply(lambda x: 0 if str(x['problem']) in missed_probs else 1, axis=1)

        df_grande_ass = df_grande_ass.append(df_temp)
    df_grande_ass.reset_index(drop=True, inplace=True)

    df_grande_ass['date'] = ''
    df_p_g = df_performance_ass.sort_values('date').iterrows()
    row_p = next(df_p_g)[1]
    for ind, row in df_grande_ass.iterrows():
        df_grande_ass.set_value(ind, 'date', row_p['date'])  # FutureWarning: set_value is deprecated and will be removed in a future release. Please use .at[] or .iat[] accessors instead
        if (row['chapter'] == int(float(row_p['end_chapter']))) and (str(row['problem']) == row_p['end_problem']):
            try:
                row_p = next(df_p_g)[1]
            except:
                print('boom!')

    # tests
    df_grande_test = pd.DataFrame()
    for ind, row in df_performance_test.iterrows():
        df_temp = pd.DataFrame()
        df_temp['problem'] = range(1, 21)
        df_temp['book'] = book
        df_temp['chapter'] = row['end_chapter']
        df_temp['date'] = row['date']

        missed_probs = row['miss_lst'][row['end_chapter']]
        df_temp['correct'] = df_temp.apply(lambda x: 0 if str(x['problem']) in missed_probs else 1, axis=1)

        df_grande_test = df_grande_test.append(df_temp)

    return df_grande_ass, df_grande_test


def performance_over_time(df, book, kid):
    def js_month(x):
        x_lst = x.split('-')
        x_lst[1] = str(int(x_lst[1]) - 1)
        if len(x_lst[1]) == 1:
            x_lst[1] = '0' + x_lst[1]
        return '-'.join(x_lst)
    df['date'] = df['date'].astype(str).apply(js_month)  # this zero indexes the month for js's benefit.

    return df['correct'].\
        groupby(df['date']).mean().\
        reset_index(drop=False). \
        assign(book=book, kid=kid, position=range(0, df['date'].unique().shape[0]))

def origin_lst_expand(df, kid):
    df = df.loc[df['problem'].astype(str).str.isdigit()].assign(kid=kid)

    df['origin_lst'] = df['origin'].str.split(', ')
    df['len_origin_lst'] = df['origin_lst'].map(len)
    df1 = df.query('len_origin_lst == 1')
    df2 = df.query('len_origin_lst == 2')
    return df1. \
        append(
        df2.assign(origin=df2['origin_lst'].map(lambda x: x[0])).append(
            df2.assign(origin=df2['origin_lst'].map(lambda x: x[1])))
        ). \
        reset_index(drop=True). \
        drop(['origin_lst', 'len_origin_lst'], 1)


def mixed_problems_correct(df):
    def counter(x):
        x['position'] = range(1, len(x) + 1)
        return x
    return df. \
        sort_values(['chapter', 'problem', 'origin']). \
        groupby([df['chapter'], df['origin']]).apply(counter)


def _sorter(x):
    x_i = x.loc[~x['origin'].str.strip().str.isdigit()]
    x_d = x.loc[x['origin'].str.strip().str.isdigit()]

    x_d['origin'] = x_d['origin'].astype(int)
    x_d.sort_values('origin', inplace=True)
    x_d['origin'] = x_d['origin'].astype(str)

    return x_d.append(x_i)

def _chapter_sort(df):
    if np.all(df['origin'].str.strip().str.isdigit()):
        df['origin'] = df['origin'].astype(int)
        df.sort_values(['mean', 'origin'], inplace=True)
    else:
        df.sort_values('mean', inplace=True)
        df = df.groupby(df['mean']).apply(_sorter)
    return df

def performance_by_chapter(df):
    df['origin'] = df['origin'].str.strip()
    return pd.concat(
        [
            df['correct'].groupby(df['origin']).agg(['mean', 'count']),
            df[['book', 'kid']].groupby(df['origin']).agg(lambda x: x.value_counts().index[0])
        ],
        axis=1
        ). \
        reset_index(drop=False). \
        pipe(_chapter_sort)


def performance_on_tests(df, kid):
    if df.empty:
        return df

    df['kid'] = kid

    df_miss = df.query('correct == 0')
    df_miss_lst = df_miss.groupby(df_miss['chapter']).apply(lambda x: x['problem'].tolist())

    df_test = pd.concat(
        [
            df['correct'].groupby(df['chapter']).mean(),
            df_miss_lst,
            df[['book', 'kid']].groupby(df['chapter']).agg(lambda x: x.value_counts().index[0])

        ],
        axis=1
    ). \
        reset_index(drop=False). \
        rename(columns={'chapter': 'test', 0: 'miss_lst', 'correct': 'perc_correct'})
    df_test['ind'] = df_test.index
    df_test['chapters'] = df_test['test'].str.split().apply(lambda x: '{0}-{1}'.format(int(x[1]) * 4 - 3, int(x[1]) * 4))
    var_lst = ['ind', 'book', 'kid', 'miss_lst', 'test', 'perc_correct', 'chapters']
    return df_test[var_lst]

@app.route("/query_performance", methods=['POST'])
def query_performance():
    js = json.loads(request.data.decode('utf-8'))
    print(js)  # kid, book


    # I need to loop over the books and not take just the one.
    df_time_data = pd.DataFrame()
    df_test_data = pd.DataFrame()
    df_prob_data = pd.DataFrame()
    df_score_data = pd.DataFrame()
    for book in ['Math_5_4', 'Math_6_5', 'Math_7_6', 'Math_8_7', 'Algebra_1_2', 'Algebra_1', 'Algebra_2']:
        perf_temp = pd.DataFrame(list(db_performance[book].find({'kid': js['kid']})))
        numb_temp = pd.DataFrame(list(db_number[book].find()))
        orig_temp = pd.DataFrame(list(db_origin[book].find()))
        print(perf_temp.shape[0])
        if perf_temp.shape[0] > 0:
            df_grande_ass, df_grande_test = the_big_one(
                book,
                numb_temp,
                orig_temp,
                perf_temp
            )
            # df_grande_ass = df_grande_ass.append(ass_temp).reset_index(drop=True)
            # df_grande_test = df_grande_test.append(test_temp).reset_index(drop=True)

            df_time_data = df_time_data.append(performance_over_time(df_grande_ass.append(df_grande_test), book, js['kid'])).reset_index(drop=True)
            df_test_data = performance_on_tests(df_grande_test, js['kid']).reset_index(drop=True)

            df_temp = df_grande_ass.pipe(origin_lst_expand, js['kid'])
            df_prob_data = mixed_problems_correct(df_temp).reset_index(drop=True)
            df_score_data = performance_by_chapter(df_temp).reset_index(drop=True)

    return jsonify(
        [
            {'df_time_data': df_time_data.to_dict('records')},
            {'df_prob_data': df_prob_data.to_dict('records')},
            {'df_score_data': df_score_data.to_dict('records')},
            {'df_test_data': df_test_data.to_dict('records')}
        ]
    )


def problem_list_count(book, start_chapter, start_problem, end_chapter, end_problem):
    prob_num = 0
    for k, v in problems(book, start_chapter, start_problem, end_chapter, end_problem).items():
        prob_num += len(v)
    return prob_num

def problems(book, start_chapter, start_problem, end_chapter, end_problem):
    """return a dictionary with chapters and lists of problems completed"""

    problems_dic = {}

    if 'test' in str(start_chapter):
        # problems_dic['start_chapter'] = list(range(1, 20))
        return {start_chapter: list(range(1, 21))}
    else:
        start_chapter_details = list(db_number[book].find({'chapter': start_chapter}))[0]
        end_chapter_details = list(db_number[book].find({'chapter': end_chapter}))[0]

        if int(end_chapter) - int(start_chapter) == 0:  # if start and end is the same chapter
            problems_dic[start_chapter] = problem_list_create(start_problem, end_problem, start_chapter_details['num_lesson_probs'])

        elif int(end_chapter) - int(start_chapter) == 1:  # if start and end is one chapter apart
            problems_dic[start_chapter] = problem_list_create(start_problem, start_chapter_details['num_mixed_probs'], start_chapter_details['num_lesson_probs'])
            problems_dic[end_chapter] = problem_list_create('a', end_problem, end_chapter_details['num_lesson_probs'])

        else:  # if start and end is multiple chapters apart
            problems_dic[start_chapter] = problem_list_create(start_problem, start_chapter_details['num_mixed_probs'], start_chapter_details['num_lesson_probs'])
            problems_dic[end_chapter] = problem_list_create('a', end_problem, end_chapter_details['num_lesson_probs'])
            for chapter in range(int(start_chapter) + 1, int(end_chapter)):
                mid_chapter_details = list(db_number[book].find({'chapter': chapter}))[0]
                problems_dic[chapter] = problem_list_create('a', mid_chapter_details['num_mixed_probs'], mid_chapter_details['num_lesson_probs'])

        return problems_dic



@app.route("/query_periods_data", methods=['POST'])
def query_periods_data():
    js = json.loads(request.data.decode('utf-8'))
    print(js)  # kid, book, periods

    df_performance_details = pd.DataFrame(list(db_performance[js['book']].find({'kid': js['kid']}))).\
        query('date == date').\
        drop(['chapter', 'miss_list'], 1). \
        sort_values(['date', 'start_chapter', 'start_problem'])
    df_performance_details['date'] = pd.to_datetime(df_performance_details['date'])
    # print(df_performance_details)

    df_book_details = pd.DataFrame(list(db_number[js['book']].find())).drop(['_id', 'book'], 1)

    threshold_date = (df_performance_details['date'] - pd.to_timedelta(int(js['periods']), unit='D')).max()
    df_merged = df_performance_details.\
        iloc[-int(js['periods']):].\
        merge(df_book_details, how='left', left_on='start_chapter', right_on='chapter')
    # df_merged = df_performance_details.\
    #     query('date > "{}"'.format(threshold_date)).\
    #     merge(df_book_details, how='left', left_on='start_chapter', right_on='chapter')

    df_merged['problem_count'] = df_merged.apply(lambda x: problem_list_count(x['book'], x['start_chapter'], x['start_problem'], x['end_chapter'], x['end_problem']), axis=1)
    df_merged['missed_count'] = df_merged['miss_lst'].apply(lambda x: len(list(x.values())[0]))
    df_merged['perc_correct'] = df_merged.apply(lambda x: (x['problem_count'] - x['missed_count']) / x['problem_count'], axis=1)

    def str_mod(x):
        x_lst = x.split('-')
        x_return = ''
        if int(x_lst[1]) < 10:
            x_return += x_lst[1][1] + '-'
        else:
            x_return += x_lst[1] + '-'
        x_return += x_lst[2]
        return x_return

    df_merged['date1'] = df_merged['date'].astype(str).apply(str_mod)
    df_merged['position1'] = range(len(df_merged))
    df_merged['value1'] = df_merged['perc_correct']
    df_merged['date2'] = df_merged['date1'].shift(-1)
    df_merged['position2'] = df_merged['position1'].shift(-1)
    df_merged['value2'] = df_merged['value1'].shift(-1)
    df_merged = df_merged.iloc[:-1]
    df_merged['position2'] = df_merged['position2'].astype(int)

    print(df_merged[['value1', 'position1', 'value2', 'position2', 'date1', 'date2']].to_dict('records'))
    return jsonify(df_merged[['value1', 'position1', 'value2', 'position2', 'date1', 'date2']].to_dict('records'))



@app.route("/enter_problem_number")
def enter_chapter_details():
    return render_template('enter_problem_number.html')


def problem_list_create(first, last, less_num):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    if str(first).isalpha():
        if str(last).isalpha():
            return [letter for letter in alphabet if (letter >= str(first) and letter <= str(last))]
        else:
            return [letter for letter in alphabet if (letter >= str(first) and letter <= less_num)] + list(map(str, range(1, int(last) + 1)))
    else:
        return list(map(str, range(int(first), int(last) + 1)))

@app.route('/query_chapter', methods=['POST', 'GET'])
def query_chapter():
    js = json.loads(request.data.decode('utf-8'))
    print(js)

    book = js['book']

    problems_dic = {}

    if js['test']:
        problems_dic[js['start_chapter']] = str(list(map(str, range(1, 21))))
    else:
        start_chapter_details = list(db_number[book].find({'chapter': js['start_chapter']}))[0]
        end_chapter_details = list(db_number[book].find({'chapter': js['end_chapter']}))[0]

        if int(js['end_chapter']) - int(js['start_chapter']) == 0:  # if start and end is the same chapter
            problems_dic[js['start_chapter']] = str(problem_list_create(js['start_problem'], js['end_problem'], start_chapter_details['num_lesson_probs']))
        elif int(js['end_chapter']) - int(js['start_chapter']) == 1:  # if start and end is one chapter apart
            problems_dic[js['start_chapter']] = str(problem_list_create(js['start_problem'], start_chapter_details['num_mixed_probs'], start_chapter_details['num_lesson_probs']))
            problems_dic[js['end_chapter']] = str(problem_list_create('a', js['end_problem'], end_chapter_details['num_lesson_probs']))
        else:  # if start and end is multiple chapters apart
            problems_dic[js['start_chapter']] = str(problem_list_create(js['start_problem'], start_chapter_details['num_mixed_probs'], start_chapter_details['num_lesson_probs']))
            problems_dic[js['end_chapter']] = str(problem_list_create('a', js['end_problem'], end_chapter_details['num_lesson_probs']))
            for chapter in range(int(js['start_chapter']) + 1, int(js['end_chapter'])):
                mid_chapter_details = list(db_number[book].find({'chapter': chapter}))[0]
                problems_dic[chapter] = str(problem_list_create('a', mid_chapter_details['num_mixed_probs'], mid_chapter_details['num_lesson_probs']))

    print(problems_dic)

    return jsonify(problems_dic)


@app.route('/add_problem_number', methods=['POST'])
def add_problem_number():
    js = json.loads(request.data.decode('utf-8'))

    collection = db_number[js['book'].replace(' ', '_').replace('/', '_')]
    y = collection.insert_one(js)
    print(y)

    print('number data inserted: {}'.format(js))
    return ''

@app.route('/remove_problem_number', methods=['POST'])
def remove_problem_number():
    js = json.loads(request.data.decode('utf-8'))

    collection = db_number[js['book'].replace(' ', '_').replace('/', '_')]
    y = collection.delete_one(js)
    print(y)

    print('number data deleted: {}'.format(js))
    return ''

@app.route("/enter_problem_origin")
def enter_problem_origin():
    return render_template('enter_problem_origins.html')


@app.route('/query_chapter2', methods=['POST', 'GET'])
def query_chapter2():
    js = json.loads(request.data.decode('utf-8'))
    print(js)

    book = js['book']

    output = list(db_number[book].find({'chapter': js['chapter']}))[0]
    print(output)
    problems_dic = {'num_lesson_probs': output['num_lesson_probs'], 'num_mixed_probs': output['num_mixed_probs']}
    return jsonify(problems_dic)


@app.route('/add_problem_origin', methods=['POST'])
def add_problem_origin():
    js = json.loads(request.data.decode('utf-8'))
    print(js)

    collection = db_origin[js['book']]
    y = collection.insert_one(js)
    print(y)

    print('data inserted: {}'.format(js))
    return ''


@app.route("/enter_performance")
def enter_performance():
    return render_template('enter_performance.html')


@app.route('/add_missed_problems', methods=['POST'])
def add_missed_problems():
    js = json.loads(request.data.decode('utf-8'))
    print(js)

    miss_lst = defaultdict(list)


    for prob in js['add_miss_list']:
        if js['test']:
            miss_lst['test {}'.format(prob['chapter'])].append(prob['problem'])
        else:
            miss_lst[prob['chapter']].append(prob['problem'])
    for prob in js['rem_miss_list']:
        if js['test']:
            miss_lst['test {}'.format(prob['chapter'])].remove(prob['problem'])
        else:
            miss_lst[prob['chapter']].remove(prob['problem'])
    k_to_del = [k for k, v in miss_lst.items() if not miss_lst[k]]
    for k in k_to_del:
        del miss_lst[k]
    js['miss_lst'] = dict(miss_lst)

    if js['test']:
        js['start_chapter'] = 'test {}'.format(js['start_chapter'])
        js['end_chapter'] = 'test {}'.format(js['end_chapter'])

    del js['add_miss_list']
    del js['rem_miss_list']
    del js['test']

    collection = db_performance[js['book']]
    y = collection.insert_one(js)

    print('data inserted: {}'.format(js))
    return ''


@app.route('/vocab/python_crash_course', methods=['POST'])
def practice():
    lesson_num = str(request.form['user_input'])
    prompt_type = str(request.form['prompt_type'])
    num_cards = len(os.listdir('static/{0}'.format(lesson_num)))
    if prompt_type == 'word':
        cards = [['static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num),
                  'static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num + 1)] for num in range(0, num_cards, 2)]
    else:
        cards = [['static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num + 1),
                  'static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num)] for num in range(0, num_cards, 2)]

    return render_template("vocab/display_card.html", cards=cards)


@app.route('/vocab/quiz', methods=['POST'])
def quiz():
    practice_type = str(request.form['python_crash_course'])
    prompt_type = str(request.form['word'])
    lesson_num = str(request.form['lesson'])
    num_cards = len(os.listdir('static/{0}'.format(lesson_num)))

    if prompt_type == 'word':
        cards = [['../static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num),
                  '../static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num + 1)] for num in range(0, num_cards, 2)]

        alternatives = []
        for card_i in range(len(cards)):
            alternatives_i = []
            for j in range(3):
                random_lesson = np.random.choice(lesson_lst)
                num_cards_in_random_lesson = len(os.listdir('static/{0}'.format(random_lesson)))
                random_card = np.random.choice(range(0, num_cards_in_random_lesson, 2))
                alternatives_i.append('../static/{0}/rc_vocab_{0}_{1}.png'.format(random_lesson, random_card + 1))
            alternatives.append(alternatives_i)

    else:
        cards = [['../static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num + 1),
                  '../static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num)] for num in range(0, num_cards, 2)]

        alternatives = []
        for card_i in range(len(cards)):
            alternatives_i = []
            for j in range(3):
                random_lesson = np.random.choice(lesson_lst)
                num_cards_in_random_lesson = len(os.listdir('static/{0}'.format(random_lesson)))
                random_card = np.random.choice(range(0, num_cards_in_random_lesson, 2))
                alternatives_i.append('../static/{0}/rc_vocab_{0}_{1}.png'.format(random_lesson, random_card))
            alternatives.append(alternatives_i)

    if practice_type == 'python_crash_course':
        return render_template('vocab/display_card.html', cards=cards)
    else:
        return render_template('vocab/quiz_card.html', cards=cards, alts=alternatives)


@app.route('/mongo_call', methods=['POST'])
def mongo_call():
    name = joblib.load('user_name.pkl')
    js = json.loads(request.data.decode('utf-8'))
    js['name'] = name

    tab = db_vocab[js['page']]
    tab.insert_one(js)

    print('data inserted: {}'.format(js))
    return ''


@app.route("/vocab")
def vocab():
    return render_template('vocabulary.html')



@app.route('/mongo_call_script', methods=['POST'])
def mongo_call_script():
    js = json.loads(request.data.decode('utf-8'))

    tab = db_script[js['name']]
    tab.insert_one(js)

    print('data inserted: {}'.format(js))
    return ''

@app.route("/scripture_commentary")
def scripture_commentary():
    return render_template('scripture_commentary.html')


@app.route('/mongo_call_forms', methods=['POST'])
def mongo_call_forms():
    js = json.loads(request.data.decode('utf-8'))

    tab = db_forms['Weekly']
    tab.insert_one(js)
    print('data inserted: {}'.format(js))

    saxon_math_helpers.latex_create(
        ['Calvin', 'Samuel', 'Kay'],
        [js['calvin_book'], js['samuel_book'], js['kay_book']],
        [datetime.datetime.strftime(datetime.datetime.strptime(js['week_start_date'], '%Y-%m-%d') + datetime.timedelta(days), '%Y-%m-%d') for days in range(0, 6)],
        [js['mon_question'], js['tue_question'], js['wed_question'], js['thu_question'], js['fri_question'], js['sat_question']],
        [js['mon_job'], js['tue_job'], js['wed_job'], js['thu_job'], js['fri_job'], js['sat_job']]
    )

    chrome = webbrowser.get('chrome')
    chrome.open_new_tab('file:///Users/travis.howe/Projects/github/data_science/flask_apps/saxon_input/weekly_time_sheet.pdf')

    return ''

@app.route("/weekly_forms_create")
def weekly_forms_create():
    today = datetime.date.today()
    date_shift = 7 - today.weekday()  # Monday is 0
    date = str(today + datetime.timedelta(date_shift))

    output = {k: v for k, v in list(db_forms['Weekly'].find())[-1].items() if k != '_id'}

    return render_template('weekly_forms_create.html', date=date, form_data=output)


@app.route('/query_book', methods=['POST'])
def query_book():
    js = json.loads(request.data.decode('utf-8'))
    print(js)

    if js['name'] == 'Choose...':
        return jsonify({})
    for book in reversed(book_lst):
        l = list(db_performance[book].find({'kid': js['name']}))
        if l:
            return jsonify(l[0]['book'])
            # break
    return ''



@app.route('/killer', methods=['POST'])
def killer():
    sys.exit(4)
    # js = json.loads(request.data.decode('utf-8'))
    # print(js)
    #
    # if js['name'] == 'Choose...':
    #     return jsonify({})
    # for book in reversed(book_lst):
    #     l = list(db_performance[book].find({'kid': js['name']}))
    #     if l:
    #         return jsonify(l[0]['book'])
    #         # break
    return ''




@app.route('/quit')
def quit():
    # sys.exit(4)
    return render_template('quit.html')
    # return ''


    # if __name__ == '__main__':
#     lesson_lst = list(range(4, 13)) + list(range(14, 75))
#     book_lst = ['Math_5_4', 'Math_6_5', 'Math_7_6', 'Math_8_7', 'Algebra_1_2', 'Algebra_1', 'Algebra_2']

    # chrome = webbrowser.get('chrome')
    # chrome.open('http://0.0.0.0:8001/login')

    # app.run(host='0.0.0.0', port=8001, debug=True)


# gunicorn --bind 0.0.0.0:8001 saxon_math_command:app