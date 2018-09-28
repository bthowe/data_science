import os
import sys
import json
import datetime
import webbrowser
import pandas as pd
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

@app.route("/")
def main_menu():
    return render_template('main_menu2.html')

# @app.route("/")
# def main_menu():
#     return render_template('main_menu.html')
#

# @app.route("/dashboards")
# def dashboards():
#     return render_template('dashboards.html')
@app.route("/dashboards")
def dashboards():
    date_thresh = '2018-09-17'
    # date_thresh = str(datetime.date.today() - datetime.timedelta(days=14))

    df = pd.DataFrame()
    for book, kid in zip(['Algebra_1_2', 'Math_7_6'], ['Calvin', 'Samuel']):
        df_performance_details = pd.DataFrame(list(db_performance[book].find({'kid': kid}))). \
                query('date == date'). \
                drop(['chapter', 'miss_list'], 1). \
                sort_values(['date', 'start_chapter', 'start_problem'])
        df_performance_details['date'] = pd.to_datetime(df_performance_details['date'])
        df_book_details = pd.DataFrame(list(db_number[book].find())).drop(['_id', 'book'], 1)

        df_merged = df_performance_details.\
            merge(df_book_details, how='left', left_on='start_chapter', right_on='chapter')

        df_merged.query('date >= "{}"'.format(date_thresh), inplace=True)

        df_merged['problem_count'] = df_merged.apply(lambda x: problem_list_count(x['book'], x['start_chapter'], x['start_problem'], x['end_chapter'], x['end_problem']), axis=1)
        df_merged['missed_count'] = df_merged['miss_lst'].apply(lambda x: len(list(x.values())[0]))

        df_grouped = df_merged[['problem_count', 'missed_count']].groupby([df_merged['book'], df_merged['kid'], df_merged['date']]).sum().reset_index(drop=False)
        df_grouped['perc_correct'] = df_grouped.apply(lambda x: (x['problem_count'] - x['missed_count']) / x['problem_count'], axis=1)
        df_grouped.drop(['problem_count', 'missed_count'], 1, inplace=True)

        # def str_mod(x):
        #     x_lst = x.split('-')
        #     x_return = ''
        #     if int(x_lst[1]) < 10:
        #         x_return += x_lst[1][1] + '-'
        #     else:
        #         x_return += x_lst[1] + '-'
        #     x_return += x_lst[2]
        #     return x_return
        # df_grouped['date1'] = df_grouped['date'].astype(str).apply(str_mod)
        df_grouped['date1'] = df_grouped['date'].astype(str)
        df_grouped['position1'] = range(len(df_grouped))
        df_grouped.rename(columns={'perc_correct': 'value1'}, inplace=True)
        df_grouped['date2'] = df_grouped['date1'].shift(-1)
        df_grouped['position2'] = df_grouped['position1'].shift(-1)
        df_grouped['value2'] = df_grouped['value1'].shift(-1)

        df_grouped.drop('date', 1, inplace=True)
        df_grouped = df_grouped.iloc[:-1]
        df_grouped['position2'] = df_grouped['position2'].astype(int)

        df = df.append(df_grouped)
        # print(df)
        print(df.query('kid == "Calvin"')['value1'].sum() + df.query('kid == "Calvin"').iloc[-1]['value2'].sum())

    return render_template("dashboards_buttons.html", score_data=df.to_dict('records'))
    # return render_template("dashboards_buttons.html", score_data=jsonify(df.to_dict('records')))


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

# todo: how should I deal with tests? Should I just throw them out for now? Count them as normal?
# todo: break down into lesson problems and mixed problems
# todo: break down into lessons and tests
# todo: percent correct on (1) tests, (2) lessons, (3) lesson problems, (4) mixed problems
# todo: maybe something by chapter and not just by day
# todo: two panels: (1) last two weeks (make the window variable) and (2) everything from the beginning of time, the parts not featured in (1) more opaque


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
    print(y)

    print('data inserted: {}'.format(js))
    return ''


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/quit')
def quit():
    shutdown_server()
    return ''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)


# todo: in performance, think of a solution to test


# todo: in performance page, maybe don't make submit button hidden after click, and make it so it refreshes if I push it again.
# todo: standardize date field in form

# todo: form styling for the three pages
# todo: complete the .sh bash file
    # -back up the database
    # -send the backup to git
    # -shutdown database and server


# todo: page to delete an entry by chapter
# todo: page to modify an entry by chapter

