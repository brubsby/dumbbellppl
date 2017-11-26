from collections import namedtuple

import flask
import sqlite3
import itertools
import datetime
import colander
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.palettes import Category20
from bokeh.layouts import gridplot
from bokeh.models import HoverTool, ColumnDataSource
import pandas

import os

Lift = namedtuple("Lift", ['Name', 'VolumeMultiplier', 'AsymmetryMultiplier'])
Lift.__new__.__defaults__ = (None, 2, 1)  # most dumbbell lifts use two dumbbells in unison

WORKOUTS = {
    'PUSH_LIFTS': [
        Lift('Chest Press'),
        Lift('Incline Fly'),
        Lift('Arnold Press'),
        Lift('Overhead Triceps Extension', 1, 1)
    ],
    'PULL_LIFTS': [
        Lift('Pull-up', 1, 1),
        Lift('Bent-Over Row', 1, 2),
        Lift('Reverse Fly'),
        Lift('Shrug'),
        Lift('Bicep Curl')
    ],
    'LEG_LIFTS': [
        Lift('Goblet Squat', 1, 1),
        Lift('Lunge', 2, 2),
        Lift('Single Leg Deadlift', 1, 2),
        Lift('Calf Raise')
    ],
    'EVERY_OTHER_LIFTS': [
        Lift('Hanging Leg Raises', 1, 2)
    ],
    'REST': [],
    'MISS': []
}

AVAILABLE_WEIGHTS = [0, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 30, 35, 40, 45, 50, 52.5]

DAY_WORKOUT_DICT = {
    0: 'REST',
    1: 'PUSH_LIFTS',
    2: 'PULL_LIFTS',
    3: 'LEG_LIFTS',
    4: 'PUSH_LIFTS',
    5: 'PULL_LIFTS',
    6: 'LEG_LIFTS'
}
app = flask.Flask(__name__)


def create_tables_if_not_exist():
    os.makedirs(os.path.join('data'), exist_ok=True)
    with sqlite3.connect(os.path.join('data', 'userdata.db'), detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS Lifts ("
            "LiftID INTEGER PRIMARY KEY, "
            "Name TEXT NOT NULL, "
            "VolumeMultiplier INTEGER DEFAULT 2, "
            "AsymmetryMultiplier INTEGER DEFAULT 1, "
            "UNIQUE(Name), "
            "CHECK (VolumeMultiplier IN (1, 2) and AsymmetryMultiplier IN (1, 2)))")
        cursor.executemany("INSERT INTO Lifts (Name, VolumeMultiplier, AsymmetryMultiplier) "
                           "SELECT ?, ?, ? WHERE NOT EXISTS(SELECT 1 FROM Lifts WHERE Name = ?);",
                           [(lift.Name, lift.VolumeMultiplier, lift.AsymmetryMultiplier, lift.Name)
                            for lift in list(set(itertools.chain.from_iterable(WORKOUTS.values())))])
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS Workouts (WorkoutID INTEGER PRIMARY KEY, Name TEXT NOT NULL, UNIQUE(Name))")
        cursor.executemany(
            "INSERT INTO Workouts (Name) SELECT ? WHERE NOT EXISTS(SELECT 1 FROM Workouts WHERE Name = ?);",
            [(lift_day, lift_day) for lift_day in WORKOUTS.keys()])
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS WorkoutContents ("
            "WorkoutContentID INTEGER PRIMARY KEY, "
            'WorkoutFK INTEGER NOT NULL, '
            'LiftFK INTEGER NOT NULL, '
            "FOREIGN KEY (WorkoutFK) REFERENCES Lifts(WorkoutID), "
            "FOREIGN KEY (LiftFK) REFERENCES Lifts(LiftID), "
            "UNIQUE(WorkoutFK, LiftFK));")
        for workout, lift_list in WORKOUTS.items():
            cursor.executemany(
                "INSERT INTO WorkoutContents (WorkoutFK, LiftFK) "
                "SELECT (SELECT WorkoutID FROM Workouts WHERE Name = ?), (SELECT LiftID FROM Lifts WHERE Name = ?) "
                "WHERE NOT EXISTS("
                "SELECT 1 FROM WorkoutContents WHERE WorkoutFK = (SELECT WorkoutID FROM Workouts WHERE Name = ?) "
                "AND LiftFK = (SELECT LiftID FROM Lifts WHERE Name = ?));",
                [(workout, lift.Name, workout, lift.Name) for lift in lift_list])
        cursor.execute(
            'CREATE TABLE IF NOT EXISTS LiftHistory ('
            'LiftHistoryID INTEGER PRIMARY KEY, '
            'LiftFK INTEGER NOT NULL, '
            'Reps1 INTEGER NOT NULL, '
            'Reps2 INTEGER NOT NULL, '
            'Reps3 INTEGER NOT NULL, '
            'Weight NUMBER NOT NULL, '
            'Date DATE NOT NULL, '
            'FOREIGN KEY (LiftFK) REFERENCES Lifts(LiftID));')
        cursor.execute(
            'CREATE TABLE IF NOT EXISTS WorkoutHistory ('
            'WorkoutHistoryID INTEGER PRIMARY KEY, '
            'WorkoutFK INTEGER NOT NULL, '
            'Date DATE NOT NULL, '
            'FOREIGN KEY (WorkoutFK) REFERENCES Workouts(WorkoutID));')


def is_todays_workout_done(conn, now_date):
    cursor = conn.cursor()
    cursor.execute("SELECT Date FROM WorkoutHistory ORDER BY Date DESC LIMIT 1;")
    todays_workout_row = cursor.fetchone()
    if not todays_workout_row:
        return False
    return todays_workout_row[0] == now_date


def get_next_workout(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT Date FROM WorkoutHistory ORDER BY Date DESC LIMIT 1;")
    most_recent_workout_date_row = cursor.fetchone()
    if not most_recent_workout_date_row:
        # no data, start on push
        return "PUSH_LIFTS"
    last_six_workouts = get_last_six_workouts(conn)
    if 'REST' not in last_six_workouts and 'MISS' not in last_six_workouts and len(last_six_workouts) == 6:
        return 'REST'
    else:
        cursor.execute(
            "SELECT Workouts.Name "
            "FROM Workouts JOIN WorkoutHistory ON Workouts.WorkoutID = WorkoutHistory.WorkoutFK "
            "WHERE Workouts.Name NOT IN ('REST', 'MISS')"
            "ORDER BY Date DESC LIMIT 1;")
        last_non_rest_or_miss_workout = cursor.fetchone()[0]
        if not last_non_rest_or_miss_workout:
            return 'PUSH_LIFTS'
        if last_non_rest_or_miss_workout == 'PUSH_LIFTS':
            return 'PULL_LIFTS'
        if last_non_rest_or_miss_workout == 'PULL_LIFTS':
            return 'LEG_LIFTS'
        if last_non_rest_or_miss_workout == 'LEG_LIFTS':
            return 'PUSH_LIFTS'
    raise RuntimeError("Couldn't determine what workout day it is.")


def get_last_six_workouts(conn):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT Workouts.Name "
        "FROM Workouts JOIN WorkoutHistory ON Workouts.WorkoutID = WorkoutHistory.WorkoutFK "
        "ORDER BY Date DESC LIMIT 6;")
    last_six_workouts = [tup[0] for tup in cursor.fetchall()]
    return last_six_workouts


def fill_rests_and_misses(conn, now_date):
    # if any workout days were missed, insert the misses/rests into workout history
    cursor = conn.cursor()
    cursor.execute("SELECT Date FROM WorkoutHistory ORDER BY Date DESC LIMIT 1;")
    most_recent_workout_date_row = cursor.fetchone()
    if not most_recent_workout_date_row:
        # no workout data, return
        return
    most_recent_workout_date = most_recent_workout_date_row[0]
    last_six_workouts = get_last_six_workouts(conn)
    workout_date_tuple_to_add = []
    iteration_date = most_recent_workout_date + datetime.timedelta(days=1)
    # rest day can only be the first day in a string of misses, can also be today
    if iteration_date <= now_date:
        if 'REST' not in last_six_workouts and 'MISS' not in last_six_workouts and len(last_six_workouts) == 6:
            workout_date_tuple_to_add.append(('REST', iteration_date))
            iteration_date = iteration_date + datetime.timedelta(days=1)
    # fill in all days up to and including yesterday with misses, if there was no workout
    while iteration_date < now_date:
        workout_date_tuple_to_add.append(('MISS', iteration_date))
        iteration_date = iteration_date + datetime.timedelta(days=1)
    cursor.executemany("INSERT INTO WorkoutHistory(WorkoutFK, Date) "
                       "VALUES ((SELECT WorkoutID FROM Workouts WHERE Name = ?), ?);",
                       (tup for tup in workout_date_tuple_to_add))


def get_todays_workout_data(conn):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT Workouts.Name "
        "FROM Workouts JOIN WorkoutHistory ON Workouts.WorkoutID = WorkoutHistory.WorkoutFK "
        "ORDER BY Date DESC LIMIT 1;")
    workout = cursor.fetchone()[0]
    lift_data = []
    for lift in WORKOUTS[workout]:
        cursor.execute(
            "SELECT LiftHistory.Weight, LiftHistory.Reps1, LiftHistory.Reps2, LiftHistory.Reps3 "
            "FROM Lifts LEFT OUTER JOIN LiftHistory ON Lifts.LiftID = LiftHistory.LiftFK WHERE Lifts.Name = ? "
            "ORDER BY Date DESC LIMIT 1", (lift,))
        lift_row = cursor.fetchone()
        lift_dict = {"name": lift.Name, "weight": lift_row[0], "previous_reps": [lift_row[1], lift_row[2], lift_row[3]]}
        lift_data.append(lift_dict)
    return {"workout_id": workout, "lift_data": lift_data}


def get_new_workout_data(conn):
    cursor = conn.cursor()
    lift_data = []
    workout = get_next_workout(conn)
    for lift in WORKOUTS[workout]:
        cursor.execute(
            "SELECT LiftHistory.Weight, LiftHistory.Reps1, LiftHistory.Reps2, LiftHistory.Reps3 "
            "FROM Lifts LEFT OUTER JOIN LiftHistory ON Lifts.LiftID = LiftHistory.LiftFK WHERE Lifts.Name = ? "
            "ORDER BY Date DESC LIMIT 3", (lift.Name,))
        data = cursor.fetchall()
        lift_dict = {"name": lift.Name, "previous_reps": []}
        if data[0][0] is None:
            lift_data.append(lift_dict)
            continue
        lift_dict['previous_reps'] = [data[0][1], data[0][2], data[0][3]]
        # if last lifts reps were all greater than or equal to 12, increase weight if possible
        if data[0][1] >= 12 and data[0][2] >= 12 and data[0][3] >= 12:
            lift_dict["weight"] = \
                AVAILABLE_WEIGHTS[min(AVAILABLE_WEIGHTS.index(data[0][0]) + 1, len(AVAILABLE_WEIGHTS) - 1)]
            lift_data.append(lift_dict)
            continue
        # if this lift has been performed at least twice and the weight hadn't increased for the last workout
        if len(data) >= 2 and data[0][0] <= data[1][0]:
            rep_total_list = []
            for i in range(2):
                rep_total_list.append(data[i][1] + data[i][2] + data[i][3])
            # if the number of total reps decreased since the last workout, decrease weight
            if rep_total_list[0] < rep_total_list[1]:
                lift_dict["weight"] = \
                         AVAILABLE_WEIGHTS[max(AVAILABLE_WEIGHTS.index(data[0][0]) - 1, 0)]
                lift_data.append(lift_dict)
                continue
        # if the workouts have been exactly the same reps and weights for 3 iterations, lower weight
        if len(data) >= 3 and data[0] == data[1] == data[2]:
            lift_dict["weight"] = \
                AVAILABLE_WEIGHTS[max(AVAILABLE_WEIGHTS.index(data[0][0]) - 1, 0)]
            lift_data.append(lift_dict)
            continue
        lift_dict["weight"] = data[0][0]
        lift_data.append(lift_dict)

    return {"workout_id": workout, "lift_data": lift_data}


class Lift(colander.MappingSchema):
    set_1_reps = colander.SchemaNode(colander.Int(), validator=colander.Range(0, 100))
    set_2_reps = colander.SchemaNode(colander.Int(), validator=colander.Range(0, 100))
    set_3_reps = colander.SchemaNode(colander.Int(), validator=colander.Range(0, 100))
    dumbbell_weight = colander.SchemaNode(colander.Float(), validator=colander.OneOf(AVAILABLE_WEIGHTS))


class Lifts(colander.SequenceSchema):
    lifts = Lift()


class Workout(colander.MappingSchema):
    workout_id = colander.SchemaNode(colander.String(), validator=colander.OneOf(WORKOUTS.keys()))
    lifts = Lifts()


def process_form(dict):
    return_dict = {'workout_id': dict.pop('workout-id'), 'lifts': []}
    weight_format_string = 'weight%i'
    lift_index = 1
    while True:
        if weight_format_string % lift_index not in dict:
            break
        lift_to_add = {'dumbbell_weight': dict.pop(weight_format_string % lift_index, None)}
        for set_index in range(1, 4):
            rep_string_key = 'lift%iset%i' % (lift_index, set_index)
            lift_to_add['set_%i_reps' % set_index] = dict.pop(rep_string_key, None)
        return_dict['lifts'].append(lift_to_add)
        lift_index += 1
    return return_dict


def save_form_to_db(form, conn, now_date):
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM WorkoutHistory WHERE Date = ? LIMIT 1",
                   (now_date,))
    data = cursor.fetchall()
    if not data:
        i = 1
        for lift in WORKOUTS[form['workout-id']]:
            cursor.execute("INSERT INTO LiftHistory(LiftFk, Reps1, Reps2, Reps3, Weight, Date) "
                           "VALUES ((SELECT LiftID FROM Lifts WHERE Name = ?), ?, ?, ?, ?, ?)",
                           (lift.Name, form['lift%iset1' % i], form['lift%iset2' % i], form['lift%iset3' % i], form['weight%i' % i], now_date))
            i += 1
        cursor.execute("INSERT INTO WorkoutHistory(WorkoutFK, Date) "
                       "VALUES ((SELECT WorkoutID FROM Workouts WHERE Name = ?), ?)",
                       (form['workout-id'], now_date))
    else:
        raise colander.Invalid(None, "You aren't allowed to submit two workouts in a day")


@app.route('/', methods=['GET', 'POST'])
def show_basic():
    with sqlite3.connect(os.path.join('data', 'userdata.db'), detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) as conn:
        kwargs = {
            'unit': "lbs",
            'available_weights': AVAILABLE_WEIGHTS
        }
        now_date = datetime.date.today() + datetime.timedelta(
            days=flask.request.args.get('offset', default=0, type=int))
        fill_rests_and_misses(conn, now_date)
        if is_todays_workout_done(conn, now_date):
            kwargs = {**kwargs, **get_todays_workout_data(conn), "done": True}
        else:
            kwargs = {**kwargs, **get_new_workout_data(conn)}

        if flask.request.method == 'POST':
            schema = Workout()
            form_dict = process_form(flask.request.form.to_dict())
            try:
                schema.deserialize(form_dict)
                save_form_to_db(flask.request.form, conn, now_date)
                kwargs = {**kwargs, **get_todays_workout_data(conn), "done": True}
                return flask.render_template("index.html", **kwargs)
            except colander.Invalid:
                kwargs['fail_validation'] = True
                return flask.render_template("index.html", **kwargs)
        else:
            return flask.render_template("index.html", **kwargs)


@app.route('/stats')
def send_stats():

    # create a new plot with a title and axis labels
    weight_plot = figure(title="Dumbbell Weight Per Lift Over Time", x_axis_label='Date', y_axis_label='Weight', x_axis_type='datetime', sizing_mode='scale_width')
    volume_plot = figure(title="Volume Per Lift Over Time", x_axis_label='Date', y_axis_label='Volume', x_axis_type='datetime', sizing_mode='scale_width')
    with sqlite3.connect(os.path.join('data', 'userdata.db'),
                         detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) as conn:
        lifts_df = pandas.read_sql_query("SELECT LiftID, Name FROM Lifts;", conn)
        for name, row in lifts_df.iterrows():
            df = pandas.read_sql_query(
                "SELECT Name, Date, Weight, Reps1, Reps2, Reps3, VolumeMultiplier, AsymmetryMultiplier, "
                "(Weight * (Reps1 + Reps2 + Reps3) * VolumeMultiplier * AsymmetryMultiplier) as Volume "
                "FROM LiftHistory INNER JOIN Lifts ON LiftHistory.LiftFK = Lifts.LiftID "
                "WHERE LiftFK = %s;" % row['LiftID'], conn)
            source = ColumnDataSource(df)
            weight_plot.line(x='Date', y='Weight', source=source, color=Category20[20][row['LiftID'] % 20], legend=(row['Name'][:13] + '..') if len(row['Name']) > 15 else row['Name'])
            weight_plot.scatter(x='Date', y='Weight', source=source, color=Category20[20][row['LiftID'] % 20], size=7)
            volume_plot.line(x='Date', y='Volume', source=source, color=Category20[20][row['LiftID'] % 20], legend=(row['Name'][:13] + '..') if len(row['Name']) > 15 else row['Name'])
            volume_plot.scatter(x='Date', y='Volume', source=source, color=Category20[20][row['LiftID'] % 20], size=7)

    weight_hover = HoverTool(tooltips=[
        ("Date", "@Date{%F}"),
        ("Lift", "@Name"),
        ("Weight", "@Weight"),
        ("Reps", "(@Reps1, @Reps2, @Reps3)")
    ], formatters={"Date": "datetime"})
    volume_hover = HoverTool(tooltips=[
        ("Date", "@Date{%F}"),
        ("Lift", "@Name"),
        ("Volume", "@Volume"),
        ("Weight", "@Weight"),
        ("Reps", "(@Reps1, @Reps2, @Reps3)")
    ], formatters={"Date": "datetime"})
    grid = gridplot([[weight_plot], [volume_plot]], sizing_mode='scale_width')
    weight_plot.add_tools(weight_hover)
    weight_plot.legend.location = 'top_left'
    weight_plot.legend.label_text_font_size = "8px"
    volume_plot.add_tools(volume_hover)
    volume_plot.legend.location = 'top_left'
    volume_plot.legend.label_text_font_size = "8px"

    script, div = components(grid)
    return flask.render_template("stats.html", bokeh_script=script, bokeh_div=div)


@app.route('/static/<path:path>')
def send_static(path):
    return flask.send_from_directory('static', path)


@app.route('/favicon.ico')
def favicon():
    return flask.send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico',
                                     mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    create_tables_if_not_exist()
    app.run(host='127.0.0.1', port=8080)


