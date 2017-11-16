import flask
import sqlite3
import itertools
import datetime

import os

WORKOUTS = {
    'PUSH_LIFTS': ['Chest Press', 'Incline Fly', 'Arnold Press', 'Overhead Triceps Extension'],
    'PULL_LIFTS': ['Pull-up', 'Bent-Over Row', 'Reverse Fly', 'Shrug', 'Bicep Curl'],
    'LEG_LIFTS': ['Goblet Squat', 'Lunge', 'Single Leg Deadlift', 'Calf Raise'],
    'EVERY_OTHER_LIFTS': ['Hanging Leg Raises'],
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
    with sqlite3.connect(os.path.join('data', 'userdata.db'), detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS Lifts (LiftID INTEGER PRIMARY KEY, Name TEXT NOT NULL, UNIQUE(Name))")
        cursor.executemany("INSERT INTO Lifts (Name) SELECT ? WHERE NOT EXISTS(SELECT 1 FROM Lifts WHERE Name = ?);",
                           [(lift, lift) for lift in list(set(itertools.chain.from_iterable(WORKOUTS.values())))])
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
                [(workout, lift, workout, lift) for lift in lift_list])
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


def get_lift_day_type(conn):
    now_date = datetime.date.today()
    cursor = conn.cursor()
    cursor.execute("SELECT Date FROM WorkoutHistory ORDER BY Date DESC LIMIT 1;")
    most_recent_workout_date = cursor.fetchone()
    if not most_recent_workout_date:
        # no data, start on push
        return "PUSH_LIFTS"
    else:
        # if a day was missed, insert the misses into workout history
        most_recent_workout_date = most_recent_workout_date[0]
        dates_to_miss = []
        while most_recent_workout_date < now_date - datetime.timedelta(days=1):
            dates_to_miss.append(most_recent_workout_date)
            most_recent_workout_date = most_recent_workout_date + datetime.timedelta(days=1)
        cursor.executemany("INSERT INTO WorkoutHistory(WorkoutFK, Date) "
                           "VALUES ((SELECT WorkoutID FROM Workouts WHERE Name = ?), ?);",
                           (('MISS', date) for date in dates_to_miss))
    cursor.execute(
        "SELECT Workouts.Name "
        "FROM Workouts JOIN WorkoutHistory ON Workouts.WorkoutID = WorkoutHistory.WorkoutFK "
        "ORDER BY Date DESC LIMIT 6;")
    last_six_workouts = [tup[0] for tup in cursor.fetchall()]
    if last_six_workouts:
        if 'REST' not in last_six_workouts and 'MISS' not in last_six_workouts and len(last_six_workouts) == 6:
            return 'REST'
        else:
            last_workout = last_six_workouts[0]
            if last_workout == 'PUSH_LIFTS':
                return 'PULL_LIFTS'
            if last_workout == 'PULL_LIFTS':
                return 'LEG_LIFTS'
            if last_workout == 'LEG_LIFTS':
                return 'PUSH_LIFTS'
    return DAY_WORKOUT_DICT[datetime.datetime.now().weekday()]


def get_todays_lift_data():
    with sqlite3.connect(os.path.join('data', 'userdata.db'), detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) as conn:
        cursor = conn.cursor()
        lift_data = []
        lift_day = get_lift_day_type(conn)
        for lift in WORKOUTS[lift_day]:
            cursor.execute(
                "SELECT LiftHistory.Weight, LiftHistory.Reps1, LiftHistory.Reps2, LiftHistory.Reps3 "
                "FROM Lifts LEFT OUTER JOIN LiftHistory ON Lifts.LiftID = LiftHistory.LiftFK WHERE Lifts.Name = ? "
                "ORDER BY Date DESC LIMIT 3", (lift,))
            data = cursor.fetchall()
            if data[0][0] is None:
                lift_data.append({"name": lift})
                continue
            if data[0][1] >= 12 and data[0][2] >= 12 and data[0][3] >= 12:
                lift_data.append(
                    {"name": lift,
                     "weight":
                         AVAILABLE_WEIGHTS[min(AVAILABLE_WEIGHTS.index(data[0][0]) + 1, len(AVAILABLE_WEIGHTS) - 1)]
                     })
                continue
            if len(data) >= 2:
                rep_total_list = []
                for i in range(2):
                    rep_total_list.append(data[i][1] + data[i][2] + data[i][3])
                if rep_total_list[0] < rep_total_list[1]:
                    lift_data.append(
                        {"name": lift,
                         "weight":
                             AVAILABLE_WEIGHTS[max(AVAILABLE_WEIGHTS.index(data[0][0]) - 1, 0)]
                         })
                    continue
            if len(data) >= 3:
                if data[0] == data[1] == data[2]:
                    lift_data.append(
                        {"name": lift,
                         "weight":
                             AVAILABLE_WEIGHTS[max(AVAILABLE_WEIGHTS.index(data[0][0]) - 1, 0)]
                         })
                    continue
            lift_data.append(
                    {"name": lift, "weight": data[0][0]})

        return lift_data


def save_form_to_db(form):
    now_date = datetime.date.today()
    with sqlite3.connect(os.path.join('data', 'userdata.db'), detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM LiftHistory WHERE Date = ? LIMIT 1",
                       (now_date,))
        data = cursor.fetchall()
        if not data:
            i = 1
            for lift in WORKOUTS[form['workout-id']]:
                cursor.execute("INSERT INTO LiftHistory(LiftFk, Reps1, Reps2, Reps3, Weight, Date) "
                               "VALUES ((SELECT LiftID FROM Lifts WHERE Name = ?), ?, ?, ?, ?, ?)",
                               (lift, form['lift%iset1' % i], form['lift%iset2' % i], form['lift%iset3' % i], form['weight%i' % i], now_date))
                i += 1
            cursor.execute("INSERT INTO WorkoutHistory(WorkoutFK, Date) "
                           "VALUES ((SELECT WorkoutID FROM Workouts WHERE Name = ?), ?)",
                           (form['workout-id'], now_date))


@app.route('/')
def show_basic():
    return flask.render_template("index.html", todays_lifts=get_todays_lift_data(), unit="lbs",
                                 available_weights=AVAILABLE_WEIGHTS, workout_id=DAY_WORKOUT_DICT[datetime.datetime.now().weekday()])


@app.route('/submit', methods=['POST'])
def submit_data():
    save_form_to_db(flask.request.form)
    return show_basic()


@app.route('/static/<path:path>')
def send_static(path):
    return flask.send_from_directory('static', path)


@app.route('/favicon.ico')
def favicon():
    return flask.send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico',
                                     mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    create_tables_if_not_exist()
    app.run(host='127.0.0.1', port=80)

