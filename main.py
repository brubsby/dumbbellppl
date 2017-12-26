import traceback
from collections import namedtuple, OrderedDict
from functools import partial

import flask
import itertools
import datetime
import colander
import math
import json

from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.palettes import Category20
from bokeh.layouts import gridplot
from bokeh.models import HoverTool, ColumnDataSource
import pandas
from sqlalchemy import literal
from flask_migrate import Migrate

from schema import db, Lift, WorkoutContent, Workout, WorkoutHistory, BodyweightHistory, LiftHistory

import os

TREAT_ZERO_REPS_AS_SKIP = True

LiftTuple = namedtuple("Lift", ['Name', 'VolumeMultiplier', 'AsymmetryMultiplier', 'BodyweightMultiplier'])
LiftTuple.__new__.__defaults__ = (None, 2, 1)  # most dumbbell lifts use two dumbbells in unison

WORKOUTS = OrderedDict(sorted({
    'PUSH_LIFTS': [
        LiftTuple('Chest Press'),
        LiftTuple('Incline Fly'),
        LiftTuple('Arnold Press'),
        LiftTuple('Overhead Triceps Extension', 1, 1)
    ],
    'PULL_LIFTS': [
        LiftTuple('Pull-up', 1, 1, 1),
        LiftTuple('Bent-Over Row', 1, 2),
        LiftTuple('Reverse Fly'),
        LiftTuple('Shrug'),
        LiftTuple('Bicep Curl')
    ],
    'LEG_LIFTS': [
        LiftTuple('Goblet Squat', 1, 1),
        LiftTuple('Lunge', 2, 2),
        LiftTuple('Single Leg Deadlift', 1, 2),
        LiftTuple('Calf Raise')
    ],
    'EVERY_OTHER_LIFTS': [
        LiftTuple('Hanging Leg Raises', 1, 2)
    ],
    'REST': [],
    'MISS': []
}.items()))

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
DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'userdata.db')
os.makedirs(os.path.split(DB_PATH)[0], exist_ok=True)
DB_URI = 'sqlite:///{}'.format(DB_PATH)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URI
db.init_app(app)
migrate = Migrate(app, db)


@app.before_first_request
def initialize_db():
    # create tables
    db.create_all()

    # add lifts
    lifts = [
        Lift(
            Name=lift.Name,
            VolumeMultiplier=lift.VolumeMultiplier,
            AsymmetryMultiplier=lift.AsymmetryMultiplier)
        for lift in list(set(itertools.chain.from_iterable(WORKOUTS.values())))
    ]
    for lift in lifts:
        if not Lift.query.filter(Lift.Name == lift.Name).count():
            db.session.add(lift)

    # add workouts
    workouts = [Workout(
            Name=workout)
        for workout in WORKOUTS.keys()
    ]
    for workout in workouts:
        if not Workout.query.filter(Workout.Name == workout.Name).count():
            db.session.add(workout)

    # add the contents of the workouts
    for workout_name, lift_list in WORKOUTS.items():
        workout_id_query = db.session.query(Workout.WorkoutID).filter(Workout.Name == workout_name)
        for lift in lift_list:
            lift_id_query = db.session.query(Lift.LiftID).filter(Lift.Name == lift.Name)
            workout_content_query = WorkoutContent.query.filter(
                WorkoutContent.WorkoutFK == workout_id_query.subquery().c.WorkoutID,
                WorkoutContent.LiftFK == lift_id_query.subquery().c.LiftID
            )
            if not workout_content_query.count():
                workout_content = WorkoutContent(WorkoutFK=workout_id_query.first()[0],
                                                 LiftFK=lift_id_query.first()[0])
                db.session.add(workout_content)

    db.session.commit()


def is_todays_workout_done(now_date):
    row = db.session.query(WorkoutHistory.Date).order_by(WorkoutHistory.Date.desc()).first()
    if not row:
        return False
    return row[0] == now_date


def get_next_workout():
    most_recent_workout_date_row = db.session.query(WorkoutHistory.Date).order_by(WorkoutHistory.Date.desc()).first()
    if not most_recent_workout_date_row:
        # no data, start on push
        return "PUSH_LIFTS"
    last_six_workouts = get_last_six_workouts()
    if 'REST' not in last_six_workouts and 'MISS' not in last_six_workouts and len(last_six_workouts) == 6:
        return 'REST'
    else:
        last_non_rest_or_miss_workout = db.session.query(Workout.Name)\
            .filter(Workout.WorkoutID == WorkoutHistory.WorkoutFK).filter(Workout.Name.notin_(('REST', 'MISS')))\
            .order_by(WorkoutHistory.Date.desc()).first()[0]
        if not last_non_rest_or_miss_workout:
            return 'PUSH_LIFTS'
        if last_non_rest_or_miss_workout == 'PUSH_LIFTS':
            return 'PULL_LIFTS'
        if last_non_rest_or_miss_workout == 'PULL_LIFTS':
            return 'LEG_LIFTS'
        if last_non_rest_or_miss_workout == 'LEG_LIFTS':
            return 'PUSH_LIFTS'
    raise RuntimeError("Couldn't determine what workout day it is.")


def get_last_six_workouts():
    rows = db.session.query(Workout.Name).filter(Workout.WorkoutID == WorkoutHistory.WorkoutFK).\
        order_by(WorkoutHistory.Date.desc()).limit(6).all()
    last_six_workouts = [tup[0] for tup in rows]
    return last_six_workouts


def get_previous_bodyweight():
    row = db.session.query(BodyweightHistory.Bodyweight).order_by(BodyweightHistory.Datetime.desc()).first()
    if row:
        return row[0]
    else:
        return None


def add_nhema_column_to_dataframe(dataframe, column_name, tau):
    nhema_series = pandas.Series()
    kwargs = {}
    for i, row in enumerate(dataframe.itertuples()):
        value = getattr(row, column_name)
        kwargs['timestamp'] = row.Index.timestamp()
        average = non_homogeneous_exponential_moving_average(value, tau, **kwargs)
        nhema_series.set_value(i, average)
        kwargs['last_value'] = value
        kwargs['last_timestamp'] = kwargs['timestamp']
        kwargs['last_average'] = average
    dataframe[column_name + '_nhema'] = nhema_series.values


def non_homogeneous_exponential_moving_average(value, tau, last_average=None, last_timestamp=None, last_value=None,
                                               timestamp=None):
    if any(x is None for x in [last_average, last_timestamp, last_value, timestamp]):
        return value
    alpha = (timestamp - last_timestamp) / float(tau)
    mu = math.exp(-alpha)
    v = (1 - mu) / alpha
    average = mu * last_average + (1 - mu) * value + (mu - v) * (value - last_value)
    return average


def fill_rests_and_misses(now_date):
    # if any workout days were missed, insert the misses/rests into workout history
    most_recent_workout_date_row = db.session.query(WorkoutHistory.Date).order_by(WorkoutHistory.Date.desc()).first()
    if not most_recent_workout_date_row:
        # no workout data, return
        return
    most_recent_workout_date = most_recent_workout_date_row[0]
    last_six_workouts = get_last_six_workouts()
    workout_histories_to_add = []
    iteration_date = most_recent_workout_date + datetime.timedelta(days=1)
    # rest day can only be the first day in a string of misses, can also be today
    if iteration_date <= now_date:
        if 'REST' not in last_six_workouts and 'MISS' not in last_six_workouts and len(last_six_workouts) == 6:
            workout_histories_to_add.append(WorkoutHistory(
                Workout=db.session.query(Workout).filter(Workout.Name == 'REST').first(),
                Date=iteration_date))
            iteration_date = iteration_date + datetime.timedelta(days=1)
    # fill in all days up to and including yesterday with misses, if there was no workout
    while iteration_date < now_date:
        workout_histories_to_add.append(WorkoutHistory(
            Workout=db.session.query(Workout).filter(Workout.Name == 'MISS').first(),
            Date=iteration_date))
        iteration_date = iteration_date + datetime.timedelta(days=1)
    db.session.add_all(workout_histories_to_add)
    db.session.commit()


def get_todays_workout_data():
    workout_data = db.session.query(Workout.Name, WorkoutHistory.Notes).filter(
        Workout.WorkoutID == WorkoutHistory.WorkoutFK).order_by(WorkoutHistory.Date.desc()).first()
    workout = workout_data[0]
    workout_notes = workout_data[1]
    lift_data = []
    for lift in get_workout_contents(workout):
        lift_row = db.session.query(
            LiftHistory.Weight,
            LiftHistory.Reps1,
            LiftHistory.Reps2,
            LiftHistory.Reps3,
            LiftHistory.Notes)\
            .filter(LiftHistory.LiftFK == Lift.LiftID)\
            .filter(Lift.Name == lift.Name)\
            .order_by(LiftHistory.Date.desc()).first()
        lift_dict = {
            "name": lift.Name,
            "weight": lift_row[0],
            "previous_reps": [lift_row[1], lift_row[2], lift_row[3]],
            "previous_notes": lift_row[4]
        }
        lift_data.append(lift_dict)
    return {"workout_id": workout, "workout_notes": workout_notes, "lift_data": lift_data}


def get_workout_contents(workout):
    return db.session.query(Lift)\
        .filter(Workout.Name == workout)\
        .filter(WorkoutContent.WorkoutFK == Workout.WorkoutID)\
        .filter(WorkoutContent.LiftFK == Lift.LiftID).all()


def get_all_workouts():
    return [row[0] for row in db.session.query(Workout.Name).all()]


def get_new_workout_data():
    lift_data = []
    workout = get_next_workout()
    for lift in get_workout_contents(workout):
        data = db.session.query(
            LiftHistory.Weight,
            LiftHistory.Reps1,
            LiftHistory.Reps2,
            LiftHistory.Reps3,
            LiftHistory.Notes)\
            .filter(Lift.LiftID == LiftHistory.LiftFK)\
            .filter(Lift.Name == lift.Name)\
            .order_by(LiftHistory.Date.desc())\
            .limit(3).all()
        lift_dict = {"name": lift.Name, "previous_reps": []}
        if not data:
            lift_data.append(lift_dict)
            continue
        lift_dict['previous_reps'] = [data[0][1], data[0][2], data[0][3]]
        if data[0][4]:
            lift_dict['previous_notes'] = data[0][4]
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


class LiftColander(colander.MappingSchema):
    set_1_reps = colander.SchemaNode(colander.Int(), validator=colander.Range(0, 100))
    set_2_reps = colander.SchemaNode(colander.Int(), validator=colander.Range(0, 100))
    set_3_reps = colander.SchemaNode(colander.Int(), validator=colander.Range(0, 100))
    dumbbell_weight = colander.SchemaNode(colander.Float(), validator=colander.OneOf(AVAILABLE_WEIGHTS))


class LiftsColander(colander.SequenceSchema):
    lifts = LiftColander()


@colander.deferred
def deferred_workout_validator(node, kw):
    workouts = kw.get('workouts', [])
    return colander.OneOf(workouts)


class WorkoutColander(colander.MappingSchema):
    workout_id = colander.SchemaNode(colander.String(), validator=deferred_workout_validator)
    lifts = LiftsColander()


class BodyweightColander(colander.MappingSchema):
    bodyweight = colander.SchemaNode(colander.Float(), validator=colander.Range(0, 500))


def process_workout_form(form_dict):
    return_dict = {'workout_id': form_dict.pop('workout-id'),
                   'notes': form_dict.pop('workout-notes'),
                   'lifts': []}
    weight_format_string = 'weight%i'
    lift_notes_format_string = 'lift%inotes'
    lift_index = 1
    while True:
        if weight_format_string % lift_index not in form_dict:
            break
        lift_to_add = {'dumbbell_weight': form_dict.pop(weight_format_string % lift_index, None)}
        for set_index in range(1, 4):
            rep_string_key = 'lift%iset%i' % (lift_index, set_index)
            lift_to_add['set_%i_reps' % set_index] = form_dict.pop(rep_string_key, None)
        lift_to_add['notes'] = form_dict.pop(lift_notes_format_string % lift_index, None)
        return_dict['lifts'].append(lift_to_add)
        lift_index += 1
    return return_dict


def save_workout_form_to_db(form, now_date):
    todays_workout_exists = db.session.query(literal(True)).filter(
        db.session.query(WorkoutHistory).filter(WorkoutHistory.Date == now_date).exists()).scalar()
    if not todays_workout_exists:
        i = 1
        lift_histories_to_add = []
        for lift in get_workout_contents(form['workout-id']):
            lift_histories_to_add.append(LiftHistory(
                Lift=db.session.query(Lift).filter(Lift.Name == lift.Name).first(),
                Reps1=form['lift%iset1' % i],
                Reps2=form['lift%iset2' % i],
                Reps3=form['lift%iset3' % i],
                Weight=form['weight%i' % i],
                Date=now_date,
                Notes=form['lift%inotes' % i]
            ))
            i += 1
        db.session.add_all(lift_histories_to_add)
        db.session.add(WorkoutHistory(
            Workout=db.session.query(Workout).filter(Workout.Name == form['workout-id']).first(),
            Date=now_date,
            Notes=form['workout-notes']
        ))
        db.session.commit()
    else:
        raise colander.Invalid(None, "You aren't allowed to submit two workouts in a day")


def save_bodyweight_form_to_db(form, now):
    db.session.add(BodyweightHistory(Bodyweight=form['bodyweight'], Datetime=now))
    db.session.commit()


@app.route('/', methods=['GET', 'POST'])
def show_basic():
    kwargs = {
        'unit': "lbs",
        'available_weights': AVAILABLE_WEIGHTS
    }
    now_date = datetime.date.today() + datetime.timedelta(
        days=flask.request.args.get('offset', default=0, type=int))
    fill_rests_and_misses(now_date)
    if is_todays_workout_done(now_date):
        kwargs = {**kwargs, **get_todays_workout_data(), "done": True}
    else:
        kwargs = {**kwargs, **get_new_workout_data()}

    if flask.request.method == 'POST':
        schema = WorkoutColander().bind(
            workouts=get_all_workouts()
        )
        form_dict = process_workout_form(flask.request.form.to_dict())
        try:
            schema.deserialize(form_dict)
            save_workout_form_to_db(flask.request.form, now_date)
            kwargs = {**kwargs, **get_todays_workout_data(), "done": True}
            return flask.render_template("index.html", **kwargs)
        except colander.Invalid:
            kwargs['fail_validation'] = True
            return flask.render_template("index.html", **kwargs)
    else:
        return flask.render_template("index.html", **kwargs)


@app.route('/bodyweight', methods=['GET', 'POST'])
def show_bodyweight_tracking():
    kwargs = {
        'unit': "lbs"
    }
    now = datetime.datetime.now() + datetime.timedelta(
        days=flask.request.args.get('offset', default=0, type=int))
    previous_bodyweight = get_previous_bodyweight()
    if previous_bodyweight:
        kwargs['previous_bodyweight'] = previous_bodyweight
    if flask.request.method == 'POST':
        schema = BodyweightColander()
        form_dict = flask.request.form.to_dict()
        try:
            schema.deserialize(form_dict)
            save_bodyweight_form_to_db(flask.request.form, now)
            return flask.redirect(flask.url_for('.show_stats', _anchor='bodyweight'), code=302)
        except colander.Invalid:
            print(traceback.format_exc())
            kwargs['fail_validation'] = True
            return flask.render_template("bodyweight.html", **kwargs)
    else:
        return flask.render_template("bodyweight.html", **kwargs)


def calculate_volume(weight, reps1, reps2, reps3,
                     volume_multiplier, asymmetry_multiplier,
                     bodyweight_multiplier, bodyweight):
    return (weight + (bodyweight * bodyweight_multiplier))\
           * (reps1 + reps2 + reps3) * volume_multiplier * asymmetry_multiplier


def calculate_volume_for_row(row):
    if not len(row):
        return None
    return calculate_volume(
        row['LiftHistory_Weight'],
        row['LiftHistory_Reps1'],
        row['LiftHistory_Reps2'],
        row['LiftHistory_Reps3'],
        row['Lifts_VolumeMultiplier'],
        row['Lifts_AsymmetryMultiplier'],
        row['Lifts_BodyweightMultiplier'],
        row['BodyweightHistory_Bodyweight_nhema']
    )


def calculate_predicted_1_rm_row(row, formula='Brzycki'):
    if not len(row):
        return None
    return predicted_1_rm(
        row['LiftHistory_Weight'],
        row['LiftHistory_Reps1'],
        row['LiftHistory_Reps2'],
        row['LiftHistory_Reps3'],
        row['Lifts_BodyweightMultiplier'],
        row['BodyweightHistory_Bodyweight_nhema'],
        formula=formula
    )


def predicted_1_rm(weight, reps1, reps2, reps3, bodyweight_multiplier, bodyweight, formula='Brzycki'):
    exercise_weight = weight + (bodyweight_multiplier * bodyweight)
    max_reps = max([reps1, reps2, reps3])
    if formula == 'Brzycki':
        return int(exercise_weight / (1.0278 - (0.0278 * max_reps)))
    elif formula == 'McGlothin':
        return int(100 * exercise_weight / (101.3 - (2.67123 * max_reps)))
    elif formula == 'Lombardi':
        return int(exercise_weight * pow(max_reps, 0.1))
    elif formula == 'Mayhew':
        return int(100 * exercise_weight / (52.2 + (41.9 * pow(math.e, (-0.055 * max_reps)))))
    elif formula == 'OConner':
        return int(exercise_weight * (1 + max_reps / 40))
    elif formula == 'Wathan':
        return int(100 * exercise_weight / (48.8 + (53.8 * pow(math.e, (-0.075 * max_reps)))))


def get_lifting_plots():
    LineScatter = namedtuple("LineScatter", ["title", "y_axis_label", "column"])
    line_scatters = [
        LineScatter("Dumbbell Weight Per Lift Over Time", "Weight", "LiftHistory_Weight"),
        LineScatter("Volume Per Lift Over Time", "Volume", "Volume"),
        LineScatter("Predicted 1RM Per Lift Over Time", "1RM", "Predicted1RM")
    ]
    # create plots
    plots = []
    for line_scatter in line_scatters:
        plots.append(figure(title=line_scatter.title, x_axis_label='Date', y_axis_label=line_scatter.y_axis_label,
                            x_axis_type='datetime', sizing_mode='scale_width'))
    lifts_df = pandas.read_sql_query(db.session.query(Lift.LiftID, Lift.Name).selectable, db.session.get_bind())
    interpolated_bodyweights_dataframe = get_interpolated_bodyweights()
    for name, row in lifts_df.iterrows():
        plot_query_base = db.session.query(
            LiftHistory.LiftHistoryID,
            Lift.Name,
            LiftHistory.Date,
            LiftHistory.Weight,
            LiftHistory.Reps1,
            LiftHistory.Reps2,
            LiftHistory.Reps3,
            Lift.VolumeMultiplier,
            Lift.AsymmetryMultiplier,
            Lift.BodyweightMultiplier
        ).filter(LiftHistory.LiftFK == Lift.LiftID).filter(LiftHistory.LiftFK == row['Lifts_LiftID'])
        if TREAT_ZERO_REPS_AS_SKIP:
            plot_query_base.filter(
                ~((LiftHistory.Reps1 == 0) & (LiftHistory.Reps2 == 0) & (LiftHistory.Reps3 == 0)))
        df = pandas.read_sql_query(plot_query_base.selectable,
                                   db.session.get_bind(), "LiftHistory_LiftHistoryID")
        df = add_interpolated_bodyweights(df, interpolated_bodyweights_dataframe=interpolated_bodyweights_dataframe)
        df['Volume'] = df.apply(calculate_volume_for_row, axis=1)
        df['Predicted1RM'] = df.apply(partial(calculate_predicted_1_rm_row, formula='Wathan'), axis=1)
        source = ColumnDataSource(df)
        for line_scatter, plot in zip(line_scatters, plots):
            plot.line(x='LiftHistory_Date', y=line_scatter.column, source=source,
                      color=Category20[20][row['Lifts_LiftID'] % 20],
                      legend=(row['Lifts_Name'][:13] + '..') if len(row['Lifts_Name']) > 15 else row['Lifts_Name'])
            plot.scatter(x='LiftHistory_Date', y=line_scatter.column, source=source,
                         color=Category20[20][row['Lifts_LiftID'] % 20],
                         size=7)

    default_tooltips = [
        ("Date", "@LiftHistory_Date{%F}"),
        ("Lift", "@Lifts_Name"),
        ("Weight", "@LiftHistory_Weight"),
        ("Reps", "(@LiftHistory_Reps1, @LiftHistory_Reps2, @LiftHistory_Reps3)")
    ]
    hovers = []
    for line_scatter in line_scatters:
        y_tooltip = (line_scatter.y_axis_label, "@%s" % line_scatter.column)
        if y_tooltip not in default_tooltips:
            tooltips = default_tooltips[:2] + [y_tooltip] + default_tooltips[2:]
        else:
            tooltips = default_tooltips
        hovers.append(HoverTool(tooltips=tooltips, formatters={"LiftHistory_Date": "datetime"}))
    for plot, hover in zip(plots, hovers):
        plot.add_tools(hover)
        plot.legend.location = 'top_left'
        plot.legend.label_text_font_size = "8px"
    return plots


def add_interpolated_bodyweights(to_add_df, interpolated_bodyweights_dataframe=None):
    if interpolated_bodyweights_dataframe is None:
        interpolated_bodyweights_dataframe = get_interpolated_bodyweights()
    return to_add_df.merge(interpolated_bodyweights_dataframe, left_on='LiftHistory_Date', right_index=True)


def get_interpolated_bodyweights():
    df = get_bodyweight_dataframe()
    df.index = df.index.round('h')
    df = df.resample('H').interpolate()
    df = df.resample('D').asfreq()

    first_workout = db.session.query(WorkoutHistory.Date).order_by(WorkoutHistory.Date.asc()).limit(1).scalar()
    last_workout = db.session.query(WorkoutHistory.Date).order_by(WorkoutHistory.Date.desc()).limit(1).scalar()
    lifts_index = pandas.DatetimeIndex(start=first_workout, end=last_workout, freq='D')
    df = df.reindex(df.index.union(lifts_index))
    df.index = df.index.date

    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    return df


def get_bodyweight_plot():
    bodyweights_df = get_bodyweight_dataframe()
    source = ColumnDataSource(bodyweights_df)
    bodyweight_plot = figure(title="Bodyweight Over Time", x_axis_label='Datetime', y_axis_label='Bodyweight',
                             x_axis_type='datetime', sizing_mode='scale_width')
    bodyweight_plot.line(x='BodyweightHistory_Datetime', y='BodyweightHistory_Bodyweight_nhema', source=source,
                         color=Category20[20][0])
    bodyweight_plot.scatter(x='BodyweightHistory_Datetime', y='BodyweightHistory_Bodyweight', source=source,
                            color=Category20[20][0], size=7)
    bodyweight_plot.add_tools(HoverTool(tooltips=[
        ("Bodyweight", "@BodyweightHistory_Bodyweight"),
        ("BodyweightEMA", "@BodyweightHistory_Bodyweight_nhema"),
        ("Datetime", "@BodyweightHistory_Datetime{%T %F}")
    ], formatters={"BodyweightHistory_Datetime": "datetime"}))
    return bodyweight_plot


def get_bodyweight_dataframe():
    bodyweights_df = \
        pandas.read_sql_query(db.session.query(BodyweightHistory.Bodyweight, BodyweightHistory.Datetime)
                              .order_by(BodyweightHistory.Datetime.asc()).selectable,
                              db.session.get_bind(),
                              parse_dates=["BodyweightHistory_Datetime"],
                              index_col="BodyweightHistory_Datetime")
    add_nhema_column_to_dataframe(bodyweights_df, 'BodyweightHistory_Bodyweight', 288000)
    return bodyweights_df


@app.route('/stats')
def show_stats():
    lifting_plots = get_lifting_plots()
    lift_grid = gridplot([[plot] for plot in lifting_plots], sizing_mode='scale_width')
    bodyweight_plot = get_bodyweight_plot()

    lift_grid_script, lift_grid_div = components(lift_grid)
    bodyweight_script, bodyweight_div = components(bodyweight_plot)
    return flask.render_template("stats.html", lift_grid_script=lift_grid_script, lift_grid_div=lift_grid_div,
                                 bodyweight_script=bodyweight_script, bodyweight_div=bodyweight_div)


@app.route('/calendar')
def show_calendar():
    workout_calendar_data = [
        # add day because graph counts midnight as day before
        {"date": row[0] + datetime.timedelta(days=1), "count": 1}
        for row
        in db.session.query(WorkoutHistory.Date)
            .filter(WorkoutHistory.WorkoutFK == Workout.WorkoutID)
            .filter((Workout.Name != 'REST') & (Workout.Name != 'MISS')).all()
    ]

    def date_handler(obj):
        return (obj.isoformat()
                if isinstance(obj, (datetime.datetime, datetime.date))
                else None)
    return flask.render_template("calendar.html", workout_calendar_data=json.dumps(workout_calendar_data,
                                                                                   default=date_handler))


@app.route('/static/<path:path>')
def send_static(path):
    return flask.send_from_directory('static', path)


@app.route('/favicon.ico')
def favicon():
    return flask.send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico',
                                     mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
