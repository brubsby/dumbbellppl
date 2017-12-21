from sqlalchemy import CheckConstraint, Column, Date, DateTime, Float, ForeignKey, Text, \
    UniqueConstraint, text, func, cast, literal, types
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship
from sqlalchemy.types import Integer
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()
metadata = db.metadata


class SqliteNumeric(types.TypeDecorator):

    @property
    def python_type(self):
        return object

    impl = types.String

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(types.VARCHAR(100))

    def process_bind_param(self, value, dialect):
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        float_val = float(value)
        if float_val.is_integer():
            return int(value)
        return float(value)

    def process_literal_param(self, value, dialect):
        return str(value)


class BodyweightHistory(db.Model):
    __tablename__ = 'BodyweightHistory'

    BodyweightHistoryID = Column(Integer, primary_key=True)
    Bodyweight = Column(Float, nullable=False)
    Datetime = Column(DateTime, nullable=False)


class LiftHistory(db.Model):
    __tablename__ = 'LiftHistory'

    LiftHistoryID = Column(Integer, primary_key=True)
    LiftFK = Column(ForeignKey('Lifts.LiftID'), nullable=False)
    Reps1 = Column(Integer, nullable=False)
    Reps2 = Column(Integer, nullable=False)
    Reps3 = Column(Integer, nullable=False)
    Weight = Column(SqliteNumeric, nullable=False)
    Date = Column(Date, nullable=False)

    Lift = relationship('Lift')


class Lift(db.Model):
    __tablename__ = 'Lifts'

    LiftID = Column(Integer, primary_key=True)
    Name = Column(Text, nullable=False, unique=True)
    VolumeMultiplier = Column(Integer, CheckConstraint('VolumeMultiplier IN (1, 2)'), server_default=text("2"))
    AsymmetryMultiplier = Column(Integer, CheckConstraint('AsymmetryMultiplier IN (1, 2)'), server_default=text("1"))
    BodyweightMultiplier = Column(SqliteNumeric, server_default=text("0"))

class WorkoutContent(db.Model):
    __tablename__ = 'WorkoutContents'
    __table_args__ = (
        UniqueConstraint('WorkoutFK', 'LiftFK'),
    )

    WorkoutContentID = Column(Integer, primary_key=True)
    WorkoutFK = Column(ForeignKey('Workouts.WorkoutID'), nullable=False)
    LiftFK = Column(ForeignKey('Lifts.LiftID'), nullable=False)

    Lift = relationship('Lift')
    Workout = relationship('Workout')


class WorkoutHistory(db.Model):
    __tablename__ = 'WorkoutHistory'

    WorkoutHistoryID = Column(Integer, primary_key=True)
    WorkoutFK = Column(ForeignKey('Workouts.WorkoutID'), nullable=False)
    Date = Column(Date, nullable=False)

    Workout = relationship('Workout')


class Workout(db.Model):
    __tablename__ = 'Workouts'

    WorkoutID = Column(Integer, primary_key=True)
    Name = Column(Text, nullable=False, unique=True)
