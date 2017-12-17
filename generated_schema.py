from sqlalchemy import CheckConstraint, Column, Date, DateTime, Float, ForeignKey, Integer, Numeric, Text, \
    UniqueConstraint, text, func, cast, literal
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship
from sqlalchemy.types import Integer
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()
metadata = db.metadata


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
    Weight = Column(Numeric, nullable=False)
    Date = Column(Date, nullable=False)

    Lift = relationship('Lift')

    @hybrid_property
    def volume(self):
        return self.Weight * (self.Reps1 + self.Reps2 + self.Reps3)\
               * Lift.VolumeMultiplier * Lift.AsymmetryMultiplier

    @hybrid_property
    def predicted_1_rm(self):
        return cast(self.Weight / (literal(1.0278) - (literal(0.0278)
                    * func.max(self.Reps1, self.Reps2, self.Reps3))), Integer)


class Lift(db.Model):
    __tablename__ = 'Lifts'

    LiftID = Column(Integer, primary_key=True)
    Name = Column(Text, nullable=False, unique=True)
    VolumeMultiplier = Column(Integer, CheckConstraint('VolumeMultiplier IN (1, 2)'), server_default=text("2"))
    AsymmetryMultiplier = Column(Integer, CheckConstraint('AsymmetryMultiplier IN (1, 2)'), server_default=text("1"))


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
