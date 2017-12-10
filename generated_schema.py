from sqlalchemy import CheckConstraint, Column, Date, DateTime, Float, ForeignKey, Integer, Numeric, Text, UniqueConstraint, text
from sqlalchemy.orm import relationship
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
