# coding: utf-8
from sqlalchemy import CheckConstraint, Column, Date, DateTime, Float, ForeignKey, Integer, Numeric, Text, UniqueConstraint, text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()
metadata = Base.metadata


class BodyweightHistory(Base):
    __tablename__ = 'BodyweightHistory'

    BodyweightHistoryID = Column(Integer, primary_key=True)
    Bodyweight = Column(Float, nullable=False)
    Datetime = Column(DateTime, nullable=False)


class LiftHistory(Base):
    __tablename__ = 'LiftHistory'

    LiftHistoryID = Column(Integer, primary_key=True)
    LiftFK = Column(ForeignKey(u'Lifts.LiftID'), nullable=False)
    Reps1 = Column(Integer, nullable=False)
    Reps2 = Column(Integer, nullable=False)
    Reps3 = Column(Integer, nullable=False)
    Weight = Column(Numeric, nullable=False)
    Date = Column(Date, nullable=False)

    Lift = relationship(u'Lift')


class Lift(Base):
    __tablename__ = 'Lifts'
    __table_args__ = (
        CheckConstraint(u'VolumeMultiplier IN (1, 2) and AsymmetryMultiplier IN (1, 2))'),
    )

    LiftID = Column(Integer, primary_key=True)
    Name = Column(Text, nullable=False, unique=True)
    VolumeMultiplier = Column(Integer, server_default=text("2"))
    AsymmetryMultiplier = Column(Integer, server_default=text("1"))


class WorkoutContent(Base):
    __tablename__ = 'WorkoutContents'
    __table_args__ = (
        UniqueConstraint('WorkoutFK', 'LiftFK'),
    )

    WorkoutContentID = Column(Integer, primary_key=True)
    WorkoutFK = Column(ForeignKey(u'Workouts.WorkoutID'), nullable=False)
    LiftFK = Column(ForeignKey(u'Lifts.LiftID'), nullable=False)

    Lift = relationship(u'Lift')
    Workout = relationship(u'Workout')


class WorkoutHistory(Base):
    __tablename__ = 'WorkoutHistory'

    WorkoutHistoryID = Column(Integer, primary_key=True)
    WorkoutFK = Column(ForeignKey(u'Workouts.WorkoutID'), nullable=False)
    Date = Column(Date, nullable=False)

    Workout = relationship(u'Workout')


class Workout(Base):
    __tablename__ = 'Workouts'

    WorkoutID = Column(Integer, primary_key=True)
    Name = Column(Text, nullable=False, unique=True)
