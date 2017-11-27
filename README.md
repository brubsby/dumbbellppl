# dumbbellppl

Small webapp to help track/guide progress on the dumbbell push push pull routine described in https://www.reddit.com/r/Fitness/comments/2e79y4/dumbbell_ppl_proposed_alternative_to_dumbbell/

## Getting Started

Cloning/forking the repo, and then running:
```
pip install -r requirements.txt
```
and running:
```
python main.py
```
should be enough to get you up and running for the time being.

The sqlite database is created on first run, and schema created as well.

### Prerequisites

Written specifically for Python 3.6+, maybe compatible with earlier versions, untested.

## Deployment

I haven't tested deploying this website outside of a development environment, it still doesn't have separate users implemented, etc.

## Built With

* [Python 3.6+](https://www.python.org/)
* [Flask](http://flask.pocoo.org/) - Web framework
* [SQLite](https://www.sqlite.org/) - RDBMS
* [Bokeh](https://bokeh.pydata.org/en/latest/) - Used for graphing lift statistics
* [Colander](https://docs.pylonsproject.org/projects/colander/en/latest/) - Backend data validation
* [JQuery](https://jquery.com/)
* [JQuery Validation Plugin](https://jqueryvalidation.org/) - Frontend data validation
* [Bootstrap 4](https://getbootstrap.com/) - Frontend component library

## Contributing

I welcome contributers, but the intended scope of this project is fairly small.

## Versioning

Currently not versioning the project, open for suggestions if necessary

## Authors

* **Tyler Busby** - *Initial work* - [typicalTYLER](https://github.com/typicalTYLER)

See also the list of [contributors](https://github.com/typicalTYLER/dumbbellppl/contributors) who participated in this project.

## Acknowledgments

* Thanks to [/u/gregariousHermit](https://www.reddit.com/user/gregariousHermit) from reddit for creating the initial program
