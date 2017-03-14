import datetime as dt

from xam.util import datetime_range


since = dt.datetime(year=2017, month=1, day=1)
until = dt.datetime(year=2017, month=2, day=1)
step = dt.timedelta(weeks=1)

dates = datetime_range(since, until, step)

for date in dates:
    print(date)
