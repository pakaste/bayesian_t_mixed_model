import datetime


def get_datetime_now():
    """
        Return the datetime now as a string, with the following format:
        YYYY_MM_DD
    """
    dt_now = datetime.datetime.now()
    year = str(dt_now.year)
    month = str(dt_now.month)
    day = str(dt_now.day)
    dt_now_string = year + '_' + month + '_' + day

    return dt_now_string