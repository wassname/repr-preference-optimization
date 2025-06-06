import re
import pandas as pd

def safe_fn(s):
    """make a safe filename from any string."""
    return "".join(
        c for c in s if c.isalpha() or c.isdigit() or c == " " or c == "_" or c == "-"
    ).rstrip()


def nice_ds_name(s):
    """make a nice name for the dataset"""
    rm = [
        "genies_preferences-",
        "genies_",
        "genies-preferences-",
        "wassname/",
        "_expression_preferences",
    ]
    for r in rm:
        s = s.replace(r, "")

    # and remove "\[\:\d+\]" and replace with space

    s = re.sub(r"\[\:\d+\]", " ", s)

    # also remove -test or -data or -test-data or -train from end
    for suffix in ["-test", "-data", "-test-data", "-train"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]

    # replace
    rp = {
        "medica-dpo-v2": "cn-medical",
        "-": "_",
    }
    for k, v in rp.items():
        s = s.replace(k, v)

    return s


def unique_ordered(lst):
    out = []
    for item in lst:
        if item not in out:
            out.append(item)
    return out


def sort_str(cols, first:list=[], last:list=[], sort_middle=False, unique=False) -> list:
    """
    Util function for sorting a list of strings, making sure some items appear first and last and the rest sorted in the middle.

    Usage:
    >>> sort_str(['banana', 'apple', 'cherry', 'date', 'cantelope'], ['apple', 'banana'], ['date'])
    ['apple', 'banana',  'cantelope', 'cherry', 'date']
    """
    middle = [x for x in cols if (x not in first) and (x not in last)]
    if sort_middle:
        middle.sort()
    first = [x for x in first if (x in cols) and (x not in last)]
    last = [x for x in last if x in cols]
    if unique:
        first = unique_ordered(first)
        last = unique_ordered(last)
        middle = unique_ordered(middle)
    return first + middle + last


def df_sort_cols(df, first:list=[], last:list=[], sort_middle=False) -> list:
    """
    Sorts the columns of a DataFrame, making sure some items appear first and last and the rest optionally sorted in the middle.
    
    Usage:
    >>> df_sort_cols(df, ['col1', 'col2'], ['col3'])
    """
    sorted_cols = sort_str(list(df.columns), first=first, last=last, sort_middle=sort_middle)
    return df[sorted_cols]


