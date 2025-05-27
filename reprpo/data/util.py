import re

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
