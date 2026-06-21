CITY_SLUGS = [
    "beijing",
    "shanghai",
    "nanjing",
    "suzhou",
    "hangzhou",
    "shenzhen",
    "chengdu",
    "wuhan",
    "guangzhou",
    "chongqing",
]

CITY_NAMES = {
    "zh": ["北京", "上海", "南京", "苏州", "杭州", "深圳", "成都", "武汉", "广州", "重庆"],
    "en": [
        "Beijing",
        "Shanghai",
        "Nanjing",
        "Suzhou",
        "Hangzhou",
        "Shenzhen",
        "Chengdu",
        "Wuhan",
        "Guangzhou",
        "Chongqing",
    ],
}

# Corrections for known English-database translation drift.  Keep these at the
# data-loading boundary so the planner, POI routing, and official commonsense
# checks all use the same canonical name without changing evaluation logic.
EN_POI_NAME_CORRECTIONS = {
    "Sola Bistro": "Bistro Sola",
}


def normalize_lang(lang=None, en_version=False):
    if en_version:
        return "en"
    if lang is None:
        return "zh"
    lang = str(lang).lower()
    if lang in {"zh", "cn", "chinese"}:
        return "zh"
    if lang in {"en", "english"}:
        return "en"
    raise ValueError("lang must be one of: zh, en")


def city_names(lang):
    return CITY_NAMES[normalize_lang(lang)]


def city_lookup(lang):
    return dict(zip(CITY_SLUGS, city_names(lang)))


def canonical_poi_name(name, lang):
    if normalize_lang(lang) == "en":
        return EN_POI_NAME_CORRECTIONS.get(name, name)
    return name


def database_dirname(lang):
    return "database_en" if normalize_lang(lang) == "en" else "database"


def relative_database_path(lang, *parts):
    dirname = database_dirname(lang)
    suffix = "/".join(parts)
    return "../../{}/{}".format(dirname, suffix).rstrip("/")
