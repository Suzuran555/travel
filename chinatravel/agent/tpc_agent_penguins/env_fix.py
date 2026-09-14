"""English-database POI-name drift correction, applied WITHOUT touching stock files.

The EN database release names one restaurant inconsistently across layers
("Sola Bistro" vs "Bistro Sola"), which breaks position grounding for plans
that book it. Our branch fixes this at the data-loading boundary
(environment/language.py canonical_poi_name). On a stock checkout we get the
same effect by patching the loader classes at import time and repairing any
instances that were built before the package was imported (run_tpc constructs
WorldEnv before the agent loads).

Everything here is idempotent and confined to the running process.
"""

EN_POI_NAME_CORRECTIONS = {
    "Sola Bistro": "Bistro Sola",
}

_applied = False


def _fix_name(name):
    return EN_POI_NAME_CORRECTIONS.get(name, name)


def fix_poi_instance(poi):
    """Re-key Poi.data ({city: {name: (lat, lon)}}) for the EN language."""
    if getattr(poi, "lang", None) not in (None, "en"):
        return
    data = getattr(poi, "data", None)
    if not isinstance(data, dict):
        return
    for city, table in data.items():
        if isinstance(table, dict):
            data[city] = { _fix_name(k): v for k, v in table.items() }


def fix_restaurants_instance(rest):
    """Rename the 'name' column of every per-city restaurants dataframe."""
    if getattr(rest, "lang", None) not in (None, "en"):
        return
    data = getattr(rest, "data", None)
    if not isinstance(data, dict):
        return
    for city, df in data.items():
        try:
            if "name" in df.columns:
                df["name"] = df["name"].map(_fix_name)
        except Exception:
            continue


def apply_class_patches():
    """Patch stock Poi/Restaurants __init__ so future instances are corrected."""
    global _applied
    if _applied:
        return
    _applied = True

    from chinatravel.environment.tools.poi.apis import Poi
    from chinatravel.environment.tools.restaurants.apis import Restaurants

    if not getattr(Poi, "_penguins_name_fix", False):
        _orig_poi_init = Poi.__init__

        def _poi_init(self, *a, **kw):
            _orig_poi_init(self, *a, **kw)
            fix_poi_instance(self)

        Poi.__init__ = _poi_init
        Poi._penguins_name_fix = True

    if not getattr(Restaurants, "_penguins_name_fix", False):
        _orig_rest_init = Restaurants.__init__

        def _rest_init(self, *a, **kw):
            _orig_rest_init(self, *a, **kw)
            fix_restaurants_instance(self)

        Restaurants.__init__ = _rest_init
        Restaurants._penguins_name_fix = True


def fix_env(env):
    """Repair Poi/Restaurants instances reachable from an already-built env."""
    seen = set()

    def walk(obj, depth=0):
        if id(obj) in seen or depth > 3 or obj is None:
            return
        seen.add(id(obj))
        cls = type(obj).__name__
        if cls == "Poi":
            fix_poi_instance(obj)
            return
        if cls == "Restaurants":
            fix_restaurants_instance(obj)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v, depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, depth + 1)
            return
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            for v in d.values():
                walk(v, depth + 1)

    walk(env)
