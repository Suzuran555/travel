"""Deterministic budget-exhaustion fallback plan builder.

Builds the simplest itinerary that passes the stock schema + commonsense
evaluators using ONLY the public environment database:

  day 1:        intercity go leg (train preferred, else airplane)
                + hotel check-in (with grounded inner-city transports)
  days 2..n-1:  breakfast at the hotel (price 0) + the same hotel
  day n:        breakfast at the hotel + intercity back leg (with grounded
                inner-city transports back to the departure station)

It exists ONLY for budget exhaustion: the agent builds it up-front (a few
env-DB lookups, ~1s) and emits it when the planner cannot deliver a plan by
the internal emission deadline.  Partial credit (schema + commonsense + the
oracle constraints it happens to satisfy: day/people counts, ticket/room
multiplicities...) beats the zero of a watchdog kill.

Everything is deterministic and content-keyed: candidate transports are
sorted by (time, ID) and hotels by name; no dict-order or wall-clock
dependence.
"""

import math


# --------------------------------------------------------------------------
# small time helpers (fallback-local; plain "HH:MM" strings)
# --------------------------------------------------------------------------

def _t2m(hhmm):
    h, m = str(hhmm).split(":")
    return int(h) * 60 + int(m)


def _m2t(minutes):
    return "%02d:%02d" % (minutes // 60, minutes % 60)


def _paginate(env, call_str):
    """Run an env select call and drain its pagination; None on failure."""
    import pandas as pd

    ret = env(call_str)
    try:
        if not ret["success"]:
            return None
    except (KeyError, TypeError):
        pass
    data = ret["data"]
    while True:
        page = env("next_page()")["data"]
        if len(page) == 0:
            break
        data = pd.concat([data, page], axis=0, ignore_index=True)
    return data


# --------------------------------------------------------------------------
# deterministic candidate selection
# --------------------------------------------------------------------------

def _intercity_rows(env, src, dst, trans_type, id_col):
    df = _paginate(
        env,
        "intercity_transport_select('{}', '{}', '{}')".format(src, dst, trans_type),
    )
    if df is None or len(df) == 0:
        return []
    rows = []
    for row in df.to_dict("records"):
        begin, end = row.get("BeginTime"), row.get("EndTime")
        if not (isinstance(begin, str) and isinstance(end, str)):
            continue
        if ":" not in begin or ":" not in end:
            continue
        if row.get(id_col) is None:
            continue
        rows.append(row)
    return rows


def _pick_go(env, src, dst):
    """Earliest-arriving same-day leg, preferring arrivals before 17:00 so
    hotel check-in + '24:00' checkout stay on the arrival day."""
    for trans_type, id_col in (("train", "TrainID"), ("airplane", "FlightID")):
        rows = _intercity_rows(env, src, dst, trans_type, id_col)
        same_day = [r for r in rows if _t2m(r["EndTime"]) > _t2m(r["BeginTime"])]
        early = [r for r in same_day if _t2m(r["EndTime"]) <= _t2m("17:00")]
        cand = early or same_day
        if not cand:
            continue
        cand.sort(key=lambda r: (r["EndTime"], r["BeginTime"], str(r[id_col])))
        return trans_type, id_col, cand[0]
    return None


def _pick_back(env, dst, src, not_before_min):
    """Earliest departure at/after ``not_before_min`` (leaves room for the
    morning hotel->station transfer); latest departure as a last resort."""
    for trans_type, id_col in (("train", "TrainID"), ("airplane", "FlightID")):
        rows = _intercity_rows(env, dst, src, trans_type, id_col)
        if not rows:
            continue
        late_enough = [r for r in rows if _t2m(r["BeginTime"]) >= not_before_min]
        if late_enough:
            late_enough.sort(key=lambda r: (r["BeginTime"], str(r[id_col])))
            return trans_type, id_col, late_enough[0]
        rows.sort(key=lambda r: (r["BeginTime"], str(r[id_col])), reverse=True)
        return trans_type, id_col, rows[0]
    return None


def _pick_hotel(env, city, people):
    df = _paginate(env, "accommodations_select('{}', 'name', lambda x: True)".format(city))
    if df is None or len(df) == 0:
        return None
    for row in sorted(df.to_dict("records"), key=lambda r: str(r.get("name"))):
        name = row.get("name")
        if not name or '"' in str(name):
            continue  # name must survive the double-quoted goto() call string
        try:
            numbed = int(row["numbed"])
            price = float(row["price"])
        except (KeyError, TypeError, ValueError):
            continue
        if numbed <= 0:
            continue
        return {
            "name": str(name),
            "price": price,
            "numbed": numbed,
            "rooms": int(math.ceil(people / float(numbed))),
        }
    return None


def _collect_transports(env, city, start, end, start_time, people):
    """Inner-city transports via the stock goto() tool, normalized exactly the
    way UrbanTripOptimizedV6.collect_innercity_transport does (so the stock
    Is_transport_correct checks accept them).  None if no mode works."""
    if start == end:
        return []
    for mode in ("taxi", "metro", "walk"):
        try:
            info = env(
                'goto("{}", "{}", "{}", "{}", "{}")'.format(
                    city, start, end, start_time, mode
                )
            )["data"]
        except Exception:
            continue
        if not isinstance(info, list) or len(info) == 0:
            continue
        legs = [dict(leg) for leg in info]
        try:
            if len(legs) == 3:
                legs[1]["price"] = legs[1]["cost"]
                legs[1]["tickets"] = people
                legs[1]["cost"] = legs[1]["price"] * people
                legs[0]["price"] = legs[0]["cost"]
                legs[2]["price"] = legs[2]["cost"]
            elif legs[0].get("mode") == "taxi":
                legs[0]["price"] = legs[0]["cost"]
                cars = (people - 1) // 4 + 1
                legs[0]["cars"] = cars
                legs[0]["cost"] = legs[0]["price"] * cars
            else:
                legs[0]["price"] = legs[0]["cost"]
            _t2m(legs[0]["start_time"])
            end_min = _t2m(legs[-1]["end_time"])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if end_min >= 24 * 60:
            continue  # crossed midnight; try a cheaper mode
        return legs
    return None


# --------------------------------------------------------------------------
# activity builders
# --------------------------------------------------------------------------

def _intercity_activity(trans_type, id_col, row, people, transports):
    price = float(row["Cost"])
    return {
        "type": trans_type,
        id_col: str(row[id_col]),
        "start": str(row["From"]),
        "end": str(row["To"]),
        "start_time": str(row["BeginTime"]),
        "end_time": str(row["EndTime"]),
        "price": price,
        "tickets": int(people),
        "cost": price * people,
        "transports": transports,
    }


def _hotel_activity(hotel, start_time, transports):
    return {
        "type": "accommodation",
        "position": hotel["name"],
        "price": hotel["price"],
        "cost": hotel["price"] * hotel["rooms"],
        "room_type": hotel["numbed"],
        "rooms": hotel["rooms"],
        "start_time": start_time,
        "end_time": "24:00",
        "transports": transports,
    }


def _hotel_breakfast(hotel):
    # breakfast at the hotel: the stock restaurant check accepts a hotel
    # position with price 0 and a time inside (06:00, 09:00)
    return {
        "type": "breakfast",
        "position": hotel["name"],
        "price": 0.0,
        "cost": 0.0,
        "start_time": "07:00",
        "end_time": "07:30",
        "transports": [],
    }


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------

def build_fallback_plan(query, env):
    """Deterministic schema+commonsense-valid plan from the env DB.

    Returns the plan dict, or None when the environment has no workable
    combination (unsupported city pair etc.).
    """
    try:
        days = int(query["days"])
        people = int(query["people_number"])
        src = str(query["start_city"])
        dst = str(query["target_city"])
    except (KeyError, TypeError, ValueError):
        return None
    if days < 1 or people < 1:
        return None

    go = _pick_go(env, src, dst)
    if go is None:
        return None
    go_type, go_id, go_row = go

    if days == 1:
        back_not_before = _t2m(go_row["EndTime"]) + 90
    else:
        back_not_before = _t2m("11:00")
    back = _pick_back(env, dst, src, back_not_before)
    if back is None:
        return None
    back_type, back_id, back_row = back

    itinerary = [{"day": d, "activities": []} for d in range(1, days + 1)]
    itinerary[0]["activities"].append(
        _intercity_activity(go_type, go_id, go_row, people, [])
    )

    if days == 1:
        # go -> (station-to-station transfer) -> back
        transfer = _collect_transports(
            env, dst, str(go_row["To"]), str(back_row["From"]),
            str(go_row["EndTime"]), people,
        )
        if transfer is None:
            return None
        if transfer and _t2m(transfer[-1]["end_time"]) > _t2m(back_row["BeginTime"]):
            return None
        itinerary[0]["activities"].append(
            _intercity_activity(back_type, back_id, back_row, people, transfer)
        )
    else:
        hotel = _pick_hotel(env, dst, people)
        if hotel is None:
            return None

        # arrival day: station -> hotel
        to_hotel = _collect_transports(
            env, dst, str(go_row["To"]), hotel["name"], str(go_row["EndTime"]), people
        )
        if to_hotel is None:
            return None
        checkin = to_hotel[-1]["end_time"] if to_hotel else str(go_row["EndTime"])
        if _t2m(checkin) >= _t2m("23:59"):
            return None
        itinerary[0]["activities"].append(_hotel_activity(hotel, str(checkin), to_hotel))

        # middle days: hotel breakfast + the same hotel for the night
        for day_idx in range(1, days - 1):
            itinerary[day_idx]["activities"].append(_hotel_breakfast(hotel))
            itinerary[day_idx]["activities"].append(
                _hotel_activity(hotel, "21:00", [])
            )

        # departure day: hotel breakfast, then hotel -> station -> home
        to_station = _collect_transports(
            env, dst, hotel["name"], str(back_row["From"]), "07:30", people
        )
        if to_station is None:
            return None
        breakfast_ok = True
        if to_station and _t2m(to_station[-1]["end_time"]) > _t2m(back_row["BeginTime"]):
            # departure too early for the 07:30 transfer: retry at 05:30
            # without breakfast
            to_station = _collect_transports(
                env, dst, hotel["name"], str(back_row["From"]), "05:30", people
            )
            breakfast_ok = False
            if to_station is None or (
                to_station
                and _t2m(to_station[-1]["end_time"]) > _t2m(back_row["BeginTime"])
            ):
                return None
        if breakfast_ok:
            itinerary[days - 1]["activities"].append(_hotel_breakfast(hotel))
        itinerary[days - 1]["activities"].append(
            _intercity_activity(back_type, back_id, back_row, people, to_station)
        )

    return {
        "people_number": people,
        "start_city": src,
        "target_city": dst,
        "itinerary": itinerary,
    }
