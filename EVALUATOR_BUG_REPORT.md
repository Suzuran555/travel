# Evaluator bug: opposing POI-name alias layers make query 20250323010327713880 unsatisfiable

**TL;DR:** Two alias layers in the official evaluator normalize the same POI in
opposite directions, so **no plan in the entire plan space can pass this query**.
Every team is capped at FPR 99.9 / C-LPR ≤99.97 on phase 1 through no fault of
their planner. One-line fix evaluator-side.

## The two layers

1. **Verification layer** — `chinatravel/symbol_verification/concept_func.py:50-52,83-86`
   ```python
   _POI_NAME_ALIASES = {"Bistro Sola": "Sola Bistro"}
   def normalize_poi_name(value): return _POI_NAME_ALIASES.get(value, value)
   ```
   Plan positions, transport endpoints, AND the constraint source
   (`normalize_concept_constraint_source`, `hard_constraint.py:491`) are all
   mapped **→ "Sola Bistro"**.

2. **Environment layer** — `chinatravel/environment/language.py:33-38`
   (`EN_POI_NAME_CORRECTIONS = {"Sola Bistro": "Bistro Sola"}`, applied by
   `canonical_poi_name` at DB load in `restaurants/apis.py:31-33` and
   `poi/apis.py:29`): the raw `database_en` rows ("Sola Bistro", Western,
   11:30–22:00, ¥583) are re-keyed **→ "Bistro Sola"**.

## Why the query is unsatisfiable

Query `20250323010327713880` hard constraint #2:
`activity_position(activity)=='Bistro Sola' and start_time<='17:00' and end_time>='18:00'`.

- **Avoiding the POI:** the constraint is exact string equality after
  normalization; exhaustive sweep of all 1,245 Shanghai POI names through the
  real `evaluate_constraints_py` shows exactly one satisfier. Any plan not
  visiting it fails hard-logic (5/6).
- **Visiting the POI:** the plan-side name (either spelling) normalizes to
  "Sola Bistro"; the DB key is "Bistro Sola"; nothing in any loaded DB matches
  "Sola Bistro". Restaurant grounding (`commonsense_constraint.py:709-756`)
  fails; transport grounding (`:862-878`) raises inside `Transportation.goto`
  (poi lookup returns an error string, `geodesic` gets a non-Point). Both
  spellings verified failing (candidate plan preserved).
- No logic escape exists: the clause is a bare existential (no disjunction /
  negation / aggregate); `activity_position` returns `''` for train/airplane;
  all five witness activity types carry unconditional grounding checks.

## Fix

Delete the `"Bistro Sola"` entry from `_POI_NAME_ALIASES` (or flip it to match
`EN_POI_NAME_CORRECTIONS`). Either way the two layers agree and plans visiting
"Bistro Sola" ground correctly and satisfy the constraint.

## Notes

- The same double-normalization is *self-cancelling* at the hard-logic layer
  (both spellings evaluate True there); the deadlock is only against the
  environment DB, which is corrected in the opposite direction.
- Related historical fix: upstream PR#25 (entity grounding) addressed the same
  family of issues; this alias pair appears to be a leftover.
- Verified on the phase-1 split, evaluator as of 2026-07-08.
