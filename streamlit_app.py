from typing import Dict, List, Optional, Tuple, Set, Literal
from snowflake.snowpark import Session # type: ignore
import streamlit as st # type: ignore
import time
import pandas as pd
from io import StringIO
import numpy as np

st.set_page_config(layout="wide", page_title="Data Editor", page_icon="🧮")
st.title("DQ Data Rule Exception Manager ❄️")
st.caption("Update the DATA_RULE_EXCEPTION_MANAGER table in snowflake. If manually adding a new record, please hit Enter after putting data into the final column")
st.markdown(
    '<span style="background:#fff59d; padding:0.05rem 0.25rem; border-radius:0.25rem; color:black">'
    "1. When making changes to a filtered data after applying a filter, please submit changes before changing or removing the filter or changes will be lost.<br>2. When adding new records, please press ENTER after editing the last column to make sure changes are saved before submission."
    '</span>',
    unsafe_allow_html=True,
)

con = st.connection("snowflake")
session: Session = con.session()

database = st.selectbox("Select the database where your table is", ("DW_CB_DEV","DW_CBDM_DEV_ABENSON"))
schema = st.selectbox("Select the schema where your table is", ("UTIL","DEV","REF"))
table_name = st.selectbox("Select the table to edit", ("DATA_RULE_EXCEPTION_MANAGER","LOOKUPS","RATE_REFERENCE","GL_MAPP","MY_FIRST_DBT_MODEL",))
table_fqn: str = f"{database}.{schema}.{table_name}"
try:
    session.sql("USE ROLE ENGINEER")
except Exception:
    pass

AUX_COLS: List[str] = ["__rowid", "_delete_flag", "_upsert_type"]




@st.cache_data(show_spinner=False)
def describe_table_cached(table_fqn: str) -> Dict[str, str]:
    info = session.sql(f"DESCRIBE TABLE {table_fqn}").collect()
    return {r["name"]: r["type"] for r in info}


@st.cache_data(show_spinner=False)
def load_table_cached(table_fqn: str) -> pd.DataFrame:
    return session.table(table_fqn).to_pandas()


def column_config_from_types(col_types: Dict[str, str]) -> Dict[str, st.column_config.Column]:
    cfg: Dict[str, st.column_config.Column] = {}
    for c, t in col_types.items():
        T = t.upper()

        if c == "STATUS":
            cfg[c] = st.column_config.SelectboxColumn(
                options=["", "ACTIVE", "INACTIVE", "PENDING"],
                # TODO: enforce these
                help="Select the current record status",
            )
        elif "VARCHAR" in T:
            cfg[c] = st.column_config.TextColumn()
        elif any(x in T for x in ["NUMBER", "DECIMAL", "INT", "FLOAT", "DOUBLE"]):
            cfg[c] = st.column_config.NumberColumn()
        elif any(x in T for x in ["TIMESTAMP", "DATETIME"]):
            cfg[c] = st.column_config.DatetimeColumn()
        elif any(x in T for x in ["DATE"]):
            cfg[c] = st.column_config.DateColumn()
        elif "BOOLEAN" in T:
            cfg[c] = st.column_config.SelectboxColumn(
                options=[None, True, False],
                format_func=lambda x: "—" if x is None else str(x),
                help="Select True or False (leave blank for NULL)",
            )
    return cfg

def is_blank(x) -> bool:
    return x is None or (isinstance(x, str) and x.strip() == "")

def normalize_boolean_value(v):
    """Convert various user-entered values into True/False/NA."""
    # If it's a true null (None, NaN, <NA>), leave it alone
    if pd.isna(v):
        return v

    # Convert to string and strip whitespace for comparison
    key = str(v).strip().lower()

    truthy_values = {"true", "t", "1"}
    falsy_values  = {"false", "f", "0"}

    if key in truthy_values:
        return True
    elif key in falsy_values:
        return False
    else:
        # If it’s not one of those, keep the original value (validation will catch it later)
        return v


def prepare_for_snowflake_with_validation(df: pd.DataFrame, col_types: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize blanks, coerce to Snowflake-friendly dtypes, and collect coercion errors.
    Returns: (coerced_df, errors_df[ row_index, column, value, reason ])
    """
    out = df.copy()
    issues = []

    for c, t in col_types.items():
        if c not in out.columns:
            continue
        U = t.upper()
        s = pd.Series(out[c], copy=True)

        # Normalize blanks/None → pd.NA (for any dtype)
        s = s.map(lambda x: pd.NA if (x is None or (isinstance(x, str) and x.strip() == "")) else x)
        nonblank = ~s.isna()  # after normalization, this now covers everything


        if "BOOLEAN" in U:
            # Accept friendly inputs; everything else is an error
            mapped = s.map(normalize_boolean_value)
            bad = nonblank & ~mapped.isin([True, False])
            out[c] = mapped.astype("boolean")
            reason = "not a boolean (True/False)"

        elif "DATE" in U:
            coerced = pd.to_datetime(s, errors="coerce")
            bad = nonblank & coerced.isna()
            out[c] = coerced.dt.date
            reason = "invalid date"

        elif any(x in U for x in ["TIMESTAMP", "DATETIME"]):
            coerced = pd.to_datetime(s, errors="coerce")
            bad = nonblank & coerced.isna()
            out[c] = coerced
            reason = "invalid timestamp"

        elif any(x in U for x in ["NUMBER", "DECIMAL", "INT", "FLOAT", "DOUBLE"]):
            coerced = pd.to_numeric(s, errors="coerce")
            bad = nonblank & coerced.isna()
            out[c] = coerced
            reason = "invalid number"

        else:
            # VARCHAR (and others) → string dtype with NA support
            out[c] = pd.Series(s, dtype=pd.StringDtype(storage="python"))
            bad = pd.Series(False, index=s.index)
            reason = None

        if bad.any():
            for idx, val in df.loc[bad, c].items():
                issues.append({"row_index": idx, "column": c, "value": val, "reason": reason})

    return out, pd.DataFrame(issues)


def ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out


def coerce_keys_as_str(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    out = df.copy()
    for k in keys:
        if k in out.columns:
            out[k] = out[k].astype(str)
    return out


def _is_datetime_series(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    try:
        pd.to_datetime(s.dropna().head(1))
        return True
    except Exception:
        return False


def build_column_filters(df: pd.DataFrame) -> Dict[str, Tuple[str, Tuple]]:
    filters: Dict[str, Tuple[str, Tuple]] = {}
    with st.expander("Filters", expanded=False):
        for c in df.columns:
            if c in ["__rowid"]:
                continue
            s = df[c]

            if pd.api.types.is_bool_dtype(s) or (
                pd.api.types.is_integer_dtype(s) 
                and set(s.dropna().unique()).issubset({0, 1})
            ):
                choice = st.selectbox(f"{c}", options=["All", "True", "False", "Blank"], index=0, key=f"boolfilter:{c}")
                if choice != "All":
                    if choice == "Blank":
                        filters[c] = ("boolean_blank", ())
                    else:
                        filters[c] = ("boolean", (choice == "True",))

            elif pd.api.types.is_numeric_dtype(s):
                vals = pd.to_numeric(s, errors="coerce")
                if vals.dropna().empty:
                    continue

                enabled = st.checkbox(f"Filter {c}", key=f"enable_num:{c}", value=False)
                if enabled:
                    vmin, vmax = float(vals.min()), float(vals.max())
                    fmin, fmax = st.slider(
                        f"{c}",
                        min_value=vmin, max_value=vmax,
                        value=(vmin, vmax),
                        key=f"numfilter:{c}"
                    )
                    include_blank = st.checkbox("Include blanks", value=True, key=f"blank_num:{c}")
                    filters[c] = ("numeric_range", (fmin, fmax, include_blank))

            elif _is_datetime_series(s):
                sdt = pd.to_datetime(s, errors="coerce")
                if sdt.dropna().empty:
                    continue

                enabled = st.checkbox(f"Filter {c}", key=f"enable:{c}", value=False)
                if enabled:
                    dmin, dmax = sdt.min().date(), sdt.max().date()
                    d1, d2 = st.date_input(f"{c}", value=(dmin, dmax), key=f"datefilter:{c}")
                    include_blank = st.checkbox("Include blanks", value=True, key=f"blank:{c}")
                    filters[c] = ("date_range", (d1, d2, include_blank))
            else:
                val = st.text_input(f"{c} contains", value="", key=f"strfilter:{c}")
                if val:
                    filters[c] = ("contains", (val,))
    return filters

def build_sort_controls(df: pd.DataFrame) -> tuple[Optional[str], bool]:
    sort_cols = [c for c in df.columns if c not in ("__rowid", "_delete_flag", "_upsert_type")]
    options = ["— None —"] + sort_cols

    with st.expander("Order by", expanded=False):
        if st.session_state.get("__orderby_col") in (None, "— None —"):
          default_label = "— None —"
        else:
          default_label = st.session_state["__orderby_col"]

        if default_label in options:
          idx_select = options.index(default_label)
        else:
          idx_select = 0

        if st.session_state.get("__orderby_asc", True):
          idx_radio = 0
        else:
          idx_radio = 1

        col_label = st.selectbox("Column", options=options, index=idx_select, key="__orderby_col")
        asc_bool  = st.radio("Direction", options=[True, False], index=idx_radio, horizontal=True, key="__orderby_asc", format_func=lambda b: "Ascending" if b else "Descending")

        if st.button("Clear order", key="__orderby_clear"):
            for k in ("__orderby_col", "__orderby_asc"):
                if k in st.session_state:
                    del st.session_state[k]   # delete widget keys, don't assign
            st.rerun()

    # Normalize without writing back to the widget keys
    sort_col = None if col_label == "— None —" else col_label
    sort_asc = bool(asc_bool)
    return sort_col, sort_asc

def apply_filters(df: pd.DataFrame, filters: Dict[str, Tuple[str, Tuple]]) -> pd.DataFrame:
    out = df.copy()
    for c, (ftype, args) in filters.items():
        if ftype == "boolean":
            out = out[out[c].eq(bool(args[0]))]
        elif ftype == "boolean_blank":
            # Special case: keep only rows where value is null/NA
            out = out[out[c].isna()]
        elif ftype == "numeric_range":
            lo, hi, include_blank = args
            ser = pd.to_numeric(out[c], errors="coerce")
            in_range = ser.between(lo, hi)
            out = out[in_range | (ser.isna() if include_blank else False)]
        elif ftype == "date_range":
            d1, d2, include_blank = args
            s = pd.to_datetime(out[c], errors="coerce").dt.date
            in_range = s.notna() & (s >= d1) & (s <= d2)
            out = out[in_range | (s.isna() if include_blank else False)]
        elif ftype == "contains":
            term = args[0]
            out = out[out[c].astype(str).str.contains(term, case=False, na=False)]
    return out


def next_negative_ids(n: int) -> np.ndarray:
    cur = st.session_state.get("__next_neg_id", -1)
    new_ids = np.arange(cur, cur - n, -1, dtype=np.int64)
    st.session_state["__next_neg_id"] = int(new_ids[-1] - 1)
    return new_ids


def assign_rowids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "__rowid" not in out.columns:
        out.insert(0, "__rowid", np.arange(len(out), dtype=np.int64))
    return out


def mark_upsert_types(base: pd.DataFrame, working: pd.DataFrame, keys: Optional[List[str]]) -> pd.Series:
    if not keys:
        base_map = base.set_index("__rowid")
        is_new = ~working["__rowid"].isin(base_map.index)
        return pd.Series(np.where(is_new, "Insert", "Existing"), index=working.index)
    keys2 = [k for k in keys if k in base.columns and k in working.columns]
    if not keys2:
        return pd.Series("Existing", index=working.index)
    b = coerce_keys_as_str(base, keys2)
    w = coerce_keys_as_str(working, keys2)
    join = w[keys2].merge(b[keys2].drop_duplicates(), on=keys2, how="left", indicator=True)["_merge"].eq("left_only")
    return pd.Series(np.where(join, "Insert", "Existing"), index=working.index)


def stage_csv_into_working(working: pd.DataFrame, base_cols: List[str], incoming: pd.DataFrame, keys: Optional[List[str]]) -> pd.DataFrame:
    inc = incoming.copy()
    inc = ensure_cols(inc, base_cols)
    if keys:
        keys2 = [k for k in keys if k in inc.columns and k in working.columns]
    else:
        keys2 = []
    if keys2:
        w = coerce_keys_as_str(working, keys2)
        i = coerce_keys_as_str(inc, keys2)
        non_keys = [c for c in base_cols if c not in keys2]
        upd = w.merge(i[keys2 + non_keys], on=keys2, how="left", suffixes=("", "_inc"))
        for c in non_keys:
            inc_col = f"{c}_inc"
            if inc_col in upd.columns:
                upd[c] = upd[inc_col].where(upd[inc_col].notna(), upd[c])
                if inc_col in upd.columns:
                    upd.drop(columns=[inc_col], inplace=True)
        w2 = upd.copy()
        i_keys = i[keys2].drop_duplicates()
        new_rows = i.merge(w[keys2].drop_duplicates(), on=keys2, how="left", indicator=True)
        new_rows = new_rows[new_rows["_merge"].eq("left_only")].drop(columns=["_merge"])
        if not new_rows.empty:
            new_rows = new_rows[base_cols]
            new_rows.insert(0, "__rowid", next_negative_ids(len(new_rows)))
            new_rows["_delete_flag"] = False
            out = pd.concat([w2, new_rows], ignore_index=True, sort=False)
        else:
            out = w2
    else:
        inc = inc[base_cols]
        inc.insert(0, "__rowid", next_negative_ids(len(inc)))
        inc["_delete_flag"] = False
        out = pd.concat([working, inc], ignore_index=True, sort=False)
    return out

def safe_replace(session: Session, database: str, schema: str, table_name: str, df: pd.DataFrame) -> None:
    """
    Replace the contents of a Snowflake table in-place using a transient staging table.
    - Uses DELETE + INSERT inside a transaction.
    - Leaves the staging table intact if the transaction fails.
    - Raises a ValueError if dropping the staging table fails after success.
    """
    if df.empty:
        raise ValueError("Refusing to overwrite table with 0 rows.")
    clean = df.drop(columns=["__rowid", "_delete_flag", "_upsert_type"], errors="ignore")
    if clean.empty:
        raise ValueError("Refusing to overwrite table with 0 rows.")

    real_table_fqn = f'"{database}"."{schema}"."{table_name}"'
    staging_table_name = f'{table_name}__staging_{int(time.time())}'
    staging_table_fqn = f'"{database}"."{schema}"."{staging_table_name}"'

    # 1 Create a transient staging table
    session.sql(f'CREATE OR REPLACE TRANSIENT TABLE {staging_table_fqn} LIKE {real_table_fqn}').collect()

    # 2 Load data into the staging table
    session.write_pandas(
        df=clean,
        table_name=staging_table_name,
        database=database,
        schema=schema,
        overwrite=False,
        quote_identifiers=True,
    )

    # 3 Verify that the staging table isn't empty
    row_count = session.sql(f'SELECT COUNT(*) AS C FROM {staging_table_fqn}').collect()[0]["C"]
    if int(row_count) == 0:
        stg_table_hint = f"\nStaging table preserved for inspection: {staging_table_fqn}"
        raise ValueError("Refusing to replace table with 0 rows (staging is empty. {stg_table_hint}).")

    # Discover real columns and fix order
    cols_rows = session.sql(f'DESCRIBE TABLE {real_table_fqn}').collect()
    real_columns = [r["name"] for r in cols_rows if r["kind"] == "COLUMN"]
    col_list = ", ".join(f'"{c}"' for c in real_columns)
    # 4 Perform the in-place replace inside a transaction
    session.sql("BEGIN").collect()
    try:
        session.sql(f'DELETE FROM {real_table_fqn}').collect()

        session.sql(
            f'INSERT INTO {real_table_fqn} ({col_list}) SELECT {col_list} FROM {staging_table_fqn}'
        ).collect()

        session.sql("COMMIT").collect()
        transaction_succeeded = True
    except Exception as err:
        session.sql("ROLLBACK").collect()
        transaction_succeeded = False
        stg_table_hint = f"\nStaging table preserved for inspection: {staging_table_fqn}"
        raise ValueError(f"Error replacing table: {err}{stg_table_hint}") from err

    # 5 Cleanup only if the transaction succeeded
    if transaction_succeeded:
        try:
            session.sql(f'DROP TABLE {staging_table_fqn}').collect()
        except Exception as drop_err:
            # Escalate as ValueError so user sees it clearly
            raise ValueError(
                f"Replace succeeded, but failed to drop staging table {staging_table_fqn}: {drop_err}"
            ) from drop_err

def revert_table_by_offset_minutes(session: Session, table_fqn: str, minutes: int) -> None:
    session.sql("BEGIN").collect()
    session.sql(f"DELETE FROM {table_fqn}").collect()
    session.sql(
        f"""
        INSERT INTO {table_fqn}
        SELECT * FROM {table_fqn}
        AT (TIMESTAMP => DATEADD('minute', -{int(minutes)}, CURRENT_TIMESTAMP()))
        """
    ).collect()
    session.sql("COMMIT").collect()

def revert_table_to_timestamp(session: Session, table_fqn: str, ts_utc: str) -> None:
    session.sql("BEGIN").collect()
    session.sql(f"DELETE FROM {table_fqn}").collect()
    session.sql(
        f"""
        INSERT INTO {table_fqn}
        SELECT * FROM {table_fqn}
        AT (TIMESTAMP => TO_TIMESTAMP_TZ('{ts_utc}'))
        """
    ).collect()
    session.sql("COMMIT").collect()

def reset_all() -> None:
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    st.session_state.clear()
    st.rerun()

def _row_changes_mask(base: pd.DataFrame, work: pd.DataFrame, cols: List[str]) -> pd.Series:
    # Align both dataframes by __rowid and focus only on the relevant columns
    base_aligned = base.set_index("__rowid")[cols].sort_index()
    work_aligned = work.set_index("__rowid")[cols].sort_index()

    # Find common rows between base and work
    aligned_ids = base_aligned.index.intersection(work_aligned.index)
    if aligned_ids.empty:
        return pd.Series(False, index=work.index)

    # Restrict both to the aligned rows
    base_rows = base_aligned.loc[aligned_ids]
    work_rows = work_aligned.loc[aligned_ids]

    # Compare elementwise, treating NaN vs NaN as equal
    not_equal_mask = (~base_rows.eq(work_rows)) & ~(base_rows.isna() & work_rows.isna())

    # A row is changed if any of its compared cells differ
    changed_row_flags = not_equal_mask.any(axis=1)

    # Return a boolean mask aligned to the original work DataFrame
    return work["__rowid"].isin(changed_row_flags[changed_row_flags].index)

def replace_string_nulls(value):
        """Return pd.NA if the value is a string that represents a null."""
        NULL_SENTINELS = {"", "null", "NULL", "NaN", "nan", "None"}
        if isinstance(value, str):
            stripped_value = value.strip()
            if stripped_value in NULL_SENTINELS:
                return pd.NA
        return value

def make_editor_friendly(df: pd.DataFrame, column_types: dict[str, str]) -> pd.DataFrame:
    """
    Convert string-based date and timestamp columns into real datetime-like objects
    so Streamlit editors (DateColumn, DatetimeColumn) work correctly.
    """
    normalized_df = df.copy()

    for column_name, snowflake_type in column_types.items():
        if column_name not in normalized_df.columns:
            continue

        normalized_type_name = snowflake_type.upper().strip()
        column_series = normalized_df[column_name]

        # Replace common string representations of nulls with pd.NA
        if column_series.dtype == object:
            normalized_df[column_name] = column_series.map(replace_string_nulls)

        # Convert TIMESTAMP or DATETIME columns to pandas datetime64[ns]
        if normalized_type_name == "DATETIME" or normalized_type_name.startswith("TIMESTAMP"):
            normalized_df[column_name] = pd.to_datetime(
                normalized_df[column_name],
                errors="coerce",
                utc=False,
            )

        # Convert DATE columns to Python datetime.date
        elif normalized_type_name == "DATE":
            parsed_dates = pd.to_datetime(
                normalized_df[column_name],
                errors="coerce",
                utc=False,
            )
            normalized_df[column_name] = parsed_dates.dt.date

    return normalized_df

# ---------- UI + State ----------

if "data_version" not in st.session_state:
    st.session_state["data_version"] = 0

if "base_df" not in st.session_state or st.session_state.get("current_fqn") != table_fqn:
    col_types = describe_table_cached(table_fqn)
    base = load_table_cached(table_fqn)
    base = ensure_cols(base, list(col_types.keys()))
    base = assign_rowids(base)
    base["_delete_flag"] = False
    st.session_state["base_df"] = base
    st.session_state["working_df"] = base.copy()
    st.session_state["current_fqn"] = table_fqn
    st.session_state["del_sel"] = set()
    st.session_state["__next_neg_id"] = -1

col_types = describe_table_cached(table_fqn)

custom_column_config = column_config_from_types(col_types)

merge_keys_raw: str = st.text_input("Key columns (optional, comma-separated)", value="")
merge_keys: List[str] = [k.strip() for k in merge_keys_raw.split(",") if k.strip()]

st.subheader("untested: stage a CSV of new or updated rows")
csv = st.file_uploader("Choose a CSV to stage", type="csv")
stage_btn = st.button("Stage CSV into editor", disabled=csv is None)

if stage_btn and csv is not None:
    try:
        inc_df = pd.read_csv(csv)
        base_cols = [c for c in st.session_state["base_df"].columns if c not in AUX_COLS]
        work = st.session_state["working_df"]
        staged = stage_csv_into_working(work, base_cols, inc_df, merge_keys if merge_keys else None)
        st.session_state["working_df"] = staged
        st.success("CSV staged.")
    except Exception as e:
        st.error(f"Staging failed: {e}")

filters = build_column_filters(st.session_state["working_df"])
if st.button("Clear all filters"):
    for k in list(st.session_state.keys()):
        if k.startswith(("filter_val:", "boolfilter:", "numfilter:", "datefilter:", "strfilter:","enable_num:", "blank_num:", "enable:", "blank:")):
            st.session_state.pop(k)
    st.rerun()


sort_col, sort_asc = build_sort_controls(st.session_state["working_df"])
view_df = apply_filters(st.session_state["working_df"], filters).copy()

# Compute _upsert_type for user feedback only
view_df["_upsert_type"] = mark_upsert_types(st.session_state["base_df"], view_df, merge_keys if merge_keys else None)

# Apply order after filters and after computing _upsert_type,
# so the “Select all to delete (current view)” button respects the sorted view.
if sort_col:
    # Best-effort: try numeric/date, else fall back to string for robust sorting
    s = view_df[sort_col]
    try:
        # try datetime sort
        ser = pd.to_datetime(s, errors="raise")
    except Exception:
        try:
            # try numeric sort
            ser = pd.to_numeric(s, errors="raise")
        except Exception:
            # fallback to string sort (case-insensitive)
            ser = s.astype(str).str.lower()

    view_df = view_df.assign(__sortkey=ser).sort_values(
        by="__sortkey", ascending=sort_asc, na_position="last", kind="mergesort"  # stable sort
    ).drop(columns="__sortkey")

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Select all to delete (current view)"):
        st.session_state["del_sel"].update(map(int, view_df["__rowid"].tolist()))
with c2:
    if st.button("Unselect all (current view)"):
        st.session_state["del_sel"].difference_update(map(int, view_df["__rowid"].tolist()))
with c3:
    if st.button("Clear all selections"):
        st.session_state["del_sel"] = set()

view_df["_delete_flag"] = view_df["__rowid"].astype(np.int64).isin(st.session_state["del_sel"])
view_df = make_editor_friendly(view_df, col_types)

with st.expander("DEBUG — dtypes before editor", expanded=False):
    st.write(view_df.dtypes)  # check REVIEWED_DATE dtype
    if "REVIEWED_DATE" in view_df.columns:
        s = view_df["REVIEWED_DATE"]
        st.write("head values:", s.head(10).tolist())
        st.write("python types:", [type(x).__name__ for x in s.head(10)])
        # is it datetime-like?
        import pandas as pd
        st.write("is datetime-like:", pd.api.types.is_datetime64_any_dtype(s) or all(hasattr(x, "year") for x in s.dropna().head(10)))
    st.write("merge_keys:", merge_keys)
    st.write("Is REVIEWED_DATE in merge_keys? ", "REVIEWED_DATE" in merge_keys)
    st.write("Sample REVIEWED_DATE parse:", pd.to_datetime(view_df["REVIEWED_DATE"], errors="coerce").head(5))


with st.form("editor_form", clear_on_submit=False):
    specials = {
        "__rowid": st.column_config.NumberColumn(disabled=True),
        "_delete_flag": st.column_config.CheckboxColumn(),
        "_upsert_type": st.column_config.SelectboxColumn(options=["Existing", "Update", "Insert"], disabled=True),
    }
    editor_config = {**custom_column_config, **specials}  # merge your generated config
    edited_view = st.data_editor(
        view_df,
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
        column_order=["__rowid", "_delete_flag", "_upsert_type"]
        + [c for c in view_df.columns if c not in ("__rowid", "_delete_flag", "_upsert_type")],
        column_config=editor_config,
        height=640,
        key="data_editor",
    )

    to_delete_count: int = int(edited_view["_delete_flag"].fillna(False).sum())
    st.write(f"{to_delete_count} records selected for removal")

    save_btn = st.form_submit_button("Submit Changes")
    #refresh_btn = st.form_submit_button("Reload from Snowflake (discard unsubmitted edits)")
    #cancel_btn = st.form_submit_button("Discard changes since load")

# Merge edited visible rows back into canonical working_df
def _merge_back_into_working(edited_visible: pd.DataFrame) -> None:
    w = st.session_state["working_df"]
    vis = edited_visible.copy()

    if "__rowid" in vis.columns and "_delete_flag" in vis.columns:
        for rid, flg in zip(
            pd.to_numeric(vis["__rowid"], errors="coerce").dropna().astype(np.int64).tolist(),
            vis["_delete_flag"].fillna(False).astype(bool).tolist(),
        ):
            if flg:
                st.session_state["del_sel"].add(int(rid))
            else:
                st.session_state["del_sel"].discard(int(rid))

    new_rows = vis[vis["__rowid"].isna()].drop(columns=["_upsert_type"], errors="ignore")
    if not new_rows.empty:
        new_rows = new_rows.drop(columns=["__rowid"], errors="ignore")
        new_rows.insert(0, "__rowid", next_negative_ids(len(new_rows)))
        new_rows["_delete_flag"] = False
        st.session_state["working_df"] = pd.concat([w, new_rows], ignore_index=True, sort=False)
        w = st.session_state["working_df"]

    vis2 = vis.dropna(subset=["__rowid"]).copy()
    if "__rowid" not in vis2.columns:
        return
    vis2["__rowid"] = pd.to_numeric(vis2["__rowid"], errors="coerce").astype(np.int64)

    edit_cols = [c for c in vis2.columns if c not in ["_upsert_type", "__rowid"]]

    if not edit_cols:
        return

    w_indexed = w.set_index("__rowid")
    for rid, row in vis2.set_index("__rowid")[edit_cols].iterrows():
        if rid in w_indexed.index:
            w_indexed.loc[rid, row.index] = row.values
    st.session_state["working_df"] = w_indexed.reset_index()


_merge_back_into_working(edited_view)
if st.session_state.get("show_confirm", False) and "pending_write" in st.session_state:
    n_delete, n_update, n_insert = st.session_state["pending_write"]["counts"]
    with st.container(border=True):
        st.warning(
            f"This will apply the following changes to **{table_fqn}**:\n\n"
            f"- 🗑️ Delete: {n_delete} rows\n"
            f"- ✏️ Update: {n_update} rows\n"
            f"- ➕ Insert: {n_insert} rows\n\n"
            "Do you want to proceed?"
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Yes, apply changes"):
                try:
                    coerced_df, errors_df = prepare_for_snowflake_with_validation(st.session_state["working_df"], col_types)

                    if not errors_df.empty:
                        st.error("Some values can’t be converted. Fix these and try again.")
                        st.dataframe(errors_df.head(200))  # show first 200
                        st.stop()  # ⛔ fail fast in the UI

                    # If we get here, it’s safe to proceed:
                    if coerced_df.empty:
                        st.error("Refusing to overwrite table with 0 rows.")
                    else:
                        safe_replace(session, database, schema, table_name, coerced_df)
                        st.success("Table updated successfully.")
                        st.session_state.pop("pending_write", None)
                        st.session_state["show_confirm"] = False
                        st.session_state["data_version"] += 1
                        st.session_state["base_df"] = assign_rowids(load_table_cached(table_fqn))
                        st.session_state["base_df"]["_delete_flag"] = False
                        st.session_state["working_df"] = st.session_state["base_df"].copy()
                        st.session_state["del_sel"] = set()
                        st.session_state["__next_neg_id"] = -1
                        reset_all()
                except Exception as e:
                    st.error(f"Error updating table: {e}")
        with c2:
            if st.button("❌ No, cancel"):
                st.info("No changes were applied.")
                st.session_state.pop("pending_write", None)
                st.session_state["show_confirm"] = False
                st.rerun()



# if refresh_btn:
#     try:
#         st.session_state["base_df"] = assign_rowids(load_table_cached(table_fqn))
#         st.session_state["base_df"]["_delete_flag"] = False
#         st.session_state["working_df"] = st.session_state["base_df"].copy()
#         st.session_state["del_sel"] = set()
#         st.session_state["__next_neg_id"] = -1
#         st.success("Reloaded from Snowflake.")
#         st.rerun()
#     except Exception as e:
#         st.error(f"Reload failed: {e}")

# if cancel_btn:
#     st.session_state["working_df"] = st.session_state["base_df"].copy()
#     st.session_state["del_sel"] = set()
#     st.session_state["__next_neg_id"] = -1
#     st.info("Local edits discarded.")
#     st.rerun()
def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]
    return out
    
def drop_all_null_data_rows(df: pd.DataFrame, aux_cols: List[str]) -> pd.DataFrame:
    data_cols: List[str] = [c for c in df.columns if c not in aux_cols]
    if not data_cols:
        return df.copy()
    keep_mask: pd.Series = df[data_cols].notna().any(axis=1)
    return df.loc[keep_mask].copy()

def _prepare_pending_write(
    base: pd.DataFrame,
    work: pd.DataFrame,
    del_sel: Set[int],
    keys: Optional[List[str]],
    aux_cols: List[str],
) -> Dict[str, object]:
    base = _dedupe_columns(base)
    work = _dedupe_columns(work)

    keep_work: pd.DataFrame = work[~work["__rowid"].isin(del_sel)].copy()
    keep_base: pd.DataFrame = base[~base["__rowid"].isin(del_sel)].copy()

    data_cols: List[str] = [c for c in keep_work.columns if c not in aux_cols]

    existing_mask: pd.Series = keep_work["__rowid"].ge(0)
    new_mask: pd.Series = keep_work["__rowid"].lt(0)

    content_mask: pd.Series = keep_work[data_cols].notna().any(axis=1)
    valid_new_mask: pd.Series = new_mask & content_mask

    update_mask: pd.Series = _row_changes_mask(
        base=keep_base[keep_base["__rowid"].ge(0)],
        work=keep_work[existing_mask],
        cols=data_cols,
    )
    update_mask = update_mask.reindex(keep_work.index, fill_value=False)

    n_delete: int = int(base["__rowid"].isin(del_sel).sum())
    n_insert: int = int(valid_new_mask.sum())
    n_update: int = int(update_mask.sum())

    final_candidate: pd.DataFrame = keep_work.drop(columns=aux_cols, errors="ignore")
    final_candidate = drop_all_null_data_rows(final_candidate, aux_cols=[])

    return {
        "counts": (n_delete, n_update, n_insert),
        "final_candidate": final_candidate,
        "base_keep": keep_base.copy(),
        "work_keep": keep_work.copy(),
    }


if save_btn:
    try:
        pending = _prepare_pending_write(
            base=_dedupe_columns(st.session_state["base_df"]),
            work=_dedupe_columns(st.session_state["working_df"]),
            del_sel=st.session_state["del_sel"],
            keys=merge_keys if merge_keys else None,
            aux_cols=AUX_COLS,
        )
        st.session_state["pending_write"] = pending
        st.session_state["show_confirm"] = True
        st.rerun()
    except Exception as e:
        st.error(f"Preparing change summary failed: {e}")



with st.expander("Revert to previous version (Time Travel)"):
    method: Literal["Minutes ago", "Exact timestamp (UTC)"] = st.radio(
        "Method", options=["Minutes ago", "Exact timestamp (UTC)"], horizontal=True, key="revert_method"
    )
    if method == "Minutes ago":
        mins: int = st.number_input("Rewind by minutes", min_value=1, max_value=7 * 24 * 60, value=10, step=1, key="revert_minutes")
        confirm = st.checkbox(
            f"I understand this will replace all rows in {table_fqn} with the snapshot from {mins} minute(s) ago.",
            key="revert_confirm_mins",
        )
        if st.button("Revert table", key="revert_btn_mins", disabled=not confirm):
            try:
                revert_table_by_offset_minutes(session, table_fqn, mins)
                st.success(f"Reverted {table_fqn} to snapshot from {mins} minute(s) ago.")
                st.session_state["data_version"] += 1
                st.session_state.pop("base_df", None)
                st.session_state.pop("working_df", None)
                st.rerun()
            except Exception as e:
                st.error(f"Revert failed: {e}")
    else:
        ts_utc: str = st.text_input("Timestamp (UTC, e.g. 2025-09-03 14:35:00+00)", key="revert_ts")
        confirm2 = st.checkbox(
            f"I understand this will replace all rows in {table_fqn} with the snapshot at {ts_utc}.", key="revert_confirm_ts"
        )
        if st.button("Revert table", key="revert_btn_ts", disabled=(not confirm2 or not ts_utc.strip())):
            try:
                revert_table_to_timestamp(session, table_fqn, ts_utc.strip())
                st.success(f"Reverted {table_fqn} to snapshot at {ts_utc}.")
                st.session_state["data_version"] += 1
                st.session_state.pop("base_df", None)
                st.session_state.pop("working_df", None)
                st.rerun()
            except Exception as e:
                st.error(f"Revert failed: {e}")
