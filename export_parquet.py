from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parent
ALLOWED_TABLES = (
    "rank_snapshots",
    "matches",
    "participants",
    "participant_frames",
    "timeline_events",
)
SAFE_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def arrow_type(data_type: str, udt_name: str) -> pa.DataType:
    if data_type in {"smallint", "integer", "bigint"}:
        return pa.int64()
    if data_type in {"real", "double precision", "numeric", "decimal"}:
        return pa.float64()
    if data_type == "boolean":
        return pa.bool_()
    if data_type == "timestamp with time zone":
        return pa.timestamp("us", tz="UTC")
    if data_type == "timestamp without time zone":
        return pa.timestamp("us")
    # JSONB is cast to text so nested event shapes cannot break Parquet schemas.
    return pa.string()


def normalise_chunk(frame: pd.DataFrame, schema: pa.Schema) -> pd.DataFrame:
    for field in schema:
        name = field.name
        if pa.types.is_integer(field.type):
            frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("Int64")
        elif pa.types.is_floating(field.type):
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
        elif pa.types.is_boolean(field.type):
            frame[name] = frame[name].astype("boolean")
        elif pa.types.is_timestamp(field.type):
            frame[name] = pd.to_datetime(
                frame[name], errors="coerce", utc=field.type.tz is not None
            )
        else:
            frame[name] = frame[name].astype("string")
    return frame


# rank_snapshots is player-level, not match-level, so a patch filter doesn't
# apply to it — it's always exported in full regardless of --patch.
TABLES_WITHOUT_PATCH_SCOPE = {"rank_snapshots"}


def export_table(
    engine, table: str, output_dir: Path, chunk_size: int, patch: str | None
) -> int:
    if table not in ALLOWED_TABLES or not SAFE_IDENTIFIER.fullmatch(table):
        raise ValueError(f"不允许导出表：{table}")

    with engine.connect() as connection:
        columns = connection.execute(
            text(
                """
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = :table
                ORDER BY ordinal_position
                """
            ),
            {"table": table},
        ).fetchall()

    if not columns:
        raise ValueError(f"数据库中不存在表：{table}")

    fields = [
        pa.field(name, arrow_type(data_type, udt_name), nullable=True)
        for name, data_type, udt_name in columns
    ]
    schema = pa.schema(fields)
    select_parts = [
        f'"{name}"::text AS "{name}"' if udt_name in {"json", "jsonb"} else f'"{name}"'
        for name, _data_type, udt_name in columns
    ]
    query = f'SELECT {", ".join(select_parts)} FROM "{table}"'
    query_params: dict[str, str] = {}
    if patch is not None and table not in TABLES_WITHOUT_PATCH_SCOPE:
        if table == "matches":
            query += ' WHERE "patch" = %(patch)s'
        else:
            query += (
                ' WHERE "match_id" IN (SELECT match_id FROM matches WHERE patch = %(patch)s)'
            )
        query_params["patch"] = patch
    target = output_dir / f"{table}.parquet"

    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        for chunk in pd.read_sql_query(
            query, engine, chunksize=chunk_size, params=query_params or None
        ):
            chunk = normalise_chunk(chunk, schema)
            arrow_table = pa.Table.from_pandas(
                chunk, schema=schema, preserve_index=False, safe=False
            )
            if writer is None:
                writer = pq.ParquetWriter(target, schema, compression="zstd")
            writer.write_table(arrow_table)
            row_count += len(chunk)
        if writer is None:
            empty = pa.Table.from_arrays(
                [pa.array([], type=field.type) for field in schema], schema=schema
            )
            pq.write_table(empty, target, compression="zstd")
    finally:
        if writer is not None:
            writer.close()
    return row_count


def main() -> int:
    parser = argparse.ArgumentParser(description="将采集数据库导出为 Parquet。")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tables", nargs="+", choices=ALLOWED_TABLES, default=ALLOWED_TABLES)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--patch",
        default=None,
        help="只导出这个 patch 的比赛相关数据（rank_snapshots 例外，始终全量导出）；"
        "不指定则导出全部版本（保留跨版本调试/对比数据）。",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        print("缺少 DATABASE_URL。", file=sys.stderr)
        return 1

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.patch:
        output_dir = Path(f"./output_v3/patch={args.patch}")
    else:
        output_dir = Path("./parquet")
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine(database_url)
    try:
        for table in args.tables:
            count = export_table(engine, table, output_dir, args.chunk_size, args.patch)
            print(f"{table}: {count:,} 行 -> {output_dir / (table + '.parquet')}")
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
