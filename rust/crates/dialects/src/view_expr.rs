// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The checks on SQL expressions taken from a metric or semantic view
//! (`rfm/view_expr.py`): why an expression is unsafe to splice into a query,
//! and why it calls a function that is not trusted.
//!
//! Python's regular expressions are matched by hand, with Python's Unicode
//! classes (`unicode.rs`), its case folding and its backtracking order, so
//! every answer, and every reason's text, is Python's.

use sdm_pql::pyrepr::str_repr;
use std::collections::BTreeSet;

use crate::Error;
use crate::unicode::{decimal_value, is_space, is_word, matches_folded};

const UNSAFE_EXPR_TOKENS: [&str; 6] = [";", "--", "/*", "*/", "|>", "->>"];
const EXPR_QUOTE_DELIMITERS: [&str; 4] = ["$$", "'", "\"", "`"];
const UNSAFE_KEYWORDS: [&str; 11] = [
    "select", "table", "update", "delete", "merge", "create", "alter", "drop", "grant", "revoke",
    "execute",
];
const CALLABLE_KEYWORDS: [&str; 2] = ["insert", "truncate"];
pub const SESSION_VALUES: [&str; 13] = [
    "current_catalog",
    "current_database",
    "current_date",
    "current_path",
    "current_role",
    "current_schema",
    "current_time",
    "current_timestamp",
    "current_user",
    "localtime",
    "localtimestamp",
    "session_user",
    "user",
];
const GRAMMAR_WORDS: &[&str] = &[
    "account",
    "add",
    "after",
    "aggregate",
    "align",
    "all",
    "alter",
    "always",
    "analyze",
    "and",
    "anonymous",
    "anti",
    "any",
    "any_value",
    "apply",
    "approx",
    "archive",
    "archived",
    "array",
    "as",
    "asc",
    "asensitive",
    "asof",
    "async",
    "at",
    "atomic",
    "authorization",
    "auto",
    "autoclusteringprofile",
    "autoclusteringupdate",
    "autocommit",
    "begin",
    "bernoulli",
    "between",
    "bigint",
    "bin",
    "bin_distribute_ratio",
    "bin_end",
    "bin_start",
    "binary",
    "binding",
    "bitemporal",
    "bloomfilter",
    "boolean",
    "both",
    "bucket",
    "buckets",
    "by",
    "byte",
    "cache",
    "call",
    "called",
    "cascade",
    "case",
    "cast",
    "catalog",
    "catalogs",
    "cdc",
    "change",
    "changes",
    "char",
    "character",
    "check",
    "checkpoint",
    "cleanroom",
    "cleanrooms",
    "clear",
    "clone",
    "close",
    "cloud",
    "cluster",
    "clustered",
    "codegen",
    "collaborators",
    "collate",
    "collation",
    "collations",
    "collect",
    "collection",
    "column",
    "columns",
    "comments",
    "commit",
    "compact",
    "compactions",
    "compensation",
    "compute",
    "concatenate",
    "condition",
    "condition_identifier",
    "conf",
    "connect",
    "connection",
    "constraint",
    "constraints",
    "contains",
    "continue",
    "convert",
    "copy",
    "copy_options",
    "create",
    "credential",
    "credentials",
    "cron",
    "cross",
    "cube",
    "current",
    "current_database",
    "current_date",
    "current_path",
    "current_recipient",
    "current_schema",
    "current_time",
    "current_timestamp",
    "current_user",
    "cursor",
    "database",
    "databases",
    "date_add",
    "date_diff",
    "date_part",
    "dateadd",
    "datediff",
    "dbfs",
    "dbproperties",
    "dec",
    "decfloat",
    "decfloat_rounding_mode",
    "decimal",
    "declare",
    "deep",
    "default",
    "default_path",
    "deferrable",
    "deferred",
    "define",
    "defined",
    "definer",
    "delay",
    "delete",
    "deletion",
    "delimited",
    "delta",
    "deny",
    "depends",
    "desc",
    "describe",
    "detail",
    "deterministic",
    "dfs",
    "diagnostics",
    "directories",
    "directory",
    "disable",
    "disk",
    "distance",
    "distinct",
    "distribute",
    "div",
    "do",
    "double",
    "drop",
    "dropped",
    "dry",
    "effective",
    "else",
    "elseif",
    "empty",
    "enable",
    "encryption",
    "end",
    "enforced",
    "environment",
    "error",
    "escape",
    "escaped",
    "every",
    "evolution",
    "exact",
    "except",
    "exchange",
    "exclude",
    "exclusive",
    "execute",
    "exists",
    "exit",
    "expect",
    "expiration",
    "explain",
    "export",
    "extend",
    "extended",
    "external",
    "extract",
    "fail",
    "false",
    "feature",
    "feed",
    "fetch",
    "fields",
    "fileformat",
    "files",
    "filter",
    "first",
    "flamegraph",
    "float",
    "flow",
    "flows",
    "fn",
    "following",
    "for",
    "force",
    "foreign",
    "format_options",
    "format_version",
    "formatted",
    "found",
    "frame",
    "freshness",
    "from",
    "fsck",
    "full",
    "function",
    "functions",
    "generate",
    "generated",
    "geography",
    "geometry",
    "get",
    "get_column_tag_value",
    "get_tag_value",
    "global",
    "governed",
    "grant",
    "grants",
    "group",
    "grouping",
    "groups",
    "gscluster",
    "handler",
    "having",
    "iceberg_compat_version",
    "identified",
    "identifier",
    "identity",
    "if",
    "ignore",
    "ilike",
    "immediate",
    "import",
    "in",
    "include",
    "inclusive",
    "increment",
    "incremental",
    "indexes",
    "inherit",
    "initially",
    "inner",
    "inout",
    "inpath",
    "input",
    "inputformat",
    "insensitive",
    "insert",
    "int",
    "integer",
    "intersect",
    "interval",
    "into",
    "inventory",
    "invoker",
    "is",
    "isolation",
    "issue",
    "items",
    "iterate",
    "jfr",
    "join",
    "json",
    "json_table",
    "keys",
    "last",
    "lateral",
    "lazy",
    "leading",
    "leave",
    "left",
    "like",
    "limit",
    "line_number",
    "lines",
    "list",
    "lite",
    "live",
    "load",
    "local",
    "localtime",
    "localtimestamp",
    "locations",
    "lock",
    "locks",
    "logical",
    "long",
    "loop",
    "macro",
    "managed",
    "manual",
    "many",
    "map",
    "mask",
    "masks",
    "match",
    "match_condition",
    "match_recognize",
    "matched",
    "matches",
    "materialize",
    "materialized",
    "max",
    "mcp",
    "measures",
    "memory",
    "merge",
    "message_arguments",
    "message_text",
    "metadata",
    "metadata_path",
    "metastore",
    "metastores",
    "metrics",
    "minus",
    "models",
    "modified",
    "modifies",
    "most",
    "move",
    "msck",
    "namespace",
    "namespaces",
    "natural",
    "nchar",
    "nearest",
    "next",
    "no",
    "nocopy",
    "none",
    "norely",
    "not",
    "novalidate",
    "null",
    "nulls",
    "numeric",
    "nvarchar",
    "object",
    "of",
    "offset",
    "omit",
    "on",
    "once",
    "one",
    "only",
    "open",
    "optimization",
    "optimize",
    "option",
    "options",
    "or",
    "order",
    "ordinality",
    "organization",
    "out",
    "outer",
    "outputformat",
    "over",
    "overlaps",
    "overlay",
    "overwrite",
    "parameter",
    "parquet",
    "partial",
    "partition",
    "partitioned",
    "partitions",
    "past",
    "pattern",
    "per",
    "permute",
    "pivot",
    "place",
    "placing",
    "point",
    "policies",
    "policy",
    "preceding",
    "predictive",
    "preserve",
    "primary",
    "principals",
    "private",
    "privileges",
    "procedure",
    "procedures",
    "profile",
    "profiler",
    "properties",
    "provider",
    "providers",
    "purge",
    "put",
    "qualify",
    "quarantine",
    "query",
    "query_tags",
    "range",
    "read",
    "reads",
    "real",
    "recipient",
    "recipients",
    "recordreader",
    "recordwriter",
    "recover",
    "recursion",
    "recursive",
    "reduce",
    "references",
    "refresh",
    "regexp",
    "relationship",
    "rely",
    "remove",
    "rename",
    "reorg",
    "repair",
    "repeat",
    "repeatable",
    "replace",
    "replicas",
    "reset",
    "resignal",
    "resolve",
    "respect",
    "restore",
    "restrict",
    "retain",
    "return",
    "returned_sqlstate",
    "returns",
    "revoke",
    "right",
    "rlike",
    "roles",
    "rollback",
    "rollup",
    "row",
    "row_count",
    "rows",
    "run",
    "sample",
    "scd",
    "schema",
    "schemas",
    "search",
    "secret",
    "secrets",
    "security",
    "select",
    "semi",
    "separated",
    "serde",
    "serdeproperties",
    "service",
    "session_user",
    "set",
    "sets",
    "shallow",
    "share",
    "shares",
    "short",
    "show",
    "shred",
    "signal",
    "similarity",
    "simple",
    "single",
    "skewed",
    "skip",
    "smallint",
    "snapshot",
    "some",
    "sort",
    "sorted",
    "specific",
    "sql",
    "sqlexception",
    "sqlstate",
    "static",
    "statistics",
    "storage",
    "stored",
    "stratify",
    "stream",
    "streaming",
    "strict",
    "string",
    "struct",
    "style",
    "subset",
    "substr",
    "substring",
    "sync",
    "system",
    "system_path",
    "system_time",
    "system_version",
    "table",
    "tableref",
    "tables",
    "tablesample",
    "tblproperties",
    "tblproperty",
    "terminated",
    "then",
    "timediff",
    "timeseries",
    "timestamp_ltz",
    "timestamp_ntz",
    "timestamp_tz",
    "timestampadd",
    "timestampdiff",
    "tinyint",
    "to",
    "touch",
    "track",
    "trailing",
    "transaction",
    "transaction_active",
    "transactions",
    "transform",
    "trigger",
    "trim",
    "true",
    "truncate",
    "try_cast",
    "unarchive",
    "unbounded",
    "uncache",
    "undrop",
    "uniform",
    "union",
    "unique",
    "unknown",
    "unlock",
    "unmatched",
    "unmaterialize",
    "unnest",
    "unpivot",
    "unset",
    "unshred",
    "until",
    "update",
    "updates",
    "upgrade",
    "use",
    "user",
    "users",
    "using",
    "vacuum",
    "validate",
    "values",
    "var",
    "varbinary",
    "varchar",
    "variable",
    "variant",
    "vector",
    "verify",
    "view",
    "views",
    "violation",
    "void",
    "volume",
    "volumes",
    "watermark",
    "when",
    "whenever",
    "where",
    "while",
    "width",
    "window",
    "with",
    "within",
    "without",
    "work",
    "zone",
    "zorder",
];
/// What may stand right before a `(` that opens a group rather than a call.
const OPENS_A_GROUP: &str = "+-*/%=<>!|&^~,([?:";
/// What stands in for quoted text, so a call to a quoted name stays visible.
const QUOTED: char = '\u{e000}';

/// Databricks' pure, deterministic, built-in scalar functions (`_DATABRICKS_FUNCTIONS`).
const DATABRICKS_FUNCTIONS: &[&str] = &[
    "abs",
    "acos",
    "acosh",
    "add_months",
    "aggregate",
    "array",
    "array_append",
    "array_compact",
    "array_contains",
    "array_distinct",
    "array_except",
    "array_insert",
    "array_intersect",
    "array_join",
    "array_max",
    "array_min",
    "array_position",
    "array_prepend",
    "array_remove",
    "array_repeat",
    "array_size",
    "array_sort",
    "array_union",
    "arrays_overlap",
    "arrays_zip",
    "ascii",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "base64",
    "bigint",
    "bin",
    "binary",
    "bit_count",
    "bit_get",
    "bit_length",
    "boolean",
    "bround",
    "btrim",
    "cardinality",
    "cast",
    "cbrt",
    "ceil",
    "ceiling",
    "char",
    "char_length",
    "character_length",
    "charindex",
    "chr",
    "coalesce",
    "concat",
    "concat_ws",
    "contains",
    "conv",
    "convert_timezone",
    "cos",
    "cosh",
    "cot",
    "crc32",
    "csc",
    "date",
    "date_add",
    "date_diff",
    "date_format",
    "date_from_unix_date",
    "date_part",
    "date_sub",
    "date_trunc",
    "dateadd",
    "datediff",
    "datepart",
    "day",
    "dayname",
    "dayofmonth",
    "dayofweek",
    "dayofyear",
    "decimal",
    "decode",
    "degrees",
    "div",
    "double",
    "e",
    "element_at",
    "elt",
    "encode",
    "endswith",
    "equal_null",
    "exists",
    "exp",
    "expm1",
    "extract",
    "factorial",
    "filter",
    "find_in_set",
    "flatten",
    "float",
    "floor",
    "forall",
    "format_number",
    "format_string",
    "from_json",
    "from_unixtime",
    "from_utc_timestamp",
    "get",
    "get_json_object",
    "getbit",
    "greatest",
    "hash",
    "hex",
    "hour",
    "hypot",
    "if",
    "iff",
    "ifnull",
    "initcap",
    "instr",
    "int",
    "is_variant_null",
    "isnan",
    "isnotnull",
    "isnull",
    "jaro_winkler_similarity",
    "json_array_length",
    "json_object_keys",
    "last_day",
    "lcase",
    "least",
    "left",
    "len",
    "length",
    "levenshtein",
    "ln",
    "locate",
    "log",
    "log10",
    "log1p",
    "log2",
    "lower",
    "lpad",
    "ltrim",
    "luhn_check",
    "make_date",
    "make_dt_interval",
    "make_interval",
    "make_timestamp",
    "make_timestamp_ntz",
    "make_ym_interval",
    "map",
    "map_concat",
    "map_contains_key",
    "map_entries",
    "map_filter",
    "map_from_arrays",
    "map_from_entries",
    "map_keys",
    "map_values",
    "map_zip_with",
    "mask",
    "md5",
    "minute",
    "mod",
    "month",
    "monthname",
    "months_between",
    "named_struct",
    "nanvl",
    "negative",
    "next_day",
    "nullif",
    "nullifzero",
    "nvl",
    "nvl2",
    "octet_length",
    "overlay",
    "parse_json",
    "parse_url",
    "pi",
    "pmod",
    "position",
    "positive",
    "pow",
    "power",
    "printf",
    "quarter",
    "quote",
    "radians",
    "reduce",
    "regexp_count",
    "regexp_extract",
    "regexp_extract_all",
    "regexp_instr",
    "regexp_like",
    "regexp_replace",
    "regexp_substr",
    "repeat",
    "replace",
    "reverse",
    "right",
    "rint",
    "rlike",
    "round",
    "rpad",
    "rtrim",
    "schema_of_json",
    "schema_of_variant",
    "sec",
    "second",
    "sentences",
    "sequence",
    "sha",
    "sha1",
    "sha2",
    "shiftleft",
    "shiftright",
    "shiftrightunsigned",
    "sign",
    "signum",
    "sin",
    "sinh",
    "size",
    "slice",
    "smallint",
    "sort_array",
    "soundex",
    "space",
    "split",
    "split_part",
    "sqrt",
    "startswith",
    "str_to_map",
    "string",
    "struct",
    "substr",
    "substring",
    "substring_index",
    "tan",
    "tanh",
    "timestamp",
    "timestamp_micros",
    "timestamp_millis",
    "timestamp_seconds",
    "timestampadd",
    "timestampdiff",
    "tinyint",
    "to_binary",
    "to_char",
    "to_date",
    "to_json",
    "to_number",
    "to_timestamp",
    "to_timestamp_ltz",
    "to_timestamp_ntz",
    "to_unix_timestamp",
    "to_utc_timestamp",
    "to_varchar",
    "transform",
    "transform_keys",
    "transform_values",
    "translate",
    "trim",
    "trim_array",
    "trunc",
    "try_add",
    "try_cast",
    "try_divide",
    "try_element_at",
    "try_make_timestamp",
    "try_mod",
    "try_multiply",
    "try_parse_json",
    "try_parse_url",
    "try_remainder",
    "try_subtract",
    "try_to_binary",
    "try_to_date",
    "try_to_number",
    "try_to_timestamp",
    "try_url_decode",
    "try_variant_get",
    "typeof",
    "ucase",
    "unbase64",
    "unhex",
    "unix_date",
    "unix_micros",
    "unix_millis",
    "unix_seconds",
    "upper",
    "url_decode",
    "url_encode",
    "variant_get",
    "weekday",
    "weekofyear",
    "width_bucket",
    "xxhash64",
    "year",
    "zeroifnull",
    "zip_with",
];

/// Snowflake's (`_SNOWFLAKE_FUNCTIONS`).
const SNOWFLAKE_FUNCTIONS: &[&str] = &[
    "abs",
    "acos",
    "acosh",
    "add_months",
    "array_append",
    "array_cat",
    "array_compact",
    "array_construct",
    "array_construct_compact",
    "array_contains",
    "array_distinct",
    "array_except",
    "array_flatten",
    "array_generate_range",
    "array_insert",
    "array_intersection",
    "array_max",
    "array_min",
    "array_position",
    "array_prepend",
    "array_remove",
    "array_remove_at",
    "array_reverse",
    "array_size",
    "array_slice",
    "array_sort",
    "array_to_string",
    "arrays_overlap",
    "arrays_to_object",
    "arrays_zip",
    "as_array",
    "as_binary",
    "as_boolean",
    "as_char",
    "as_date",
    "as_decimal",
    "as_double",
    "as_integer",
    "as_number",
    "as_object",
    "as_real",
    "as_time",
    "as_timestamp_ltz",
    "as_timestamp_ntz",
    "as_timestamp_tz",
    "as_varchar",
    "ascii",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "base64_decode_string",
    "base64_encode",
    "bit_length",
    "bitand",
    "bitnot",
    "bitor",
    "bitshiftleft",
    "bitshiftright",
    "bitxor",
    "booland",
    "boolnot",
    "boolor",
    "boolxor",
    "cast",
    "cbrt",
    "ceil",
    "char",
    "charindex",
    "check_json",
    "chr",
    "coalesce",
    "concat",
    "concat_ws",
    "contains",
    "convert_timezone",
    "cos",
    "cosh",
    "cot",
    "date_from_parts",
    "date_part",
    "date_trunc",
    "dateadd",
    "datediff",
    "datefromparts",
    "day",
    "dayname",
    "dayofmonth",
    "dayofweek",
    "dayofweekiso",
    "dayofyear",
    "decode",
    "degrees",
    "div0",
    "div0null",
    "editdistance",
    "endswith",
    "equal_null",
    "exp",
    "extract",
    "factorial",
    "filter",
    "floor",
    "get",
    "get_ignore_case",
    "get_path",
    "greatest",
    "greatest_ignore_nulls",
    "hash",
    "haversine",
    "hex_decode_string",
    "hex_encode",
    "hour",
    "iff",
    "ifnull",
    "ilike",
    "initcap",
    "insert",
    "is_array",
    "is_boolean",
    "is_date",
    "is_decimal",
    "is_double",
    "is_integer",
    "is_null_value",
    "is_object",
    "is_real",
    "is_time",
    "is_timestamp_ltz",
    "is_timestamp_ntz",
    "is_timestamp_tz",
    "is_varchar",
    "jarowinkler_similarity",
    "json_extract_path_text",
    "last_day",
    "least",
    "least_ignore_nulls",
    "left",
    "len",
    "length",
    "like",
    "ln",
    "log",
    "lower",
    "lpad",
    "ltrim",
    "md5",
    "md5_hex",
    "md5_number_lower64",
    "md5_number_upper64",
    "minute",
    "mod",
    "month",
    "monthname",
    "months_between",
    "next_day",
    "nullif",
    "nullifzero",
    "nvl",
    "nvl2",
    "object_construct",
    "object_construct_keep_null",
    "object_delete",
    "object_insert",
    "object_keys",
    "object_pick",
    "octet_length",
    "parse_json",
    "parse_url",
    "pi",
    "position",
    "pow",
    "power",
    "previous_day",
    "quarter",
    "radians",
    "reduce",
    "regexp_count",
    "regexp_instr",
    "regexp_like",
    "regexp_replace",
    "regexp_substr",
    "regexp_substr_all",
    "regr_valx",
    "regr_valy",
    "repeat",
    "replace",
    "reverse",
    "right",
    "rlike",
    "round",
    "rpad",
    "rtrim",
    "second",
    "sha1",
    "sha1_hex",
    "sha2",
    "sha2_hex",
    "sign",
    "sin",
    "sinh",
    "soundex",
    "soundex_p123",
    "space",
    "split",
    "split_part",
    "sqrt",
    "square",
    "startswith",
    "strip_null_value",
    "strtok",
    "strtok_to_array",
    "substr",
    "substring",
    "tan",
    "tanh",
    "time_from_parts",
    "time_slice",
    "timeadd",
    "timediff",
    "timefromparts",
    "timestamp_from_parts",
    "timestamp_ltz_from_parts",
    "timestamp_ntz_from_parts",
    "timestamp_tz_from_parts",
    "timestampadd",
    "timestampdiff",
    "timestampfromparts",
    "to_array",
    "to_binary",
    "to_boolean",
    "to_char",
    "to_date",
    "to_decimal",
    "to_double",
    "to_number",
    "to_numeric",
    "to_object",
    "to_time",
    "to_timestamp",
    "to_timestamp_ltz",
    "to_timestamp_ntz",
    "to_timestamp_tz",
    "to_varchar",
    "to_variant",
    "transform",
    "translate",
    "trim",
    "trunc",
    "truncate",
    "try_base64_decode_string",
    "try_cast",
    "try_hex_decode_string",
    "try_parse_json",
    "try_to_binary",
    "try_to_boolean",
    "try_to_date",
    "try_to_decimal",
    "try_to_double",
    "try_to_number",
    "try_to_numeric",
    "try_to_time",
    "try_to_timestamp",
    "try_to_timestamp_ltz",
    "try_to_timestamp_ntz",
    "try_to_timestamp_tz",
    "typeof",
    "unicode",
    "upper",
    "week",
    "weekiso",
    "weekofyear",
    "width_bucket",
    "year",
    "yearofweek",
    "yearofweekiso",
    "zeroifnull",
];

/// Keywords an expression may follow with `(` without calling anything.
const KEYWORDS: &[&str] = &[
    "all", "and", "any", "as", "between", "both", "by", "case", "distinct", "else", "end",
    "escape", "exists", "for", "from", "ilike", "in", "interval", "is", "leading", "like", "not",
    "or", "over", "placing", "regexp", "rlike", "some", "then", "trailing", "when",
];

/// Type names, which take `(` for their parameters behind `AS` or `::`.
const TYPE_NAMES: &[&str] = &[
    "array",
    "binary",
    "char",
    "character",
    "dec",
    "decimal",
    "double",
    "float",
    "map",
    "nchar",
    "number",
    "numeric",
    "nvarchar",
    "object",
    "string",
    "struct",
    "text",
    "time",
    "timestamp",
    "timestamp_ltz",
    "timestamp_ntz",
    "timestamp_tz",
    "varbinary",
    "varchar",
    "vector",
];

/// Keywords SQL reserves, which therefore cannot name a function.
/// The keyword pairs SQL actually writes (`_FOLLOWS_A_KEYWORD`).
const FOLLOWS_A_KEYWORD: &[(&str, &[&str])] = &[
    ("case", &["when"]),
    ("distinct", &["from"]),
    ("ilike", &["all", "any", "some"]),
    ("is", &["distinct", "not"]),
    ("like", &["all", "any", "some"]),
    (
        "not",
        &[
            "between", "distinct", "exists", "ilike", "in", "like", "regexp", "rlike",
        ],
    ),
];

fn contains(set: &[&str], word: &str) -> bool {
    set.binary_search(&word).is_ok()
}

fn follows(before: &str) -> &'static [&'static str] {
    FOLLOWS_A_KEYWORD
        .iter()
        .find(|(k, _)| *k == before)
        .map_or(&[], |(_, words)| *words)
}

/// The quoted spans of `chars`, as `_QUOTED_EXPR_TEXT` finds them: at each
/// position the first alternative that matches, `$$...$$`, `'...'`,
/// `"..."` or `` `...` ``, scanning on from its end.
fn quoted_spans(chars: &[char]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut at = 0;
    while at < chars.len() {
        match quoted_at(chars, at) {
            Some(end) => {
                spans.push((at, end));
                at = end;
            }
            None => at += 1,
        }
    }
    spans
}

fn quoted_at(chars: &[char], at: usize) -> Option<usize> {
    match chars[at] {
        '$' if chars.get(at + 1) == Some(&'$') => (at + 2..chars.len().saturating_sub(1))
            .find(|&j| chars[j] == '$' && chars[j + 1] == '$')
            .map(|j| j + 2),
        '\'' => quoted_literal(chars, at, '\'', true),
        '"' => quoted_literal(chars, at, '"', true),
        '`' => quoted_literal(chars, at, '`', false),
        _ => None,
    }
}

/// `Q(?:QQ|\\[\s\S]|[^Q\\])*Q`, or without the backslash escape: the items
/// between the quotes are read greedily; where no closing quote follows the
/// last of them, the match backs off to the last doubled quote, whose first
/// half then closes it.
fn quoted_literal(chars: &[char], at: usize, quote: char, escapes: bool) -> Option<usize> {
    let mut last_pair = None;
    let mut p = at + 1;
    while p < chars.len() {
        let c = chars[p];
        if c == quote {
            if chars.get(p + 1) == Some(&quote) {
                last_pair = Some(p);
                p += 2;
                continue;
            }
            return Some(p + 1);
        }
        if escapes && c == '\\' {
            if p + 1 >= chars.len() {
                break;
            }
            p += 2;
            continue;
        }
        p += 1;
    }
    last_pair.map(|p| p + 1)
}

/// `expr` with each quoted span replaced by `with`.
fn replaced(chars: &[char], spans: &[(usize, usize)], with: char) -> Vec<char> {
    let mut out = Vec::with_capacity(chars.len());
    let mut at = 0;
    for &(start, end) in spans {
        out.extend_from_slice(&chars[at..start]);
        out.push(with);
        at = end;
    }
    out.extend_from_slice(&chars[at..]);
    out
}

fn contains_text(chars: &[char], needle: &str) -> bool {
    let needle: Vec<char> = needle.chars().collect();
    chars.windows(needle.len()).any(|w| w == needle.as_slice())
}

/// Whether `word`, lower-case ASCII, matches `chars` from `at` under
/// `re.IGNORECASE`; the end if it does.
fn folded_at(chars: &[char], at: usize, word: &str) -> Option<usize> {
    let mut p = at;
    for letter in word.chars() {
        let c = *chars.get(p)?;
        let same = if letter.is_ascii_alphabetic() {
            matches_folded(c, letter)
        } else {
            c == letter
        };
        if !same {
            return None;
        }
        p += 1;
    }
    Some(p)
}

/// `\b` at `at`.
fn boundary(chars: &[char], at: usize) -> bool {
    let before = at > 0 && is_word(chars[at - 1]);
    let after = chars.get(at).is_some_and(|&c| is_word(c));
    before != after
}

/// `(?!\s*\()` at `at`.
fn not_called(chars: &[char], at: usize) -> bool {
    let mut p = at;
    while chars.get(p).is_some_and(|&c| is_space(c)) {
        p += 1;
    }
    chars.get(p) != Some(&'(')
}

/// The first match of `_UNSAFE_EXPR_KEYWORDS` in `chars`.
fn unsafe_keyword(chars: &[char]) -> Option<String> {
    for at in 0..chars.len() {
        if at > 0 && (is_word(chars[at - 1]) || chars[at - 1] == '.') {
            continue;
        }
        for word in UNSAFE_KEYWORDS {
            if let Some(end) = folded_at(chars, at, word)
                && boundary(chars, end)
            {
                return Some(chars[at..end].iter().collect());
            }
        }
        for word in CALLABLE_KEYWORDS {
            if let Some(end) = folded_at(chars, at, word)
                && boundary(chars, end)
                && not_called(chars, end)
            {
                return Some(chars[at..end].iter().collect());
            }
        }
    }
    None
}

/// Whether `_QUERY_START`, `(?:^|\()\s*from\b`, matches in `chars`.
fn query_start(chars: &[char]) -> bool {
    let from_at = |start: usize| {
        let mut p = start;
        while chars.get(p).is_some_and(|&c| is_space(c)) {
            p += 1;
        }
        folded_at(chars, p, "from").is_some_and(|end| boundary(chars, end))
    };
    if from_at(0) {
        return true;
    }
    chars
        .iter()
        .enumerate()
        .any(|(at, &c)| c == '(' && from_at(at + 1))
}

/// `_unsafe_expr_reason(expr, dialect)`: why a view expression is unsafe,
/// or `None`.
pub fn unsafe_expr_reason(expr: &str, dialect: Option<&str>) -> Option<String> {
    for token in UNSAFE_EXPR_TOKENS {
        if expr.contains(token) {
            return Some(format!("it contains {}", str_repr(token)));
        }
    }
    if dialect == Some("databricks") && expr.contains("$$") {
        return Some("it contains '$$', which Databricks SQL does not quote with".into());
    }
    let chars: Vec<char> = expr.chars().collect();
    let spans = quoted_spans(&chars);
    if dialect == Some("snowflake") {
        for &(start, end) in &spans {
            if chars[start] == '"' && chars[start..end].contains(&'\\') {
                return Some("it has a backslash inside a double-quoted name".into());
            }
        }
    }
    let unquoted = replaced(&chars, &spans, ' ');
    for delimiter in EXPR_QUOTE_DELIMITERS {
        if contains_text(&unquoted, delimiter) {
            return Some(format!(
                "it leaves {} unpaired, so which part of it is quoted text cannot be established",
                str_repr(delimiter)
            ));
        }
    }
    let mut depth: i64 = 0;
    for &c in &unquoted {
        depth += match c {
            '(' => 1,
            ')' => -1,
            _ => 0,
        };
        if depth < 0 {
            return Some("it closes a '(' it never opened".into());
        }
    }
    if depth != 0 {
        return Some("it leaves a '(' unclosed".into());
    }
    if let Some(keyword) = unsafe_keyword(&unquoted) {
        return Some(format!("it contains the SQL keyword '{keyword}'"));
    }
    if query_start(&unquoted) {
        return Some("it contains a sub-query".into());
    }
    None
}

fn rstrip(chars: &[char]) -> &[char] {
    let end = chars
        .iter()
        .rposition(|&c| !is_space(c))
        .map_or(0, |p| p + 1);
    &chars[..end]
}

fn is_token_char(c: char) -> bool {
    is_word(c) || c == '$' || c == QUOTED
}

/// `_LAST_TOKEN.search(lead)` of a stripped `lead`: where its last run of
/// `[\w$\u{e000}]` starts, if it ends with one.
fn last_token(lead: &[char]) -> Option<usize> {
    if !lead.last().is_some_and(|&c| is_token_char(c)) {
        return None;
    }
    let start = lead
        .iter()
        .rposition(|&c| !is_token_char(c))
        .map_or(0, |p| p + 1);
    Some(start)
}

fn lower(chars: &[char]) -> String {
    chars.iter().collect::<String>().to_lowercase()
}

/// `_follows_a_value(lead, strong)`.
fn follows_a_value(lead: &[char], columns: &BTreeSet<String>) -> bool {
    let lead = rstrip(lead);
    let Some(&last) = lead.last() else {
        return false;
    };
    if matches!(last, ')' | ']' | QUOTED) {
        return true;
    }
    let Some(start) = last_token(lead) else {
        return false;
    };
    let before = rstrip(&lead[..start]);
    if qualified(before) || before.ends_with(&[':', ':']) || datetime_field(before) {
        return true;
    }
    let word = lower(&lead[start..]);
    if word.chars().next().is_some_and(is_digit)
        || matches!(word.as_str(), "true" | "false" | "null")
    {
        return true;
    }
    if word == "end" {
        return closes_a_case(before) && follows_a_value(before, columns);
    }
    columns.contains(&word) && !contains(GRAMMAR_WORDS, &word)
}

/// `_acts_as_keyword(word, lead)`.
fn acts_as_keyword(word: &str, lead: &[char], group: &[char], columns: &BTreeSet<String>) -> bool {
    match word {
        "not" => true,
        "case" => opens_a_simple_case(lead, group),
        _ => in_keyword_position(word, lead, columns),
    }
}

/// `_in_keyword_position(word, lead, strong)`.
fn in_keyword_position(word: &str, lead: &[char], columns: &BTreeSet<String>) -> bool {
    if follows_a_value(lead, columns) {
        return true;
    }
    let lead = rstrip(lead);
    let Some(start) = last_token(lead) else {
        return false;
    };
    let before = lower(&lead[start..]);
    if !follows(&before).contains(&word) {
        return false;
    }
    if before == "case" {
        return true;
    }
    in_keyword_position(&before, &lead[..start], columns)
}

/// `_QUALIFIED`, `\.\s*$`.
fn qualified(lead: &[char]) -> bool {
    rstrip(lead).last() == Some(&'.')
}

/// `_TYPE_CONTEXT`, `(?:(?<![\w$])as|::)\s*$`, under `re.IGNORECASE`.
fn type_context(lead: &[char]) -> bool {
    let lead = rstrip(lead);
    let n = lead.len();
    if n >= 2 && lead[n - 2..] == [':', ':'] {
        return true;
    }
    if !(n >= 2
        && matches_folded(lead[n - 2], 'a')
        && matches_folded(lead[n - 1], 's')
        && !(n > 2 && (is_word(lead[n - 3]) || lead[n - 3] == '$')))
    {
        return false;
    }
    let mut depth = 0;
    for index in (0..n - 2).rev() {
        depth += match lead[index] {
            ')' => 1,
            '(' => -1,
            _ => 0,
        };
        if depth < 0 {
            let opener = rstrip(&lead[..index]);
            return last_token(opener).is_some_and(|start| {
                matches!(lower(&opener[start..]).as_str(), "cast" | "try_cast")
                    && !qualified(&opener[..start])
            });
        }
    }
    false
}

fn datetime_field(lead: &[char]) -> bool {
    let lead = rstrip(lead);
    if lead.last() != Some(&'(') {
        return false;
    }
    let opener = rstrip(&lead[..lead.len() - 1]);
    last_token(opener).is_some_and(|start| {
        matches!(lower(&opener[start..]).as_str(), "extract" | "date_part")
            && !(start > 0
                && (is_word(opener[start - 1]) || matches!(opener[start - 1], '$' | '.')))
    })
}

fn is_digit(token: char) -> bool {
    decimal_value(token).is_some()
        || matches!(token as u32,
        178..=179 | 185 | 4969..=4977 | 6618 | 8304 | 8308..=8313 | 8320..=8329 |
        9312..=9320 | 9332..=9340 | 9352..=9360 | 9450 | 9461..=9469 | 9471 |
        10102..=10110 | 10112..=10120 | 10122..=10130 | 68160..=68163 |
        69216..=69224 | 69714..=69722 | 127232..=127242)
}

fn closes_a_case(lead: &[char]) -> bool {
    let mut open_cases = vec![0usize];
    let mut at = 0;
    while at < lead.len() {
        match lead[at] {
            '(' | '[' => open_cases.push(0),
            ')' | ']' => {
                if open_cases.len() > 1 {
                    open_cases.pop();
                }
            }
            token if is_token_char(token) => {
                let start = at;
                while at < lead.len() && is_token_char(lead[at]) {
                    at += 1;
                }
                if !qualified(&lead[..start]) {
                    let cases = open_cases.last_mut().expect("one scope");
                    match lower(&lead[start..at]).as_str() {
                        "case" => *cases += 1,
                        "end" if *cases > 0 => *cases -= 1,
                        _ => {}
                    }
                }
                continue;
            }
            _ => {}
        }
        at += 1;
    }
    open_cases.last().copied().unwrap_or(0) > 0
}

fn opens_a_simple_case(lead: &[char], group: &[char]) -> bool {
    if rstrip(lead)
        .last()
        .is_some_and(|last| !matches!(last, '(' | ','))
    {
        return false;
    }
    let mut depth = 0;
    for (index, token) in group.iter().enumerate() {
        depth += match token {
            '(' => 1,
            ')' => -1,
            _ => 0,
        };
        if depth == 0 {
            let mut at = index + 1;
            while group.get(at).is_some_and(|token| is_space(*token)) {
                at += 1;
            }
            return folded_at(group, at, "when").is_some_and(|end| boundary(group, end));
        }
    }
    false
}

fn lambda_parameters(text: &[char]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for at in 0..text.len().saturating_sub(1) {
        if text[at..at + 2] != ['-', '>'] {
            continue;
        }
        let lead = rstrip(&text[..at]);
        let parameter = |token: char| is_word(token) || token == '$';
        if lead.last() == Some(&')') {
            if let Some(start) = lead.iter().rposition(|token| *token == '(') {
                let group = &lead[start + 1..lead.len() - 1];
                let parts: Vec<_> = group.split(|token| *token == ',').collect();
                let valid = parts.iter().all(|part| {
                    let name: Vec<_> = part
                        .iter()
                        .copied()
                        .skip_while(|token| is_space(*token))
                        .collect();
                    let name = rstrip(&name);
                    !name.is_empty() && name.iter().all(|token| parameter(*token))
                });
                if valid {
                    for part in parts {
                        names.insert(lower(part).trim_matches(is_space).to_string());
                    }
                }
            }
        } else if lead.last().is_some_and(|token| parameter(*token)) {
            let start = lead
                .iter()
                .rposition(|token| !parameter(*token))
                .map_or(0, |index| index + 1);
            names.insert(lower(&lead[start..]));
        }
    }
    names
}

/// The first match of `_SESSION_VALUES` in `chars`: the session value read.
/// Its lookbehind holds the stand-in for quoted text, so a session value
/// right after a quoted name is not read as one.
fn session_value(chars: &[char], dialect: &str, columns: &BTreeSet<String>) -> Option<String> {
    for at in 0..chars.len() {
        // The lookbehind, `(?<![\w$.\u{e000}])`, skips quoted text too.
        if at > 0 && (is_word(chars[at - 1]) || matches!(chars[at - 1], '$' | '.' | QUOTED)) {
            continue;
        }
        for word in SESSION_VALUES {
            if let Some(end) = folded_at(chars, at, word)
                && boundary(chars, end)
                && not_called(chars, end)
                && !(dialect == "databricks"
                    && word != "current_path"
                    && lower(&chars[at..end]) == word
                    && columns.contains(word))
            {
                return Some(chars[at..end].iter().collect());
            }
        }
    }
    None
}

/// `_untrusted_call_reason(expr, dialect)`: why `expr` calls a function that
/// is not trusted on `dialect`, `databricks` or `snowflake`, or `None`.
pub fn untrusted_call_reason(
    expr: &str,
    dialect: &str,
    columns: &[String],
) -> Result<Option<String>, Error> {
    let trusted = match dialect {
        "databricks" => DATABRICKS_FUNCTIONS,
        "snowflake" => SNOWFLAKE_FUNCTIONS,
        other => return Err(Error::new(format!("KeyError: {other:?}"))),
    };
    let chars: Vec<char> = expr.chars().collect();
    let text = replaced(&chars, &quoted_spans(&chars), QUOTED);
    let named: BTreeSet<_> = columns.iter().map(|column| column.to_lowercase()).collect();
    let mut visible = named.clone();
    visible.extend(lambda_parameters(&text));
    for (index, &c) in text.iter().enumerate() {
        if c != '(' {
            continue;
        }
        let before = rstrip(&text[..index]);
        let Some(&last) = before.last() else {
            continue;
        };
        if OPENS_A_GROUP.contains(last) {
            continue;
        }
        if matches!(last, ')' | ']') {
            return Ok(Some("it calls the result of an expression".into()));
        }
        let Some(start) = last_token(before) else {
            let tail: String = before[before.len().saturating_sub(10)..].iter().collect();
            return Ok(Some(format!(
                "it calls what precedes '(' after {}",
                str_repr(&tail)
            )));
        };
        let name: String = before[start..].iter().collect();
        if name.contains(QUOTED) {
            return Ok(Some("it calls a function by a quoted name".into()));
        }
        let lead = &before[..start];
        if qualified(lead) {
            return Ok(Some(format!("it calls the qualified function '{name}'")));
        }
        let folded = name.to_lowercase();
        if folded == "identifier" {
            return Ok(Some("it names a function through IDENTIFIER()".into()));
        }
        if contains(KEYWORDS, &folded) && acts_as_keyword(&folded, lead, &text[index..], &visible) {
            continue;
        }
        if contains(TYPE_NAMES, &folded) && type_context(lead) {
            continue;
        }
        if !contains(trusted, &folded) {
            return Ok(Some(format!(
                "it calls '{name}', which is not a pure built-in SQL function"
            )));
        }
    }
    if let Some(value) = session_value(&text, dialect, &named) {
        return Ok(Some(format!("it reads the session value '{value}'")));
    }
    let quote = if dialect == "databricks" { '`' } else { '"' };
    for (start, end) in quoted_spans(&chars) {
        if chars[start] != quote || qualified(&chars[..start]) {
            continue;
        }
        let word = lower(&chars[start + 1..end - 1]);
        if SESSION_VALUES.contains(&word.as_str())
            && !(dialect == "databricks" && named.contains(&word))
        {
            let quoted: String = chars[start..end].iter().collect();
            return Ok(Some(format!("it reads the session value {quoted}")));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sets_are_sorted() {
        for set in [
            DATABRICKS_FUNCTIONS,
            SNOWFLAKE_FUNCTIONS,
            KEYWORDS,
            TYPE_NAMES,
            GRAMMAR_WORDS,
        ] {
            assert!(set.windows(2).all(|w| w[0] < w[1]));
        }
    }

    #[test]
    fn quoting_backs_off_to_the_last_doubled_quote() {
        let chars: Vec<char> = "'a''b''c".chars().collect();
        assert_eq!(quoted_spans(&chars), [(0, 6)]);
        let chars: Vec<char> = "'a\\'b' x".chars().collect();
        assert_eq!(quoted_spans(&chars), [(0, 6)]);
        let chars: Vec<char> = "$$$".chars().collect();
        assert!(quoted_spans(&chars).is_empty());
    }

    #[test]
    fn answers_as_python() {
        assert_eq!(unsafe_expr_reason("price * qty", None), None);
        assert_eq!(
            unsafe_expr_reason("ſelect 1", None).as_deref(),
            Some("it contains the SQL keyword 'ſelect'")
        );
        assert_eq!(
            unsafe_expr_reason("'unterminated", None).as_deref(),
            Some(
                "it leaves \"'\" unpaired, so which part of it is quoted text cannot be established"
            )
        );
        assert_eq!(unsafe_expr_reason("insert(x, 1)", None), None);
        assert_eq!(
            untrusted_call_reason("NOT escape(secret())", "databricks", &[])
                .unwrap()
                .as_deref(),
            Some("it calls 'escape', which is not a pure built-in SQL function")
        );
        assert_eq!(
            untrusted_call_reason("x NOT LIKE ('a%')", "snowflake", &["x".into()]).unwrap(),
            None
        );
        assert_eq!(
            untrusted_call_reason("localtimestamp + 1", "snowflake", &[])
                .unwrap()
                .as_deref(),
            Some("it reads the session value 'localtimestamp'")
        );
    }

    #[test]
    fn scope_does_not_fold_session_names_like_a_regex() {
        assert_eq!(
            untrusted_call_reason("current_ſchema", "databricks", &["current_schema".into()])
                .unwrap()
                .as_deref(),
            Some("it reads the session value 'current_ſchema'")
        );
        assert_eq!(
            untrusted_call_reason("½ IN (1)", "databricks", &[])
                .unwrap()
                .as_deref(),
            Some("it calls 'IN', which is not a pure built-in SQL function")
        );
    }
}
