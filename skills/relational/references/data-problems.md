# When the data is wrong

What the client does, before anything is sent, when the tables have a problem
in them: which problems it refuses, which it warns about and answers anyway,
and which it answers without a word. The last kind is the one to look for
yourself, because nothing will tell you.

Everything here happens in `Schema.from_data` and in `materialize()`, so it
needs no endpoint. An endpoint can still refuse a request this client was
willing to build; that is in [`../SKILL.md`](../SKILL.md).

## What happens to each problem

A refusal is a `StructuredDataError` carrying the code shown. A warning is an
`SDMWarning`, and the request is still built. Each message is longer than the
part quoted here, and names the table and column.

| In the data | The client | What it says | What to do |
|---|---|---|---|
| A primary key repeats in a few rows | warns | `identifies a row`, `it has already been used`, `the graph was built without some of the rows` | Drop the repeats, or pass the composite key the message names. The repeated rows are left out of the graph |
| A primary key repeats in more than 1% of rows | refuses | `INVALID_REQUEST`, `is not unique enough to identify a row` | The column is not a key: name another, or a composite key |
| A relationship joins two different types | refuses | `INVALID_REQUEST`, `which cannot match` | Store both columns as one type. Left to inference, the link is not made, and a warning says so |
| A foreign key matches no row of its parent | warns | `joins none of the rows given` | It is usually the wrong column, or ids that were re-keyed or zero-padded on one side only |
| A foreign key matches only some rows of its parent | says nothing | | Count the unmatched rows yourself: they are linked to nothing |
| Integer ids stored as floats | says nothing | | Nothing for the join, which matches them. Cast them back before showing them to anyone |
| A table no relationship reaches | warns | `is in the schema and not in this request` | Declare the relationship, or leave the table out of the schema |
| A column holding lists | refuses | `UNSUPPORTED_REQUEST`, `holds a container` | Give one value per row: move the list into a table of its own, one row per element |
| A dot in a column name | refuses | `INVALID_REQUEST`, `contains one or more invalid characters` | Rename the column |
| A time column of integers, such as 20250630 | refuses | `INVALID_REQUEST`, `which cannot mean 'timestamp'` | Confirm what the integer encodes, then parse it to a datetime |
| A date outside 1677 to 2262, such as 9999-12-31 | refuses | `INVALID_REQUEST`, `Out of bounds nanosecond timestamp` | It is a placeholder for "none": set it to null |
| Times written as text, some with a UTC offset and some without | says nothing | | Parse them to one zone yourself. The column is taken as the time column as it stands |
| Several date columns, and none named as the time column | warns | `was taken as the time column` | Name the right one in `time_columns` |
| No time column, and a query with a time window | refuses | `INVALID_REQUEST`, `Missing a "Create date" column` | Name the time column, or ask a question without a window |
| Less history than the query's window | refuses | `INVALID_REQUEST`, `Context anchor timestamp is too early` | Shorten the window, or bring more history |
| Numbers, lists or prose stored as text | warns | `is typed text, which this model does not read` | Parse numbers into numbers, and type labels categorical. Otherwise the column is not used |
| An infinite number | refuses | `INVALID_REQUEST`, `encountered a non-finite value` | Replace it with a null or a finite value |
| Yes and no spelled several ways | says nothing | | Each spelling is read as a category of its own: map them to one boolean |
| A query naming a column the table does not have | refuses | `INVALID_REQUEST`, `does not exist` | Check whether the column was renamed or lives on another table |
| A query comparing a text column with a number | refuses | `INVALID_REQUEST`, `has type string but constant 1 has type int` | Quote the value, or store the column as numbers |

## Keys the client finds for itself

Leave `primary_keys`, `time_columns` or `relationships` out and they are
inferred. Inference refuses nothing; where it is unsure it warns, and in one
case below it says nothing:

- A key is inferred only from a column whose name ends in `id` or `key` and
  whose values are at least 99% distinct, and the first such column wins.
  `customer_code`, `sku` or `item` is never inferred as a key, however unique.
  So a table holding a surrogate `customer_id` beside the `customer_code` that
  other tables point at is keyed by `customer_id`, and says nothing. The link
  through `customer_code` is not made, and a query whose `FOR EACH` names it
  is refused as `is not a primary key`. Pass the real key in `primary_keys`.
- A table with no such column gets no key, and every relationship to it is
  dropped. Where one of its unique columns is named by another table, the
  warning says `has no primary key` and gives the `primary_keys` override.
- A foreign key whose type differs from the key it names makes no
  relationship. The warning says `so no relationship was made from it`.
- Of several date columns, the one whose values start earliest is taken as
  the time column, and the warning names the others.

Inference matches on names. A foreign key spelled differently from the key it
points at, such as `sku` for `product_id`, is not found: declare it.
