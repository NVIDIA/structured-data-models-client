# Snowflake semantic-model fixture

Five Arrow IPC streams hold a semantic definition, two catalog descriptions and their preview rows.
The native-client test uses explicit schema, materialization and refusal expectations; it does not compare with a running old client.
These streams were extracted unchanged from the historical Snowflake capture; provenance.json records the original commits and SHA256 hashes.
Catalog and schema placeholders are replaced by the test's local names when loaded.
The complete original capture remains available in the migration archive for effort 3.
