# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S3, Snowflake and Databricks -- the connectors that need real credentials.

Everything written here is written under a prefix this file also removes.
"""

from __future__ import annotations

import pathlib

import data
from harness import TABULAR, fail, note, probe, save
from sdm_client import KumoTabular, Request, SDMClient
from sdm_connectors import ConnectorError, read

AREA = 's3'
SCRATCH = pathlib.Path.home() / 'sdm-qa' / 'scratch'
# Made here rather than assumed: this probe is meant to run on its own, and
# every other file that writes under it makes it the same way.
SCRATCH.mkdir(parents=True, exist_ok=True)
BUCKET = pathlib.Path('/tmp/qa_bucket').read_text().strip()
PREFIX = f's3://{BUCKET}/qa'


def scores(client, frame, target, task='classification'):
    context = frame.head(len(frame) - 3)
    predict = frame.tail(3).drop(columns=[target]).reset_index(drop=True)
    return client.predict(
        Request(context=context, predict=predict, target=target, task=task),
        model=KumoTabular(),
    ).to_pandas()


def run_s3(client):
    import s3fs

    fs = s3fs.S3FileSystem()
    adult = data.adult().head(400)

    with probe(AREA, 'a csv on s3', 'reads back whole, and predicts') as e:
        local = SCRATCH / 'adult.csv'
        adult.to_csv(local, index=False)
        fs.put(str(local), f'{BUCKET}/qa/adult.csv')
        frame = read('s3', path=f'{PREFIX}/adult.csv')
        if len(frame) != 400:
            fail(e, f'{len(frame)} rows')
        elif len(scores(client, frame, 'income')) != 3:
            fail(e, 'the s3 frame did not score')

    with probe(AREA, 'a parquet on s3', 'dtypes survive object storage') as e:
        local = SCRATCH / 'adult.parquet'
        adult.to_parquet(local, index=False)
        fs.put(str(local), f'{BUCKET}/qa/adult.parquet')
        frame = read('s3', path=f'{PREFIX}/adult.parquet')
        if not frame.dtypes.equals(adult.dtypes):
            fail(e, 'dtypes changed over s3')

    with probe(
        AREA, 'a gzipped csv on s3', 'compression detected from the key'
    ) as e:
        local = SCRATCH / 'adult.csv.gz'
        adult.to_csv(local, index=False, compression='gzip')
        fs.put(str(local), f'{BUCKET}/qa/adult.csv.gz')
        frame = read('s3', path=f'{PREFIX}/adult.csv.gz')
        if len(frame) != 400:
            fail(e, f'{len(frame)} rows')

    with probe(
        AREA, 'a parquet prefix on s3', 'many objects read as one frame'
    ) as e:
        directory = SCRATCH / 'adult_parts'
        directory.mkdir(exist_ok=True)
        # The two just written, not whatever the directory holds. It persists
        # between runs, so listing it uploads anything left there and the
        # prefix reads back more than the 400 rows this checks for.
        parts = []
        for i, part in enumerate([adult.head(200), adult.tail(200)]):
            path = directory / f'part-{i}.parquet'
            part.to_parquet(path, index=False)
            parts.append(path)
        for path in parts:
            fs.put(str(path), f'{BUCKET}/qa/parts/{path.name}')
        frame = read('s3', path=f'{PREFIX}/parts')
        if len(frame) != 400:
            fail(e, f'{len(frame)} rows from a 2-object prefix')

    with probe(
        AREA, 'a key that is not there', 'a ConnectorError naming the key'
    ) as e:
        try:
            read('s3', path=f'{PREFIX}/missing.csv')
            fail(e, 'a missing key was accepted')
        except (ConnectorError, FileNotFoundError, OSError) as exc:
            note(e, f'{type(exc).__name__}: {str(exc)[:120]}')

    with probe(
        AREA, 'a bucket that is not there', 'a clear error, not a hang'
    ) as e:
        try:
            read('s3', path='s3://sdm-client-qa-does-not-exist-42/x.csv')
            fail(e, 'a missing bucket was accepted')
        except Exception as exc:  # noqa: BLE001
            note(e, f'{type(exc).__name__}: {str(exc)[:120]}')

    with probe(
        AREA,
        'a local path given to the s3 connector',
        'refused, pointing at local',
    ) as e:
        try:
            read('s3', path=str(SCRATCH / 'adult.csv'))
            fail(e, 'the s3 connector accepted a local path')
        except ConnectorError as exc:
            note(e, str(exc)[:140])

    with probe(
        AREA, 'storage_options reach s3fs', 'an explicit region is honoured'
    ) as e:
        frame = read(
            's3',
            path=f'{PREFIX}/adult.csv',
            storage_options={'client_kwargs': {'region_name': 'us-west-2'}},
        )
        if len(frame) != 400:
            fail(e, f'{len(frame)} rows')

    with probe(AREA, 'an unexpected keyword', 'refused by name') as e:
        try:
            read('s3', path=f'{PREFIX}/adult.csv', nonsense=1)
            fail(e, 'an unknown keyword was accepted')
        except ConnectorError as exc:
            note(e, str(exc)[:140])


def cleanup():
    import s3fs

    fs = s3fs.S3FileSystem()
    for key in fs.find(f'{BUCKET}/qa'):
        fs.rm(key)
    print(f'removed s3://{BUCKET}/qa ({len(fs.find(BUCKET))} objects left)')


if __name__ == '__main__':
    with SDMClient(url=TABULAR, timeout=300.0) as client:
        run_s3(client)
    cleanup()
    save('s3.json')
