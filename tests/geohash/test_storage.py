# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import os
import fsspec
import pytest
import pyinterp.geohash.storage as storage


@pytest.fixture(scope="session")
def unqlite_db(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data").join("database.unqlite")
    return fn


@pytest.fixture(scope="session")
def temp_root(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data")
    return fn


def test_unqlite(unqlite_db):
    assert not os.path.exists(unqlite_db)
    with storage.UnQlite(unqlite_db, mode="w") as db:
        db[b'0'] = 1
    assert os.path.exists(unqlite_db)


def test_memory():
    with storage.MutableMapping() as db:
        db[b'0'] = 1
        assert db[b'0'] == [1]


def test_file_system(temp_root):
    fs = fsspec.filesystem("file")
    with storage.FileSystem(fs, str(temp_root)) as db:
        assert list(db.keys()) == []
        db[b'0'] = 1
        assert db[b'0'] == [1]
        assert b'0' in db
        assert list(db.keys()) == [b'0']
        db.rollback()
        assert b'0' not in db
        assert list(db.keys()) == []

        db[b'0'] = 1
        db[b'0'] = [2, 3, 4]
        assert db[b'0'] == [2, 3, 4]
        db.extend(((b'0', [5]), ))
        db.extend(((b'0', 6), ))
        assert db[b'0'] == [2, 3, 4, 5, 6]
        assert list(db.values()) == [[2, 3, 4, 5, 6]]
        assert list(db.items()) == [(b'0', [2, 3, 4, 5, 6])]
        db.commit()
        assert list(db.keys()) == [b'0']

        db.update(((b'0', [0, 1]), ))
        assert db[b'0'] == [0, 1]

        del db[b'0']
        assert b'0' not in db
        db[b'0'] = 9
        assert b'0' in db
        db.rollback()
        assert db[b'0'] == [2, 3, 4, 5, 6]
        del db[b'0']
        assert b'0' not in db
        db.commit()
        assert b'0' not in db
