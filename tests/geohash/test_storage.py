# Copyright (c) 2020 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import os
import pytest
import pyinterp.geohash.storage as storage


@pytest.fixture(scope="session")
def unqlite_db(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data").join("database.unqlite")
    return fn


def test_unqlite(unqlite_db):
    assert not os.path.exists(unqlite_db)
    with storage.UnQlite(unqlite_db, mode="w") as db:
        db[b'0'] = 1
    assert os.path.exists(unqlite_db)
