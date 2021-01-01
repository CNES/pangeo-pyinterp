# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import pickle
from pyinterp.core import storage


def test_pickle():
    p = storage.Marshaller()
    s = "#" * 256
    d1 = p.dumps(s)
    d2 = pickle.dumps(s, protocol=-1)
    assert p.loads(d1) == s
    assert d1 == d2


def test_long():
    p = storage.Marshaller()
    for _ in range(4096):
        s = "#" * 256
        d1 = p.dumps(s)
        d2 = pickle.dumps(s, protocol=-1)
        assert p.loads(d1) == s
        assert d1 == d2
