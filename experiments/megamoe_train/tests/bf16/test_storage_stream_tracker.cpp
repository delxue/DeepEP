// Copyright 2026 Huawei Technologies Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/ATen.h>
#include <cassert>
#include <iostream>

#include "storage_stream_tracker.hpp"

int main()
{
    using deepep::megamoe_train::SavedStorageStreams;
    const c10::Device device(c10::DeviceType::PrivateUse1, 0);
    const c10::Stream stream(c10::Stream::UNSAFE, device, 1);
    const c10::Stream other(c10::Stream::UNSAFE, device, 2);
    SavedStorageStreams tracker;
    auto tensor = at::empty({64}, at::kByte);
    auto foreign = at::empty({64}, at::kByte);
    assert(!tracker.all_on({tensor.storage()}, stream));
    tracker.remember({tensor.storage()}, stream);
    assert(tracker.all_on({tensor.storage()}, stream));
    assert(tracker.all_on({tensor.narrow(0, 8, 16).storage()}, stream));
    assert(!tracker.all_on({tensor.storage()}, other));
    assert(!tracker.all_on({foreign.storage()}, stream));
    assert(!tracker.all_on({tensor.storage(), foreign.storage()}, stream));
    assert(!tracker.all_on({tensor.clone().storage()}, stream));
    tensor.storage().set_nbytes(32);
    assert(!tracker.all_on({tensor.storage()}, stream));
    tensor.storage().set_nbytes(64);
    auto weak = tensor.storage().getWeakStorageImpl();
    tensor = at::Tensor();
    assert(weak.expired());
    for (int i = 0; i < 1000; ++i) {
        auto current = at::empty({64}, at::kByte);
        assert(!tracker.all_on({current.storage()}, stream));
        tracker.remember({current.storage()}, stream);
        assert(tracker.all_on({current.storage()}, stream));
    }
    tracker.remember({foreign.storage()}, other);
    assert(tracker.all_on({foreign.storage()}, other));
    assert(!tracker.all_on({foreign.storage()}, stream));
    tracker.clear();
    assert(!tracker.all_on({foreign.storage()}, other));
    std::cout << "PASS: storage identity, aliases, stream mismatch, transformed/external storage, "
                 "resized storage, weak lifetime, stale pruning and close\n";
}
