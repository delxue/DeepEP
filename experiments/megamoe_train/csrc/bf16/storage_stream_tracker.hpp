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

#pragma once

#include <algorithm>
#include <initializer_list>
#include <vector>

#include <c10/core/Storage.h>
#include <c10/core/Stream.h>

namespace deepep::megamoe_train {

// Weak storage identity preserves aliases without retaining activation payloads.
class SavedStorageStreams {
public:
    void remember(std::initializer_list<c10::Storage> storages, c10::Stream stream)
    {
        entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                      [](const Entry& entry) { return entry.storage.expired(); }),
                       entries_.end());
        for (const auto& storage : storages) {
            entries_.push_back({storage.getWeakStorageImpl(), storage.data(), storage.data_ptr().get_context(),
                                storage.nbytes(), stream});
        }
    }

    bool all_on(std::initializer_list<c10::Storage> storages, c10::Stream stream) const
    {
        return std::all_of(storages.begin(), storages.end(), [&](const c10::Storage& storage) {
            return std::any_of(entries_.begin(), entries_.end(), [&](const Entry& entry) {
                return entry.stream == stream && entry.storage.lock().get() == storage.unsafeGetStorageImpl() &&
                       entry.data == storage.data() && entry.context == storage.data_ptr().get_context() &&
                       entry.bytes == storage.nbytes();
            });
        });
    }

    void clear()
    {
        entries_.clear();
    }

private:
    struct Entry {
        c10::weak_intrusive_ptr<c10::StorageImpl> storage;
        const void* data;
        const void* context;
        size_t bytes;
        c10::Stream stream;
    };
    std::vector<Entry> entries_;
};

}  // namespace deepep::megamoe_train
