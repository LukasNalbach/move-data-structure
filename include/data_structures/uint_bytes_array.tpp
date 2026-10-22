/**
 * part of LukasNalbach/move-data-structure
 *
 * MIT License
 *
 * Copyright (c) Lukas Nalbach
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef MOVE_DATA_STRUCTURE_DATA_STRUCTURES_UINT_BYTES_ARRAY_TPP
#define MOVE_DATA_STRUCTURE_DATA_STRUCTURES_UINT_BYTES_ARRAY_TPP

#include <data_structures/uint_bytes_array.hpp>

template <uint8_t num_fields>
void uint_bytes_array<num_fields>::reset(uint64_t size, uint64_t max_value)
{
    _size = size;
    _width = ::bytes_per_value(max_value);
    _mask = (uint64_t { 1 } << (8 * _width)) - 1;
    resize_no_init(_data, size * num_fields * _width + sizeof(uint64_t));
}

template <uint8_t num_fields>
void uint_bytes_array<num_fields>::clear()
{
    _data = no_init_vector<uint8_t>();
    _size = 0;
}

template <uint8_t num_fields>
template <uint8_t field>
inline uint64_t uint_bytes_array<num_fields>::get(uint64_t i) const
{
    uint64_t value;
    std::memcpy(&value, _data.data() + (i * num_fields + field) * _width, sizeof(uint64_t));
    return value & _mask;
}

template <uint8_t num_fields>
inline void uint_bytes_array<num_fields>::store(uint8_t* dst, uint64_t value) const
{
    // one memcpy with a constant length per width
    switch (_width) {
        case 4: std::memcpy(dst, &value, 4); break;
        case 5: std::memcpy(dst, &value, 5); break;
        default: std::memcpy(dst, &value, 6); break;
    }
}

template <uint8_t num_fields>
template <uint8_t field>
void uint_bytes_array<num_fields>::sort_by(uint64_t num_threads)
{
    switch (_width) {
        case 4: sort_records_by<field, 4>(num_threads); break;
        case 5: sort_records_by<field, 5>(num_threads); break;
        default: sort_records_by<field, 6>(num_threads); break;
    }
}

template <uint8_t num_fields>
template <uint8_t field, uint8_t value_bytes>
void uint_bytes_array<num_fields>::sort_records_by(uint64_t num_threads)
{
    // a group as a record of a fixed size, whose key is read with exactly value_bytes bytes
    struct record_t {
        uint8_t bytes[num_fields * value_bytes];
    };

    auto key = [](const record_t& record) {
        uint64_t value = 0;
        std::memcpy(&value, record.bytes + field * value_bytes, value_bytes);
        return value;
    };

    record_t* records = reinterpret_cast<record_t*>(_data.data());
    radix_sort(records, records + _size, key, num_threads);
}

#endif // MOVE_DATA_STRUCTURE_DATA_STRUCTURES_UINT_BYTES_ARRAY_TPP
