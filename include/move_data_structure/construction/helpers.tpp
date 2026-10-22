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

#ifndef MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_HELPERS_TPP
#define MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_HELPERS_TPP

#include <move_data_structure/construction/construction.hpp>

template <typename elem_t, typename key_t, typename second_fnc_t>
inline void mds_sort_unique(no_init_vector<elem_t>& v, uint64_t num_threads, key_t key, second_fnc_t second)
{
    radix_sort(v.begin(), v.end(), key, num_threads);

    // split v into chunks that begin at the start of a run; sort and deduplicate the runs of each chunk, then
    // concatenate the chunks
    // one chunk per thread, with at least one cache line of elements
    uint64_t chunks = std::max<uint64_t>(1, std::min<uint64_t>(num_threads, v.size() * sizeof(elem_t) / cache_line_bytes));
    // beg[c] = index of the first element of chunk c; num_unique[c + 1] = number of distinct elements of chunk c
    std::vector<uint64_t> beg(chunks + 1), num_unique(chunks + 1, 0);
    beg[chunks] = v.size();

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t c = 0; c < chunks; c++) {
        uint64_t b = (c * v.size()) / chunks;
        while (b > 0 && b < v.size() && key(v[b]) == key(v[b - 1])) b++;
        beg[c] = b;
    }

    beg[0] = 0;

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t c = 0; c < chunks; c++) {
        uint64_t e = std::max(beg[c], beg[c + 1]);
        uint64_t out = beg[c];

        for (uint64_t b = beg[c], f; b < e; b = f) {
            f = b + 1;
            while (f < e && key(v[f]) == key(v[b])) f++;
            if (f - b > 1) std::sort(v.begin() + b, v.begin() + f, [&](const elem_t& e1, const elem_t& e2) { return second(e1) < second(e2); });

            for (uint64_t i = b; i < f; i++) {
                if (i == b || second(v[i]) != second(v[i - 1])) v[out++] = v[i];
            }
        }

        num_unique[c + 1] = out - beg[c];
    }

    // concatenate the distinct elements of the chunks
    for (uint64_t c = 0; c < chunks; c++) num_unique[c + 1] += num_unique[c];
    no_init_vector<elem_t> w;
    resize_no_init(w, num_unique[chunks]);

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t c = 0; c < chunks; c++) std::copy(v.begin() + beg[c], v.begin() + beg[c] + (num_unique[c + 1] - num_unique[c]), w.begin() + num_unique[c]);

    v.swap(w);
}

inline void mds_build_word_ranks(const no_init_vector<uint64_t>& words, no_init_vector<uint64_t>& ranks, uint64_t num_threads)
{
    uint64_t num_words = words.size();
    resize_no_init(ranks, num_words + 1);
    // one part per thread, with at least one cache line of words
    uint64_t parts = std::max<uint64_t>(1, std::min<uint64_t>(num_threads, num_words * sizeof(uint64_t) / cache_line_bytes));

    // part_ones[c + 1] = number of ones in part c, then prefix sums over the parts
    std::vector<uint64_t> part_ones(parts + 1, 0);

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t c = 0; c < parts; c++) {
        uint64_t ones = 0;
        for (uint64_t w = (c * num_words) / parts; w < ((c + 1) * num_words) / parts; w++) ones += std::popcount(words[w]);
        part_ones[c + 1] = ones;
    }

    for (uint64_t c = 0; c < parts; c++) part_ones[c + 1] += part_ones[c];

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t c = 0; c < parts; c++) {
        uint64_t ones = part_ones[c];

        for (uint64_t w = (c * num_words) / parts; w < ((c + 1) * num_words) / parts; w++) {
            ranks[w] = ones;
            ones += std::popcount(words[w]);
        }
    }

    ranks[num_words] = part_ones[parts];
}

inline uint64_t mds_elias_fano::low(uint64_t i) const
{
    if (low_width == 0) return 0;
    uint64_t bit = i * low_width;
    uint64_t w = bit >> 6;
    uint64_t shift = bit & 63;
    uint64_t v = lower[w] >> shift;
    if (shift + low_width > 64) v |= lower[w + 1] << (64 - shift);
    return v & ((uint64_t { 1 } << low_width) - 1);
}

inline uint64_t mds_elias_fano::select1(uint64_t j) const
{
    uint64_t w = select1_samples[j >> 6];
    uint64_t e = select1_samples[(j >> 6) + 1] + 1; // ones_before_word[e] > j

    if (e - w <= linear_scan_length) {
        while (ones_before_word[w + 1] <= j) w++;
    } else {
        while (e - w > 1) {
            uint64_t m = w + (e - w) / 2;
            if (ones_before_word[m] <= j) w = m;
            else e = m;
        }
    }

    return 64 * w + sux::select64(upper[w], j - ones_before_word[w]);
}

inline uint64_t mds_elias_fano::select0(uint64_t j) const
{
    uint64_t w = select0_samples[j >> 6];
    uint64_t e = select0_samples[(j >> 6) + 1] + 1; // zeros_before(e) > j

    if (e - w <= linear_scan_length) {
        while (zeros_before(w + 1) <= j) w++;
    } else {
        while (e - w > 1) {
            uint64_t m = w + (e - w) / 2;
            if (zeros_before(m) <= j) w = m;
            else e = m;
        }
    }

    return 64 * w + sux::select64(~upper[w], j - zeros_before(w));
}

template <typename read_fnc_t>
mds_elias_fano::mds_elias_fano(read_fnc_t read, uint64_t num_values, uint64_t universe_, uint64_t num_threads)
    : num_values(num_values), universe(universe_)
{
    if (num_values == 0) return;
    low_width = universe <= num_values ? 0 : uint8_t(std::bit_width(universe / num_values) - 1);
    uint64_t num_upper = num_values + (universe >> low_width) + 1;
    uint64_t words = num_upper / 64 + 2;
    resize_no_init(lower, (num_values * low_width) / 64 + 2);
    resize_no_init(upper, words);

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t w = 0; w < lower.size(); w++) lower[w] = 0;

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t w = 0; w < upper.size(); w++) upper[w] = 0;

    // write lower and upper; each chunk starts at a multiple of 64 values, so no two chunks write to the same
    // word of lower; a word of upper can contain ones of two chunks and is written atomically
    uint64_t chunks = std::max<uint64_t>(1, std::min<uint64_t>(num_threads, (num_values + 63) / 64));

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t c = 0; c < chunks; c++) {
        uint64_t b = (((c * ((num_values + 63) / 64)) / chunks) * 64);
        uint64_t e = std::min(num_values, ((((c + 1) * ((num_values + 63) / 64)) / chunks) * 64));
        uint64_t word_idx = 0, word = 0;

        for (uint64_t i = b; i < e; i++) {
            uint64_t v = read(i);

            if (low_width != 0) {
                uint64_t bit = i * low_width;
                uint64_t lv = v & ((uint64_t { 1 } << low_width) - 1);
                lower[bit >> 6] |= lv << (bit & 63);
                if ((bit & 63) + low_width > 64) lower[(bit >> 6) + 1] |= lv >> (64 - (bit & 63));
            }

            uint64_t pos = (v >> low_width) + i;

            if (pos / 64 != word_idx) {
                if (word != 0) __atomic_fetch_or(&upper[word_idx], word, __ATOMIC_RELAXED);
                word_idx = pos / 64;
                word = 0;
            }

            word |= uint64_t { 1 } << (pos % 64);
        }

        if (word != 0) __atomic_fetch_or(&upper[word_idx], word, __ATOMIC_RELAXED);
    }

    mds_build_word_ranks(upper, ones_before_word, num_threads);

    // build select1_samples and select0_samples; the entries after the last sample are set to the last word
    uint64_t num_zeros = 64 * words - num_values;
    resize_no_init(select1_samples, num_values / 64 + 2);
    resize_no_init(select0_samples, num_zeros / 64 + 2);

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t w = 0; w < words; w++) {
        for (uint64_t j = (ones_before_word[w] + 63) / 64; j * 64 < ones_before_word[w + 1]; j++) select1_samples[j] = w;
        for (uint64_t j = (zeros_before(w) + 63) / 64; j * 64 < zeros_before(w + 1); j++) select0_samples[j] = w;
    }

    for (uint64_t j = (num_values + 63) / 64; j < select1_samples.size(); j++) select1_samples[j] = words - 1;
    for (uint64_t j = (num_zeros + 63) / 64; j < select0_samples.size(); j++) select0_samples[j] = words - 1;
}

inline uint64_t mds_elias_fano::rank(uint64_t v) const
{
    if (num_values == 0 || v >= universe) return num_values;
    uint64_t h = v >> low_width;
    uint64_t lv = v & ((uint64_t { 1 } << low_width) - 1);
    // the values with a high part smaller than h are the ones before the (h-1)-th zero of upper
    uint64_t pos = h == 0 ? 0 : select0(h - 1) + 1;
    uint64_t i = pos - h;

    // compare the low parts of the first linear_scan_length values with the high part h
    for (uint64_t t = 0; t < linear_scan_length; t++, pos++, i++) {
        if (!((upper[pos >> 6] >> (pos & 63)) & 1)) return i;
        if (low(i) >= lv) return i;
    }

    // binary search over the low parts of the remaining values with the high part h (they end at the next
    // zero of upper)
    uint64_t w = pos >> 6;
    uint64_t word = ~upper[w] & (~uint64_t { 0 } << (pos & 63));
    while (word == 0) word = ~upper[++w];
    uint64_t e = 64 * w + std::countr_zero(word) - h;

    while (i < e) {
        uint64_t mid = i + (e - i) / 2;
        if (low(mid) < lv) i = mid + 1;
        else e = mid;
    }

    return i;
}

inline void mds_piece_marks::build(uint64_t num_threads)
{
    uint64_t num_words = bits.size();
    mds_build_word_ranks(bits, ones_before_word, num_threads);
    resize_no_init(select_samples, ones_before_word[num_words] / 64 + 1);

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t w = 0; w < num_words; w++) {
        for (uint64_t i = (ones_before_word[w] + 63) / 64; i * 64 < ones_before_word[w + 1]; i++) select_samples[i] = w;
    }
}

template <typename write_fnc_t>
inline void mds_deferred_writes<write_fnc_t>::push(const void* word, uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
    __builtin_prefetch(word, 1, 3);
    std::array<uint64_t, 4>& e = queue[num_queued % depth];
    if (num_queued >= depth) write(e[0], e[1], e[2], e[3]);
    e = { a, b, c, d };
    num_queued++;
}

template <typename write_fnc_t>
inline void mds_deferred_writes<write_fnc_t>::flush()
{
    for (uint64_t i = num_queued > depth ? num_queued - depth : 0; i < num_queued; i++) {
        const std::array<uint64_t, 4>& e = queue[i % depth];
        write(e[0], e[1], e[2], e[3]);
    }

    num_queued = 0;
}

template <move_pos_encoding_t enc, move_mode_t mode>
inline void move_data_structure<enc, mode>::construction::pieces_t::init(uint64_t len_, uint64_t l_max_,
    const uint_bytes_array<1>& offsets_, std::pair<uint64_t, uint64_t> splits)
{
    offsets = &offsets_;
    len = len_;
    l_max = l_max_;
    next_multiple = l_max;
    next_split = splits.first;
    first_split = splits.first;
    end_split = splits.second;
    beg = 0;
    piece_idx = 0;
    end = next_piece_offset();
}

template <move_pos_encoding_t enc, move_mode_t mode>
inline bool move_data_structure<enc, mode>::construction::pieces_t::next()
{
    if (end == len) return false;
    beg = end;
    piece_idx++;
    if (next_multiple == beg) next_multiple += l_max;
    if (next_split != end_split && offsets->get(next_split) == beg) next_split++;
    end = next_piece_offset();
    return true;
}

template <move_pos_encoding_t enc, move_mode_t mode>
inline void move_data_structure<enc, mode>::construction::pieces_t::seek(uint64_t d, uint64_t num_splits)
{
    // the piece starts at the last multiple of l_max or the last stored split offset that is <= d
    uint64_t num_multiples = d / l_max;
    beg = num_multiples * l_max;
    if (num_splits != 0 && offsets->get(first_split + num_splits - 1) > beg) beg = offsets->get(first_split + num_splits - 1);
    piece_idx = num_multiples + num_splits;
    next_multiple = (num_multiples + 1) * l_max;
    next_split = first_split + num_splits;
    end = next_piece_offset();
}

template <move_pos_encoding_t enc, move_mode_t mode>
inline void move_data_structure<enc, mode>::construction::input_cursor::go_to_pair(uint64_t x_)
{
    x = x_;
    p = c->p_of(x);

    if (x + 1 < c->k) {
        p_cursor = c->P.cursor_at(x + 1);
        p_next = p_cursor.value();
    } else {
        p_next = c->n;
    }

    pieces.init(p_next - p, c->l_max, c->splits.offsets, c->splits.splits_of(x));
}

template <move_pos_encoding_t enc, move_mode_t mode>
inline void move_data_structure<enc, mode>::construction::input_cursor::go_to_next_pair()
{
    x++;
    p = p_next;

    if (x + 1 < c->k) {
        p_cursor.next();
        p_next = p_cursor.value();
    } else {
        p_next = c->n;
    }

    pieces.init(p_next - p, c->l_max, c->splits.offsets, c->splits.splits_of(x));
}

template <move_pos_encoding_t enc, move_mode_t mode>
inline void move_data_structure<enc, mode>::construction::output_cursor::go_to_output(uint64_t y_)
{
    y = y_;
    q = c->q_of_output(y);
    q_next = c->q_of_output(y + 1);

    if (c->splits.output_has_splits(y)) {
        pair = c->pair_of_output(y);
        pieces.init(q_next - q, c->l_max, c->splits.offsets, c->splits.splits_of(pair));
    } else {
        pieces.init(q_next - q, c->l_max, c->splits.offsets, { 0, 0 });
    }
}

#endif // MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_HELPERS_TPP
