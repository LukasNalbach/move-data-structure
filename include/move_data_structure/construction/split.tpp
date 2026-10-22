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

#ifndef MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_SPLIT_TPP
#define MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_SPLIT_TPP

#include <move_data_structure/construction/construction.hpp>

// ############################# SPLIT LISTS #############################

template <move_pos_encoding_t enc, move_mode_t mode>
void move_data_structure<enc, mode>::construction::split_lists::init(uint64_t k, uint64_t n_)
{
    n = n_;
    has_splits_p.assign(k / 64 + 1, 0);
    rank_has_splits_p.assign(k / 64 + 2, 0);
    has_splits_q.assign(k / 64 + 1, 0);
    pairs.clear();
    pairs_y.clear();
    offsets_begin.assign(1, 0);
    offsets.clear();
    long_before_p.clear();
    long_before_q.clear();
    split_pos_p = mds_elias_fano();
    split_pos_q = mds_elias_fano();
}

template <move_pos_encoding_t enc, move_mode_t mode>
void move_data_structure<enc, mode>::construction::split_lists::merge(const no_init_vector<split_t>& new_splits, uint64_t num_threads)
{
    /* the lists are merged in chunks of pairs; each chunk begins at a multiple of 64, so no two chunks write to
       the same word of has_splits_p. First, the pairs and the split offsets of each chunk are counted, then each
       chunk is written to the positions given by the prefix sums of the counts. */
    uint64_t num_pairs = 64 * (has_splits_p.size());
    uint64_t chunks = std::max<uint64_t>(1, std::min<uint64_t>(num_threads, has_splits_p.size()));
    // x_beg[c] = first pair of chunk c; i_beg[c] / j_beg[c] = index of its first pair in pairs / in new_splits
    std::vector<uint64_t> x_beg(chunks + 1), i_beg(chunks + 1), j_beg(chunks + 1);
    // cnt_pairs[c + 1] / cnt_offs[c + 1] = number of pairs / split offsets of chunk c after the merge, then their
    // prefix sums
    std::vector<uint64_t> cnt_pairs(chunks + 1, 0), cnt_offs(chunks + 1, 0);

    for (uint64_t c = 0; c <= chunks; c++) {
        x_beg[c] = c == chunks ? num_pairs : 64 * ((c * has_splits_p.size()) / chunks);
        i_beg[c] = c == chunks ? pairs.size() : rank_has_splits_p[x_beg[c] >> 6];
        j_beg[c] = std::lower_bound(new_splits.begin(), new_splits.end(), x_beg[c], [](const split_t& s, uint64_t x) { return s.x < x; }) - new_splits.begin();
    }

    constexpr uint64_t none = std::numeric_limits<uint64_t>::max();

    // merges the stored and the new splits of the chunk c; write = false: counts the resulting pairs and split
    // offsets, write = true: writes them to the new lists
    auto merge_chunk = [&](uint64_t c, bool write, no_init_vector<uint40_t>& pairs_new, no_init_vector<uint40_t>& pairs_y_new,
                           no_init_vector<uint40_t>& offsets_begin_new, uint_bytes_array<1>& offsets_new) {
        uint64_t i = i_beg[c];
        uint64_t j = j_beg[c];
        uint64_t num_p = write ? cnt_pairs[c] : 0;
        uint64_t num_o = write ? cnt_offs[c] : 0;

        while (i < i_beg[c + 1] || j < j_beg[c + 1]) {
            uint64_t x_old = i < i_beg[c + 1] ? uint64_t(pairs[i]) : none;
            uint64_t x_new = j < j_beg[c + 1] ? new_splits[j].x : none;
            uint64_t x = std::min(x_old, x_new);
            uint64_t b = x == x_old ? uint64_t(offsets_begin[i]) : 0;
            uint64_t e = x == x_old ? uint64_t(offsets_begin[i + 1]) : 0;
            uint64_t y = x == x_old ? uint64_t(pairs_y[i]) : new_splits[j].y;
            if (write) offsets_begin_new[num_p] = num_o;

            // merge the stored split offsets [b, e) of the pair x with its new split offsets
            while (b < e || (j < j_beg[c + 1] && new_splits[j].x == x)) {
                uint64_t o;
                if (j == j_beg[c + 1] || new_splits[j].x != x || (b < e && offsets.get(b) < new_splits[j].o)) o = offsets.get(b++);
                else o = new_splits[j++].o;
                if (write) offsets_new.set(num_o, o);
                num_o++;
            }

            if (write) {
                pairs_new[num_p] = x;
                pairs_y_new[num_p] = y;
            }

            num_p++;
            if (x == x_old) i++;
        }

        if (!write) {
            cnt_pairs[c + 1] = num_p;
            cnt_offs[c + 1] = num_o;
        }
    };

    no_init_vector<uint40_t> pairs_new, pairs_y_new, offsets_begin_new;
    uint_bytes_array<1> offsets_new;

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t c = 0; c < chunks; c++) merge_chunk(c, false, pairs_new, pairs_y_new, offsets_begin_new, offsets_new);

    for (uint64_t c = 0; c < chunks; c++) {
        cnt_pairs[c + 1] += cnt_pairs[c];
        cnt_offs[c + 1] += cnt_offs[c];
    }

    resize_no_init(pairs_new, cnt_pairs[chunks]);
    resize_no_init(pairs_y_new, cnt_pairs[chunks]);
    resize_no_init(offsets_begin_new, cnt_pairs[chunks] + 1);
    offsets_new.reset(cnt_offs[chunks], n);
    offsets_begin_new[cnt_pairs[chunks]] = cnt_offs[chunks];

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t c = 0; c < chunks; c++) {
        merge_chunk(c, true, pairs_new, pairs_y_new, offsets_begin_new, offsets_new);

        // mark the pairs of the new splits in has_splits_p
        for (uint64_t j = j_beg[c]; j < j_beg[c + 1]; j++) has_splits_p[new_splits[j].x >> 6] |= uint64_t { 1 } << (new_splits[j].x & 63);
    }

    pairs.swap(pairs_new);
    pairs_y.swap(pairs_y_new);
    offsets_begin.swap(offsets_begin_new);
    std::swap(offsets, offsets_new);

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t j = 0; j < new_splits.size(); j++) __atomic_fetch_or(&has_splits_q[new_splits[j].y >> 6], uint64_t { 1 } << (new_splits[j].y & 63), __ATOMIC_RELAXED);

    mds_build_word_ranks(has_splits_p, rank_has_splits_p, num_threads);
}

template <move_pos_encoding_t enc, move_mode_t mode>
void move_data_structure<enc, mode>::construction::split_lists::clear()
{
    no_init_vector<uint64_t>().swap(has_splits_p);
    no_init_vector<uint64_t>().swap(rank_has_splits_p);
    no_init_vector<uint64_t>().swap(has_splits_q);
    no_init_vector<uint40_t>().swap(pairs);
    no_init_vector<uint40_t>().swap(pairs_y);
    no_init_vector<uint40_t>().swap(offsets_begin);
    offsets.clear();
    no_init_vector<uint40_t>().swap(long_before_p);
    no_init_vector<uint40_t>().swap(long_before_q);
    split_pos_p = mds_elias_fano();
    split_pos_q = mds_elias_fano();
}

template <move_pos_encoding_t enc, move_mode_t mode>
void move_data_structure<enc, mode>::construction::index_splits()
{
    // collect the long pairs (see split_lists::max_short_splits) and build long_before_p
    uint64_t num_pairs = splits.pairs.size();
    auto num_offsets = [&](uint64_t i) { return uint64_t(splits.offsets_begin[i + 1]) - uint64_t(splits.offsets_begin[i]); };
    auto is_long = [&](uint64_t i) { return num_offsets(i) > split_lists::max_short_splits; };
    resize_no_init(splits.long_before_p, num_pairs + 1);
    resize_no_init(splits.long_before_q, num_pairs + 1);
    // one chunk per thread, each writing at least one cache line of long_before_p
    uint64_t chunks = std::max<uint64_t>(1, std::min<uint64_t>(num_threads, num_pairs * sizeof(uint40_t) / cache_line_bytes));
    // chunk_splits[c + 1] / chunk_long_pairs[c + 1] = number of split offsets of long pairs / number of long pairs
    // in chunk c, then their prefix sums
    std::vector<uint64_t> chunk_splits(chunks + 1, 0), chunk_long_pairs(chunks + 1, 0);

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t c = 0; c < chunks; c++) {
        for (uint64_t i = (c * num_pairs) / chunks; i < ((c + 1) * num_pairs) / chunks; i++) {
            if (!is_long(i)) continue;
            chunk_splits[c + 1] += num_offsets(i);
            chunk_long_pairs[c + 1]++;
        }
    }

    for (uint64_t c = 0; c < chunks; c++) {
        chunk_splits[c + 1] += chunk_splits[c];
        chunk_long_pairs[c + 1] += chunk_long_pairs[c];
    }

    uint64_t num_splits = chunk_splits[chunks];
    no_init_vector<uint40_t> long_pairs; // indices in splits.pairs of the long pairs
    resize_no_init(long_pairs, chunk_long_pairs[chunks]);

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t c = 0; c < chunks; c++) {
        uint64_t num_before = chunk_splits[c];
        uint64_t j = chunk_long_pairs[c];

        for (uint64_t i = (c * num_pairs) / chunks; i < ((c + 1) * num_pairs) / chunks; i++) {
            splits.long_before_p[i] = num_before;
            if (!is_long(i)) continue;
            long_pairs[j++] = i;
            num_before += num_offsets(i);
        }
    }

    splits.long_before_p[num_pairs] = num_splits;
    no_init_vector<uint64_t> split_pos;
    resize_no_init(split_pos, num_splits);

    // build split_pos_p: the split positions p_x + o of the long pairs, in the order of p
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, dynamic_chunk_length(long_pairs.size()))
    for (uint64_t j = 0; j < long_pairs.size(); j++) {
        uint64_t i = long_pairs[j];
        uint64_t p_x = p_of(splits.pairs[i]);
        uint64_t pos = splits.long_before_p[i];
        for (uint64_t c = splits.offsets_begin[i]; c < splits.offsets_begin[i + 1]; c++) split_pos[pos++] = p_x + splits.offsets.get(c);
    }

    splits.split_pos_p = mds_elias_fano([&](uint64_t c) { return split_pos[c]; }, num_splits, n, num_threads);

    // build split_pos_q and long_before_q: sort the long pairs by the rank of their output intervals
    radix_sort(long_pairs.begin(), long_pairs.end(), key_of_index<uint40_t> { splits.pairs_y.data() }, num_threads);
    no_init_vector<uint64_t> before; // before[j] = number of split offsets of the long pairs before long_pairs[j]
    resize_no_init(before, long_pairs.size() + 1);
    before[0] = 0;
    for (uint64_t j = 0; j < long_pairs.size(); j++) before[j + 1] = before[j] + num_offsets(long_pairs[j]);

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, dynamic_chunk_length(long_pairs.size()))
    for (uint64_t j = 0; j < long_pairs.size(); j++) {
        uint64_t i = long_pairs[j];
        uint64_t q_x = q_of_output(splits.pairs_y[i]);
        uint64_t pos = before[j];
        splits.long_before_q[i] = pos;
        for (uint64_t c = splits.offsets_begin[i]; c < splits.offsets_begin[i + 1]; c++) split_pos[pos++] = q_x + splits.offsets.get(c);
    }

    splits.split_pos_q = mds_elias_fano([&](uint64_t c) { return split_pos[c]; }, num_splits, n, num_threads);
    if (mf != nullptr) *mf << " long_pairs_round_" << num_rounds << "=" << long_pairs.size() << " long_cuts_round_" << num_rounds << "=" << num_splits;
}

#endif // MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_SPLIT_TPP
