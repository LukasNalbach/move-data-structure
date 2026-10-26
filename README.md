# move-data-structure

This repository contains a C++ implementation of the move data structure [1] with the balancing of [2]. It stores a
disjoint interval sequence of [0, n) and answers move queries in constant time.

- `move_data_structure<enc, mode>` is built in parallel from the pairs of the interval sequence. A move query maps a
  position and the index of its input interval to its image and the index of the input interval containing it.
- `enc` is `POS` (default) or `EF` (Elias-Fano coded starting positions, smaller).
- `mode` is `UNI` (default) or `BI` (also move queries on the inverse function, only with `POS`).
- Up to 5 extra rows with one value of a given bit width per input interval can be stored.

## Build

Requirements: a C++20 compiler (GCC 13 or later, Clang 18 or later), CMake 3.21 or later and OpenMP.

```shell
git clone https://github.com/LukasNalbach/move-data-structure.git
cd move-data-structure
cmake -B build
cmake --build build -j
```

`cmake` downloads the submodule [string-utils](https://github.com/LukasNalbach/string-utils). The tests are written to
`build/tests/` (`MOVE_DATA_STRUCTURE_BUILD_TESTS`, `ON` if move-data-structure is the top-level project). The general
options and environment variables (`MOVE_DATA_STRUCTURE_MARCH_NATIVE`, ...) are described in string-utils.

## C++ usage

Add move-data-structure to a CMake project and link the target `move_data_structure`:

```cmake
add_subdirectory(move-data-structure/ EXCLUDE_FROM_ALL)
target_link_libraries(<target> PRIVATE move_data_structure)
```

The construction parameters (`mds_params`) are the number of threads, the balancing parameter `a` (default 8, at
least 2) and whether to log the construction.

### Building from pairs and move queries

```cpp
#include <move_data_structure/move_data_structure.hpp>

int main()
{
    // I = (0,1), (1,0) with n = 2 (swaps the positions 0 and 1)
    move_data_structure mds(make_mds_pairs({ { 0, 1 }, { 1, 0 } }, 2), 2);
    // 2 2
    std::cout << mds.num_intervals() << " " << mds.max_value() << std::endl;

    // move queries on a pair (position, index of its input interval)
    auto ix = mds.move(std::pair<uint64_t, uint64_t>{ 0, 0 });
    std::cout << ix.first << "," << ix.second << std::endl; // 1,1
    uint64_t i = 1, x = 1;
    mds.move(i, x);
    std::cout << i << "," << x << std::endl; // 0,0

    // a sequence that is not 2-balanced: the output interval [0, 4) contains
    // 4 = 2a starting positions; the balancing splits the input interval
    // [4, 8) into [4, 6) and [6, 8), i.e., adds the pair (6, 2)
    move_data_structure balanced(
        make_mds_pairs({ { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }, { 4, 0 } }, 8),
        8, { .num_threads = 1, .a = 2 });
    for (uint64_t y = 0; y < balanced.num_intervals(); y++)
        std::cout << "(" << balanced.p(y) << "," << balanced.q(y) << ") ";
    std::cout << std::endl; // (0,4) (1,5) (2,6) (3,7) (4,0) (6,2)

    // the fields of an input interval: its length, idx(x) (the input interval
    // containing q(x)), offs(x) = q(x) - p(idx(x))
    // 2 2 0
    std::cout << balanced.len(5) << " " << balanced.idx(5) << " "
              << balanced.offs(5) << std::endl;

    // the input interval of a position and the offset in it (binary search),
    // then a move query
    auto [x7, o7] = balanced.interval_and_offset(7);
    std::cout << x7 << "," << o7 << std::endl; // 5,1
    uint64_t j = 7;
    balanced.move(j, x7);
    std::cout << j << "," << x7 << std::endl; // 3,3
}
```

### Encodings, the inverse and extra rows

```cpp
#include <move_data_structure/move_data_structure.hpp>

int main()
{
    std::vector<std::pair<uint64_t, uint64_t>> pairs =
        { { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }, { 4, 0 } };

    // EF: move takes and returns positions, move_offset the offset in the
    // input interval
    move_data_structure<EF> mds_ef(make_mds_pairs(pairs, 8), 8, { .a = 2 });
    uint64_t i = 7, x = 5;
    mds_ef.move(i, x);
    std::cout << i << "," << x << std::endl; // 3,3
    uint64_t o = 1; // position 7 = p(5) + 1
    x = 5;
    mds_ef.move_offset(o, x);
    std::cout << mds_ef.pos(x, o) << "," << x << std::endl; // 3,3

    // BI: move queries on the inverse function with the index y of the output
    // interval of a position
    move_data_structure<POS, BI> mds_bi(make_mds_pairs(pairs, 8), 8,
        { .a = 2 });
    i = 3;
    uint64_t y = mds_bi.output_interval_of(i, 3);
    mds_bi.move_inv(i, y);
    std::cout << i << " " << mds_bi.p_inv(y) << std::endl; // 7 7

    // one extra row of 3 bits per input interval
    move_data_structure mds_rows(make_mds_pairs(pairs, 8), 8, { .a = 2 },
        { 3 });
    for (uint64_t z = 0; z < mds_rows.num_intervals(); z++)
        mds_rows.set_row<0>(z, z + 1);
    // 6 3
    std::cout << mds_rows.row<0>(5) << " " << int(mds_rows.row_width<0>())
              << std::endl;
}
```

### Storing and loading

```cpp
#include <filesystem>
#include <move_data_structure/move_data_structure.hpp>

int main()
{
    move_data_structure mds(make_mds_pairs({ { 0, 1 }, { 1, 0 } }, 2), 2);

    std::ofstream out("mds.bin");
    mds.serialize(out); // or: mds >> out;
    out.close();

    move_data_structure loaded;
    std::ifstream in("mds.bin");
    // the number of extra rows (read from the file with POS)
    loaded.load(in, 0);
    in.close();
    // 2 1
    std::cout << loaded.num_intervals() << " " << loaded.q(0) << std::endl;

    std::filesystem::remove("mds.bin");
}
```

## Tests

```shell
build/tests/test-move-data-structure
```

The tests check random inputs for at most `MOVE_DATA_STRUCTURE_TEST_SECONDS` seconds (default 10) or
`MOVE_DATA_STRUCTURE_TEST_ITERATIONS` inputs (default 1000) per function. A failed test prints its seed;
`MOVE_DATA_STRUCTURE_TEST_SEED=<seed>` repeats its inputs.

## References

[1] Takaaki Nishimoto and Yasuo Tabei. Optimal-time queries on bwt-runs compressed indexes.
In 48th International Colloquium on Automata, Languages, and Programming (ICALP), 2021. ([paper](https://arxiv.org/abs/2006.05104))

[2] Nico Bertram, Johannes Fischer and Lukas Nalbach. Move-r: Optimizing the r-index.
In Symposium on Experimental Algorithms (SEA), 2024. ([paper](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SEA.2024.1))

## License

This project is licensed under the MIT License, see [LICENSE](LICENSE). The submodules in [external/](external/) have
their own licenses.
