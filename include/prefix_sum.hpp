// include/prefix_sum.hpp
// Header-only implementation of sequential and parallel reduce / scan.
// Parallel strategy: chunking + offset (reduce-then-scan, three phases).
#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <numeric>
#include <thread>
#include <vector>

namespace psum
{

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  struct Range
  {
    std::size_t begin;
    std::size_t end; // exclusive
  };

  // Split [0, n) into p contiguous chunks whose sizes differ by at most 1.
  inline Range chunk_range(std::size_t n, std::size_t p, std::size_t i)
  {
    const std::size_t base = n / p;
    const std::size_t rem = n % p;
    const std::size_t begin = i * base + std::min(i, rem);
    const std::size_t end = begin + base + (i < rem ? 1 : 0);
    return {begin, end};
  }

  // Clamp thread count: 0 means "use hardware_concurrency"; never exceed n;
  // always at least 1 (so empty input is handled without dividing by zero).
  inline std::size_t effective_threads(std::size_t n, std::size_t requested)
  {
    if (requested == 0)
    {
      const unsigned hw = std::thread::hardware_concurrency();
      requested = hw == 0 ? 1 : hw; // hardware_concurrency() may return 0
    }
    return std::max<std::size_t>(1, std::min(requested, n));
  }

  // One cache line per slot to avoid false sharing between threads that write
  // neighbouring partial results.
  template <typename T>
  struct alignas(64) Padded
  {
    T value{};
  };

  // Run fn(0..p-1) in parallel: p-1 worker threads + the calling thread.
  // Exceptions thrown inside a worker are captured and rethrown after join,
  // because an exception escaping a std::thread would call std::terminate.
  template <typename Fn>
  void run_parallel(std::size_t p, Fn &&fn)
  {
    std::vector<std::exception_ptr> errors(p);
    auto task = [&](std::size_t i)
    {
      try
      {
        fn(i);
      }
      catch (...)
      {
        errors[i] = std::current_exception();
      }
    };

    std::vector<std::thread> workers;
    workers.reserve(p > 0 ? p - 1 : 0);
    for (std::size_t i = 1; i < p; ++i)
      workers.emplace_back(task, i);
    task(0); // calling thread does chunk 0
    for (auto &t : workers)
      t.join(); // join acts as the barrier

    for (auto &e : errors)
      if (e)
        std::rethrow_exception(e);
  }

  // ---------------------------------------------------------------------------
  // Sequential baselines
  // ---------------------------------------------------------------------------

  template <typename T, typename Op>
  T sequential_reduce(const std::vector<T> &a, T identity, Op op)
  {
    T acc = identity;
    for (const T &x : a)
      acc = op(acc, x);
    return acc;
  }

  enum class ScanKind
  {
    Inclusive,
    Exclusive
  };

  template <ScanKind Kind, typename T, typename Op>
  void sequential_scan(const std::vector<T> &in, std::vector<T> &out, T identity, Op op)
  {
    out.resize(in.size());
    T acc = identity;
    for (std::size_t i = 0; i < in.size(); ++i)
    {
      const T x = in[i]; // copy first: safe when &in == &out (in-place)
      if constexpr (Kind == ScanKind::Exclusive)
      {
        out[i] = acc;
        acc = op(acc, x);
      }
      else
      {
        acc = op(acc, x);
        out[i] = acc;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Parallel reduce: phase 1 (local reduce) + sequential combine of p partials
  // ---------------------------------------------------------------------------

  template <typename T, typename Op>
  T parallel_reduce(const std::vector<T> &a, T identity, Op op, std::size_t threads)
  {
    const std::size_t n = a.size();
    const std::size_t p = effective_threads(n, threads);
    std::vector<Padded<T>> partial(p);

    run_parallel(p, [&](std::size_t i)
                 {
        const Range r = chunk_range(n, p, i);
        T acc = identity;
        for (std::size_t k = r.begin; k < r.end; ++k) acc = op(acc, a[k]);
        partial[i].value = acc; });

    // Combine IN ORDER (0..p-1) so non-commutative operators stay correct.
    T result = identity;
    for (std::size_t i = 0; i < p; ++i)
      result = op(result, partial[i].value);
    return result;
  }

  // ---------------------------------------------------------------------------
  // Parallel scan (reduce-then-scan, three phases)
  // Works in place (&in == &out) because every thread finishes reading its own
  // chunk in phase 1 before any thread writes in phase 3 (join in between), and
  // in phase 3 element k is read before out[k] is written.
  // ---------------------------------------------------------------------------

  template <ScanKind Kind, typename T, typename Op>
  void parallel_scan(const std::vector<T> &in, std::vector<T> &out, T identity, Op op,
                     std::size_t threads)
  {
    const std::size_t n = in.size();
    out.resize(n);
    const std::size_t p = effective_threads(n, threads);

    // Phase 1 (parallel): chunk sums.
    std::vector<Padded<T>> sums(p);
    run_parallel(p, [&](std::size_t i)
                 {
        const Range r = chunk_range(n, p, i);
        T acc = identity;
        for (std::size_t k = r.begin; k < r.end; ++k) acc = op(acc, in[k]);
        sums[i].value = acc; }); // <-- barrier

    // Phase 2 (sequential, O(p)): exclusive scan of chunk sums -> offsets.
    std::vector<T> offset(p);
    T running = identity;
    for (std::size_t i = 0; i < p; ++i)
    {
      offset[i] = running;
      running = op(running, sums[i].value);
    }

    // Phase 3 (parallel): local scan seeded with the chunk's offset.
    run_parallel(p, [&](std::size_t i)
                 {
        const Range r = chunk_range(n, p, i);
        T acc = offset[i];
        for (std::size_t k = r.begin; k < r.end; ++k) {
            const T x = in[k];
            if constexpr (Kind == ScanKind::Exclusive) {
                out[k] = acc;
                acc = op(acc, x);
            } else {
                acc = op(acc, x);
                out[k] = acc;
            }
        } });
  }

} // namespace psum