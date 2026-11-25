// src/bench.cpp
// Benchmark driver: measures sequential and parallel reduce / inclusive scan
// and writes ONE ROW PER REPETITION to a CSV file.
//
// Usage:
//   ./bench --out results/raw.csv --reps 30 --warmup 5 \
//           --sizes 65536,1048576,16777216 --threads 1,2,4,8
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "prefix_sum.hpp"

using Clock = std::chrono::steady_clock;
using u64 = std::uint64_t;
using psum::ScanKind;

// Sink prevents the compiler from removing the measured computation.
static volatile u64 g_sink = 0;

static std::vector<std::size_t> parse_list(const std::string &s)
{
  std::vector<std::size_t> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ','))
  {
    if (!tok.empty())
      v.push_back(static_cast<std::size_t>(std::stoull(tok)));
  }
  return v;
}

struct Config
{
  std::string out_path = "results/raw.csv";
  int reps = 30;
  int warmup = 5;
  std::vector<std::size_t> sizes = {1u << 16, 1u << 18, 1u << 20,
                                    1u << 22, 1u << 24, 1u << 25};
  std::vector<std::size_t> threads = {1, 2, 4, 8};
};

static Config parse_args(int argc, char **argv)
{
  Config c;
  for (int i = 1; i + 1 < argc; i += 2)
  {
    const std::string key = argv[i];
    const std::string val = argv[i + 1];
    if (key == "--out")
      c.out_path = val;
    else if (key == "--reps")
      c.reps = std::stoi(val);
    else if (key == "--warmup")
      c.warmup = std::stoi(val);
    else if (key == "--sizes")
      c.sizes = parse_list(val);
    else if (key == "--threads")
      c.threads = parse_list(val);
    else
      std::cerr << "Unknown option: " << key << "\n";
  }
  return c;
}

// Runs `body` (warmup + reps) and appends one CSV row per measured repetition.
template <typename Body>
static void measure(std::ofstream &csv, const std::string &algo, const std::string &op,
                    std::size_t n, std::size_t threads, bool correct, const Config &cfg,
                    Body &&body)
{
  for (int r = -cfg.warmup; r < cfg.reps; ++r)
  {
    const auto t0 = Clock::now();
    const u64 result = body();
    const auto t1 = Clock::now();
    g_sink = result;
    if (r >= 0)
    {
      const auto ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      csv << algo << ',' << op << ",u64," << n << ',' << threads << ',' << r << ','
          << ns << ',' << (correct ? 1 : 0) << '\n';
    }
  }
}

int main(int argc, char **argv)
{
  const Config cfg = parse_args(argc, argv);
  std::ofstream csv(cfg.out_path);
  if (!csv)
  {
    std::cerr << "Cannot open output file: " << cfg.out_path << "\n";
    return 1;
  }
  csv << "algo,op,dtype,n,threads,rep,time_ns,correct\n";

  auto add = [](u64 a, u64 b)
  { return a + b; };
  std::mt19937_64 rng(12345); // fixed seed => reproducible data

  for (const std::size_t n : cfg.sizes)
  {
    // Allocation and initialisation are OUTSIDE the timed region.
    std::vector<u64> in(n), out(n), ref(n);
    for (auto &x : in)
      x = rng();

    // Reference results (also used as the sequential baseline).
    psum::sequential_scan<ScanKind::Inclusive>(in, ref, u64{0}, add);
    const u64 ref_sum = psum::sequential_reduce(in, u64{0}, add);

    std::cerr << "n = " << n << "\n";

    // --- sequential baselines (threads = 1) ---
    measure(csv, "seq", "reduce", n, 1, true, cfg,
            [&]
            { return psum::sequential_reduce(in, u64{0}, add); });
    measure(csv, "seq", "scan_inc", n, 1, true, cfg, [&]
            {
              psum::sequential_scan<ScanKind::Inclusive>(in, out, u64{0}, add);
              return out.back(); // keeps the result observable
            });

    // --- parallel versions ---
    for (const std::size_t p : cfg.threads)
    {
      // Correctness check once per configuration (not timed).
      const bool ok_reduce = psum::parallel_reduce(in, u64{0}, add, p) == ref_sum;
      psum::parallel_scan<ScanKind::Inclusive>(in, out, u64{0}, add, p);
      const bool ok_scan = (out == ref);
      if (!ok_reduce || !ok_scan)
        std::cerr << "WARNING: mismatch n=" << n << " p=" << p << "\n";

      measure(csv, "par", "reduce", n, p, ok_reduce, cfg,
              [&]
              { return psum::parallel_reduce(in, u64{0}, add, p); });
      measure(csv, "par", "scan_inc", n, p, ok_scan, cfg, [&]
              {
          psum::parallel_scan<ScanKind::Inclusive>(in, out, u64{0}, add, p);
          return out.back(); });
    }
  }
  return 0;
}