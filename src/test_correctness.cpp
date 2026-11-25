// src/test_correctness.cpp
// Compares parallel results with a sequential reference for many (n, threads)
// combinations, for commutative and non-commutative operators.
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "prefix_sum.hpp"

using u64 = std::uint64_t;
using psum::ScanKind;

static int g_failures = 0;

#define CHECK(cond, msg)                    \
  do                                        \
  {                                         \
    if (!(cond))                            \
    {                                       \
      ++g_failures;                         \
      std::cerr << "FAIL: " << msg << "\n"; \
    }                                       \
  } while (0)

// Unsigned 64-bit addition wraps around (modular arithmetic): it is still
// associative, so results must match EXACTLY (no floating-point tolerance).
static void test_u64_sum()
{
  std::mt19937_64 rng(42);
  const std::vector<std::size_t> sizes = {0, 1, 2, 3, 7, 8, 100, 12345, 1000003};
  const std::vector<std::size_t> threads = {1, 2, 3, 4, 8, 16, 64};
  auto add = [](u64 a, u64 b)
  { return a + b; };

  for (std::size_t n : sizes)
  {
    std::vector<u64> a(n);
    for (auto &x : a)
      x = rng();

    std::vector<u64> ref_inc, ref_exc;
    psum::sequential_scan<ScanKind::Inclusive>(a, ref_inc, u64{0}, add);
    psum::sequential_scan<ScanKind::Exclusive>(a, ref_exc, u64{0}, add);
    const u64 ref_sum = psum::sequential_reduce(a, u64{0}, add);

    for (std::size_t p : threads)
    {
      const u64 s = psum::parallel_reduce(a, u64{0}, add, p);
      CHECK(s == ref_sum, "reduce n=" << n << " p=" << p);

      std::vector<u64> inc, exc;
      psum::parallel_scan<ScanKind::Inclusive>(a, inc, u64{0}, add, p);
      psum::parallel_scan<ScanKind::Exclusive>(a, exc, u64{0}, add, p);
      CHECK(inc == ref_inc, "inclusive n=" << n << " p=" << p);
      CHECK(exc == ref_exc, "exclusive n=" << n << " p=" << p);

      // In-place variant
      std::vector<u64> inplace = a;
      psum::parallel_scan<ScanKind::Inclusive>(inplace, inplace, u64{0}, add, p);
      CHECK(inplace == ref_inc, "in-place n=" << n << " p=" << p);
    }
  }
}

// String concatenation is associative but NOT commutative: it detects any
// bug that combines chunks in the wrong order.
static void test_non_commutative()
{
  std::vector<std::string> a;
  for (int i = 0; i < 1000; ++i)
    a.push_back(std::string(1, char('a' + i % 26)));
  auto cat = [](const std::string &x, const std::string &y)
  { return x + y; };

  std::vector<std::string> ref;
  psum::sequential_scan<ScanKind::Inclusive>(a, ref, std::string{}, cat);

  for (std::size_t p : {1u, 2u, 3u, 7u, 16u})
  {
    std::vector<std::string> out;
    psum::parallel_scan<ScanKind::Inclusive>(a, out, std::string{}, cat, p);
    CHECK(out == ref, "string concat scan p=" << p);
    CHECK(psum::parallel_reduce(a, std::string{}, cat, p) == ref.back(),
          "string concat reduce p=" << p);
  }
}

// Max is associative, commutative, idempotent. Identity = lowest value.
static void test_max()
{
  std::mt19937_64 rng(7);
  std::vector<long long> a(50001);
  for (auto &x : a)
    x = static_cast<long long>(rng() % 2000001) - 1000000;
  auto mx = [](long long x, long long y)
  { return x > y ? x : y; };
  const long long id = std::numeric_limits<long long>::lowest();

  std::vector<long long> ref;
  psum::sequential_scan<ScanKind::Inclusive>(a, ref, id, mx);
  for (std::size_t p : {1u, 4u, 9u})
  {
    std::vector<long long> out;
    psum::parallel_scan<ScanKind::Inclusive>(a, out, id, mx, p);
    CHECK(out == ref, "max scan p=" << p);
  }
}

// Floating point: addition is NOT associative, so only compare with tolerance.
static void test_double_tolerance()
{
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<double> d(0.0, 1.0);
  std::vector<double> a(1'000'000);
  for (auto &x : a)
    x = d(rng);
  auto add = [](double x, double y)
  { return x + y; };

  const double ref = psum::sequential_reduce(a, 0.0, add);
  const double par = psum::parallel_reduce(a, 0.0, add, 8);
  const double rel = std::abs(ref - par) / std::abs(ref);
  CHECK(rel < 1e-12, "double reduce relative error " << rel);
}

int main()
{
  test_u64_sum();
  test_non_commutative();
  test_max();
  test_double_tolerance();
  if (g_failures == 0)
  {
    std::cout << "ALL TESTS PASSED\n";
    return 0;
  }
  std::cerr << g_failures << " FAILURE(S)\n";
  return 1;
}