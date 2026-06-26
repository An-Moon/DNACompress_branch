#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr uint32_t STATE_BITS = 32;
constexpr uint64_t FULL_RANGE = uint64_t{1} << STATE_BITS;
constexpr uint64_t HALF_RANGE = FULL_RANGE >> 1;
constexpr uint64_t QUARTER_RANGE = HALF_RANGE >> 1;
constexpr uint64_t MASK = FULL_RANGE - 1;

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point &start, const Clock::time_point &end) {
  return std::chrono::duration<double>(end - start).count();
}

struct Interval {
  uint32_t low;
  uint32_t high;
  uint32_t total;
};

template <typename scalar_t>
Interval quantize_interval_from_row_sorted(const scalar_t *row, int64_t vocab_size, int64_t symbol, int64_t total_i64) {
  if (vocab_size <= 0) {
    throw std::runtime_error("probability row must have at least one symbol");
  }
  if (symbol < 0 || symbol >= vocab_size) {
    throw std::runtime_error("target symbol is outside the probability row vocabulary");
  }
  if (total_i64 <= vocab_size || total_i64 > static_cast<int64_t>((uint64_t{1} << 30) + 2)) {
    throw std::runtime_error("frequency total is outside the supported 32-bit arithmetic range");
  }

  const auto total = static_cast<uint32_t>(total_i64);
  double row_sum = 0.0;
  for (int64_t i = 0; i < vocab_size; ++i) {
    const double value = static_cast<double>(row[i]);
    if (std::isfinite(value) && value > 0.0) {
      row_sum += value;
    }
  }
  row_sum = std::max(row_sum, 1e-300);

  std::vector<uint32_t> freq(static_cast<size_t>(vocab_size));
  std::vector<double> fractional(static_cast<size_t>(vocab_size));
  int64_t freq_sum = 0;
  const double scale = static_cast<double>(total_i64 - vocab_size);
  for (int64_t i = 0; i < vocab_size; ++i) {
    double probability = static_cast<double>(row[i]);
    if (!std::isfinite(probability) || probability < 0.0) {
      probability = 0.0;
    }
    const double scaled = (probability / row_sum) * scale;
    const double floored = std::floor(scaled);
    const auto value = static_cast<uint32_t>(floored) + 1U;
    freq[static_cast<size_t>(i)] = value;
    fractional[static_cast<size_t>(i)] = scaled - floored;
    freq_sum += value;
  }

  int64_t remainder = total_i64 - freq_sum;
  std::vector<int64_t> order(static_cast<size_t>(vocab_size));
  std::iota(order.begin(), order.end(), 0);
  if (remainder > 0) {
    std::stable_sort(order.begin(), order.end(), [&](int64_t lhs, int64_t rhs) {
      return fractional[static_cast<size_t>(lhs)] > fractional[static_cast<size_t>(rhs)];
    });
    int64_t cursor = 0;
    while (remainder > 0) {
      const int64_t index = order[static_cast<size_t>(cursor % vocab_size)];
      ++freq[static_cast<size_t>(index)];
      --remainder;
      ++cursor;
    }
  } else if (remainder < 0) {
    std::stable_sort(order.begin(), order.end(), [&](int64_t lhs, int64_t rhs) {
      return freq[static_cast<size_t>(lhs)] > freq[static_cast<size_t>(rhs)];
    });
    int64_t debt = -remainder;
    while (debt > 0) {
      bool made_progress = false;
      for (int64_t index : order) {
        auto &value = freq[static_cast<size_t>(index)];
        if (value <= 1U) {
          continue;
        }
        --value;
        --debt;
        made_progress = true;
        if (debt == 0) {
          break;
        }
      }
      if (!made_progress) {
        throw std::runtime_error("failed to normalize arithmetic coding frequencies");
      }
    }
  }

  uint32_t low = 0;
  for (int64_t i = 0; i < symbol; ++i) {
    low += freq[static_cast<size_t>(i)];
  }
  const uint32_t high = low + freq[static_cast<size_t>(symbol)];
  if (!(low < high && high <= total)) {
    throw std::runtime_error("invalid quantized target interval");
  }
  return Interval{low, high, total};
}

template <typename scalar_t>
Interval quantize_interval_from_row(const scalar_t *row, int64_t vocab_size, int64_t symbol, int64_t total_i64) {
  return quantize_interval_from_row_sorted(row, vocab_size, symbol, total_i64);
}

template <typename scalar_t>
std::vector<uint32_t> quantize_freqs_from_row_fast_floor(const scalar_t *row, int64_t vocab_size, int64_t total_i64) {
  if (vocab_size <= 0) {
    throw std::runtime_error("probability row must have at least one symbol");
  }
  if (total_i64 <= 0 || total_i64 > static_cast<int64_t>((uint64_t{1} << 30) + 2)) {
    throw std::runtime_error("frequency total is outside the supported 32-bit arithmetic range");
  }

  std::vector<uint32_t> freqs(static_cast<size_t>(vocab_size));
  uint64_t freq_sum = 0;
  for (int64_t i = 0; i < vocab_size; ++i) {
    double probability = static_cast<double>(row[i]);
    if (!std::isfinite(probability) || probability < 0.0) {
      probability = 0.0;
    }
    const double scaled = probability * static_cast<double>(total_i64);
    uint64_t value = 1;
    if (std::isfinite(scaled) && scaled > 1.0) {
      value = static_cast<uint64_t>(std::floor(scaled));
    }
    if (value > static_cast<uint64_t>((uint64_t{1} << 30) + 2)) {
      throw std::runtime_error("fast-floor frequency is outside the supported 32-bit arithmetic range");
    }
    freqs[static_cast<size_t>(i)] = static_cast<uint32_t>(value);
    freq_sum += value;
  }
  if (freq_sum == 0 || freq_sum > static_cast<uint64_t>((uint64_t{1} << 30) + 2)) {
    throw std::runtime_error("fast-floor row total is outside the supported 32-bit arithmetic range");
  }
  return freqs;
}

template <typename scalar_t>
Interval quantize_interval_from_row_fast_floor(const scalar_t *row, int64_t vocab_size, int64_t symbol, int64_t total_i64) {
  if (symbol < 0 || symbol >= vocab_size) {
    throw std::runtime_error("target symbol is outside the probability row vocabulary");
  }
  std::vector<uint32_t> freqs = quantize_freqs_from_row_fast_floor(row, vocab_size, total_i64);
  uint32_t low = 0;
  for (int64_t i = 0; i < symbol; ++i) {
    low += freqs[static_cast<size_t>(i)];
  }
  const uint32_t high = low + freqs[static_cast<size_t>(symbol)];
  uint32_t row_total = 0;
  for (uint32_t freq : freqs) {
    row_total += freq;
  }
  if (!(low < high && high <= row_total)) {
    throw std::runtime_error("invalid fast-floor quantized target interval");
  }
  return Interval{low, high, row_total};
}

template <typename scalar_t>
std::vector<uint32_t> quantize_freqs_from_row(const scalar_t *row, int64_t vocab_size, int64_t total_i64) {
  std::vector<uint32_t> freqs(static_cast<size_t>(vocab_size));
  for (int64_t symbol = 0; symbol < vocab_size; ++symbol) {
    const Interval interval = quantize_interval_from_row(row, vocab_size, symbol, total_i64);
    freqs[static_cast<size_t>(symbol)] = interval.high - interval.low;
  }
  return freqs;
}

template <typename scalar_t, typename symbol_t>
std::vector<Interval> quantize_probability_rows_impl(
    const at::Tensor &probabilities,
    const at::Tensor &symbols,
    int64_t total) {
  if (probabilities.dim() != 2) {
    throw std::runtime_error("probabilities must be a 2D CPU tensor");
  }
  if (symbols.dim() != 1 || symbols.size(0) != probabilities.size(0)) {
    throw std::runtime_error("symbols must be a 1D tensor matching probability rows");
  }

  const int64_t rows = probabilities.size(0);
  const int64_t vocab_size = probabilities.size(1);
  const scalar_t *prob_ptr = probabilities.data_ptr<scalar_t>();
  const symbol_t *symbol_ptr = symbols.data_ptr<symbol_t>();
  std::vector<Interval> intervals;
  intervals.reserve(static_cast<size_t>(rows));
  for (int64_t row = 0; row < rows; ++row) {
    intervals.push_back(quantize_interval_from_row(
        prob_ptr + row * vocab_size,
        vocab_size,
        static_cast<int64_t>(symbol_ptr[row]),
        total));
  }
  return intervals;
}

template <typename scalar_t, typename symbol_t>
std::vector<Interval> quantize_probability_rows_fast_floor_impl(
    const at::Tensor &probabilities,
    const at::Tensor &symbols,
    int64_t total) {
  if (probabilities.dim() != 2) {
    throw std::runtime_error("probabilities must be a 2D CPU tensor");
  }
  if (symbols.dim() != 1 || symbols.size(0) != probabilities.size(0)) {
    throw std::runtime_error("symbols must be a 1D tensor matching probability rows");
  }

  const int64_t rows = probabilities.size(0);
  const int64_t vocab_size = probabilities.size(1);
  const scalar_t *prob_ptr = probabilities.data_ptr<scalar_t>();
  const symbol_t *symbol_ptr = symbols.data_ptr<symbol_t>();
  std::vector<Interval> intervals;
  intervals.reserve(static_cast<size_t>(rows));
  for (int64_t row = 0; row < rows; ++row) {
    intervals.push_back(quantize_interval_from_row_fast_floor(
        prob_ptr + row * vocab_size,
        vocab_size,
        static_cast<int64_t>(symbol_ptr[row]),
        total));
  }
  return intervals;
}

std::vector<Interval> quantize_probability_rows(const at::Tensor &probabilities, const at::Tensor &symbols, int64_t total) {
  if (probabilities.device().is_cuda() || symbols.device().is_cuda()) {
    throw std::runtime_error("fast arithmetic encoder expects CPU tensors");
  }
  at::Tensor probs = probabilities.contiguous();
  at::Tensor syms = symbols.contiguous();

  if (probs.scalar_type() == at::kDouble) {
    if (syms.scalar_type() == at::kLong) {
      return quantize_probability_rows_impl<double, int64_t>(probs, syms, total);
    }
    if (syms.scalar_type() == at::kInt) {
      return quantize_probability_rows_impl<double, int32_t>(probs, syms, total);
    }
  }
  if (probs.scalar_type() == at::kFloat) {
    if (syms.scalar_type() == at::kLong) {
      return quantize_probability_rows_impl<float, int64_t>(probs, syms, total);
    }
    if (syms.scalar_type() == at::kInt) {
      return quantize_probability_rows_impl<float, int32_t>(probs, syms, total);
    }
  }
  throw std::runtime_error("fast arithmetic supports float32/float64 probabilities and int32/int64 symbols");
}

std::vector<Interval> quantize_probability_rows_fast_floor(const at::Tensor &probabilities, const at::Tensor &symbols, int64_t total) {
  if (probabilities.device().is_cuda() || symbols.device().is_cuda()) {
    throw std::runtime_error("fast arithmetic encoder expects CPU tensors");
  }
  at::Tensor probs = probabilities.contiguous();
  at::Tensor syms = symbols.contiguous();

  if (probs.scalar_type() == at::kDouble) {
    if (syms.scalar_type() == at::kLong) {
      return quantize_probability_rows_fast_floor_impl<double, int64_t>(probs, syms, total);
    }
    if (syms.scalar_type() == at::kInt) {
      return quantize_probability_rows_fast_floor_impl<double, int32_t>(probs, syms, total);
    }
  }
  if (probs.scalar_type() == at::kFloat) {
    if (syms.scalar_type() == at::kLong) {
      return quantize_probability_rows_fast_floor_impl<float, int64_t>(probs, syms, total);
    }
    if (syms.scalar_type() == at::kInt) {
      return quantize_probability_rows_fast_floor_impl<float, int32_t>(probs, syms, total);
    }
  }
  throw std::runtime_error("fast arithmetic supports float32/float64 probabilities and int32/int64 symbols");
}

template <typename int_t>
std::vector<Interval> intervals_from_tensors_impl(
    const at::Tensor &lows,
    const at::Tensor &highs,
    const at::Tensor &totals) {
  const int64_t rows = lows.size(0);
  const int_t *low_ptr = lows.data_ptr<int_t>();
  const int_t *high_ptr = highs.data_ptr<int_t>();
  const int_t *total_ptr = totals.data_ptr<int_t>();
  std::vector<Interval> intervals;
  intervals.reserve(static_cast<size_t>(rows));
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t low_i64 = static_cast<int64_t>(low_ptr[row]);
    const int64_t high_i64 = static_cast<int64_t>(high_ptr[row]);
    const int64_t total_i64 = static_cast<int64_t>(total_ptr[row]);
    if (!(0 <= low_i64 && low_i64 < high_i64 && high_i64 <= total_i64)) {
      throw std::runtime_error("invalid arithmetic interval tensors");
    }
    if (total_i64 <= 0 || total_i64 > static_cast<int64_t>((uint64_t{1} << 30) + 2)) {
      throw std::runtime_error("interval total is outside the supported 32-bit arithmetic range");
    }
    intervals.push_back(Interval{
        static_cast<uint32_t>(low_i64),
        static_cast<uint32_t>(high_i64),
        static_cast<uint32_t>(total_i64)});
  }
  return intervals;
}

std::vector<Interval> intervals_from_tensors(
    const at::Tensor &lows,
    const at::Tensor &highs,
    const at::Tensor &totals) {
  if (lows.device().is_cuda() || highs.device().is_cuda() || totals.device().is_cuda()) {
    throw std::runtime_error("interval tensors must be on CPU");
  }
  if (lows.dim() != 1 || highs.dim() != 1 || totals.dim() != 1 ||
      highs.size(0) != lows.size(0) || totals.size(0) != lows.size(0)) {
    throw std::runtime_error("interval tensors must be 1D tensors with equal length");
  }
  at::Tensor low_tensor = lows.contiguous();
  at::Tensor high_tensor = highs.contiguous();
  at::Tensor total_tensor = totals.contiguous();
  if (low_tensor.scalar_type() == at::kLong &&
      high_tensor.scalar_type() == at::kLong &&
      total_tensor.scalar_type() == at::kLong) {
    return intervals_from_tensors_impl<int64_t>(low_tensor, high_tensor, total_tensor);
  }
  if (low_tensor.scalar_type() == at::kInt &&
      high_tensor.scalar_type() == at::kInt &&
      total_tensor.scalar_type() == at::kInt) {
    return intervals_from_tensors_impl<int32_t>(low_tensor, high_tensor, total_tensor);
  }
  throw std::runtime_error("interval tensors must all be int32 or all be int64");
}

std::vector<Interval> quantize_grouped_steps(
    const std::vector<at::Tensor> &step_probabilities,
    const std::vector<at::Tensor> &step_symbols,
    const std::vector<at::Tensor> &step_row_positions,
    int64_t row_count,
    int64_t total) {
  if (step_probabilities.size() != step_symbols.size() ||
      step_probabilities.size() != step_row_positions.size()) {
    throw std::runtime_error("grouped step probability, symbol, and position lists must have equal length");
  }
  std::vector<Interval> intervals;
  for (int64_t row = 0; row < row_count; ++row) {
    for (size_t step = 0; step < step_probabilities.size(); ++step) {
      at::Tensor positions = step_row_positions[step].contiguous();
      if (positions.dim() != 1 || positions.size(0) != row_count) {
        throw std::runtime_error("step row positions must be 1D tensors with row_count entries");
      }
      int64_t position = 0;
      if (positions.scalar_type() == at::kLong) {
        position = positions.data_ptr<int64_t>()[row];
      } else if (positions.scalar_type() == at::kInt) {
        position = positions.data_ptr<int32_t>()[row];
      } else {
        throw std::runtime_error("step row positions must be int32 or int64");
      }
      if (position < 0) {
        continue;
      }

      at::Tensor probs = step_probabilities[step].contiguous();
      at::Tensor symbols = step_symbols[step].contiguous();
      if (probs.dim() != 2 || symbols.dim() != 1 || position >= probs.size(0) || position >= symbols.size(0)) {
        throw std::runtime_error("invalid grouped step probability/symbol shape or position");
      }

      int64_t symbol = 0;
      if (symbols.scalar_type() == at::kLong) {
        symbol = symbols.data_ptr<int64_t>()[position];
      } else if (symbols.scalar_type() == at::kInt) {
        symbol = symbols.data_ptr<int32_t>()[position];
      } else {
        throw std::runtime_error("step symbols must be int32 or int64");
      }

      const int64_t vocab_size = probs.size(1);
      if (probs.scalar_type() == at::kDouble) {
        intervals.push_back(quantize_interval_from_row(
            probs.data_ptr<double>() + position * vocab_size, vocab_size, symbol, total));
      } else if (probs.scalar_type() == at::kFloat) {
        intervals.push_back(quantize_interval_from_row(
            probs.data_ptr<float>() + position * vocab_size, vocab_size, symbol, total));
      } else {
        throw std::runtime_error("step probabilities must be float32 or float64");
      }
    }
  }
  return intervals;
}

class BitWriter {
 public:
  void write(int bit) {
    current_byte_ = static_cast<uint8_t>((current_byte_ << 1) | (bit & 1));
    ++num_bits_filled_;
    if (num_bits_filled_ == 8) {
      buffer_.push_back(current_byte_);
      current_byte_ = 0;
      num_bits_filled_ = 0;
    }
  }

  std::string finish() {
    if (num_bits_filled_ > 0) {
      current_byte_ <<= static_cast<uint8_t>(8 - num_bits_filled_);
      buffer_.push_back(current_byte_);
      current_byte_ = 0;
      num_bits_filled_ = 0;
    }
    return std::string(reinterpret_cast<const char *>(buffer_.data()), buffer_.size());
  }

 private:
  std::vector<uint8_t> buffer_;
  uint8_t current_byte_ = 0;
  int num_bits_filled_ = 0;
};

class BitReader {
 public:
  explicit BitReader(std::string data) : data_(std::move(data)) {}

  int read() {
    if (byte_index_ >= data_.size()) {
      return 0;
    }
    const uint8_t byte = static_cast<uint8_t>(data_[byte_index_]);
    const int bit = (byte >> (7 - bit_index_)) & 1;
    ++bit_index_;
    if (bit_index_ == 8) {
      bit_index_ = 0;
      ++byte_index_;
    }
    return bit;
  }

 private:
  std::string data_;
  size_t byte_index_ = 0;
  int bit_index_ = 0;
};

class FastArithmeticEncoder {
  friend class BatchedFastArithmeticEncoder;

 public:
  py::dict encode_probability_rows(const at::Tensor &probabilities, const at::Tensor &symbols, int64_t total) {
    const auto quantize_start = Clock::now();
    std::vector<Interval> intervals = quantize_probability_rows(probabilities, symbols, total);
    const auto quantize_end = Clock::now();
    const auto range_start = Clock::now();
    for (const auto &interval : intervals) {
      update(interval);
    }
    const auto range_end = Clock::now();
    py::dict result;
    result["quantize_seconds"] = elapsed_seconds(quantize_start, quantize_end);
    result["range_seconds"] = elapsed_seconds(range_start, range_end);
    result["emitted_count"] = static_cast<int64_t>(intervals.size());
    return result;
  }

  py::dict encode_probability_rows_fast_floor(const at::Tensor &probabilities, const at::Tensor &symbols, int64_t total) {
    const auto quantize_start = Clock::now();
    std::vector<Interval> intervals = quantize_probability_rows_fast_floor(probabilities, symbols, total);
    const auto quantize_end = Clock::now();
    const auto range_start = Clock::now();
    for (const auto &interval : intervals) {
      update(interval);
    }
    const auto range_end = Clock::now();
    py::dict result;
    result["quantize_seconds"] = elapsed_seconds(quantize_start, quantize_end);
    result["range_seconds"] = elapsed_seconds(range_start, range_end);
    result["emitted_count"] = static_cast<int64_t>(intervals.size());
    return result;
  }

  py::dict encode_intervals(const at::Tensor &lows, const at::Tensor &highs, const at::Tensor &totals) {
    const auto range_start = Clock::now();
    std::vector<Interval> intervals = intervals_from_tensors(lows, highs, totals);
    for (const auto &interval : intervals) {
      update(interval);
    }
    const auto range_end = Clock::now();
    py::dict result;
    result["quantize_seconds"] = 0.0;
    result["range_seconds"] = elapsed_seconds(range_start, range_end);
    result["emitted_count"] = static_cast<int64_t>(intervals.size());
    return result;
  }

  py::dict encode_grouped_steps(
      const std::vector<at::Tensor> &step_probabilities,
      const std::vector<at::Tensor> &step_symbols,
      const std::vector<at::Tensor> &step_row_positions,
      int64_t row_count,
      int64_t total) {
    const auto quantize_start = Clock::now();
    std::vector<Interval> intervals = quantize_grouped_steps(
        step_probabilities, step_symbols, step_row_positions, row_count, total);
    const auto quantize_end = Clock::now();
    const auto range_start = Clock::now();
    for (const auto &interval : intervals) {
      update(interval);
    }
    const auto range_end = Clock::now();
    py::dict result;
    result["quantize_seconds"] = elapsed_seconds(quantize_start, quantize_end);
    result["range_seconds"] = elapsed_seconds(range_start, range_end);
    result["emitted_count"] = static_cast<int64_t>(intervals.size());
    return result;
  }

  py::bytes finish() {
    pending_underflow_ += 1;
    if (low_ < QUARTER_RANGE) {
      shift(0);
    } else {
      shift(1);
    }
    return py::bytes(writer_.finish());
  }

 private:
  void update(const Interval &interval) {
    const uint64_t current_range = high_ - low_ + 1;
    high_ = low_ + (current_range * interval.high / interval.total) - 1;
    low_ = low_ + (current_range * interval.low / interval.total);

    while (((low_ ^ high_) & HALF_RANGE) == 0) {
      const int bit = static_cast<int>(low_ >> (STATE_BITS - 1));
      shift(bit);
      low_ = (low_ << 1) & MASK;
      high_ = ((high_ << 1) & MASK) | 1;
    }

    while ((low_ & ~high_ & QUARTER_RANGE) != 0) {
      ++pending_underflow_;
      low_ = (low_ << 1) & (MASK >> 1);
      high_ = ((high_ ^ HALF_RANGE) << 1) | HALF_RANGE | 1;
    }
  }

  void shift(int bit) {
    writer_.write(bit);
    while (pending_underflow_ > 0) {
      writer_.write(bit ^ 1);
      --pending_underflow_;
    }
  }

  uint64_t low_ = 0;
  uint64_t high_ = MASK;
  uint64_t pending_underflow_ = 0;
  BitWriter writer_;
};

class BatchedFastArithmeticEncoder {
 public:
  explicit BatchedFastArithmeticEncoder(int64_t stream_count) {
    if (stream_count <= 0) {
      throw std::runtime_error("batched encoder requires at least one stream");
    }
    encoders_.resize(static_cast<size_t>(stream_count));
  }

  py::dict encode_interval_matrix(const at::Tensor &lows, const at::Tensor &highs, const at::Tensor &totals) {
    if (lows.device().is_cuda() || highs.device().is_cuda() || totals.device().is_cuda()) {
      throw std::runtime_error("interval matrix tensors must be on CPU");
    }
    if (lows.dim() != 2 || highs.dim() != 2 || totals.dim() != 2 ||
        highs.sizes() != lows.sizes() || totals.sizes() != lows.sizes()) {
      throw std::runtime_error("interval matrix tensors must be 2D tensors with equal shapes");
    }
    if (lows.size(0) != static_cast<int64_t>(encoders_.size())) {
      throw std::runtime_error("interval matrix row count must match batched encoder stream count");
    }
    at::Tensor low_tensor = lows.contiguous();
    at::Tensor high_tensor = highs.contiguous();
    at::Tensor total_tensor = totals.contiguous();
    const int64_t rows = low_tensor.size(0);
    const int64_t steps = low_tensor.size(1);
    const auto range_start = Clock::now();
    int64_t emitted = 0;

    const auto encode_impl = [&](auto *low_ptr, auto *high_ptr, auto *total_ptr) {
      for (int64_t row = 0; row < rows; ++row) {
        FastArithmeticEncoder &encoder = encoders_[static_cast<size_t>(row)];
        for (int64_t step = 0; step < steps; ++step) {
          const int64_t index = row * steps + step;
          const int64_t low_i64 = static_cast<int64_t>(low_ptr[index]);
          const int64_t high_i64 = static_cast<int64_t>(high_ptr[index]);
          const int64_t total_i64 = static_cast<int64_t>(total_ptr[index]);
          if (!(0 <= low_i64 && low_i64 < high_i64 && high_i64 <= total_i64)) {
            throw std::runtime_error("invalid arithmetic interval matrix");
          }
          if (total_i64 <= 0 || total_i64 > static_cast<int64_t>((uint64_t{1} << 30) + 2)) {
            throw std::runtime_error("interval matrix total is outside the supported 32-bit arithmetic range");
          }
          encoder.update(Interval{
              static_cast<uint32_t>(low_i64),
              static_cast<uint32_t>(high_i64),
              static_cast<uint32_t>(total_i64)});
          ++emitted;
        }
      }
    };

    if (low_tensor.scalar_type() == at::kLong &&
        high_tensor.scalar_type() == at::kLong &&
        total_tensor.scalar_type() == at::kLong) {
      encode_impl(low_tensor.data_ptr<int64_t>(), high_tensor.data_ptr<int64_t>(), total_tensor.data_ptr<int64_t>());
    } else if (low_tensor.scalar_type() == at::kInt &&
               high_tensor.scalar_type() == at::kInt &&
               total_tensor.scalar_type() == at::kInt) {
      encode_impl(low_tensor.data_ptr<int32_t>(), high_tensor.data_ptr<int32_t>(), total_tensor.data_ptr<int32_t>());
    } else {
      throw std::runtime_error("interval matrix tensors must all be int32 or all be int64");
    }

    const auto range_end = Clock::now();
    py::dict result;
    result["range_seconds"] = elapsed_seconds(range_start, range_end);
    result["emitted_count"] = emitted;
    return result;
  }

  py::dict encode_interval_matrix_with_lengths(
      const at::Tensor &lows,
      const at::Tensor &highs,
      const at::Tensor &totals,
      const at::Tensor &lengths) {
    if (lows.device().is_cuda() || highs.device().is_cuda() || totals.device().is_cuda() || lengths.device().is_cuda()) {
      throw std::runtime_error("interval matrix tensors and lengths must be on CPU");
    }
    if (lows.dim() != 2 || highs.dim() != 2 || totals.dim() != 2 || lengths.dim() != 1 ||
        highs.sizes() != lows.sizes() || totals.sizes() != lows.sizes() || lengths.size(0) != lows.size(0)) {
      throw std::runtime_error("interval matrix tensors must be [rows, steps] and lengths must be [rows]");
    }
    if (lows.size(0) != static_cast<int64_t>(encoders_.size())) {
      throw std::runtime_error("interval matrix row count must match batched encoder stream count");
    }
    at::Tensor low_tensor = lows.contiguous();
    at::Tensor high_tensor = highs.contiguous();
    at::Tensor total_tensor = totals.contiguous();
    at::Tensor length_tensor = lengths.contiguous();
    const int64_t rows = low_tensor.size(0);
    const int64_t steps = low_tensor.size(1);
    const auto range_start = Clock::now();
    int64_t emitted = 0;

    const auto length_at = [&](int64_t row) -> int64_t {
      int64_t value = 0;
      if (length_tensor.scalar_type() == at::kLong) {
        value = length_tensor.data_ptr<int64_t>()[row];
      } else if (length_tensor.scalar_type() == at::kInt) {
        value = length_tensor.data_ptr<int32_t>()[row];
      } else {
        throw std::runtime_error("lengths must be int64 or int32");
      }
      if (value < 0 || value > steps) {
        throw std::runtime_error("row length is outside interval matrix step count");
      }
      return value;
    };

    const auto encode_impl = [&](auto *low_ptr, auto *high_ptr, auto *total_ptr) {
      for (int64_t row = 0; row < rows; ++row) {
        FastArithmeticEncoder &encoder = encoders_[static_cast<size_t>(row)];
        const int64_t row_steps = length_at(row);
        for (int64_t step = 0; step < row_steps; ++step) {
          const int64_t index = row * steps + step;
          const int64_t low_i64 = static_cast<int64_t>(low_ptr[index]);
          const int64_t high_i64 = static_cast<int64_t>(high_ptr[index]);
          const int64_t total_i64 = static_cast<int64_t>(total_ptr[index]);
          if (!(0 <= low_i64 && low_i64 < high_i64 && high_i64 <= total_i64)) {
            throw std::runtime_error("invalid arithmetic interval matrix");
          }
          if (total_i64 <= 0 || total_i64 > static_cast<int64_t>((uint64_t{1} << 30) + 2)) {
            throw std::runtime_error("interval matrix total is outside the supported 32-bit arithmetic range");
          }
          encoder.update(Interval{
              static_cast<uint32_t>(low_i64),
              static_cast<uint32_t>(high_i64),
              static_cast<uint32_t>(total_i64)});
          ++emitted;
        }
      }
    };

    if (low_tensor.scalar_type() == at::kLong &&
        high_tensor.scalar_type() == at::kLong &&
        total_tensor.scalar_type() == at::kLong) {
      encode_impl(low_tensor.data_ptr<int64_t>(), high_tensor.data_ptr<int64_t>(), total_tensor.data_ptr<int64_t>());
    } else if (low_tensor.scalar_type() == at::kInt &&
               high_tensor.scalar_type() == at::kInt &&
               total_tensor.scalar_type() == at::kInt) {
      encode_impl(low_tensor.data_ptr<int32_t>(), high_tensor.data_ptr<int32_t>(), total_tensor.data_ptr<int32_t>());
    } else {
      throw std::runtime_error("interval matrix tensors must all be int32 or all be int64");
    }

    const auto range_end = Clock::now();
    py::dict result;
    result["range_seconds"] = elapsed_seconds(range_start, range_end);
    result["emitted_count"] = emitted;
    return result;
  }

  std::vector<py::bytes> finish() {
    std::vector<py::bytes> streams;
    streams.reserve(encoders_.size());
    for (auto &encoder : encoders_) {
      streams.push_back(encoder.finish());
    }
    return streams;
  }

  int64_t size() const {
    return static_cast<int64_t>(encoders_.size());
  }

 private:
  std::vector<FastArithmeticEncoder> encoders_;
};

class FastArithmeticDecoder {
 public:
  explicit FastArithmeticDecoder(const py::bytes &encoded) : reader_(std::string(encoded)) {
    for (uint32_t i = 0; i < STATE_BITS; ++i) {
      code_ = ((code_ << 1) & MASK) | static_cast<uint64_t>(reader_.read());
    }
  }

  at::Tensor decode_probability_rows(const at::Tensor &probabilities, int64_t total) {
    if (probabilities.device().is_cuda() || probabilities.dim() != 2) {
      throw std::runtime_error("decoder probabilities must be a 2D CPU tensor");
    }
    at::Tensor probs = probabilities.contiguous();
    const int64_t rows = probs.size(0);
    const int64_t vocab_size = probs.size(1);
    std::vector<int64_t> decoded;
    decoded.reserve(static_cast<size_t>(rows));

    for (int64_t row = 0; row < rows; ++row) {
      std::vector<uint32_t> freqs;
      if (probs.scalar_type() == at::kDouble) {
        freqs = quantize_freqs_from_row(probs.data_ptr<double>() + row * vocab_size, vocab_size, total);
      } else if (probs.scalar_type() == at::kFloat) {
        freqs = quantize_freqs_from_row(probs.data_ptr<float>() + row * vocab_size, vocab_size, total);
      } else {
        throw std::runtime_error("decoder probabilities must be float32 or float64");
      }
      decoded.push_back(decode_symbol(freqs, static_cast<uint32_t>(total)));
    }

    return torch::from_blob(decoded.data(), {rows}, torch::TensorOptions().dtype(torch::kInt64)).clone();
  }

  at::Tensor decode_probability_rows_fast_floor(const at::Tensor &probabilities, int64_t total) {
    if (probabilities.device().is_cuda() || probabilities.dim() != 2) {
      throw std::runtime_error("decoder probabilities must be a 2D CPU tensor");
    }
    at::Tensor probs = probabilities.contiguous();
    const int64_t rows = probs.size(0);
    const int64_t vocab_size = probs.size(1);
    std::vector<int64_t> decoded;
    decoded.reserve(static_cast<size_t>(rows));

    for (int64_t row = 0; row < rows; ++row) {
      std::vector<uint32_t> freqs;
      if (probs.scalar_type() == at::kDouble) {
        freqs = quantize_freqs_from_row_fast_floor(probs.data_ptr<double>() + row * vocab_size, vocab_size, total);
      } else if (probs.scalar_type() == at::kFloat) {
        freqs = quantize_freqs_from_row_fast_floor(probs.data_ptr<float>() + row * vocab_size, vocab_size, total);
      } else {
        throw std::runtime_error("decoder probabilities must be float32 or float64");
      }
      uint32_t row_total = 0;
      for (uint32_t freq : freqs) {
        row_total += freq;
      }
      decoded.push_back(decode_symbol(freqs, row_total));
    }

    return torch::from_blob(decoded.data(), {rows}, torch::TensorOptions().dtype(torch::kInt64)).clone();
  }

  int64_t decode_frequency_row(const at::Tensor &frequencies) {
    if (frequencies.device().is_cuda() || frequencies.dim() != 1) {
      throw std::runtime_error("frequency row must be a 1D CPU tensor");
    }
    at::Tensor freqs_tensor = frequencies.contiguous();
    const int64_t vocab_size = freqs_tensor.size(0);
    if (vocab_size <= 0) {
      throw std::runtime_error("frequency row must have at least one symbol");
    }
    std::vector<uint32_t> freqs(static_cast<size_t>(vocab_size));
    uint64_t total = 0;
    if (freqs_tensor.scalar_type() == at::kLong) {
      const int64_t *ptr = freqs_tensor.data_ptr<int64_t>();
      for (int64_t i = 0; i < vocab_size; ++i) {
        if (ptr[i] <= 0) {
          throw std::runtime_error("frequency row entries must be positive");
        }
        freqs[static_cast<size_t>(i)] = static_cast<uint32_t>(ptr[i]);
        total += static_cast<uint64_t>(ptr[i]);
      }
    } else if (freqs_tensor.scalar_type() == at::kInt) {
      const int32_t *ptr = freqs_tensor.data_ptr<int32_t>();
      for (int64_t i = 0; i < vocab_size; ++i) {
        if (ptr[i] <= 0) {
          throw std::runtime_error("frequency row entries must be positive");
        }
        freqs[static_cast<size_t>(i)] = static_cast<uint32_t>(ptr[i]);
        total += static_cast<uint64_t>(ptr[i]);
      }
    } else {
      throw std::runtime_error("frequency row must be int32 or int64");
    }
    if (total == 0 || total > static_cast<uint64_t>((uint64_t{1} << 30) + 2)) {
      throw std::runtime_error("frequency row total is outside the supported 32-bit arithmetic range");
    }
    return decode_symbol(freqs, static_cast<uint32_t>(total));
  }

  template <typename freq_t>
  int64_t decode_frequency_row_ptr(const freq_t *ptr, int64_t vocab_size) {
    if (vocab_size <= 0) {
      throw std::runtime_error("frequency row must have at least one symbol");
    }
    uint64_t total = 0;
    for (int64_t i = 0; i < vocab_size; ++i) {
      const auto raw_value = ptr[i];
      if constexpr (std::is_signed<freq_t>::value) {
        if (raw_value <= 0) {
          throw std::runtime_error("frequency row entries must be positive");
        }
      } else {
        if (raw_value == 0) {
          throw std::runtime_error("frequency row entries must be positive");
        }
      }
      total += static_cast<uint64_t>(raw_value);
    }
    if (total == 0 || total > static_cast<uint64_t>((uint64_t{1} << 30) + 2)) {
      throw std::runtime_error("frequency row total is outside the supported 32-bit arithmetic range");
    }

    const auto total_u32 = static_cast<uint32_t>(total);
    const uint64_t current_range = high_ - low_ + 1;
    const uint64_t offset = code_ - low_;
    const uint64_t value = ((offset + 1) * total_u32 - 1) / current_range;

    uint32_t cumulative = 0;
    int64_t symbol = -1;
    uint32_t symbol_low = 0;
    uint32_t symbol_high = 0;
    for (int64_t i = 0; i < vocab_size; ++i) {
      const uint32_t freq = static_cast<uint32_t>(ptr[i]);
      const uint32_t next = cumulative + freq;
      if (value < next) {
        symbol = i;
        symbol_low = cumulative;
        symbol_high = next;
        break;
      }
      cumulative = next;
    }
    if (symbol < 0) {
      throw std::runtime_error("failed to decode arithmetic symbol");
    }

    high_ = low_ + (current_range * symbol_high / total_u32) - 1;
    low_ = low_ + (current_range * symbol_low / total_u32);

    while (((low_ ^ high_) & HALF_RANGE) == 0) {
      low_ = (low_ << 1) & MASK;
      high_ = ((high_ << 1) & MASK) | 1;
      code_ = ((code_ << 1) & MASK) | static_cast<uint64_t>(reader_.read());
    }
    while ((low_ & ~high_ & QUARTER_RANGE) != 0) {
      low_ = (low_ << 1) & (MASK >> 1);
      high_ = ((high_ ^ HALF_RANGE) << 1) | HALF_RANGE | 1;
      code_ = ((code_ - QUARTER_RANGE) << 1) | static_cast<uint64_t>(reader_.read());
      code_ &= MASK;
    }
    return symbol;
  }

  template <typename freq_t>
  int64_t decode_frequency_row_ptr_with_total(const freq_t *ptr, int64_t vocab_size, uint32_t total_u32) {
    if (vocab_size <= 0) {
      throw std::runtime_error("frequency row must have at least one symbol");
    }
    if (total_u32 == 0 || total_u32 > static_cast<uint32_t>((uint64_t{1} << 30) + 2)) {
      throw std::runtime_error("frequency row total is outside the supported 32-bit arithmetic range");
    }
    const uint64_t current_range = high_ - low_ + 1;
    const uint64_t offset = code_ - low_;
    const uint64_t value = ((offset + 1) * total_u32 - 1) / current_range;

    uint32_t cumulative = 0;
    int64_t symbol = -1;
    uint32_t symbol_low = 0;
    uint32_t symbol_high = 0;
    for (int64_t i = 0; i < vocab_size; ++i) {
      const uint32_t freq = static_cast<uint32_t>(ptr[i]);
      const uint32_t next = cumulative + freq;
      if (value < next) {
        symbol = i;
        symbol_low = cumulative;
        symbol_high = next;
        break;
      }
      cumulative = next;
    }
    if (symbol < 0 || symbol_high > total_u32) {
      throw std::runtime_error("failed to decode arithmetic symbol");
    }

    high_ = low_ + (current_range * symbol_high / total_u32) - 1;
    low_ = low_ + (current_range * symbol_low / total_u32);

    while (((low_ ^ high_) & HALF_RANGE) == 0) {
      low_ = (low_ << 1) & MASK;
      high_ = ((high_ << 1) & MASK) | 1;
      code_ = ((code_ << 1) & MASK) | static_cast<uint64_t>(reader_.read());
    }
    while ((low_ & ~high_ & QUARTER_RANGE) != 0) {
      low_ = (low_ << 1) & (MASK >> 1);
      high_ = ((high_ ^ HALF_RANGE) << 1) | HALF_RANGE | 1;
      code_ = ((code_ - QUARTER_RANGE) << 1) | static_cast<uint64_t>(reader_.read());
      code_ &= MASK;
    }
    return symbol;
  }

 private:
  int64_t decode_symbol(const std::vector<uint32_t> &freqs, uint32_t total) {
    const uint64_t current_range = high_ - low_ + 1;
    const uint64_t offset = code_ - low_;
    const uint64_t value = ((offset + 1) * total - 1) / current_range;

    uint32_t cumulative = 0;
    int64_t symbol = -1;
    uint32_t symbol_low = 0;
    uint32_t symbol_high = 0;
    for (size_t i = 0; i < freqs.size(); ++i) {
      const uint32_t next = cumulative + freqs[i];
      if (value < next) {
        symbol = static_cast<int64_t>(i);
        symbol_low = cumulative;
        symbol_high = next;
        break;
      }
      cumulative = next;
    }
    if (symbol < 0) {
      throw std::runtime_error("failed to decode arithmetic symbol");
    }

    high_ = low_ + (current_range * symbol_high / total) - 1;
    low_ = low_ + (current_range * symbol_low / total);

    while (((low_ ^ high_) & HALF_RANGE) == 0) {
      low_ = (low_ << 1) & MASK;
      high_ = ((high_ << 1) & MASK) | 1;
      code_ = ((code_ << 1) & MASK) | static_cast<uint64_t>(reader_.read());
    }
    while ((low_ & ~high_ & QUARTER_RANGE) != 0) {
      low_ = (low_ << 1) & (MASK >> 1);
      high_ = ((high_ ^ HALF_RANGE) << 1) | HALF_RANGE | 1;
      code_ = ((code_ - QUARTER_RANGE) << 1) | static_cast<uint64_t>(reader_.read());
      code_ &= MASK;
    }
    return symbol;
  }

  uint64_t low_ = 0;
  uint64_t high_ = MASK;
  uint64_t code_ = 0;
  BitReader reader_;
};

class BatchedFastArithmeticDecoder {
 public:
  BatchedFastArithmeticDecoder(const std::vector<py::bytes> &encoded_streams, int64_t threads = 0)
      : thread_count_(threads) {
    if (encoded_streams.empty()) {
      throw std::runtime_error("batched decoder requires at least one stream");
    }
    decoders_.reserve(encoded_streams.size());
    for (const auto &encoded : encoded_streams) {
      decoders_.emplace_back(encoded);
    }
  }

  at::Tensor decode_frequency_rows(const at::Tensor &frequencies) {
    if (frequencies.device().is_cuda() || frequencies.dim() != 2) {
      throw std::runtime_error("frequency rows must be a 2D CPU tensor");
    }
    if (frequencies.size(0) != static_cast<int64_t>(decoders_.size())) {
      throw std::runtime_error("frequency row count must match batched decoder stream count");
    }
    at::Tensor freq_tensor = frequencies.contiguous();
    std::vector<int64_t> decoded(decoders_.size());
    const int64_t rows = freq_tensor.size(0);
    const int64_t vocab_size = freq_tensor.size(1);

    const auto run_range = [&](int64_t start, int64_t end) {
      if (freq_tensor.scalar_type() == at::kLong) {
        const int64_t *ptr = freq_tensor.data_ptr<int64_t>();
        for (int64_t row = start; row < end; ++row) {
          decoded[static_cast<size_t>(row)] = decoders_[static_cast<size_t>(row)].decode_frequency_row_ptr(
              ptr + row * vocab_size, vocab_size);
        }
      } else if (freq_tensor.scalar_type() == at::kInt) {
        const int32_t *ptr = freq_tensor.data_ptr<int32_t>();
        for (int64_t row = start; row < end; ++row) {
          decoded[static_cast<size_t>(row)] = decoders_[static_cast<size_t>(row)].decode_frequency_row_ptr(
              ptr + row * vocab_size, vocab_size);
        }
      } else if (freq_tensor.scalar_type() == at::kShort) {
        const int16_t *ptr = freq_tensor.data_ptr<int16_t>();
        for (int64_t row = start; row < end; ++row) {
          decoded[static_cast<size_t>(row)] = decoders_[static_cast<size_t>(row)].decode_frequency_row_ptr(
              ptr + row * vocab_size, vocab_size);
        }
      } else if (freq_tensor.scalar_type() == at::kUInt16) {
        const uint16_t *ptr = freq_tensor.data_ptr<uint16_t>();
        for (int64_t row = start; row < end; ++row) {
          decoded[static_cast<size_t>(row)] = decoders_[static_cast<size_t>(row)].decode_frequency_row_ptr(
              ptr + row * vocab_size, vocab_size);
        }
      } else if (freq_tensor.scalar_type() == at::kByte) {
        const uint8_t *ptr = freq_tensor.data_ptr<uint8_t>();
        for (int64_t row = start; row < end; ++row) {
          decoded[static_cast<size_t>(row)] = decoders_[static_cast<size_t>(row)].decode_frequency_row_ptr(
              ptr + row * vocab_size, vocab_size);
        }
      } else {
        throw std::runtime_error("frequency rows must be int64, int32, int16, uint16, or uint8");
      }
    };

    const int64_t requested_threads = thread_count_ > 0
        ? thread_count_
        : static_cast<int64_t>(std::min(8U, std::max(1U, std::thread::hardware_concurrency())));
    const int64_t worker_count = std::max<int64_t>(1, std::min<int64_t>(requested_threads, rows));
    if (worker_count == 1) {
      run_range(0, rows);
    } else {
      std::vector<std::thread> workers;
      workers.reserve(static_cast<size_t>(worker_count));
      for (int64_t worker = 0; worker < worker_count; ++worker) {
        const int64_t start = rows * worker / worker_count;
        const int64_t end = rows * (worker + 1) / worker_count;
        workers.emplace_back(run_range, start, end);
      }
      for (auto &worker : workers) {
        worker.join();
      }
    }

    return torch::from_blob(decoded.data(), {rows}, torch::TensorOptions().dtype(torch::kInt64)).clone();
  }

  at::Tensor decode_frequency_rows_with_totals(const at::Tensor &frequencies, const at::Tensor &totals) {
    if (frequencies.device().is_cuda() || totals.device().is_cuda() || frequencies.dim() != 2 || totals.dim() != 1) {
      throw std::runtime_error("frequency rows must be a 2D CPU tensor and totals must be a 1D CPU tensor");
    }
    if (frequencies.size(0) != static_cast<int64_t>(decoders_.size()) || totals.size(0) != frequencies.size(0)) {
      throw std::runtime_error("frequency row count and total count must match batched decoder stream count");
    }
    at::Tensor freq_tensor = frequencies.contiguous();
    at::Tensor total_tensor = totals.contiguous();
    std::vector<int64_t> decoded(decoders_.size());
    const int64_t rows = freq_tensor.size(0);
    const int64_t vocab_size = freq_tensor.size(1);

    const auto total_at = [&](int64_t row) -> uint32_t {
      int64_t value = 0;
      if (total_tensor.scalar_type() == at::kLong) {
        value = total_tensor.data_ptr<int64_t>()[row];
      } else if (total_tensor.scalar_type() == at::kInt) {
        value = total_tensor.data_ptr<int32_t>()[row];
      } else {
        throw std::runtime_error("totals must be int64 or int32");
      }
      if (value <= 0 || value > static_cast<int64_t>((uint64_t{1} << 30) + 2)) {
        throw std::runtime_error("frequency row total is outside the supported 32-bit arithmetic range");
      }
      return static_cast<uint32_t>(value);
    };

    const auto run_range = [&](int64_t start, int64_t end) {
      if (freq_tensor.scalar_type() == at::kLong) {
        const int64_t *ptr = freq_tensor.data_ptr<int64_t>();
        for (int64_t row = start; row < end; ++row) {
          decoded[static_cast<size_t>(row)] = decoders_[static_cast<size_t>(row)].decode_frequency_row_ptr_with_total(
              ptr + row * vocab_size, vocab_size, total_at(row));
        }
      } else if (freq_tensor.scalar_type() == at::kInt) {
        const int32_t *ptr = freq_tensor.data_ptr<int32_t>();
        for (int64_t row = start; row < end; ++row) {
          decoded[static_cast<size_t>(row)] = decoders_[static_cast<size_t>(row)].decode_frequency_row_ptr_with_total(
              ptr + row * vocab_size, vocab_size, total_at(row));
        }
      } else if (freq_tensor.scalar_type() == at::kUInt16) {
        const uint16_t *ptr = freq_tensor.data_ptr<uint16_t>();
        for (int64_t row = start; row < end; ++row) {
          decoded[static_cast<size_t>(row)] = decoders_[static_cast<size_t>(row)].decode_frequency_row_ptr_with_total(
              ptr + row * vocab_size, vocab_size, total_at(row));
        }
      } else {
        throw std::runtime_error("frequency rows must be int64, int32, or uint16 when totals are provided");
      }
    };

    const int64_t requested_threads = thread_count_ > 0
        ? thread_count_
        : static_cast<int64_t>(std::min(8U, std::max(1U, std::thread::hardware_concurrency())));
    const int64_t worker_count = std::max<int64_t>(1, std::min<int64_t>(requested_threads, rows));
    if (worker_count == 1) {
      run_range(0, rows);
    } else {
      std::vector<std::thread> workers;
      workers.reserve(static_cast<size_t>(worker_count));
      for (int64_t worker = 0; worker < worker_count; ++worker) {
        const int64_t start = rows * worker / worker_count;
        const int64_t end = rows * (worker + 1) / worker_count;
        workers.emplace_back(run_range, start, end);
      }
      for (auto &worker : workers) {
        worker.join();
      }
    }

    return torch::from_blob(decoded.data(), {rows}, torch::TensorOptions().dtype(torch::kInt64)).clone();
  }

  at::Tensor decode_frequency_rows_with_totals_and_active(
      const at::Tensor &frequencies,
      const at::Tensor &totals,
      const at::Tensor &active) {
    if (frequencies.device().is_cuda() || totals.device().is_cuda() || active.device().is_cuda() ||
        frequencies.dim() != 2 || totals.dim() != 1 || active.dim() != 1) {
      throw std::runtime_error("frequency rows must be [rows, vocab], totals [rows], active [rows] on CPU");
    }
    if (frequencies.size(0) != static_cast<int64_t>(decoders_.size()) ||
        totals.size(0) != frequencies.size(0) ||
        active.size(0) != frequencies.size(0)) {
      throw std::runtime_error("frequency, total, active row counts must match batched decoder stream count");
    }
    at::Tensor active_tensor = active.contiguous();
    std::vector<int64_t> active_rows;
    active_rows.reserve(static_cast<size_t>(active_tensor.size(0)));
    for (int64_t row = 0; row < active_tensor.size(0); ++row) {
      bool is_active = false;
      if (active_tensor.scalar_type() == at::kBool) {
        is_active = active_tensor.data_ptr<bool>()[row];
      } else if (active_tensor.scalar_type() == at::kByte) {
        is_active = active_tensor.data_ptr<uint8_t>()[row] != 0;
      } else if (active_tensor.scalar_type() == at::kLong) {
        is_active = active_tensor.data_ptr<int64_t>()[row] != 0;
      } else if (active_tensor.scalar_type() == at::kInt) {
        is_active = active_tensor.data_ptr<int32_t>()[row] != 0;
      } else {
        throw std::runtime_error("active mask must be bool, uint8, int32, or int64");
      }
      if (is_active) {
        active_rows.push_back(row);
      }
    }
    if (active_rows.empty()) {
      return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64));
    }

    at::Tensor freq_tensor = frequencies.contiguous();
    at::Tensor total_tensor = totals.contiguous();
    std::vector<int64_t> decoded(active_rows.size());
    const int64_t vocab_size = freq_tensor.size(1);

    const auto total_at = [&](int64_t row) -> uint32_t {
      int64_t value = 0;
      if (total_tensor.scalar_type() == at::kLong) {
        value = total_tensor.data_ptr<int64_t>()[row];
      } else if (total_tensor.scalar_type() == at::kInt) {
        value = total_tensor.data_ptr<int32_t>()[row];
      } else {
        throw std::runtime_error("totals must be int64 or int32");
      }
      if (value <= 0 || value > static_cast<int64_t>((uint64_t{1} << 30) + 2)) {
        throw std::runtime_error("frequency row total is outside the supported 32-bit arithmetic range");
      }
      return static_cast<uint32_t>(value);
    };

    const auto run_range = [&](int64_t start, int64_t end) {
      if (freq_tensor.scalar_type() == at::kLong) {
        const int64_t *ptr = freq_tensor.data_ptr<int64_t>();
        for (int64_t index = start; index < end; ++index) {
          const int64_t row = active_rows[static_cast<size_t>(index)];
          decoded[static_cast<size_t>(index)] = decoders_[static_cast<size_t>(row)].decode_frequency_row_ptr_with_total(
              ptr + row * vocab_size, vocab_size, total_at(row));
        }
      } else if (freq_tensor.scalar_type() == at::kInt) {
        const int32_t *ptr = freq_tensor.data_ptr<int32_t>();
        for (int64_t index = start; index < end; ++index) {
          const int64_t row = active_rows[static_cast<size_t>(index)];
          decoded[static_cast<size_t>(index)] = decoders_[static_cast<size_t>(row)].decode_frequency_row_ptr_with_total(
              ptr + row * vocab_size, vocab_size, total_at(row));
        }
      } else if (freq_tensor.scalar_type() == at::kUInt16) {
        const uint16_t *ptr = freq_tensor.data_ptr<uint16_t>();
        for (int64_t index = start; index < end; ++index) {
          const int64_t row = active_rows[static_cast<size_t>(index)];
          decoded[static_cast<size_t>(index)] = decoders_[static_cast<size_t>(row)].decode_frequency_row_ptr_with_total(
              ptr + row * vocab_size, vocab_size, total_at(row));
        }
      } else {
        throw std::runtime_error("frequency rows must be int64, int32, or uint16 when totals are provided");
      }
    };

    const int64_t rows = static_cast<int64_t>(active_rows.size());
    const int64_t requested_threads = thread_count_ > 0
        ? thread_count_
        : static_cast<int64_t>(std::min(8U, std::max(1U, std::thread::hardware_concurrency())));
    const int64_t worker_count = std::max<int64_t>(1, std::min<int64_t>(requested_threads, rows));
    if (worker_count == 1) {
      run_range(0, rows);
    } else {
      std::vector<std::thread> workers;
      workers.reserve(static_cast<size_t>(worker_count));
      for (int64_t worker = 0; worker < worker_count; ++worker) {
        const int64_t start = rows * worker / worker_count;
        const int64_t end = rows * (worker + 1) / worker_count;
        workers.emplace_back(run_range, start, end);
      }
      for (auto &worker : workers) {
        worker.join();
      }
    }

    return torch::from_blob(decoded.data(), {rows}, torch::TensorOptions().dtype(torch::kInt64)).clone();
  }

  int64_t size() const {
    return static_cast<int64_t>(decoders_.size());
  }

 private:
  std::vector<FastArithmeticDecoder> decoders_;
  int64_t thread_count_ = 0;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<FastArithmeticEncoder>(m, "FastArithmeticEncoder")
      .def(py::init<>())
      .def("encode_probability_rows", &FastArithmeticEncoder::encode_probability_rows)
      .def("encode_probability_rows_fast_floor", &FastArithmeticEncoder::encode_probability_rows_fast_floor)
      .def("encode_intervals", &FastArithmeticEncoder::encode_intervals)
      .def("encode_grouped_steps", &FastArithmeticEncoder::encode_grouped_steps)
      .def("finish", &FastArithmeticEncoder::finish);

  py::class_<BatchedFastArithmeticEncoder>(m, "BatchedFastArithmeticEncoder")
      .def(py::init<int64_t>(), py::arg("stream_count"))
      .def("encode_interval_matrix", &BatchedFastArithmeticEncoder::encode_interval_matrix)
      .def("encode_interval_matrix_with_lengths", &BatchedFastArithmeticEncoder::encode_interval_matrix_with_lengths)
      .def("finish", &BatchedFastArithmeticEncoder::finish)
      .def("size", &BatchedFastArithmeticEncoder::size);

  py::class_<FastArithmeticDecoder>(m, "FastArithmeticDecoder")
      .def(py::init<const py::bytes &>())
      .def("decode_probability_rows", &FastArithmeticDecoder::decode_probability_rows)
      .def("decode_probability_rows_fast_floor", &FastArithmeticDecoder::decode_probability_rows_fast_floor)
      .def("decode_frequency_row", &FastArithmeticDecoder::decode_frequency_row);

  py::class_<BatchedFastArithmeticDecoder>(m, "BatchedFastArithmeticDecoder")
      .def(py::init<const std::vector<py::bytes> &, int64_t>(), py::arg("encoded_streams"), py::arg("threads") = 0)
      .def("decode_frequency_rows", &BatchedFastArithmeticDecoder::decode_frequency_rows)
      .def("decode_frequency_rows_with_totals", &BatchedFastArithmeticDecoder::decode_frequency_rows_with_totals)
      .def("decode_frequency_rows_with_totals_and_active", &BatchedFastArithmeticDecoder::decode_frequency_rows_with_totals_and_active)
      .def("size", &BatchedFastArithmeticDecoder::size);
}
