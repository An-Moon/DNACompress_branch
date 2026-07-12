#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace py = pybind11;

namespace {

constexpr uint64_t CONTEXT_HASH_MULTIPLIER = 1315423911ULL;
constexpr uint64_t CONTEXT_HASH_SYMBOL_SALT = 2654435761ULL;
constexpr uint64_t CONTEXT_HASH_MASK = (uint64_t{1} << 63) - 1;
constexpr int64_t DENSE_CONTEXT_MAX_ENTRIES = 1000000;
constexpr uint16_t COUNTER_MAX = std::numeric_limits<uint16_t>::max();
constexpr int64_t DEFAULT_HASH_BUCKETS = 0;
constexpr int64_t DEFAULT_HASH_SLOTS = 4;
constexpr int64_t NC_PREFIX_PRESET_GECO2_STRICT = 2;
constexpr int64_t GECO2_HASH_TABLE_BEGIN_CTX = 15;
constexpr int64_t GECO2_HASH_SIZE = 33554471;
constexpr uint16_t GECO2_HASH_COUNTER_MAX = 15;
constexpr uint16_t GECO2_MX_PMODEL = 65535;
constexpr bool COMPUTE_MIXED_FLOAT_BITS = false;
constexpr int64_t DEFAULT_PIPELINE_BLOCK_WINDOWS = 64;
constexpr size_t LARGE_TABLE_ALIGNMENT_BYTES = 2ULL << 20;
constexpr uint32_t FUSED_STATE_BITS = 32;
constexpr uint64_t FUSED_FULL_RANGE = uint64_t{1} << FUSED_STATE_BITS;
constexpr uint64_t FUSED_HALF_RANGE = FUSED_FULL_RANGE >> 1;
constexpr uint64_t FUSED_QUARTER_RANGE = FUSED_HALF_RANGE >> 1;
constexpr uint64_t FUSED_MASK = FUSED_FULL_RANGE - 1;

using Clock = std::chrono::steady_clock;

bool env_flag_enabled(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return false;
  }
  const std::string text(value);
  return text != "0" && text != "false" && text != "FALSE" && text != "no" && text != "NO" &&
      text != "off" && text != "OFF";
}

template <typename T>
class LargeZeroArray {
 public:
  LargeZeroArray() = default;
  explicit LargeZeroArray(size_t count) { allocate(count); }
  LargeZeroArray(const LargeZeroArray &) = delete;
  LargeZeroArray &operator=(const LargeZeroArray &) = delete;

  LargeZeroArray(LargeZeroArray &&other) noexcept { move_from(std::move(other)); }
  LargeZeroArray &operator=(LargeZeroArray &&other) noexcept {
    if (this != &other) {
      release();
      move_from(std::move(other));
    }
    return *this;
  }

  ~LargeZeroArray() { release(); }

  void allocate(size_t count) {
    release();
    size_ = count;
    if (count == 0) {
      return;
    }
    const size_t bytes = count * sizeof(T);
    requested_bytes_ = bytes;
#ifdef __linux__
    const size_t alignment = LARGE_TABLE_ALIGNMENT_BYTES;
    size_t page_size = 4096;
    const long detected_page_size = sysconf(_SC_PAGESIZE);
    if (detected_page_size > 0) {
      page_size = static_cast<size_t>(detected_page_size);
    }
    const size_t allocation_bytes = round_up(bytes, page_size);
    const size_t map_bytes = allocation_bytes + alignment;
    void *mapped = mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapped != MAP_FAILED) {
      const uintptr_t mapped_addr = reinterpret_cast<uintptr_t>(mapped);
      const uintptr_t aligned_addr = (mapped_addr + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
      const size_t prefix_bytes = static_cast<size_t>(aligned_addr - mapped_addr);
      const size_t suffix_bytes = map_bytes - prefix_bytes - allocation_bytes;
      if (prefix_bytes > 0) {
        munmap(mapped, prefix_bytes);
      }
      if (suffix_bytes > 0) {
        void *suffix = reinterpret_cast<void *>(aligned_addr + allocation_bytes);
        munmap(suffix, suffix_bytes);
      }
      void *aligned = reinterpret_cast<void *>(aligned_addr);
      data_ = static_cast<T *>(aligned);
      mapped_bytes_ = allocation_bytes;
      mapped_ = true;
      mapped_alignment_bytes_ = alignment;
      mapped_2mb_aligned_ = aligned_addr % alignment == 0;
      hugepage_advised_ = madvise(aligned, allocation_bytes, MADV_HUGEPAGE) == 0;
      populate_requested_ = env_flag_enabled("DNA_COMPRESS_NC_PREFIX_POPULATE_TABLES");
      if (populate_requested_) {
#ifdef MADV_POPULATE_WRITE
        if (madvise(aligned, allocation_bytes, MADV_POPULATE_WRITE) == 0) {
          populate_succeeded_ = true;
          populate_mode_ = "madvise_populate_write";
        } else
#endif
        {
          touch_pages(aligned, allocation_bytes);
          populate_succeeded_ = true;
          populate_mode_ = "manual_page_touch";
        }
      }
      return;
    }
#endif
    fallback_.assign(count, T{});
    data_ = fallback_.data();
  }

  T *data() { return data_; }
  const T *data() const { return data_; }
  T &operator[](size_t index) { return data_[index]; }
  const T &operator[](size_t index) const { return data_[index]; }
  size_t size() const { return size_; }
  bool mapped() const { return mapped_; }
  bool hugepage_advised() const { return hugepage_advised_; }
  size_t requested_bytes() const { return requested_bytes_; }
  size_t mapped_bytes() const { return mapped_bytes_; }
  size_t mapped_alignment_bytes() const { return mapped_alignment_bytes_; }
  bool mapped_2mb_aligned() const { return mapped_2mb_aligned_; }
  bool populate_requested() const { return populate_requested_; }
  bool populate_succeeded() const { return populate_succeeded_; }
  const std::string &populate_mode() const { return populate_mode_; }

 private:
  T *data_ = nullptr;
  size_t size_ = 0;
  size_t requested_bytes_ = 0;
  size_t mapped_bytes_ = 0;
  size_t mapped_alignment_bytes_ = 0;
  bool mapped_ = false;
  bool hugepage_advised_ = false;
  bool mapped_2mb_aligned_ = false;
  bool populate_requested_ = false;
  bool populate_succeeded_ = false;
  std::string populate_mode_ = "none";
  std::vector<T> fallback_;

  static void touch_pages(void *mapped, size_t bytes) {
    if (bytes == 0) {
      return;
    }
    size_t page_size = 4096;
#ifdef __linux__
    const long detected = sysconf(_SC_PAGESIZE);
    if (detected > 0) {
      page_size = static_cast<size_t>(detected);
    }
#endif
    volatile char *cursor = static_cast<volatile char *>(mapped);
    for (size_t offset = 0; offset < bytes; offset += page_size) {
      cursor[offset] = 0;
    }
    cursor[bytes - 1] = 0;
  }

  static size_t round_up(size_t value, size_t alignment) {
    if (alignment == 0) {
      return value;
    }
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
  }

  void release() {
#ifdef __linux__
    if (mapped_) {
      munmap(data_, mapped_bytes_);
    }
#endif
    data_ = nullptr;
    size_ = 0;
    requested_bytes_ = 0;
    mapped_bytes_ = 0;
    mapped_alignment_bytes_ = 0;
    mapped_ = false;
    hugepage_advised_ = false;
    mapped_2mb_aligned_ = false;
    populate_requested_ = false;
    populate_succeeded_ = false;
    populate_mode_ = "none";
    fallback_.clear();
  }

  void move_from(LargeZeroArray &&other) {
    data_ = other.data_;
    size_ = other.size_;
    requested_bytes_ = other.requested_bytes_;
    mapped_bytes_ = other.mapped_bytes_;
    mapped_alignment_bytes_ = other.mapped_alignment_bytes_;
    mapped_ = other.mapped_;
    hugepage_advised_ = other.hugepage_advised_;
    mapped_2mb_aligned_ = other.mapped_2mb_aligned_;
    populate_requested_ = other.populate_requested_;
    populate_succeeded_ = other.populate_succeeded_;
    populate_mode_ = std::move(other.populate_mode_);
    fallback_ = std::move(other.fallback_);
    if (!mapped_ && !fallback_.empty()) {
      data_ = fallback_.data();
    }
    other.data_ = nullptr;
    other.size_ = 0;
    other.requested_bytes_ = 0;
    other.mapped_bytes_ = 0;
    other.mapped_alignment_bytes_ = 0;
    other.mapped_ = false;
    other.hugepage_advised_ = false;
    other.mapped_2mb_aligned_ = false;
    other.populate_requested_ = false;
    other.populate_succeeded_ = false;
    other.populate_mode_ = "none";
  }
};

double elapsed_seconds(const Clock::time_point &start, const Clock::time_point &end) {
  return std::chrono::duration<double>(end - start).count();
}

void add_elapsed(double &accumulator, const Clock::time_point &start) {
  accumulator += elapsed_seconds(start, Clock::now());
}

Clock::time_point stage_start(bool enabled) {
  return enabled ? Clock::now() : Clock::time_point{};
}

void add_stage_elapsed(bool enabled, double &accumulator, const Clock::time_point &start) {
  if (enabled) {
    add_elapsed(accumulator, start);
  }
}

uint64_t mix_context_key(uint64_t previous_key, int64_t symbol) {
  return ((previous_key * CONTEXT_HASH_MULTIPLIER) ^ (static_cast<uint64_t>(symbol + 1) * CONTEXT_HASH_SYMBOL_SALT)) &
      CONTEXT_HASH_MASK;
}

uint64_t zhash(uint64_t value) {
  value = (~value) + (value << 21);
  value = value ^ (value >> 24);
  value = (value + (value << 3)) + (value << 8);
  value = value ^ (value >> 14);
  value = (value + (value << 2)) + (value << 4);
  value = value ^ (value >> 28);
  value = value + (value << 31);
  return value;
}

struct FusedInterval {
  uint32_t low = 0;
  uint32_t high = 1;
  uint32_t total = 1;
};

class FusedBitWriter {
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

class FusedArithmeticEncoder {
 public:
  void update(const FusedInterval &interval) {
    const uint64_t low_count = interval.low;
    const uint64_t high_count = interval.high;
    const uint64_t total = interval.total;
    if (!(low_count < high_count && high_count <= total)) {
      throw std::runtime_error("invalid fused arithmetic interval");
    }

    const uint64_t range = high_ - low_ + 1;
    high_ = low_ + (range * high_count / total) - 1;
    low_ = low_ + (range * low_count / total);

    while (true) {
      if (high_ < FUSED_HALF_RANGE) {
        write_bit_and_follow(0);
      } else if (low_ >= FUSED_HALF_RANGE) {
        write_bit_and_follow(1);
        low_ -= FUSED_HALF_RANGE;
        high_ -= FUSED_HALF_RANGE;
      } else if (low_ >= FUSED_QUARTER_RANGE && high_ < 3 * FUSED_QUARTER_RANGE) {
        ++pending_bits_;
        low_ -= FUSED_QUARTER_RANGE;
        high_ -= FUSED_QUARTER_RANGE;
      } else {
        break;
      }
      low_ = (low_ << 1) & FUSED_MASK;
      high_ = ((high_ << 1) & FUSED_MASK) | 1;
    }
  }

  std::string finish() {
    ++pending_bits_;
    if (low_ < FUSED_QUARTER_RANGE) {
      write_bit_and_follow(0);
    } else {
      write_bit_and_follow(1);
    }
    return writer_.finish();
  }

 private:
  uint64_t low_ = 0;
  uint64_t high_ = FUSED_MASK;
  uint64_t pending_bits_ = 0;
  FusedBitWriter writer_;

  void write_bit_and_follow(int bit) {
    writer_.write(bit);
    const int follow = bit ^ 1;
    while (pending_bits_ > 0) {
      writer_.write(follow);
      --pending_bits_;
    }
  }
};

struct DenseTable {
  bool enabled = false;
  int64_t entries = 0;
  int64_t vocab_size = 0;
  LargeZeroArray<uint16_t> counts;

  DenseTable() = default;

  DenseTable(int64_t entry_count, int64_t vocab) : enabled(true), entries(entry_count), vocab_size(vocab) {
    counts.allocate(static_cast<size_t>(entries * vocab_size));
  }

  uint16_t *row(int64_t key) {
    return counts.data() + static_cast<size_t>(key * vocab_size);
  }

  const uint16_t *row(int64_t key) const {
    return counts.data() + static_cast<size_t>(key * vocab_size);
  }
};

uint8_t complement_symbol(uint8_t symbol) {
  return static_cast<uint8_t>(3 - symbol);
}

struct alignas(64) StrictHashBucket {
  std::array<uint32_t, 8> keys{};
  std::array<uint16_t, 8> packed_counts{};
  uint8_t next_slot = 0;
  std::array<uint8_t, 15> padding{};
};

static_assert(sizeof(StrictHashBucket) == 64, "strict hash bucket must fit one cache line");

struct StrictHashTable {
  int64_t bucket_count = 0;
  int64_t slots = 0;
  LargeZeroArray<StrictHashBucket> buckets;

  StrictHashTable() = default;

  StrictHashTable(int64_t buckets, int64_t slot_count)
      : bucket_count(std::max<int64_t>(1, buckets)),
        slots(std::max<int64_t>(1, slot_count)) {
    if (slots > 8) {
      throw std::runtime_error("strict hash supports at most 8 slots per bucket");
    }
    this->buckets.allocate(static_cast<size_t>(bucket_count));
  }

  int64_t bucket_for_hashed(uint64_t hashed_key) const {
    return static_cast<int64_t>(hashed_key % static_cast<uint64_t>(bucket_count));
  }

  int64_t find_slot_in_bucket(const StrictHashBucket &bucket, uint32_t low_key) const {
    for (int64_t slot = 0; slot < slots; ++slot) {
      if (bucket.keys[static_cast<size_t>(slot)] == low_key) {
        return slot;
      }
    }
    return -1;
  }

  void prefetch_hashed(uint64_t hashed, bool for_write = false) const {
    const int64_t bucket_index = bucket_for_hashed(hashed);
    __builtin_prefetch(buckets.data() + bucket_index, for_write ? 1 : 0, 1);
  }

  uint32_t freqs_hashed(uint64_t hashed, uint32_t alpha_den, std::array<uint32_t, 4> &out) const {
    const int64_t bucket_index = bucket_for_hashed(hashed);
    const StrictHashBucket &bucket = buckets[static_cast<size_t>(bucket_index)];
    const int64_t slot = find_slot_in_bucket(bucket, static_cast<uint32_t>(hashed & 0xffffffffULL));
    if (slot < 0) {
      out = {1, 1, 1, 1};
      return 4;
    }
    const uint16_t packed = bucket.packed_counts[static_cast<size_t>(slot)];
    uint32_t sum = 0;
    for (int64_t symbol = 0; symbol < 4; ++symbol) {
      out[static_cast<size_t>(symbol)] =
          1U + alpha_den * static_cast<uint32_t>((packed >> (symbol << 2)) & 0x0fU);
      sum += out[static_cast<size_t>(symbol)];
    }
    return sum;
  }

  uint32_t freqs(uint64_t raw_key, uint32_t alpha_den, std::array<uint32_t, 4> &out) const {
    return freqs_hashed(zhash(raw_key), alpha_den, out);
  }

  void update_hashed(uint64_t hashed, uint8_t symbol) {
    const int64_t bucket_index = bucket_for_hashed(hashed);
    const uint32_t low_key = static_cast<uint32_t>(hashed & 0xffffffffULL);
    StrictHashBucket &bucket = buckets[static_cast<size_t>(bucket_index)];
    for (int64_t slot = 0; slot < slots; ++slot) {
      if (bucket.keys[static_cast<size_t>(slot)] == low_key) {
        uint16_t packed = bucket.packed_counts[static_cast<size_t>(slot)];
        uint16_t sc = static_cast<uint16_t>((packed >> (symbol << 2)) & 0x0fU);
        if (sc == GECO2_HASH_COUNTER_MAX) {
          uint16_t decayed = 0;
          for (int64_t s = 0; s < 4; ++s) {
            const uint16_t counter = static_cast<uint16_t>(((packed >> (s << 2)) & 0x0fU) >> 1);
            decayed |= static_cast<uint16_t>(counter << (s << 2));
          }
          packed = decayed;
          sc = static_cast<uint16_t>((packed >> (symbol << 2)) & 0x0fU);
        }
        ++sc;
        packed &= static_cast<uint16_t>(~(0x0fU << (symbol << 2)));
        packed |= static_cast<uint16_t>(sc << (symbol << 2));
        bucket.packed_counts[static_cast<size_t>(slot)] = packed;
        return;
      }
    }

    uint8_t &cursor = bucket.next_slot;
    ++cursor;
    if (cursor == slots) {
      cursor = 0;
    }
    bucket.keys[static_cast<size_t>(cursor)] = low_key;
    bucket.packed_counts[static_cast<size_t>(cursor)] = static_cast<uint16_t>(1U << (symbol << 2));
  }

  void update(uint64_t raw_key, uint8_t symbol) {
    update_hashed(zhash(raw_key), symbol);
  }
};

struct StrictModel {
  int64_t context_len = 0;
  uint32_t alpha_den = 1;
  double gamma = 0.9;
  int64_t hash_slots = 1;
  bool ir = false;
  uint32_t edits = 0;
  uint32_t edit_alpha_den = 0;
  double edit_gamma = 0.0;
  std::string label;
  uint64_t n_contexts = 1;
  uint64_t multiplier = 1;
  DenseTable dense;
  StrictHashTable hash;
  std::vector<uint64_t> pidx;
  std::vector<uint64_t> pidx_ir;
  std::vector<uint64_t> edit_pidx;
  std::vector<uint32_t> edit_masks;
  std::vector<uint8_t> edit_in;
  std::vector<uint8_t> edit_symbols;
  std::vector<int8_t> edit_best_ids;
  int64_t queries = 0;
  int64_t hits = 0;
  int64_t ir_updates = 0;
  int64_t edit_hits = 0;
  int64_t edit_fails = 0;
};

struct StrictPredictor {
  int64_t model_index = 0;
  bool edit = false;
  std::string label;
  uint32_t alpha_den = 1;
  double gamma = 0.9;
  int64_t queries = 0;
  int64_t hits = 0;
};

uint64_t pow4_i64(int64_t context_len) {
  if (context_len >= 32) {
    return std::numeric_limits<uint64_t>::max();
  }
  return uint64_t{1} << (2 * context_len);
}

std::vector<StrictModel> make_strict_level10_models(
    int64_t window_count,
    int64_t hash_bucket_count,
    bool disable_edit_experts,
    bool disable_ir,
    std::vector<StrictPredictor> &predictors) {
  struct Spec {
    int64_t ctx;
    uint32_t alpha_den;
    bool ir;
    int64_t hash_slots;
    double gamma;
    uint32_t edits;
    uint32_t edit_alpha_den;
    double edit_gamma;
  };
  const std::vector<Spec> specs = {
      {1, 1, false, 0, 0.90, 0, 0, 0},
      {3, 1, false, 0, 0.90, 0, 0, 0},
      {6, 1, true, 0, 0.82, 0, 0, 0},
      {9, 10, false, 0, 0.90, 0, 0, 0},
      {11, 10, false, 0, 0.90, 0, 0, 0},
      {13, 10, true, 0, 0.90, 0, 20, 0.94},
      {17, 100, true, 8, 0.89, 5, 10, 0.90},
  };

  std::vector<StrictModel> models;
  predictors.clear();
  for (const Spec &spec : specs) {
    StrictModel model;
    model.context_len = spec.ctx;
    model.alpha_den = spec.alpha_den;
    model.gamma = spec.gamma;
    model.hash_slots = std::max<int64_t>(1, spec.hash_slots);
    model.ir = spec.ir && !disable_ir;
    model.edits = disable_edit_experts ? 0 : spec.edits;
    model.edit_alpha_den = spec.edit_alpha_den;
    model.edit_gamma = spec.edit_gamma;
    model.label = "ctx" + std::to_string(spec.ctx);
    model.n_contexts = pow4_i64(spec.ctx);
    model.multiplier = spec.ctx <= 0 ? 1 : pow4_i64(spec.ctx - 1);
    if (spec.ctx < GECO2_HASH_TABLE_BEGIN_CTX) {
      model.dense = DenseTable(static_cast<int64_t>(model.n_contexts), 4);
    } else {
      model.hash = StrictHashTable(hash_bucket_count, model.hash_slots);
    }
    model.pidx.assign(static_cast<size_t>(window_count), 0);
    if (model.ir) {
      model.pidx_ir.assign(static_cast<size_t>(window_count), model.n_contexts - 1);
    }
    if (model.edits > 0) {
      model.edit_pidx.assign(static_cast<size_t>(window_count), 0);
      model.edit_masks.assign(static_cast<size_t>(window_count), 0);
      model.edit_in.assign(static_cast<size_t>(window_count), 0);
    }
    const int64_t model_index = static_cast<int64_t>(models.size());
    models.push_back(std::move(model));

    StrictPredictor base;
    base.model_index = model_index;
    base.edit = false;
    base.label = "ctx" + std::to_string(spec.ctx);
    base.alpha_den = spec.alpha_den;
    base.gamma = spec.gamma;
    predictors.push_back(base);

    if (!disable_edit_experts && spec.edits > 0) {
      StrictPredictor edit;
      edit.model_index = model_index;
      edit.edit = true;
      edit.label = "ctx" + std::to_string(spec.ctx) + "_edit" + std::to_string(spec.edits);
      edit.alpha_den = spec.edit_alpha_den;
      edit.gamma = spec.edit_gamma;
      predictors.push_back(edit);
    }
  }
  return models;
}

uint32_t strict_model_freqs(
    const StrictModel &model,
    uint64_t key,
    uint32_t alpha_den,
    std::array<uint32_t, 4> &freqs) {
  if (model.dense.enabled) {
    const uint16_t *counts = model.dense.row(static_cast<int64_t>(key));
    uint32_t sum = 0;
    for (int64_t symbol = 0; symbol < 4; ++symbol) {
      freqs[static_cast<size_t>(symbol)] = 1U + alpha_den * static_cast<uint32_t>(counts[symbol]);
      sum += freqs[static_cast<size_t>(symbol)];
    }
    return sum;
  }
  return model.hash.freqs(key, alpha_den, freqs);
}

void strict_model_update(StrictModel &model, uint64_t key, uint8_t symbol) {
  if (model.dense.enabled) {
    uint16_t *counts = model.dense.row(static_cast<int64_t>(key));
    if (counts[symbol] == COUNTER_MAX - 1) {
      ++counts[symbol];
      for (int64_t s = 0; s < 4; ++s) {
        counts[s] >>= 1;
      }
    } else {
      ++counts[symbol];
    }
    return;
  }
  model.hash.update(key, symbol);
}

int64_t configured_pipeline_block_windows() {
  const char *value = std::getenv("DNA_COMPRESS_NC_PREFIX_PIPELINE_BLOCK_WINDOWS");
  if (value == nullptr || *value == '\0') {
    return DEFAULT_PIPELINE_BLOCK_WINDOWS;
  }
  const long parsed = std::strtol(value, nullptr, 10);
  return std::max<int64_t>(16, std::min<int64_t>(parsed, 4096));
}

struct PipelineQueryAddress {
  uint64_t key = 0;
  uint64_t hashed = 0;
};

void pipeline_prefetch(const StrictModel &model, uint64_t key, uint64_t hashed, bool for_write = false) {
  if (!model.dense.enabled) {
    model.hash.prefetch_hashed(hashed, for_write);
    return;
  }
  if (model.context_len == 11) {
    __builtin_prefetch(model.dense.row(static_cast<int64_t>(key)), for_write ? 1 : 0, 2);
  } else {
    __builtin_prefetch(model.dense.row(static_cast<int64_t>(key)), for_write ? 1 : 0, 1);
  }
}

uint32_t strict_model_freqs_planned(
    const StrictModel &model,
    uint64_t key,
    uint64_t hashed,
    uint32_t alpha_den,
    std::array<uint32_t, 4> &freqs) {
  if (model.dense.enabled) {
    return strict_model_freqs(model, key, alpha_den, freqs);
  }
  return model.hash.freqs_hashed(hashed, alpha_den, freqs);
}

int64_t strict_best_id(const std::array<uint32_t, 4> &freqs, uint32_t sum) {
  if (sum == 4) {
    return -2;
  }
  uint32_t max_freq = freqs[0];
  int64_t best = 0;
  for (int64_t symbol = 1; symbol < 4; ++symbol) {
    if (freqs[static_cast<size_t>(symbol)] > max_freq) {
      max_freq = freqs[static_cast<size_t>(symbol)];
      best = symbol;
    }
  }
  for (int64_t symbol = 0; symbol < 4; ++symbol) {
    if (symbol != best && freqs[static_cast<size_t>(symbol)] == max_freq) {
      return -1;
    }
  }
  return best;
}

uint32_t strict_popcount_mask(uint32_t mask, int64_t context_len) {
  const uint32_t limit_mask = context_len >= 32
      ? std::numeric_limits<uint32_t>::max()
      : ((uint32_t{1} << context_len) - 1U);
  return static_cast<uint32_t>(__builtin_popcount(mask & limit_mask));
}

uint8_t strict_correct_edit_symbol(
    StrictModel &model,
    int64_t window_id,
    int64_t best,
    uint8_t target) {
  if (model.edits == 0) {
    return target;
  }
  uint8_t edited_symbol = target;
  uint8_t &in = model.edit_in[static_cast<size_t>(window_id)];
  uint32_t &mask = model.edit_masks[static_cast<size_t>(window_id)];
  const uint32_t limit_mask = model.context_len >= 32
      ? std::numeric_limits<uint32_t>::max()
      : ((uint32_t{1} << model.context_len) - 1U);

  auto hit = [&]() {
    mask = (mask << 1) & limit_mask;
    ++model.edit_hits;
  };
  auto fail = [&]() {
    const uint32_t fails = strict_popcount_mask(mask, model.context_len);
    if (fails <= model.edits) {
      mask = ((mask << 1) | 1U) & limit_mask;
      ++model.edit_fails;
    } else {
      in = 0;
    }
  };

  if (best == -2) {
    if (in != 0) {
      fail();
    }
  } else if (best == -1) {
    if (in != 0) {
      hit();
    }
  } else if (in == 0) {
    in = 1;
    mask = 0;
  } else if (best == target) {
    hit();
  } else {
    fail();
    edited_symbol = static_cast<uint8_t>(best);
  }
  return edited_symbol;
}

py::dict strict_weight_snapshot(
    int64_t depth,
    const std::vector<StrictPredictor> &predictors,
    const std::vector<double> &weights) {
  py::dict snapshot;
  snapshot["depth"] = depth;
  py::dict weight_dict;
  for (size_t index = 0; index < predictors.size(); ++index) {
    weight_dict[py::str(predictors[index].label)] = weights[index];
  }
  snapshot["weights"] = weight_dict;
  return snapshot;
}

std::vector<double> strict_average_window_weights(
    const std::vector<double> &window_weights,
    int64_t window_count,
    size_t predictor_count) {
  std::vector<double> averaged(predictor_count, 0.0);
  if (window_count <= 0 || predictor_count == 0) {
    return averaged;
  }
  for (int64_t window_id = 0; window_id < window_count; ++window_id) {
    const double *weights = window_weights.data() + static_cast<size_t>(window_id) * predictor_count;
    for (size_t predictor_index = 0; predictor_index < predictor_count; ++predictor_index) {
      averaged[predictor_index] += weights[predictor_index];
    }
  }
  for (double &weight : averaged) {
    weight /= static_cast<double>(window_count);
  }
  return averaged;
}

py::dict strict_window_weight_snapshot(
    int64_t depth,
    const std::vector<StrictPredictor> &predictors,
    const std::vector<double> &window_weights,
    int64_t window_count) {
  return strict_weight_snapshot(
      depth,
      predictors,
      strict_average_window_weights(window_weights, window_count, predictors.size()));
}

class FusedNcPrefixStreamingEncoder {
 public:
  FusedNcPrefixStreamingEncoder(
      int64_t window_count,
      int64_t window_bases,
      int64_t hash_bucket_count,
      int64_t arithmetic_frequency_total,
      double fusion_eta,
      double initial_lm_weight,
      bool encode_arithmetic,
      bool collect_diagnostics)
      : window_count_(window_count),
        window_bases_(window_bases),
        arithmetic_frequency_total_(arithmetic_frequency_total),
        fusion_eta_(fusion_eta),
        encode_arithmetic_(encode_arithmetic),
        collect_diagnostics_(collect_diagnostics) {
    if (window_count_ <= 0) {
      throw std::runtime_error("window_count must be positive");
    }
    if (window_bases_ <= 0) {
      throw std::runtime_error("window_bases must be positive");
    }
    if (hash_bucket_count < 0) {
      throw std::runtime_error("hash_bucket_count must be non-negative; use 0 for GECO2 default");
    }
    if (arithmetic_frequency_total_ <= 4 ||
        arithmetic_frequency_total_ > static_cast<int64_t>((uint64_t{1} << 30) + 2)) {
      throw std::runtime_error("arithmetic_frequency_total is outside the supported range");
    }
    if (!(fusion_eta_ >= 0.0 && fusion_eta_ < 1.0)) {
      throw std::runtime_error("fusion_eta must be in [0, 1)");
    }
    if (!(initial_lm_weight >= 0.0 && initial_lm_weight <= 1.0)) {
      throw std::runtime_error("initial_lm_weight must be in [0, 1]");
    }
    const int64_t effective_hash_buckets = hash_bucket_count > 0 ? hash_bucket_count : GECO2_HASH_SIZE;
    models_ = make_strict_level10_models(
        window_count_,
        effective_hash_buckets,
        false,
        false,
        predictors_);
    for (StrictModel &model : models_) {
      if (model.edits > 0) {
        model.edit_symbols.assign(static_cast<size_t>(window_count_ * model.context_len), 0);
        model.edit_best_ids.assign(static_cast<size_t>(window_count_), int8_t{-2});
      }
    }
    predictor_count_ = predictors_.size();
    edit_predictor_by_model_.assign(models_.size(), -1);
    for (size_t predictor_index = 0; predictor_index < predictors_.size(); ++predictor_index) {
      if (predictors_[predictor_index].edit) {
        edit_predictor_by_model_[static_cast<size_t>(predictors_[predictor_index].model_index)] =
            static_cast<int64_t>(predictor_index);
      }
    }
    nc_window_weights_.assign(
        static_cast<size_t>(window_count_) * predictor_count_,
        1.0 / static_cast<double>(predictor_count_));
    lm_source_weights_.assign(static_cast<size_t>(window_count_), initial_lm_weight);
    nc_source_weights_.assign(static_cast<size_t>(window_count_), 1.0 - initial_lm_weight);
    predictor_target_probs_.assign(predictor_count_, 0.25);
    step_target_probs_.assign(static_cast<size_t>(window_count_) * predictor_count_, 0.25);
    step_nc_probs_.resize(static_cast<size_t>(window_count_));
    step_freqs_.resize(static_cast<size_t>(window_count_) * predictor_count_);
    step_sums_.assign(static_cast<size_t>(window_count_) * predictor_count_, 4);
    history_.assign(static_cast<size_t>(window_count_ * 17), 0);
    pipeline_block_windows_ = configured_pipeline_block_windows();
    pipeline_predictor_to_high_.assign(predictor_count_, -1);
    for (size_t predictor_index = 0; predictor_index < predictor_count_; ++predictor_index) {
      const StrictPredictor &predictor = predictors_[predictor_index];
      const StrictModel &model = models_[static_cast<size_t>(predictor.model_index)];
      if (model.context_len >= 11) {
        pipeline_predictor_to_high_[predictor_index] =
            static_cast<int64_t>(pipeline_high_predictors_.size());
        pipeline_high_predictors_.push_back(static_cast<int64_t>(predictor_index));
      }
      if (!predictor.edit && model.context_len == 17) {
        pipeline_ctx17_model_index_ = predictor.model_index;
      }
    }
    const size_t query_slots =
        static_cast<size_t>(pipeline_block_windows_) * pipeline_high_predictors_.size();
    pipeline_query_buffers_[0].resize(query_slots);
    pipeline_query_buffers_[1].resize(query_slots);
    pipeline_freqs_.resize(static_cast<size_t>(pipeline_block_windows_) * predictor_count_);
    pipeline_sums_.assign(static_cast<size_t>(pipeline_block_windows_) * predictor_count_, 4);
    pipeline_ctx17_base_hashes_.resize(static_cast<size_t>(window_count_));
    pipeline_ctx17_ir_hashes_.resize(static_cast<size_t>(window_count_));
    pipeline_ctx17_ir_keys_.resize(static_cast<size_t>(window_count_));
    if (encode_arithmetic_) {
      arithmetic_streams_.resize(static_cast<size_t>(window_count_));
    }
    initialized_seconds_ = 0.0;
  }

  py::dict encode_base_step(const at::Tensor &lm_probabilities, const at::Tensor &target_symbols) {
    if (finished_) {
      throw std::runtime_error("cannot encode after finish()");
    }
    if (depth_ >= window_bases_) {
      throw std::runtime_error("received more base steps than window_bases");
    }
    if (lm_probabilities.device().is_cuda() || target_symbols.device().is_cuda()) {
      throw std::runtime_error("fused streaming encoder expects CPU tensors");
    }
    if (lm_probabilities.dim() != 2 || lm_probabilities.size(1) != 4 || target_symbols.dim() != 1 ||
        target_symbols.size(0) != lm_probabilities.size(0)) {
      throw std::runtime_error("lm_probabilities must be [active_windows,4] and target_symbols [active_windows]");
    }
    const int64_t active_count = lm_probabilities.size(0);
    if (active_count < 0 || active_count > window_count_) {
      throw std::runtime_error("active window count exceeds encoder window_count");
    }
    at::Tensor lm = lm_probabilities.contiguous();
    at::Tensor targets = target_symbols.contiguous();
    if (lm.scalar_type() != at::kFloat && lm.scalar_type() != at::kDouble) {
      throw std::runtime_error("lm_probabilities must be float32 or float64");
    }
    if (targets.scalar_type() != at::kLong &&
        targets.scalar_type() != at::kInt &&
        targets.scalar_type() != at::kShort) {
      throw std::runtime_error("target_symbols must be int16, int32, or int64");
    }

    const auto started = Clock::now();
    double step_fused_bits = 0.0;
    double step_lm_bits = 0.0;
    double step_nc_bits = 0.0;
    int64_t emitted = 0;

    auto body = [&]<typename prob_t, typename target_t>(const prob_t *lm_ptr, const target_t *target_ptr) {
      predict_pipeline_step(active_count);
      fuse_update_pipeline_step(
          active_count,
          lm_ptr,
          target_ptr,
          step_fused_bits,
          step_lm_bits,
          step_nc_bits,
          emitted,
          4,
          1);
    };

    {
      py::gil_scoped_release release;
      if (lm.scalar_type() == at::kFloat && targets.scalar_type() == at::kShort) {
        body(lm.data_ptr<float>(), targets.data_ptr<int16_t>());
      } else if (lm.scalar_type() == at::kFloat && targets.scalar_type() == at::kInt) {
        body(lm.data_ptr<float>(), targets.data_ptr<int32_t>());
      } else if (lm.scalar_type() == at::kFloat && targets.scalar_type() == at::kLong) {
        body(lm.data_ptr<float>(), targets.data_ptr<int64_t>());
      } else if (lm.scalar_type() == at::kDouble && targets.scalar_type() == at::kShort) {
        body(lm.data_ptr<double>(), targets.data_ptr<int16_t>());
      } else if (lm.scalar_type() == at::kDouble && targets.scalar_type() == at::kInt) {
        body(lm.data_ptr<double>(), targets.data_ptr<int32_t>());
      } else {
        body(lm.data_ptr<double>(), targets.data_ptr<int64_t>());
      }
    }

    record_encoded_base(active_count, step_fused_bits, step_lm_bits, step_nc_bits, emitted);
    encode_seconds_ += elapsed_seconds(started, Clock::now());
    py::dict result;
    result["active_windows"] = active_count;
    result["fused_bits"] = step_fused_bits;
    result["lm_bits"] = step_lm_bits;
    result["nc_bits"] = step_nc_bits;
    result["emitted_count"] = emitted;
    return result;
  }

  py::dict encode_token_step(const at::Tensor &lm_probabilities, const at::Tensor &target_symbols) {
    if (finished_) {
      throw std::runtime_error("cannot encode after finish()");
    }
    if (lm_probabilities.device().is_cuda() || target_symbols.device().is_cuda()) {
      throw std::runtime_error("fused streaming encoder expects CPU tensors");
    }
    if (lm_probabilities.dim() != 3 || lm_probabilities.size(2) != 4 || target_symbols.dim() != 2 ||
        target_symbols.size(0) != lm_probabilities.size(0) ||
        target_symbols.size(1) != lm_probabilities.size(1)) {
      throw std::runtime_error(
          "lm_probabilities must be [active_windows,token_merge_size,4] and target_symbols [active_windows,token_merge_size]");
    }
    const int64_t active_count = lm_probabilities.size(0);
    const int64_t token_merge_size = lm_probabilities.size(1);
    if (active_count < 0 || active_count > window_count_) {
      throw std::runtime_error("active window count exceeds encoder window_count");
    }
    if (token_merge_size <= 0) {
      throw std::runtime_error("token_merge_size must be positive");
    }
    if (depth_ + token_merge_size > window_bases_) {
      throw std::runtime_error("token step crosses nc_prefix window boundary");
    }
    at::Tensor lm = lm_probabilities.contiguous();
    at::Tensor targets = target_symbols.contiguous();
    if (lm.scalar_type() != at::kFloat && lm.scalar_type() != at::kDouble) {
      throw std::runtime_error("lm_probabilities must be float32 or float64");
    }
    if (targets.scalar_type() != at::kLong &&
        targets.scalar_type() != at::kInt &&
        targets.scalar_type() != at::kShort) {
      throw std::runtime_error("target_symbols must be int16, int32, or int64");
    }

    const auto started = Clock::now();
    double token_fused_bits = 0.0;
    double token_lm_bits = 0.0;
    double token_nc_bits = 0.0;
    int64_t token_emitted = 0;

    auto body = [&]<typename prob_t, typename target_t>(const prob_t *lm_ptr, const target_t *target_ptr) {
      for (int64_t base_offset = 0; base_offset < token_merge_size; ++base_offset) {
        double step_fused_bits = 0.0;
        double step_lm_bits = 0.0;
        double step_nc_bits = 0.0;
        int64_t emitted = 0;
        predict_pipeline_step(active_count);
        fuse_update_pipeline_step(
            active_count,
            lm_ptr + base_offset * 4,
            target_ptr + base_offset,
            step_fused_bits,
            step_lm_bits,
            step_nc_bits,
            emitted,
            token_merge_size * 4,
            token_merge_size);
        record_encoded_base(active_count, step_fused_bits, step_lm_bits, step_nc_bits, emitted);
        token_fused_bits += step_fused_bits;
        token_lm_bits += step_lm_bits;
        token_nc_bits += step_nc_bits;
        token_emitted += emitted;
      }
    };

    {
      py::gil_scoped_release release;
      if (lm.scalar_type() == at::kFloat && targets.scalar_type() == at::kShort) {
        body(lm.data_ptr<float>(), targets.data_ptr<int16_t>());
      } else if (lm.scalar_type() == at::kFloat && targets.scalar_type() == at::kInt) {
        body(lm.data_ptr<float>(), targets.data_ptr<int32_t>());
      } else if (lm.scalar_type() == at::kFloat && targets.scalar_type() == at::kLong) {
        body(lm.data_ptr<float>(), targets.data_ptr<int64_t>());
      } else if (lm.scalar_type() == at::kDouble && targets.scalar_type() == at::kShort) {
        body(lm.data_ptr<double>(), targets.data_ptr<int16_t>());
      } else if (lm.scalar_type() == at::kDouble && targets.scalar_type() == at::kInt) {
        body(lm.data_ptr<double>(), targets.data_ptr<int32_t>());
      } else {
        body(lm.data_ptr<double>(), targets.data_ptr<int64_t>());
      }
    }

    const double elapsed = elapsed_seconds(started, Clock::now());
    encode_seconds_ += elapsed;
    token_encode_seconds_ += elapsed;
    ++native_token_steps_;
    py::dict result;
    result["active_windows"] = active_count;
    result["token_merge_size"] = token_merge_size;
    result["fused_bits"] = token_fused_bits;
    result["lm_bits"] = token_lm_bits;
    result["nc_bits"] = token_nc_bits;
    result["emitted_count"] = token_emitted;
    return result;
  }

  py::dict finish() {
    if (finished_) {
      throw std::runtime_error("finish() was already called");
    }
    finished_ = true;
    const auto started = Clock::now();
    py::list streams;
    int64_t bytes = 0;
    if (encode_arithmetic_) {
      for (FusedArithmeticEncoder &encoder : arithmetic_streams_) {
        std::string stream = encoder.finish();
        bytes += static_cast<int64_t>(stream.size());
        streams.append(py::bytes(stream));
      }
    }
    finish_seconds_ += elapsed_seconds(started, Clock::now());
    py::dict result;
    result["codec"] = "fused_lm_nc_prefix_streaming_token_native";
    result["window_count"] = window_count_;
    result["window_bases"] = window_bases_;
    result["base_count"] = base_count_;
    result["emitted_arithmetic_symbol_count"] = emitted_symbols_;
    result["encode_arithmetic"] = encode_arithmetic_;
    result["diagnostics_collected"] = collect_diagnostics_;
    result["arithmetic_coded_bytes"] = encode_arithmetic_ ? py::cast(bytes) : py::none();
    result["arithmetic_bits_per_base"] =
        encode_arithmetic_ && base_count_ > 0 ? py::cast(static_cast<double>(bytes) * 8.0 / base_count_) : py::none();
    result["fused_theoretical_bits"] = collect_diagnostics_ ? py::cast(fused_bits_) : py::none();
    result["fused_theoretical_bits_per_base"] =
        collect_diagnostics_ ? py::cast(fused_bits_ / std::max<int64_t>(base_count_, 1)) : py::none();
    result["lm_only_theoretical_bits"] = collect_diagnostics_ ? py::cast(lm_bits_) : py::none();
    result["lm_only_theoretical_bits_per_base"] =
        collect_diagnostics_ ? py::cast(lm_bits_ / std::max<int64_t>(base_count_, 1)) : py::none();
    result["nc_prefix_only_theoretical_bits"] = collect_diagnostics_ ? py::cast(nc_bits_) : py::none();
    result["nc_prefix_only_theoretical_bits_per_base"] =
        collect_diagnostics_ ? py::cast(nc_bits_ / std::max<int64_t>(base_count_, 1)) : py::none();
    result["fusion_final_mean_lm_weight"] = mean(lm_source_weights_);
    result["encode_seconds"] = encode_seconds_;
    result["finish_seconds"] = finish_seconds_;
    result["streams"] = streams;
    result["model_metadata"] = metadata();
    return result;
  }

  py::dict metadata() const {
    py::dict metadata;
    metadata["algorithm"] = "geco2_level10_per_window_weights_streaming";
    metadata["window_count"] = window_count_;
    metadata["window_bases"] = window_bases_;
    metadata["predictor_count"] = static_cast<int64_t>(predictor_count_);
    metadata["hash_bucket_count"] = models_.empty() ? 0 : models_.back().hash.bucket_count;
    metadata["source_fusion"] = "online_hedge_lm_nc";
    metadata["diagnostics_collected"] = collect_diagnostics_;
    metadata["fused_cache_pipeline"] = true;
    metadata["pipeline_mode"] = "streaming_token";
    metadata["pipeline_block_windows"] = pipeline_block_windows_;
    metadata["pipeline_scratch_bytes"] = static_cast<int64_t>(pipeline_scratch_bytes());
    metadata["fusion_eta"] = fusion_eta_;
    metadata["arithmetic_frequency_total"] = arithmetic_frequency_total_;
    const double predict_seconds =
        pipeline_address_prepare_seconds_ + pipeline_low_lookup_seconds_ + pipeline_high_lookup_seconds_;
    const double update_seconds = pipeline_update_prepare_seconds_ + pipeline_update_commit_seconds_ +
        pipeline_edit_state_update_seconds_ + pipeline_context_update_seconds_;
    metadata["native_encode_seconds"] = encode_seconds_;
    metadata["native_token_steps"] = native_token_steps_;
    metadata["token_encode_seconds"] = token_encode_seconds_;
    metadata["native_nc_predict_seconds"] = predict_seconds;
    metadata["native_fusion_update_seconds"] = pipeline_fusion_seconds_ + update_seconds;
    metadata["predict_seconds"] = predict_seconds;
    metadata["fusion_encode_seconds"] = pipeline_fusion_seconds_;
    metadata["update_seconds"] = update_seconds;
    metadata["pipeline_address_prepare_seconds"] = pipeline_address_prepare_seconds_;
    metadata["pipeline_low_lookup_seconds"] = pipeline_low_lookup_seconds_;
    metadata["pipeline_high_lookup_seconds"] = pipeline_high_lookup_seconds_;
    metadata["pipeline_fusion_seconds"] = pipeline_fusion_seconds_;
    metadata["pipeline_update_prepare_seconds"] = pipeline_update_prepare_seconds_;
    metadata["pipeline_update_commit_seconds"] = pipeline_update_commit_seconds_;
    metadata["pipeline_edit_state_update_seconds"] = pipeline_edit_state_update_seconds_;
    metadata["pipeline_context_update_seconds"] = pipeline_context_update_seconds_;
    metadata["initialized_seconds"] = initialized_seconds_;
    py::list model_list;
    for (const StrictModel &model : models_) {
      py::dict item;
      item["label"] = model.label;
      item["context_len"] = model.context_len;
      item["alpha_den"] = model.alpha_den;
      item["gamma"] = model.gamma;
      item["ir"] = model.ir;
      item["edits"] = model.edits;
      item["dense"] = model.dense.enabled;
      item["hash_slots"] = model.hash_slots;
      item["queries"] = model.queries;
      item["hits"] = model.hits;
      item["edit_hits"] = model.edit_hits;
      item["edit_fails"] = model.edit_fails;
      item["storage_bytes"] = static_cast<int64_t>(model.dense.enabled
          ? model.dense.counts.requested_bytes()
          : model.hash.buckets.requested_bytes());
      model_list.append(item);
    }
    metadata["models"] = model_list;
    return metadata;
  }

 private:
  int64_t window_count_ = 0;
  int64_t window_bases_ = 0;
  int64_t depth_ = 0;
  int64_t window_group_count_ = 0;
  int64_t arithmetic_frequency_total_ = 65536;
  double fusion_eta_ = 0.05;
  bool encode_arithmetic_ = true;
  bool collect_diagnostics_ = true;
  bool finished_ = false;
  std::vector<StrictPredictor> predictors_;
  std::vector<StrictModel> models_;
  size_t predictor_count_ = 0;
  std::vector<int64_t> edit_predictor_by_model_;
  std::vector<double> nc_window_weights_;
  std::vector<double> lm_source_weights_;
  std::vector<double> nc_source_weights_;
  std::vector<double> predictor_target_probs_;
  std::vector<double> step_target_probs_;
  std::vector<std::array<double, 4>> step_nc_probs_;
  std::vector<std::array<uint32_t, 4>> step_freqs_;
  std::vector<uint32_t> step_sums_;
  std::vector<uint8_t> history_;
  int64_t pipeline_block_windows_ = DEFAULT_PIPELINE_BLOCK_WINDOWS;
  std::vector<int64_t> pipeline_high_predictors_;
  std::vector<int64_t> pipeline_predictor_to_high_;
  std::array<std::vector<PipelineQueryAddress>, 2> pipeline_query_buffers_;
  std::vector<std::array<uint32_t, 4>> pipeline_freqs_;
  std::vector<uint32_t> pipeline_sums_;
  std::vector<uint64_t> pipeline_ctx17_base_hashes_;
  std::vector<uint64_t> pipeline_ctx17_ir_hashes_;
  std::vector<uint64_t> pipeline_ctx17_ir_keys_;
  int64_t pipeline_ctx17_model_index_ = -1;
  std::vector<FusedArithmeticEncoder> arithmetic_streams_;
  int64_t base_count_ = 0;
  int64_t emitted_symbols_ = 0;
  double fused_bits_ = 0.0;
  double lm_bits_ = 0.0;
  double nc_bits_ = 0.0;
  double encode_seconds_ = 0.0;
  double finish_seconds_ = 0.0;
  double initialized_seconds_ = 0.0;
  int64_t native_token_steps_ = 0;
  double token_encode_seconds_ = 0.0;
  double pipeline_address_prepare_seconds_ = 0.0;
  double pipeline_low_lookup_seconds_ = 0.0;
  double pipeline_high_lookup_seconds_ = 0.0;
  double pipeline_fusion_seconds_ = 0.0;
  double pipeline_update_prepare_seconds_ = 0.0;
  double pipeline_update_commit_seconds_ = 0.0;
  double pipeline_edit_state_update_seconds_ = 0.0;
  double pipeline_context_update_seconds_ = 0.0;

  static double mean(const std::vector<double> &values) {
    if (values.empty()) {
      return 0.0;
    }
    double total = 0.0;
    for (double value : values) {
      total += value;
    }
    return total / static_cast<double>(values.size());
  }

  void record_encoded_base(
      int64_t active_count,
      double step_fused_bits,
      double step_lm_bits,
      double step_nc_bits,
      int64_t emitted) {
    fused_bits_ += step_fused_bits;
    lm_bits_ += step_lm_bits;
    nc_bits_ += step_nc_bits;
    emitted_symbols_ += emitted;
    base_count_ += active_count;
    ++depth_;
    if (depth_ == window_bases_) {
      depth_ = 0;
      ++window_group_count_;
    }
  }

  FusedInterval quantize_fused_interval(const std::array<double, 4> &probabilities, uint8_t target) const {
    std::array<uint32_t, 4> freqs{};
    uint32_t sum = 0;
    for (int64_t symbol = 0; symbol < 4; ++symbol) {
      const double probability = std::isfinite(probabilities[static_cast<size_t>(symbol)])
          ? std::max(probabilities[static_cast<size_t>(symbol)], 0.0)
          : 0.0;
      uint32_t value = static_cast<uint32_t>(
          std::floor(probability * static_cast<double>(arithmetic_frequency_total_)));
      value = std::max<uint32_t>(value, 1U);
      freqs[static_cast<size_t>(symbol)] = value;
      sum += value;
    }
    uint32_t low = 0;
    for (uint8_t symbol = 0; symbol < target; ++symbol) {
      low += freqs[static_cast<size_t>(symbol)];
    }
    const uint32_t high = low + freqs[static_cast<size_t>(target)];
    return FusedInterval{low, high, sum};
  }

  uint8_t old_symbol(int64_t window_id, int64_t context_len) const {
    if (depth_ < context_len) {
      return 0;
    }
    const int64_t slot = (depth_ - context_len) % 17;
    return history_[static_cast<size_t>(window_id * 17 + slot)];
  }

  void normalize_nc_weights(double *weights) {
    double weight_total = 0.0;
    for (size_t predictor_index = 0; predictor_index < predictor_count_; ++predictor_index) {
      weight_total += weights[predictor_index];
    }
    if (!(weight_total > 0.0) || !std::isfinite(weight_total)) {
      std::fill(weights, weights + predictor_count_, 1.0 / static_cast<double>(predictor_count_));
      return;
    }
    for (size_t predictor_index = 0; predictor_index < predictor_count_; ++predictor_index) {
      weights[predictor_index] /= weight_total;
    }
  }

  void update_source_weights(int64_t window_id, double lm_target, double nc_target) {
    const size_t index = static_cast<size_t>(window_id);
    const double lm_weight = lm_source_weights_[index];
    const double nc_weight = nc_source_weights_[index];
    double lm_new = 0.0;
    double nc_new = 0.0;
    if (fusion_eta_ > 0.0) {
      lm_new = std::pow(lm_weight, 1.0 - fusion_eta_) * lm_target;
      nc_new = std::pow(nc_weight, 1.0 - fusion_eta_) * nc_target;
    } else {
      lm_new = lm_weight * lm_target;
      nc_new = nc_weight * nc_target;
    }
    const double source_total = std::max(lm_new + nc_new, 1e-300);
    lm_source_weights_[index] = lm_new / source_total;
    nc_source_weights_[index] = nc_new / source_total;
  }

  size_t pipeline_scratch_bytes() const {
    return pipeline_query_buffers_[0].capacity() * sizeof(PipelineQueryAddress) +
        pipeline_query_buffers_[1].capacity() * sizeof(PipelineQueryAddress) +
        pipeline_freqs_.capacity() * sizeof(std::array<uint32_t, 4>) +
        pipeline_sums_.capacity() * sizeof(uint32_t) +
        pipeline_ctx17_base_hashes_.capacity() * sizeof(uint64_t) +
        pipeline_ctx17_ir_hashes_.capacity() * sizeof(uint64_t) +
        pipeline_ctx17_ir_keys_.capacity() * sizeof(uint64_t);
  }

  void predict_pipeline_step(int64_t active_count) {
    const int64_t block_count =
        (active_count + pipeline_block_windows_ - 1) / pipeline_block_windows_;

    auto prepare_block = [&](int64_t block_index, std::vector<PipelineQueryAddress> &buffer) {
      const auto started = Clock::now();
      const int64_t begin = block_index * pipeline_block_windows_;
      const int64_t end = std::min<int64_t>(active_count, begin + pipeline_block_windows_);
      for (int64_t window_id = begin; window_id < end; ++window_id) {
        const size_t local_window = static_cast<size_t>(window_id - begin);
        for (size_t high_index = 0; high_index < pipeline_high_predictors_.size(); ++high_index) {
          const int64_t predictor_index = pipeline_high_predictors_[high_index];
          const StrictPredictor &predictor = predictors_[static_cast<size_t>(predictor_index)];
          const StrictModel &model = models_[static_cast<size_t>(predictor.model_index)];
          const uint64_t key = predictor.edit
              ? model.edit_pidx[static_cast<size_t>(window_id)]
              : model.pidx[static_cast<size_t>(window_id)];
          const uint64_t hashed = model.dense.enabled ? 0 : zhash(key);
          PipelineQueryAddress &query =
              buffer[local_window * pipeline_high_predictors_.size() + high_index];
          query.key = key;
          query.hashed = hashed;
          pipeline_prefetch(model, key, hashed, false);
        }
      }
      pipeline_address_prepare_seconds_ += elapsed_seconds(started, Clock::now());
    };

    if (block_count > 0) {
      prepare_block(0, pipeline_query_buffers_[0]);
    }
    for (int64_t block_index = 0; block_index < block_count; ++block_index) {
      const int current_buffer_index = static_cast<int>(block_index & 1);
      const int next_buffer_index = current_buffer_index ^ 1;
      if (block_index + 1 < block_count) {
        prepare_block(block_index + 1, pipeline_query_buffers_[next_buffer_index]);
      }
      const int64_t begin = block_index * pipeline_block_windows_;
      const int64_t end = std::min<int64_t>(active_count, begin + pipeline_block_windows_);
      const std::vector<PipelineQueryAddress> &query_buffer =
          pipeline_query_buffers_[current_buffer_index];

      const auto low_started = Clock::now();
      for (int64_t window_id = begin; window_id < end; ++window_id) {
        const size_t local_window = static_cast<size_t>(window_id - begin);
        for (size_t predictor_index = 0; predictor_index < predictor_count_; ++predictor_index) {
          if (pipeline_predictor_to_high_[predictor_index] >= 0) {
            continue;
          }
          StrictPredictor &predictor = predictors_[predictor_index];
          StrictModel &model = models_[static_cast<size_t>(predictor.model_index)];
          const uint64_t key = model.pidx[static_cast<size_t>(window_id)];
          const size_t slot = static_cast<size_t>(window_id) * predictor_count_ + predictor_index;
          const uint32_t sum =
              strict_model_freqs(model, key, predictor.alpha_den, step_freqs_[slot]);
          step_sums_[slot] = sum;
          ++predictor.queries;
          ++model.queries;
          if (sum > 4) {
            ++predictor.hits;
            ++model.hits;
          }
        }
      }
      pipeline_low_lookup_seconds_ += elapsed_seconds(low_started, Clock::now());

      const auto high_started = Clock::now();
      for (int64_t window_id = begin; window_id < end; ++window_id) {
        const size_t local_window = static_cast<size_t>(window_id - begin);
        uint64_t ctx17_base_key = 0;
        size_t ctx17_base_slot = 0;
        bool have_ctx17_base = false;
        for (size_t high_index = 0; high_index < pipeline_high_predictors_.size(); ++high_index) {
          const size_t predictor_index =
              static_cast<size_t>(pipeline_high_predictors_[high_index]);
          StrictPredictor &predictor = predictors_[predictor_index];
          StrictModel &model = models_[static_cast<size_t>(predictor.model_index)];
          const PipelineQueryAddress &query =
              query_buffer[local_window * pipeline_high_predictors_.size() + high_index];
          const size_t slot = static_cast<size_t>(window_id) * predictor_count_ + predictor_index;
          uint32_t sum = 0;
          if (predictor.edit && model.context_len == 17 && have_ctx17_base &&
              query.key == ctx17_base_key) {
            for (int64_t symbol = 0; symbol < 4; ++symbol) {
              const uint32_t base_freq =
                  step_freqs_[ctx17_base_slot][static_cast<size_t>(symbol)];
              const uint32_t count = (base_freq - 1U) / model.alpha_den;
              step_freqs_[slot][static_cast<size_t>(symbol)] =
                  1U + predictor.alpha_den * count;
              sum += step_freqs_[slot][static_cast<size_t>(symbol)];
            }
          } else {
            sum = strict_model_freqs_planned(
                model,
                query.key,
                query.hashed,
                predictor.alpha_den,
                step_freqs_[slot]);
          }
          step_sums_[slot] = sum;
          ++predictor.queries;
          ++model.queries;
          if (sum > 4) {
            ++predictor.hits;
            ++model.hits;
          }
          if (predictor.edit) {
            model.edit_best_ids[static_cast<size_t>(window_id)] =
                static_cast<int8_t>(strict_best_id(step_freqs_[slot], sum));
          } else if (model.context_len == 17) {
            ctx17_base_key = query.key;
            ctx17_base_slot = slot;
            have_ctx17_base = true;
            pipeline_ctx17_base_hashes_[static_cast<size_t>(window_id)] = query.hashed;
          }
        }
      }
      pipeline_high_lookup_seconds_ += elapsed_seconds(high_started, Clock::now());

      for (int64_t window_id = begin; window_id < end; ++window_id) {
        const size_t slot_base = static_cast<size_t>(window_id) * predictor_count_;
        const double *nc_weights =
            nc_window_weights_.data() + static_cast<size_t>(window_id) * predictor_count_;
        std::array<double, 4> nc_mixed{};
        for (size_t predictor_index = 0; predictor_index < predictor_count_; ++predictor_index) {
          const size_t slot = slot_base + predictor_index;
          const uint32_t sum = step_sums_[slot];
          const double factor = nc_weights[predictor_index] / static_cast<double>(sum);
          for (int64_t symbol = 0; symbol < 4; ++symbol) {
            nc_mixed[static_cast<size_t>(symbol)] +=
                static_cast<double>(step_freqs_[slot][static_cast<size_t>(symbol)]) * factor;
          }
        }
        std::array<uint32_t, 4> nc_freqs{};
        uint32_t nc_sum = 0;
        for (int64_t symbol = 0; symbol < 4; ++symbol) {
          const double probability = std::max(nc_mixed[static_cast<size_t>(symbol)], 0.0);
          nc_freqs[static_cast<size_t>(symbol)] =
              1U + static_cast<uint32_t>(std::floor(probability * static_cast<double>(GECO2_MX_PMODEL)));
          nc_sum += nc_freqs[static_cast<size_t>(symbol)];
        }
        for (int64_t symbol = 0; symbol < 4; ++symbol) {
          step_nc_probs_[static_cast<size_t>(window_id)][static_cast<size_t>(symbol)] =
              static_cast<double>(nc_freqs[static_cast<size_t>(symbol)]) / static_cast<double>(nc_sum);
        }
      }
    }
  }

  template <typename prob_t, typename target_t>
  void fuse_update_pipeline_step(
      int64_t active_count,
      const prob_t *lm_ptr,
      const target_t *target_ptr,
      double &step_fused_bits,
      double &step_lm_bits,
      double &step_nc_bits,
      int64_t &emitted,
      int64_t lm_window_stride,
      int64_t target_window_stride) {
    const int64_t block_count =
        (active_count + pipeline_block_windows_ - 1) / pipeline_block_windows_;
    for (int64_t block_index = 0; block_index < block_count; ++block_index) {
      const int64_t begin = block_index * pipeline_block_windows_;
      const int64_t end = std::min<int64_t>(active_count, begin + pipeline_block_windows_);
      const auto fusion_started = Clock::now();
      for (int64_t window_id = begin; window_id < end; ++window_id) {
        const int64_t target_i64 = static_cast<int64_t>(target_ptr[window_id * target_window_stride]);
        if (target_i64 < 0 || target_i64 >= 4) {
          throw std::runtime_error("target symbol must be in ACGT");
        }
        const uint8_t target = static_cast<uint8_t>(target_i64);
        const size_t slot_base = static_cast<size_t>(window_id) * predictor_count_;
        double *nc_weights = nc_window_weights_.data() + static_cast<size_t>(window_id) * predictor_count_;
        for (size_t predictor_index = 0; predictor_index < predictor_count_; ++predictor_index) {
          const size_t slot = slot_base + predictor_index;
          const uint32_t sum = step_sums_[slot];
          const double target_probability = std::max(
              static_cast<double>(step_freqs_[slot][static_cast<size_t>(target)]) /
                  static_cast<double>(sum),
              1e-300);
          nc_weights[predictor_index] =
              std::pow(nc_weights[predictor_index], predictors_[predictor_index].gamma) *
              target_probability;
        }
        normalize_nc_weights(nc_weights);
        const std::array<double, 4> &nc_probs = step_nc_probs_[static_cast<size_t>(window_id)];

        std::array<double, 4> lm_probs{};
        double lm_sum = 0.0;
        for (int64_t symbol = 0; symbol < 4; ++symbol) {
          double value = static_cast<double>(lm_ptr[window_id * lm_window_stride + symbol]);
          if (!std::isfinite(value) || value < 0.0) {
            value = 0.0;
          }
          lm_probs[static_cast<size_t>(symbol)] = value;
          lm_sum += value;
        }
        lm_sum = std::max(lm_sum, 1e-300);
        for (double &value : lm_probs) {
          value /= lm_sum;
        }

        const double lm_weight = lm_source_weights_[static_cast<size_t>(window_id)];
        const double nc_weight = nc_source_weights_[static_cast<size_t>(window_id)];
        const double lm_target = std::max(lm_probs[static_cast<size_t>(target)], 1e-300);
        const double nc_target = std::max(nc_probs[static_cast<size_t>(target)], 1e-300);
        std::array<double, 4> fused{};
        double fused_target = 1.0;
        if (encode_arithmetic_ || collect_diagnostics_) {
          double fused_sum = 0.0;
          for (int64_t symbol = 0; symbol < 4; ++symbol) {
            fused[static_cast<size_t>(symbol)] =
                lm_weight * lm_probs[static_cast<size_t>(symbol)] +
                nc_weight * nc_probs[static_cast<size_t>(symbol)];
            fused_sum += fused[static_cast<size_t>(symbol)];
          }
          fused_sum = std::max(fused_sum, 1e-300);
          for (double &value : fused) {
            value /= fused_sum;
          }
          fused_target = std::max(fused[static_cast<size_t>(target)], 1e-300);
        }
        if (collect_diagnostics_) {
          step_lm_bits += -std::log2(lm_target);
          step_nc_bits += -std::log2(nc_target);
          step_fused_bits += -std::log2(fused_target);
        }
        update_source_weights(window_id, lm_target, nc_target);

        if (encode_arithmetic_) {
          arithmetic_streams_[static_cast<size_t>(window_id)].update(
              quantize_fused_interval(fused, target));
          ++emitted;
        }
      }
      pipeline_fusion_seconds_ += elapsed_seconds(fusion_started, Clock::now());
    }

    update_nc_counters_and_contexts_pipeline(active_count, target_ptr, target_window_stride);
  }

  template <typename target_t>
  void update_nc_counters_and_contexts_pipeline(
      int64_t active_count,
      const target_t *target_ptr,
      int64_t target_window_stride) {
    const auto prepare_started = Clock::now();
    StrictModel &ctx17_model = models_[static_cast<size_t>(pipeline_ctx17_model_index_)];
    for (int64_t window_id = 0; window_id < active_count; ++window_id) {
      const uint8_t target = static_cast<uint8_t>(target_ptr[window_id * target_window_stride]);
      uint64_t &ir_key = ctx17_model.pidx_ir[static_cast<size_t>(window_id)];
      ir_key = (ir_key >> 2) +
          (static_cast<uint64_t>(complement_symbol(target)) * ctx17_model.multiplier);
      pipeline_ctx17_ir_keys_[static_cast<size_t>(window_id)] = ir_key;
      pipeline_ctx17_ir_hashes_[static_cast<size_t>(window_id)] = zhash(ir_key);
      ++ctx17_model.ir_updates;
    }
    pipeline_update_prepare_seconds_ += elapsed_seconds(prepare_started, Clock::now());

    const auto commit_started = Clock::now();
    const int64_t block_count =
        (active_count + pipeline_block_windows_ - 1) / pipeline_block_windows_;
    for (int64_t block_index = 0; block_index < block_count; ++block_index) {
      const int64_t begin = block_index * pipeline_block_windows_;
      const int64_t end = std::min<int64_t>(active_count, begin + pipeline_block_windows_);
      const int64_t next_begin = end;
      const int64_t next_end = std::min<int64_t>(active_count, next_begin + pipeline_block_windows_);
      for (int64_t future_window = next_begin; future_window < next_end; ++future_window) {
        ctx17_model.hash.prefetch_hashed(
            pipeline_ctx17_base_hashes_[static_cast<size_t>(future_window)], true);
        ctx17_model.hash.prefetch_hashed(
            pipeline_ctx17_ir_hashes_[static_cast<size_t>(future_window)], true);
        for (StrictModel &model : models_) {
          if (model.dense.enabled && model.context_len == 13) {
            pipeline_prefetch(model, model.pidx[static_cast<size_t>(future_window)], 0, true);
          }
        }
      }

      for (int64_t window_id = begin; window_id < end; ++window_id) {
        const uint8_t target = static_cast<uint8_t>(target_ptr[window_id * target_window_stride]);
        for (StrictModel &model : models_) {
          const uint64_t key = model.pidx[static_cast<size_t>(window_id)];
          if (model.context_len == 17) {
            model.hash.update_hashed(
                pipeline_ctx17_base_hashes_[static_cast<size_t>(window_id)], target);
            model.hash.update_hashed(
                pipeline_ctx17_ir_hashes_[static_cast<size_t>(window_id)],
                complement_symbol(old_symbol(window_id, model.context_len)));
            continue;
          }
          strict_model_update(model, key, target);
          if (model.ir) {
            const uint8_t old = old_symbol(window_id, model.context_len);
            uint64_t &ir_key = model.pidx_ir[static_cast<size_t>(window_id)];
            ir_key = (ir_key >> 2) +
                (static_cast<uint64_t>(complement_symbol(target)) * model.multiplier);
            strict_model_update(model, ir_key, complement_symbol(old));
            ++model.ir_updates;
          }
        }
      }
    }
    pipeline_update_commit_seconds_ += elapsed_seconds(commit_started, Clock::now());

    const auto edit_started = Clock::now();
    for (int64_t window_id = 0; window_id < active_count; ++window_id) {
      const uint8_t target = static_cast<uint8_t>(target_ptr[window_id * target_window_stride]);
      for (size_t model_index = 0; model_index < models_.size(); ++model_index) {
        StrictModel &model = models_[model_index];
        if (model.edits == 0) {
          continue;
        }
        const int64_t edit_predictor_index = edit_predictor_by_model_[model_index];
        uint8_t edited_symbol = target;
        if (edit_predictor_index >= 0) {
          edited_symbol = strict_correct_edit_symbol(
              model,
              window_id,
              static_cast<int64_t>(model.edit_best_ids[static_cast<size_t>(window_id)]),
              target);
        }
        const size_t edit_ring_slot =
            static_cast<size_t>(window_id * model.context_len + depth_ % model.context_len);
        const uint8_t old_edit_symbol = depth_ >= model.context_len
            ? model.edit_symbols[edit_ring_slot]
            : uint8_t{0};
        uint64_t &edit_key = model.edit_pidx[static_cast<size_t>(window_id)];
        edit_key = ((edit_key - static_cast<uint64_t>(old_edit_symbol) * model.multiplier) << 2) +
            edited_symbol;
        edit_key &= (model.n_contexts - 1);
        model.edit_symbols[edit_ring_slot] = edited_symbol;
      }
    }
    pipeline_edit_state_update_seconds_ += elapsed_seconds(edit_started, Clock::now());

    const auto context_started = Clock::now();
    for (int64_t window_id = 0; window_id < active_count; ++window_id) {
      const uint8_t target = static_cast<uint8_t>(target_ptr[window_id * target_window_stride]);
      for (StrictModel &model : models_) {
        const uint8_t old = old_symbol(window_id, model.context_len);
        uint64_t &key = model.pidx[static_cast<size_t>(window_id)];
        key = ((key - static_cast<uint64_t>(old) * model.multiplier) << 2) + target;
        key &= (model.n_contexts - 1);
      }
      history_[static_cast<size_t>(window_id * 17 + depth_ % 17)] = target;
    }
    pipeline_context_update_seconds_ += elapsed_seconds(context_started, Clock::now());
  }

  void predict_nc_step(int64_t active_count) {
    for (int64_t window_id = 0; window_id < active_count; ++window_id) {
      uint64_t ctx17_base_key = 0;
      size_t ctx17_base_slot = 0;
      bool have_ctx17_base = false;
      for (size_t predictor_index = 0; predictor_index < predictor_count_; ++predictor_index) {
        StrictPredictor &predictor = predictors_[predictor_index];
        StrictModel &model = models_[static_cast<size_t>(predictor.model_index)];
        const uint64_t key = predictor.edit
            ? model.edit_pidx[static_cast<size_t>(window_id)]
            : model.pidx[static_cast<size_t>(window_id)];
        const size_t slot = static_cast<size_t>(window_id) * predictor_count_ + predictor_index;
        uint32_t sum = 0;
        if (predictor.edit && model.context_len == 17 && have_ctx17_base && key == ctx17_base_key) {
          for (int64_t symbol = 0; symbol < 4; ++symbol) {
            const uint32_t base_freq = step_freqs_[ctx17_base_slot][static_cast<size_t>(symbol)];
            const uint32_t count = (base_freq - 1U) / model.alpha_den;
            step_freqs_[slot][static_cast<size_t>(symbol)] = 1U + predictor.alpha_den * count;
            sum += step_freqs_[slot][static_cast<size_t>(symbol)];
          }
        } else {
          sum = strict_model_freqs(model, key, predictor.alpha_den, step_freqs_[slot]);
        }
        step_sums_[slot] = sum;
        ++predictor.queries;
        ++model.queries;
        if (sum > 4) {
          ++predictor.hits;
          ++model.hits;
        }
        if (predictor.edit) {
          model.edit_best_ids[static_cast<size_t>(window_id)] =
              static_cast<int8_t>(strict_best_id(step_freqs_[slot], sum));
        } else if (model.context_len == 17) {
          ctx17_base_key = key;
          ctx17_base_slot = slot;
          have_ctx17_base = true;
        }
      }
    }
  }

  void compute_nc_probability_for_window(
      int64_t window_id,
      uint8_t target,
      std::array<double, 4> &nc_probs) {
    std::array<double, 4> mixed{};
    double *weights = nc_window_weights_.data() + static_cast<size_t>(window_id) * predictor_count_;
    const size_t slot_base = static_cast<size_t>(window_id) * predictor_count_;
    for (size_t predictor_index = 0; predictor_index < predictor_count_; ++predictor_index) {
      const size_t slot = slot_base + predictor_index;
      const uint32_t sum = step_sums_[slot];
      step_target_probs_[slot] = std::max(
          static_cast<double>(step_freqs_[slot][static_cast<size_t>(target)]) / static_cast<double>(sum),
          1e-300);
      const double factor = weights[predictor_index] / static_cast<double>(sum);
      for (int64_t symbol = 0; symbol < 4; ++symbol) {
        mixed[static_cast<size_t>(symbol)] +=
            static_cast<double>(step_freqs_[slot][static_cast<size_t>(symbol)]) * factor;
      }
    }
    uint32_t mx_sum = 0;
    std::array<uint32_t, 4> mx_freqs{};
    for (int64_t symbol = 0; symbol < 4; ++symbol) {
      const double probability = std::max(mixed[static_cast<size_t>(symbol)], 0.0);
      mx_freqs[static_cast<size_t>(symbol)] =
          1U + static_cast<uint32_t>(std::floor(probability * static_cast<double>(GECO2_MX_PMODEL)));
      mx_sum += mx_freqs[static_cast<size_t>(symbol)];
    }
    for (int64_t symbol = 0; symbol < 4; ++symbol) {
      nc_probs[static_cast<size_t>(symbol)] =
          static_cast<double>(mx_freqs[static_cast<size_t>(symbol)]) / static_cast<double>(mx_sum);
    }
  }

  void update_nc_weights(int64_t active_count) {
    for (int64_t window_id = 0; window_id < active_count; ++window_id) {
      double *weights = nc_window_weights_.data() + static_cast<size_t>(window_id) * predictor_count_;
      const size_t slot_base = static_cast<size_t>(window_id) * predictor_count_;
      double weight_total = 0.0;
      for (size_t predictor_index = 0; predictor_index < predictor_count_; ++predictor_index) {
        const size_t slot = slot_base + predictor_index;
        const double target_probability = step_target_probs_[slot];
        weights[predictor_index] =
            std::pow(weights[predictor_index], predictors_[predictor_index].gamma) * target_probability;
        weight_total += weights[predictor_index];
        (void)slot;
      }
      if (!(weight_total > 0.0) || !std::isfinite(weight_total)) {
        std::fill(weights, weights + predictor_count_, 1.0 / static_cast<double>(predictor_count_));
      } else {
        for (size_t predictor_index = 0; predictor_index < predictor_count_; ++predictor_index) {
          weights[predictor_index] /= weight_total;
        }
      }
    }
  }

  template <typename target_t>
  void update_nc_counters_and_contexts(int64_t active_count, const target_t *target_ptr) {
    for (int64_t window_id = 0; window_id < active_count; ++window_id) {
      const uint8_t target = static_cast<uint8_t>(target_ptr[window_id]);
      for (StrictModel &model : models_) {
        const uint64_t key = model.pidx[static_cast<size_t>(window_id)];
        strict_model_update(model, key, target);
        if (model.ir) {
          const uint8_t old = old_symbol(window_id, model.context_len);
          uint64_t &ir_key = model.pidx_ir[static_cast<size_t>(window_id)];
          ir_key = (ir_key >> 2) +
              (static_cast<uint64_t>(complement_symbol(target)) * model.multiplier);
          strict_model_update(model, ir_key, complement_symbol(old));
          ++model.ir_updates;
        }
      }
    }

    for (int64_t window_id = 0; window_id < active_count; ++window_id) {
      const uint8_t target = static_cast<uint8_t>(target_ptr[window_id]);
      for (size_t model_index = 0; model_index < models_.size(); ++model_index) {
        StrictModel &model = models_[model_index];
        if (model.edits == 0) {
          continue;
        }
        const int64_t edit_predictor_index = edit_predictor_by_model_[model_index];
        uint8_t edited_symbol = target;
        if (edit_predictor_index >= 0) {
          edited_symbol = strict_correct_edit_symbol(
              model,
              window_id,
              static_cast<int64_t>(model.edit_best_ids[static_cast<size_t>(window_id)]),
              target);
        }
        const size_t edit_ring_slot =
            static_cast<size_t>(window_id * model.context_len + depth_ % model.context_len);
        const uint8_t old_edit_symbol = depth_ >= model.context_len
            ? model.edit_symbols[edit_ring_slot]
            : uint8_t{0};
        uint64_t &edit_key = model.edit_pidx[static_cast<size_t>(window_id)];
        edit_key = ((edit_key - static_cast<uint64_t>(old_edit_symbol) * model.multiplier) << 2) +
            edited_symbol;
        edit_key &= (model.n_contexts - 1);
        model.edit_symbols[edit_ring_slot] = edited_symbol;
      }
    }

    for (int64_t window_id = 0; window_id < active_count; ++window_id) {
      const uint8_t target = static_cast<uint8_t>(target_ptr[window_id]);
      for (StrictModel &model : models_) {
        const uint8_t old = old_symbol(window_id, model.context_len);
        uint64_t &key = model.pidx[static_cast<size_t>(window_id)];
        key = ((key - static_cast<uint64_t>(old) * model.multiplier) << 2) + target;
        key &= (model.n_contexts - 1);
      }
      history_[static_cast<size_t>(window_id * 17 + depth_ % 17)] = target;
    }
  }
};

template <typename scalar_t>
py::dict compute_current_nc_prefix_impl(
    const at::Tensor &symbols_tensor,
    int64_t window_bases,
    int64_t vocab_size,
    bool return_probabilities,
    bool summary_only,
    int64_t hash_bucket_count,
    bool disable_edit_experts,
    bool disable_ir,
    const std::string &update_mode,
    const std::string &profile_mode,
    int64_t threads) {
  if (vocab_size != 4) {
    throw std::runtime_error("nc_prefix requires vocab_size=4 (ACGT)");
  }
  if (window_bases <= 0) {
    throw std::runtime_error("window_bases must be positive");
  }
  if (hash_bucket_count < 0) {
    throw std::runtime_error("hash_bucket_count must be non-negative; use 0 for GECO2 default");
  }
  const scalar_t *symbols = symbols_tensor.data_ptr<scalar_t>();
  const int64_t n = symbols_tensor.size(0);
  if (n <= 0) {
    throw std::runtime_error("sequence must contain at least one base");
  }
  if (update_mode != "cache_pipeline") {
    throw std::runtime_error("nc_prefix now exposes only update_mode='cache_pipeline'");
  }
  if (profile_mode != "normal") {
    throw std::runtime_error("nc_prefix now exposes only profile_mode='normal'");
  }
  const bool cache_pipeline = true;
  const int64_t pipeline_block_windows = configured_pipeline_block_windows();
  const bool profile_skip_mixture = false;
  const bool profile_skip_weight_update = false;
  const int threads_effective = 1;
  const bool collect_timing = env_flag_enabled("DNA_COMPRESS_NC_PREFIX_PROFILE_TIMING");
  const auto started = Clock::now();
  const int64_t window_count = (n + window_bases - 1) / window_bases;
  const int64_t max_window_len = std::min<int64_t>(window_bases, n);
  std::vector<int64_t> window_starts(static_cast<size_t>(window_count));
  for (int64_t window = 0; window < window_count; ++window) {
    window_starts[static_cast<size_t>(window)] = window * window_bases;
  }

  if (summary_only && return_probabilities) {
    throw std::runtime_error("summary_only cannot return a probability matrix");
  }
  at::Tensor bpb_tensor = summary_only
      ? torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat64))
      : torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat64));
  at::Tensor target_tensor = summary_only
      ? torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64))
      : torch::empty({n}, torch::TensorOptions().dtype(torch::kInt64));
  at::Tensor emit_order_tensor = summary_only
      ? torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64))
      : torch::empty({n}, torch::TensorOptions().dtype(torch::kInt64));
  at::Tensor probabilities_tensor = return_probabilities && !summary_only
      ? torch::empty({n, 4}, torch::TensorOptions().dtype(torch::kFloat64))
      : torch::empty({0, 4}, torch::TensorOptions().dtype(torch::kFloat64));
  double *bpb = summary_only ? nullptr : bpb_tensor.data_ptr<double>();
  int64_t *output_targets = summary_only ? nullptr : target_tensor.data_ptr<int64_t>();
  int64_t *emit_order = summary_only ? nullptr : emit_order_tensor.data_ptr<int64_t>();
  double *probabilities = return_probabilities && !summary_only ? probabilities_tensor.data_ptr<double>() : nullptr;
  const scalar_t *targets = symbols;
  for (int64_t pos = 0; pos < n; ++pos) {
    const int64_t symbol = static_cast<int64_t>(symbols[pos]);
    if (symbol < 0 || symbol >= 4) {
      throw std::runtime_error("nc_prefix symbols must be in ACGT alphabet");
    }
    if (!summary_only) {
      output_targets[pos] = symbol;
    }
  }

  const int64_t effective_hash_buckets = hash_bucket_count > 0 ? hash_bucket_count : GECO2_HASH_SIZE;
  std::vector<StrictPredictor> predictors;
  std::vector<StrictModel> models = make_strict_level10_models(
      window_count,
      effective_hash_buckets,
      disable_edit_experts,
      disable_ir,
      predictors);
  for (StrictModel &model : models) {
    if (model.edits > 0) {
      model.edit_symbols.assign(static_cast<size_t>(window_count * model.context_len), 0);
      model.edit_best_ids.assign(static_cast<size_t>(window_count), int8_t{-2});
    }
  }
  const size_t predictor_count = predictors.size();
  std::vector<int64_t> edit_predictor_by_model(models.size(), -1);
  for (size_t predictor_index = 0; predictor_index < predictors.size(); ++predictor_index) {
    if (predictors[predictor_index].edit) {
      edit_predictor_by_model[static_cast<size_t>(predictors[predictor_index].model_index)] =
          static_cast<int64_t>(predictor_index);
    }
  }
  std::vector<double> window_weights(
      static_cast<size_t>(window_count) * predictor_count,
      1.0 / static_cast<double>(predictor_count));
  std::vector<double> predictor_target_probs(predictors.size(), 0.25);
  std::vector<double> mixed_float(4, 0.0);
  std::array<uint32_t, 4> mx_freqs{1, 1, 1, 1};
  uint32_t mx_sum = 4;
  std::vector<int64_t> pipeline_high_predictors;
  std::vector<int64_t> pipeline_predictor_to_high(predictor_count, -1);
  std::array<std::vector<PipelineQueryAddress>, 2> pipeline_query_buffers;
  std::vector<std::array<uint32_t, 4>> pipeline_freqs;
  std::vector<uint32_t> pipeline_sums;
  std::vector<uint64_t> pipeline_ctx17_base_hashes;
  std::vector<uint64_t> pipeline_ctx17_ir_hashes;
  std::vector<uint64_t> pipeline_ctx17_ir_keys;
  int64_t pipeline_ctx17_model_index = -1;
  if (cache_pipeline) {
    for (size_t predictor_index = 0; predictor_index < predictor_count; ++predictor_index) {
      const StrictPredictor &predictor = predictors[predictor_index];
      const StrictModel &model = models[static_cast<size_t>(predictor.model_index)];
      if (model.context_len >= 11) {
        pipeline_predictor_to_high[predictor_index] = static_cast<int64_t>(pipeline_high_predictors.size());
        pipeline_high_predictors.push_back(static_cast<int64_t>(predictor_index));
      }
      if (!predictor.edit && model.context_len == 17) {
        pipeline_ctx17_model_index = predictor.model_index;
      }
    }
    const size_t query_slots = static_cast<size_t>(pipeline_block_windows) * pipeline_high_predictors.size();
    pipeline_query_buffers[0].resize(query_slots);
    pipeline_query_buffers[1].resize(query_slots);
    pipeline_freqs.resize(static_cast<size_t>(pipeline_block_windows) * predictor_count);
    pipeline_sums.resize(static_cast<size_t>(pipeline_block_windows) * predictor_count, 4);
    pipeline_ctx17_base_hashes.resize(static_cast<size_t>(window_count));
    pipeline_ctx17_ir_hashes.resize(static_cast<size_t>(window_count));
    pipeline_ctx17_ir_keys.resize(static_cast<size_t>(window_count));
  }

  py::list weight_history;
  if (collect_timing) {
    weight_history.append(strict_window_weight_snapshot(0, predictors, window_weights, window_count));
  }
  const int64_t snapshot_stride = std::max<int64_t>(1, max_window_len / 32);
  int64_t emit_index = 0;
  int64_t depth_count = 0;
  double mixed_float_bits = 0.0;
  double quantized_bits = 0.0;
  double setup_seconds = 0.0;
  double emit_order_seconds = 0.0;
  double prediction_weight_seconds = 0.0;
  double pipeline_address_prepare_seconds = 0.0;
  double pipeline_low_lookup_seconds = 0.0;
  double pipeline_high_lookup_seconds = 0.0;
  double pipeline_fusion_seconds = 0.0;
  double pipeline_update_prepare_seconds = 0.0;
  double pipeline_update_commit_seconds = 0.0;
  double base_counter_update_seconds = 0.0;
  double edit_state_update_seconds = 0.0;
  double context_state_update_seconds = 0.0;
  double weight_snapshot_seconds = 0.0;
  setup_seconds = collect_timing ? elapsed_seconds(started, Clock::now()) : 0.0;

  for (int64_t depth = 0; depth < max_window_len; ++depth) {
    int64_t active_count = 0;

    if (cache_pipeline) {
      const auto prediction_started = stage_start(collect_timing);
      for (int64_t window_id = 0; window_id < window_count; ++window_id) {
        const int64_t position = window_starts[static_cast<size_t>(window_id)] + depth;
        if (position < n) {
          if (!summary_only) {
            emit_order[emit_index++] = position;
          }
          ++active_count;
        }
      }

      const int64_t block_count =
          (window_count + pipeline_block_windows - 1) / pipeline_block_windows;
      auto prepare_block = [&](int64_t block_index, std::vector<PipelineQueryAddress> &buffer) {
        const auto prepare_started = stage_start(collect_timing);
        const int64_t begin = block_index * pipeline_block_windows;
        const int64_t end = std::min<int64_t>(window_count, begin + pipeline_block_windows);
        for (int64_t window_id = begin; window_id < end; ++window_id) {
          const int64_t position = window_starts[static_cast<size_t>(window_id)] + depth;
          if (position >= n) {
            continue;
          }
          const size_t local_window = static_cast<size_t>(window_id - begin);
          for (size_t high_index = 0; high_index < pipeline_high_predictors.size(); ++high_index) {
            const int64_t predictor_index = pipeline_high_predictors[high_index];
            const StrictPredictor &predictor = predictors[static_cast<size_t>(predictor_index)];
            const StrictModel &model = models[static_cast<size_t>(predictor.model_index)];
            const uint64_t key = predictor.edit
                ? model.edit_pidx[static_cast<size_t>(window_id)]
                : model.pidx[static_cast<size_t>(window_id)];
            const uint64_t hashed = model.dense.enabled ? 0 : zhash(key);
            PipelineQueryAddress &query =
                buffer[local_window * pipeline_high_predictors.size() + high_index];
            query.key = key;
            query.hashed = hashed;
            pipeline_prefetch(model, key, hashed, false);
          }
        }
        add_stage_elapsed(collect_timing, pipeline_address_prepare_seconds, prepare_started);
      };

      if (block_count > 0) {
        prepare_block(0, pipeline_query_buffers[0]);
      }
      for (int64_t block_index = 0; block_index < block_count; ++block_index) {
        const int current_buffer_index = static_cast<int>(block_index & 1);
        const int next_buffer_index = current_buffer_index ^ 1;
        if (block_index + 1 < block_count) {
          prepare_block(block_index + 1, pipeline_query_buffers[next_buffer_index]);
        }
        const int64_t begin = block_index * pipeline_block_windows;
        const int64_t end = std::min<int64_t>(window_count, begin + pipeline_block_windows);
        const std::vector<PipelineQueryAddress> &query_buffer =
            pipeline_query_buffers[current_buffer_index];

        const auto low_started = stage_start(collect_timing);
        for (int64_t window_id = begin; window_id < end; ++window_id) {
          const int64_t position = window_starts[static_cast<size_t>(window_id)] + depth;
          if (position >= n) {
            continue;
          }
          const size_t local_window = static_cast<size_t>(window_id - begin);
          for (size_t predictor_index = 0; predictor_index < predictor_count; ++predictor_index) {
            if (pipeline_predictor_to_high[predictor_index] >= 0) {
              continue;
            }
            StrictPredictor &predictor = predictors[predictor_index];
            StrictModel &model = models[static_cast<size_t>(predictor.model_index)];
            const uint64_t key = model.pidx[static_cast<size_t>(window_id)];
            const size_t slot = local_window * predictor_count + predictor_index;
            const uint32_t sum = strict_model_freqs(
                model, key, predictor.alpha_den, pipeline_freqs[slot]);
            pipeline_sums[slot] = sum;
            ++predictor.queries;
            if (sum > 4) {
              ++predictor.hits;
            }
          }
        }
        add_stage_elapsed(collect_timing, pipeline_low_lookup_seconds, low_started);

        const auto high_started = stage_start(collect_timing);
        for (int64_t window_id = begin; window_id < end; ++window_id) {
          const int64_t position = window_starts[static_cast<size_t>(window_id)] + depth;
          if (position >= n) {
            continue;
          }
          const size_t local_window = static_cast<size_t>(window_id - begin);
          uint64_t ctx17_base_key = 0;
          size_t ctx17_base_slot = 0;
          bool have_ctx17_base = false;
          for (size_t high_index = 0; high_index < pipeline_high_predictors.size(); ++high_index) {
            const size_t predictor_index =
                static_cast<size_t>(pipeline_high_predictors[high_index]);
            StrictPredictor &predictor = predictors[predictor_index];
            StrictModel &model = models[static_cast<size_t>(predictor.model_index)];
            const PipelineQueryAddress &query =
                query_buffer[local_window * pipeline_high_predictors.size() + high_index];
            const size_t slot = local_window * predictor_count + predictor_index;
            uint32_t sum = 0;
            if (predictor.edit && model.context_len == 17 && have_ctx17_base &&
                query.key == ctx17_base_key) {
              for (int64_t symbol = 0; symbol < 4; ++symbol) {
                const uint32_t base_freq =
                    pipeline_freqs[ctx17_base_slot][static_cast<size_t>(symbol)];
                const uint32_t count = (base_freq - 1U) / model.alpha_den;
                pipeline_freqs[slot][static_cast<size_t>(symbol)] =
                    1U + predictor.alpha_den * count;
                sum += pipeline_freqs[slot][static_cast<size_t>(symbol)];
              }
            } else {
              sum = strict_model_freqs_planned(
                  model,
                  query.key,
                  query.hashed,
                  predictor.alpha_den,
                  pipeline_freqs[slot]);
            }
            pipeline_sums[slot] = sum;
            ++predictor.queries;
            if (sum > 4) {
              ++predictor.hits;
            }
            if (predictor.edit) {
              model.edit_best_ids[static_cast<size_t>(window_id)] =
                  static_cast<int8_t>(strict_best_id(pipeline_freqs[slot], sum));
            } else if (model.context_len == 17) {
              ctx17_base_key = query.key;
              ctx17_base_slot = slot;
              have_ctx17_base = true;
              pipeline_ctx17_base_hashes[static_cast<size_t>(window_id)] = query.hashed;
            }
          }
        }
        add_stage_elapsed(collect_timing, pipeline_high_lookup_seconds, high_started);

        const auto fusion_started = stage_start(collect_timing);
        for (int64_t window_id = begin; window_id < end; ++window_id) {
          const int64_t position = window_starts[static_cast<size_t>(window_id)] + depth;
          if (position >= n) {
            continue;
          }
          const uint8_t target = static_cast<uint8_t>(targets[position]);
          const size_t local_window = static_cast<size_t>(window_id - begin);
          const size_t slot_base = local_window * predictor_count;
          double *weights = window_weights.data() + static_cast<size_t>(window_id) * predictor_count;
          std::fill(mixed_float.begin(), mixed_float.end(), 0.0);
          for (size_t predictor_index = 0; predictor_index < predictor_count; ++predictor_index) {
            const size_t slot = slot_base + predictor_index;
            predictor_target_probs[predictor_index] = std::max(
                static_cast<double>(pipeline_freqs[slot][target]) /
                    static_cast<double>(pipeline_sums[slot]),
                1e-300);
            if (!profile_skip_mixture) {
              const double factor = weights[predictor_index] /
                  static_cast<double>(pipeline_sums[slot]);
              for (int64_t symbol = 0; symbol < 4; ++symbol) {
                mixed_float[static_cast<size_t>(symbol)] +=
                    static_cast<double>(pipeline_freqs[slot][static_cast<size_t>(symbol)]) * factor;
              }
            }
          }

          double qbits = 0.0;
          double fbits = 0.0;
          if (!profile_skip_mixture) {
            mx_sum = 0;
            for (int64_t symbol = 0; symbol < 4; ++symbol) {
              const double probability = std::max(mixed_float[static_cast<size_t>(symbol)], 0.0);
              mx_freqs[static_cast<size_t>(symbol)] =
                  1U + static_cast<uint32_t>(std::floor(probability * static_cast<double>(GECO2_MX_PMODEL)));
              mx_sum += mx_freqs[static_cast<size_t>(symbol)];
            }
            const double qprob = std::max(
                static_cast<double>(mx_freqs[static_cast<size_t>(target)]) /
                    static_cast<double>(mx_sum),
                1e-300);
            qbits = -std::log2(qprob);
            if (COMPUTE_MIXED_FLOAT_BITS) {
              fbits = -std::log2(std::max(mixed_float[static_cast<size_t>(target)], 1e-300));
            }
          } else {
            mx_sum = 4;
            mx_freqs = {1, 1, 1, 1};
          }
          if (!summary_only) {
            bpb[position] = qbits;
          }
          quantized_bits += qbits;
          mixed_float_bits += fbits;
          if (return_probabilities) {
            double *row = probabilities + position * 4;
            for (int64_t symbol = 0; symbol < 4; ++symbol) {
              row[symbol] = static_cast<double>(mx_freqs[static_cast<size_t>(symbol)]) /
                  static_cast<double>(mx_sum);
            }
          }

          if (!profile_skip_weight_update) {
            double weight_total = 0.0;
            for (size_t predictor_index = 0; predictor_index < predictor_count; ++predictor_index) {
              weights[predictor_index] =
                  std::pow(weights[predictor_index], predictors[predictor_index].gamma) *
                  predictor_target_probs[predictor_index];
              weight_total += weights[predictor_index];
            }
            if (!(weight_total > 0.0) || !std::isfinite(weight_total)) {
              std::fill(weights, weights + predictor_count, 1.0 / static_cast<double>(predictor_count));
            } else {
              for (size_t predictor_index = 0; predictor_index < predictor_count; ++predictor_index) {
                weights[predictor_index] /= weight_total;
              }
            }
          }
        }
        add_stage_elapsed(collect_timing, pipeline_fusion_seconds, fusion_started);
      }
      add_stage_elapsed(collect_timing, prediction_weight_seconds, prediction_started);
    }

    if (active_count == 0) {
      continue;
    }
    ++depth_count;

    if (cache_pipeline) {
      const auto base_update_started = stage_start(collect_timing);
      StrictModel &ctx17_model = models[static_cast<size_t>(pipeline_ctx17_model_index)];
      const auto update_prepare_started = stage_start(collect_timing);
      for (int64_t window_id = 0; window_id < window_count; ++window_id) {
        const int64_t position = window_starts[static_cast<size_t>(window_id)] + depth;
        if (position >= n) {
          continue;
        }
        const uint8_t target = static_cast<uint8_t>(targets[position]);
        const uint8_t old_symbol = depth >= ctx17_model.context_len
            ? static_cast<uint8_t>(targets[position - ctx17_model.context_len])
            : uint8_t{0};
        uint64_t &ir_key = ctx17_model.pidx_ir[static_cast<size_t>(window_id)];
        ir_key = (ir_key >> 2) +
            (static_cast<uint64_t>(complement_symbol(target)) * ctx17_model.multiplier);
        pipeline_ctx17_ir_keys[static_cast<size_t>(window_id)] = ir_key;
        pipeline_ctx17_ir_hashes[static_cast<size_t>(window_id)] = zhash(ir_key);
        ++ctx17_model.ir_updates;
      }
      add_stage_elapsed(collect_timing, pipeline_update_prepare_seconds, update_prepare_started);

      const auto update_commit_started = stage_start(collect_timing);
      const int64_t update_block_count =
          (window_count + pipeline_block_windows - 1) / pipeline_block_windows;
      for (int64_t block_index = 0; block_index < update_block_count; ++block_index) {
        const int64_t begin = block_index * pipeline_block_windows;
        const int64_t end = std::min<int64_t>(window_count, begin + pipeline_block_windows);
        const int64_t next_begin = end;
        const int64_t next_end = std::min<int64_t>(window_count, next_begin + pipeline_block_windows);
        for (int64_t future_window = next_begin; future_window < next_end; ++future_window) {
          const int64_t future_position = window_starts[static_cast<size_t>(future_window)] + depth;
          if (future_position >= n) {
            continue;
          }
          ctx17_model.hash.prefetch_hashed(
              pipeline_ctx17_base_hashes[static_cast<size_t>(future_window)], true);
          ctx17_model.hash.prefetch_hashed(
              pipeline_ctx17_ir_hashes[static_cast<size_t>(future_window)], true);
          for (StrictModel &model : models) {
            if (model.dense.enabled && model.context_len == 13) {
              pipeline_prefetch(
                  model,
                  model.pidx[static_cast<size_t>(future_window)],
                  0,
                  true);
            }
          }
        }

        for (int64_t window_id = begin; window_id < end; ++window_id) {
          const int64_t position = window_starts[static_cast<size_t>(window_id)] + depth;
          if (position >= n) {
            continue;
          }
          const uint8_t target = static_cast<uint8_t>(targets[position]);
          for (StrictModel &model : models) {
            const uint64_t key = model.pidx[static_cast<size_t>(window_id)];
            if (model.context_len == 17) {
              model.hash.update_hashed(
                  pipeline_ctx17_base_hashes[static_cast<size_t>(window_id)], target);
              model.hash.update_hashed(
                  pipeline_ctx17_ir_hashes[static_cast<size_t>(window_id)],
                  complement_symbol(depth >= model.context_len
                      ? static_cast<uint8_t>(targets[position - model.context_len])
                      : uint8_t{0}));
              continue;
            }
            strict_model_update(model, key, target);
            if (model.ir) {
              const uint8_t old_symbol = depth >= model.context_len
                  ? static_cast<uint8_t>(targets[position - model.context_len])
                  : uint8_t{0};
              const uint8_t ir_symbol = complement_symbol(old_symbol);
              uint64_t &ir_key = model.pidx_ir[static_cast<size_t>(window_id)];
              ir_key = (ir_key >> 2) +
                  (static_cast<uint64_t>(complement_symbol(target)) * model.multiplier);
              strict_model_update(model, ir_key, ir_symbol);
              ++model.ir_updates;
            }
          }
        }
      }
      add_stage_elapsed(collect_timing, pipeline_update_commit_seconds, update_commit_started);
      add_stage_elapsed(collect_timing, base_counter_update_seconds, base_update_started);

      const auto edit_update_started = stage_start(collect_timing);
      for (int64_t window_id = 0; window_id < window_count; ++window_id) {
        const int64_t position = window_starts[static_cast<size_t>(window_id)] + depth;
        if (position >= n) {
          continue;
        }
        const uint8_t target = static_cast<uint8_t>(targets[position]);
        for (StrictModel &model : models) {
          if (model.edits == 0) {
            continue;
          }
          const int64_t edit_predictor_index =
              edit_predictor_by_model[static_cast<size_t>(&model - models.data())];
          uint8_t edited_symbol = target;
          if (edit_predictor_index >= 0) {
            edited_symbol = strict_correct_edit_symbol(
                model,
                window_id,
                static_cast<int64_t>(model.edit_best_ids[static_cast<size_t>(window_id)]),
                target);
          }
          const size_t edit_ring_slot =
              static_cast<size_t>(window_id * model.context_len + depth % model.context_len);
          const uint8_t old_edit_symbol = depth >= model.context_len
              ? model.edit_symbols[edit_ring_slot]
              : uint8_t{0};
          uint64_t &edit_key = model.edit_pidx[static_cast<size_t>(window_id)];
          edit_key = ((edit_key - static_cast<uint64_t>(old_edit_symbol) * model.multiplier) << 2) +
              edited_symbol;
          edit_key &= (model.n_contexts - 1);
          model.edit_symbols[edit_ring_slot] = edited_symbol;
        }
      }
      add_stage_elapsed(collect_timing, edit_state_update_seconds, edit_update_started);

      const auto context_update_started = stage_start(collect_timing);
      for (int64_t window_id = 0; window_id < window_count; ++window_id) {
        const int64_t position = window_starts[static_cast<size_t>(window_id)] + depth;
        if (position >= n) {
          continue;
        }
        const uint8_t target = static_cast<uint8_t>(targets[position]);
        for (StrictModel &model : models) {
          const uint8_t old_symbol = depth >= model.context_len
              ? static_cast<uint8_t>(targets[position - model.context_len])
              : uint8_t{0};
          uint64_t &key = model.pidx[static_cast<size_t>(window_id)];
          key = ((key - static_cast<uint64_t>(old_symbol) * model.multiplier) << 2) + target;
          key &= (model.n_contexts - 1);
        }
      }
      add_stage_elapsed(collect_timing, context_state_update_seconds, context_update_started);
    }

    if (collect_timing && (depth + 1 == max_window_len || (depth + 1) % snapshot_stride == 0)) {
      const auto snapshot_started = stage_start(collect_timing);
      weight_history.append(strict_window_weight_snapshot(depth + 1, predictors, window_weights, window_count));
      add_stage_elapsed(collect_timing, weight_snapshot_seconds, snapshot_started);
    }
  }

  if (!summary_only && emit_index != n) {
    throw std::runtime_error("internal error: emitted symbol count does not match input length");
  }
  const auto finished = Clock::now();

  py::list model_list;
  for (const StrictModel &model : models) {
    py::dict item;
    item["label"] = model.label;
    item["context_len"] = model.context_len;
    item["alpha_den"] = model.alpha_den;
    item["gamma"] = model.gamma;
    item["ir"] = model.ir;
    item["edits"] = model.edits;
    item["edit_alpha_den"] = model.edit_alpha_den;
    item["edit_gamma"] = model.edit_gamma;
    item["dense"] = model.dense.enabled;
    item["hash_slots"] = model.hash_slots;
    item["storage_mmap"] = model.dense.enabled
        ? model.dense.counts.mapped()
        : model.hash.buckets.mapped();
    item["hugepage_advised"] = model.dense.enabled
        ? model.dense.counts.hugepage_advised()
        : model.hash.buckets.hugepage_advised();
    item["storage_bytes"] = static_cast<int64_t>(model.dense.enabled
        ? model.dense.counts.requested_bytes()
        : model.hash.buckets.requested_bytes());
    item["storage_mapped_bytes"] = static_cast<int64_t>(model.dense.enabled
        ? model.dense.counts.mapped_bytes()
        : model.hash.buckets.mapped_bytes());
    item["storage_alignment_bytes"] = static_cast<int64_t>(model.dense.enabled
        ? model.dense.counts.mapped_alignment_bytes()
        : model.hash.buckets.mapped_alignment_bytes());
    item["storage_2mb_aligned"] = model.dense.enabled
        ? model.dense.counts.mapped_2mb_aligned()
        : model.hash.buckets.mapped_2mb_aligned();
    item["populate_requested"] = model.dense.enabled
        ? model.dense.counts.populate_requested()
        : model.hash.buckets.populate_requested();
    item["populate_succeeded"] = model.dense.enabled
        ? model.dense.counts.populate_succeeded()
        : model.hash.buckets.populate_succeeded();
    item["populate_mode"] = model.dense.enabled
        ? model.dense.counts.populate_mode()
        : model.hash.buckets.populate_mode();
    item["ir_updates"] = model.ir_updates;
    item["edit_hits"] = model.edit_hits;
    item["edit_fails"] = model.edit_fails;
    model_list.append(item);
  }
  py::list predictor_list;
  py::dict final_weights;
  py::dict hit_rates;
  const std::vector<double> final_mean_weights =
      strict_average_window_weights(window_weights, window_count, predictor_count);
  for (size_t predictor_index = 0; predictor_index < predictors.size(); ++predictor_index) {
    const StrictPredictor &predictor = predictors[predictor_index];
    py::dict item;
    item["label"] = predictor.label;
    item["model_index"] = predictor.model_index;
    item["edit"] = predictor.edit;
    item["alpha_den"] = predictor.alpha_den;
    item["gamma"] = predictor.gamma;
    item["queries"] = predictor.queries;
    item["hits"] = predictor.hits;
    item["hit_rate"] = predictor.queries > 0 && predictor.hits >= 0
        ? static_cast<double>(predictor.hits) / static_cast<double>(predictor.queries)
        : 0.0;
    predictor_list.append(item);
    final_weights[py::str(predictor.label)] = final_mean_weights[predictor_index];
    hit_rates[py::str(predictor.label)] = predictor.queries > 0 && predictor.hits >= 0
        ? static_cast<double>(predictor.hits) / static_cast<double>(predictor.queries)
        : 0.0;
  }

  const double compute_seconds = elapsed_seconds(started, finished);
  const double timed_stage_seconds =
      setup_seconds +
      emit_order_seconds +
      prediction_weight_seconds +
      base_counter_update_seconds +
      edit_state_update_seconds +
      context_state_update_seconds +
      weight_snapshot_seconds;
  py::dict timing;
  timing["setup_seconds"] = setup_seconds;
  timing["emit_order_seconds"] = emit_order_seconds;
  timing["prediction_and_weight_seconds"] = prediction_weight_seconds;
  timing["pipeline_address_prepare_seconds"] = pipeline_address_prepare_seconds;
  timing["pipeline_low_lookup_seconds"] = pipeline_low_lookup_seconds;
  timing["pipeline_high_lookup_seconds"] = pipeline_high_lookup_seconds;
  timing["pipeline_fusion_seconds"] = pipeline_fusion_seconds;
  timing["pipeline_update_prepare_seconds"] = pipeline_update_prepare_seconds;
  timing["pipeline_update_commit_seconds"] = pipeline_update_commit_seconds;
  timing["base_counter_update_seconds"] = base_counter_update_seconds;
  timing["edit_state_update_seconds"] = edit_state_update_seconds;
  timing["context_state_update_seconds"] = context_state_update_seconds;
  timing["weight_snapshot_seconds"] = weight_snapshot_seconds;
  timing["timed_stage_seconds"] = timed_stage_seconds;
  timing["untimed_seconds"] = std::max(0.0, compute_seconds - timed_stage_seconds);

  py::dict metadata;
  metadata["backend"] = "fast_cpp";
  metadata["update_mode"] = update_mode;
  metadata["profile_mode"] = profile_mode;
  metadata["threads_requested"] = threads;
  metadata["threads_effective"] = threads_effective;
  metadata["threads_used_for_prediction"] = 1;
  metadata["prediction_parallel"] = false;
  metadata["dense_update_mode"] = "serial_counter_max_halving";
  metadata["hash_update_mode"] = "serial_inline";
  metadata["update_iteration_order"] = "window_major_block_prefetch";
  metadata["parallel_predictor_hit_counts"] = "exact";
  metadata["preset"] = "nc_prefix";
  metadata["algorithm"] = "geco2_level10_per_window_weights";
  metadata["geco2_level"] = 10;
  metadata["base_count"] = n;
  metadata["summary_only"] = summary_only;
  metadata["artifact_tensor_bytes"] = summary_only
      ? int64_t{0}
      : static_cast<int64_t>(
            bpb_tensor.nbytes() + target_tensor.nbytes() + emit_order_tensor.nbytes() + probabilities_tensor.nbytes());
  metadata["edit_state_bytes"] = static_cast<int64_t>(window_count) * 2 +
      static_cast<int64_t>(window_count) * 17;
  metadata["window_bases"] = window_bases;
  metadata["window_count"] = window_count;
  metadata["depth_count"] = depth_count;
  metadata["max_window_len"] = max_window_len;
  metadata["alphabet"] = "ACGT";
  metadata["model_count"] = static_cast<int64_t>(models.size());
  metadata["predictor_count"] = static_cast<int64_t>(predictors.size());
  metadata["models"] = model_list;
  metadata["predictors"] = predictor_list;
  metadata["disable_edit_experts"] = disable_edit_experts;
  metadata["disable_ir"] = disable_ir;
  metadata["emit_order"] = "depth_major_nonoverlap_windows";
  metadata["updates"] = window_count == 1
      ? "serial GECO2 order: predict, quantize, weight update, counter/IR/SUBS update"
      : "depth-major GECO2-style order: all predictions at a depth precede counter updates at that depth";
  metadata["fusion_mode"] = window_count == 1
      ? "geco2_serial_weight_power_times_model_probability"
      : "geco2_per_window_weight_power_times_model_probability_depth_major_shared_counters";
  metadata["weight_power_mode"] = "std_pow_exact";
  metadata["mixed_float_bits_computed"] = COMPUTE_MIXED_FLOAT_BITS;
  metadata["weight_scope"] = window_count == 1 ? "single_serial_stream" : "per_window_local_weights";
  metadata["weight_history_summary"] = collect_timing ? "mean of per-window weights" : "not collected";
  metadata["context_key_mode"] = "geco2_acgt_2bit_full_context_zero_padded";
  metadata["reverse_complement_update"] = !disable_ir;
  metadata["edit_experts"] = !disable_edit_experts;
  metadata["edit_counter_sharing"] = "edit predictors query the owning base model counter table";
  metadata["dense_context_rule"] = "ctx < 15";
  metadata["dense_counter_mode"] = "uint16_default_max_count_halving";
  metadata["hash_bucket_count"] = effective_hash_buckets;
  metadata["hash_bucket_count_requested"] = hash_bucket_count;
  metadata["hash_counter_mode"] = "geco2_4bit_packed_counter_halve_on_15";
  metadata["hash_replacement"] = "geco2_style_bucket_cursor_oldest_replacement";
  metadata["cache_pipeline"] = cache_pipeline;
  metadata["pipeline_block_windows"] = pipeline_block_windows;
  metadata["pipeline_prefetch_hints"] = "ctx11=locality2,ctx13_ctx17=locality1";
  metadata["pipeline_scratch_bytes"] = static_cast<int64_t>(
      pipeline_query_buffers[0].capacity() * sizeof(PipelineQueryAddress) +
      pipeline_query_buffers[1].capacity() * sizeof(PipelineQueryAddress) +
      pipeline_freqs.capacity() * sizeof(std::array<uint32_t, 4>) +
      pipeline_sums.capacity() * sizeof(uint32_t) +
      pipeline_ctx17_base_hashes.capacity() * sizeof(uint64_t) +
      pipeline_ctx17_ir_hashes.capacity() * sizeof(uint64_t) +
      pipeline_ctx17_ir_keys.capacity() * sizeof(uint64_t));
  metadata["query_order"] = "window_block_double_buffer_prefetch";
  metadata["hash_bucket_bytes"] = static_cast<int64_t>(sizeof(StrictHashBucket));
  metadata["hash_bucket_cacheline_aligned"] = alignof(StrictHashBucket) == 64;
  metadata["large_table_allocator"] = "anonymous_mmap_2mb_aligned_with_madvise_hugepage_fallback_contiguous";
  metadata["large_table_alignment_bytes"] = static_cast<int64_t>(LARGE_TABLE_ALIGNMENT_BYTES);
  metadata["populate_tables_requested"] = env_flag_enabled("DNA_COMPRESS_NC_PREFIX_POPULATE_TABLES");
#ifdef __linux__
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    metadata["process_peak_rss_bytes"] = static_cast<int64_t>(usage.ru_maxrss) * 1024;
  }
#endif
  metadata["mixed_float_bits"] = mixed_float_bits;
  metadata["mixed_float_bits_per_base"] = mixed_float_bits / static_cast<double>(std::max<int64_t>(n, 1));
  metadata["geco2_quantized_bits"] = quantized_bits;
  metadata["geco2_quantized_bits_per_base"] = quantized_bits / static_cast<double>(std::max<int64_t>(n, 1));
  metadata["theoretical_bits"] = quantized_bits;
  metadata["theoretical_bits_per_base"] = quantized_bits / static_cast<double>(std::max<int64_t>(n, 1));
  metadata["final_order_weights"] = final_weights;
  metadata["expert_hit_rates"] = hit_rates;
  metadata["weight_history"] = weight_history;
  metadata["fine_timing_enabled"] = collect_timing;
  metadata["compute_seconds"] = compute_seconds;
  metadata["timing"] = timing;

  py::dict result;
  result["probabilities"] = probabilities_tensor;
  result["bpb"] = bpb_tensor;
  result["target_symbols"] = target_tensor;
  result["emit_order"] = emit_order_tensor;
  result["metadata"] = metadata;
  return result;
}

py::dict compute_nc_prefix(
    const at::Tensor &symbols,
    const at::Tensor &orders_tensor,
    int64_t window_bases,
    int64_t vocab_size,
    double alpha,
    double eta,
    double local_beta,
    bool use_local_stats,
    bool return_probabilities,
    bool summary_only,
    int64_t preset_id,
    int64_t geco2_level,
    bool disable_edit_experts,
    bool disable_ir,
    int64_t dense_max_entries,
    int64_t hash_bucket_count,
    int64_t hash_slots,
    const std::string &update_mode,
    const std::string &profile_mode,
    int64_t threads) {
  (void)alpha;
  (void)eta;
  (void)local_beta;
  (void)use_local_stats;
  (void)preset_id;
  (void)dense_max_entries;
  (void)hash_slots;
  if (symbols.device().is_cuda() || orders_tensor.device().is_cuda()) {
    throw std::runtime_error("fast nc_prefix expects CPU tensors");
  }
  if (symbols.dim() != 1 || orders_tensor.dim() != 1) {
    throw std::runtime_error("symbols and orders must be 1D tensors");
  }
  at::Tensor symbols_contig = symbols.contiguous();
  if (geco2_level != 10) {
    throw std::runtime_error("nc_prefix currently supports only GECO2 level 10");
  }

  if (symbols_contig.scalar_type() == at::kLong) {
    return compute_current_nc_prefix_impl<int64_t>(
        symbols_contig,
        window_bases,
        vocab_size,
        return_probabilities,
        summary_only,
        hash_bucket_count,
        disable_edit_experts,
        disable_ir,
        update_mode,
        profile_mode,
        threads);
  }
  if (symbols_contig.scalar_type() == at::kShort) {
    return compute_current_nc_prefix_impl<int16_t>(
        symbols_contig,
        window_bases,
        vocab_size,
        return_probabilities,
        summary_only,
        hash_bucket_count,
        disable_edit_experts,
        disable_ir,
        update_mode,
        profile_mode,
        threads);
  }
  if (symbols_contig.scalar_type() == at::kInt) {
    return compute_current_nc_prefix_impl<int32_t>(
        symbols_contig,
        window_bases,
        vocab_size,
        return_probabilities,
        summary_only,
        hash_bucket_count,
        disable_edit_experts,
        disable_ir,
        update_mode,
        profile_mode,
        threads);
  }
  throw std::runtime_error("symbols must be int16, int32, or int64");
}

py::dict compute_nc_prefix_current(
    const at::Tensor &symbols,
    int64_t window_bases,
    int64_t vocab_size,
    bool return_probabilities,
    bool summary_only,
    int64_t hash_bucket_count) {
  if (symbols.device().is_cuda()) {
    throw std::runtime_error("fast nc_prefix expects a CPU tensor");
  }
  if (symbols.dim() != 1) {
    throw std::runtime_error("symbols must be a 1D tensor");
  }
  at::Tensor symbols_contig = symbols.contiguous();
  if (symbols_contig.scalar_type() == at::kLong) {
    return compute_current_nc_prefix_impl<int64_t>(
        symbols_contig,
        window_bases,
        vocab_size,
        return_probabilities,
        summary_only,
        hash_bucket_count,
        false,
        false,
        "cache_pipeline",
        "normal",
        0);
  }
  if (symbols_contig.scalar_type() == at::kShort) {
    return compute_current_nc_prefix_impl<int16_t>(
        symbols_contig,
        window_bases,
        vocab_size,
        return_probabilities,
        summary_only,
        hash_bucket_count,
        false,
        false,
        "cache_pipeline",
        "normal",
        0);
  }
  if (symbols_contig.scalar_type() == at::kInt) {
    return compute_current_nc_prefix_impl<int32_t>(
        symbols_contig,
        window_bases,
        vocab_size,
        return_probabilities,
        summary_only,
        hash_bucket_count,
        false,
        false,
        "cache_pipeline",
        "normal",
        0);
  }
  throw std::runtime_error("symbols must be int16, int32, or int64");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<FusedNcPrefixStreamingEncoder>(m, "FusedNcPrefixStreamingEncoder")
      .def(
          py::init<int64_t, int64_t, int64_t, int64_t, double, double, bool, bool>(),
          py::arg("window_count"),
          py::arg("window_bases"),
          py::arg("hash_bucket_count"),
          py::arg("arithmetic_frequency_total"),
          py::arg("fusion_eta"),
          py::arg("initial_lm_weight"),
          py::arg("encode_arithmetic"),
          py::arg("collect_diagnostics") = true)
      .def("encode_base_step", &FusedNcPrefixStreamingEncoder::encode_base_step)
      .def("encode_token_step", &FusedNcPrefixStreamingEncoder::encode_token_step)
      .def("finish", &FusedNcPrefixStreamingEncoder::finish)
      .def("metadata", &FusedNcPrefixStreamingEncoder::metadata);
  m.def(
      "compute_nc_prefix_current",
      &compute_nc_prefix_current,
      py::arg("symbols"),
      py::arg("window_bases"),
      py::arg("vocab_size"),
      py::arg("return_probabilities"),
      py::arg("summary_only") = false,
      py::arg("hash_bucket_count") = DEFAULT_HASH_BUCKETS);
  m.def(
      "compute_nc_prefix",
      &compute_nc_prefix,
      py::arg("symbols"),
      py::arg("orders"),
      py::arg("window_bases"),
      py::arg("vocab_size"),
      py::arg("alpha"),
      py::arg("eta"),
      py::arg("local_beta"),
      py::arg("use_local_stats"),
      py::arg("return_probabilities"),
      py::arg("summary_only") = false,
      py::arg("preset_id") = NC_PREFIX_PRESET_GECO2_STRICT,
      py::arg("geco2_level") = 10,
      py::arg("disable_edit_experts") = false,
      py::arg("disable_ir") = false,
      py::arg("dense_max_entries") = DENSE_CONTEXT_MAX_ENTRIES,
      py::arg("hash_bucket_count") = DEFAULT_HASH_BUCKETS,
      py::arg("hash_slots") = DEFAULT_HASH_SLOTS,
      py::arg("update_mode") = "cache_pipeline",
      py::arg("profile_mode") = "normal",
      py::arg("threads") = 0);
}
