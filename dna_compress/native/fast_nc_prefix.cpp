#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr uint64_t CONTEXT_HASH_MULTIPLIER = 1315423911ULL;
constexpr uint64_t CONTEXT_HASH_SYMBOL_SALT = 2654435761ULL;
constexpr uint64_t CONTEXT_HASH_MASK = (uint64_t{1} << 63) - 1;
constexpr int64_t DENSE_CONTEXT_MAX_ENTRIES = 1000000;
constexpr uint16_t COUNTER_MAX = std::numeric_limits<uint16_t>::max();
constexpr int64_t DEFAULT_HASH_BUCKETS = 1 << 16;
constexpr int64_t DEFAULT_HASH_SLOTS = 4;
constexpr int64_t NC_PREFIX_PRESET_LEGACY = 0;
constexpr int64_t NC_PREFIX_PRESET_GECO2_PARALLEL = 1;
constexpr int64_t NC_PREFIX_PRESET_GECO2_STRICT = 2;
constexpr int64_t GECO2_HASH_TABLE_BEGIN_CTX = 15;
constexpr int64_t GECO2_HASH_SIZE = 33554471;
constexpr uint16_t GECO2_HASH_COUNTER_MAX = 15;
constexpr uint16_t GECO2_MX_PMODEL = 65535;

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point &start, const Clock::time_point &end) {
  return std::chrono::duration<double>(end - start).count();
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

void increment_counter_row(uint16_t *counts, int64_t vocab_size, int64_t target) {
  if (counts[target] >= COUNTER_MAX) {
    for (int64_t symbol = 0; symbol < vocab_size; ++symbol) {
      counts[symbol] >>= 1;
    }
  }
  ++counts[target];
}

double row_sum(const uint16_t *counts, int64_t vocab_size) {
  double total = 0.0;
  for (int64_t symbol = 0; symbol < vocab_size; ++symbol) {
    total += static_cast<double>(counts[symbol]);
  }
  return total;
}

struct DenseTable {
  bool enabled = false;
  int64_t entries = 0;
  int64_t vocab_size = 0;
  std::vector<uint16_t> counts;

  DenseTable() = default;

  DenseTable(int64_t entry_count, int64_t vocab) : enabled(true), entries(entry_count), vocab_size(vocab) {
    counts.assign(static_cast<size_t>(entries * vocab_size), 0);
  }

  uint16_t *row(int64_t key) {
    return counts.data() + static_cast<size_t>(key * vocab_size);
  }

  const uint16_t *row(int64_t key) const {
    return counts.data() + static_cast<size_t>(key * vocab_size);
  }
};

struct HashTable {
  int64_t bucket_count = 0;
  int64_t slots = 0;
  int64_t vocab_size = 0;
  std::vector<uint64_t> keys;
  std::vector<uint8_t> used;
  std::vector<uint8_t> next_slot;
  std::vector<uint16_t> counts;

  HashTable() = default;

  HashTable(int64_t buckets, int64_t slot_count, int64_t vocab)
      : bucket_count(std::max<int64_t>(1, buckets)),
        slots(std::max<int64_t>(1, slot_count)),
        vocab_size(vocab) {
    const size_t entry_count = static_cast<size_t>(bucket_count * slots);
    keys.assign(entry_count, 0);
    used.assign(entry_count, 0);
    next_slot.assign(static_cast<size_t>(bucket_count), 0);
    counts.assign(entry_count * static_cast<size_t>(vocab_size), 0);
  }

  int64_t bucket_for(uint64_t key) const {
    return static_cast<int64_t>(zhash(key) % static_cast<uint64_t>(bucket_count));
  }

  const uint16_t *find(uint64_t key) const {
    if (bucket_count <= 0) {
      return nullptr;
    }
    const int64_t bucket = bucket_for(key);
    const int64_t base = bucket * slots;
    for (int64_t slot = 0; slot < slots; ++slot) {
      const int64_t index = base + slot;
      if (used[static_cast<size_t>(index)] && keys[static_cast<size_t>(index)] == key) {
        return counts.data() + static_cast<size_t>(index * vocab_size);
      }
    }
    return nullptr;
  }

  uint16_t *find_or_insert(uint64_t key) {
    if (bucket_count <= 0) {
      throw std::runtime_error("attempted to insert into an uninitialized hash table");
    }
    const int64_t bucket = bucket_for(key);
    const int64_t base = bucket * slots;
    for (int64_t slot = 0; slot < slots; ++slot) {
      const int64_t index = base + slot;
      if (used[static_cast<size_t>(index)] && keys[static_cast<size_t>(index)] == key) {
        return counts.data() + static_cast<size_t>(index * vocab_size);
      }
    }
    int64_t slot_to_write = -1;
    for (int64_t slot = 0; slot < slots; ++slot) {
      const int64_t index = base + slot;
      if (!used[static_cast<size_t>(index)]) {
        slot_to_write = slot;
        break;
      }
    }
    if (slot_to_write < 0) {
      auto &cursor = next_slot[static_cast<size_t>(bucket)];
      slot_to_write = static_cast<int64_t>(cursor);
      cursor = static_cast<uint8_t>((cursor + 1) % slots);
    }
    const int64_t index = base + slot_to_write;
    used[static_cast<size_t>(index)] = 1;
    keys[static_cast<size_t>(index)] = key;
    uint16_t *row_ptr = counts.data() + static_cast<size_t>(index * vocab_size);
    std::fill(row_ptr, row_ptr + vocab_size, uint16_t{0});
    return row_ptr;
  }
};

struct LocalRow {
  std::vector<uint16_t> counts;
};

using LocalTables = std::vector<std::vector<std::unordered_map<uint64_t, LocalRow>>>;

const uint16_t *local_find(const LocalTables &tables, int64_t window_id, int64_t context_len, uint64_t key) {
  const auto &table = tables[static_cast<size_t>(window_id)][static_cast<size_t>(context_len)];
  auto iter = table.find(key);
  if (iter == table.end()) {
    return nullptr;
  }
  return iter->second.counts.data();
}

uint16_t *local_find_or_insert(LocalTables &tables, int64_t window_id, int64_t context_len, uint64_t key, int64_t vocab_size) {
  auto &table = tables[static_cast<size_t>(window_id)][static_cast<size_t>(context_len)];
  auto iter = table.find(key);
  if (iter == table.end()) {
    LocalRow row;
    row.counts.assign(static_cast<size_t>(vocab_size), 0);
    iter = table.emplace(key, std::move(row)).first;
  }
  return iter->second.counts.data();
}

std::vector<DenseTable> make_dense_tables(int64_t max_order, int64_t vocab_size, int64_t max_entries) {
  std::vector<DenseTable> tables(static_cast<size_t>(max_order + 1));
  int64_t entries = 1;
  for (int64_t context_len = 0; context_len <= max_order; ++context_len) {
    if (context_len == 0) {
      entries = 1;
    } else if (entries <= max_entries) {
      entries *= vocab_size;
    }
    if (entries <= max_entries) {
      tables[static_cast<size_t>(context_len)] = DenseTable(entries, vocab_size);
    }
  }
  return tables;
}

std::vector<HashTable> make_hash_tables(
    int64_t max_order,
    int64_t vocab_size,
    int64_t buckets,
    int64_t slots,
    const std::vector<DenseTable> &dense_tables) {
  std::vector<HashTable> tables(static_cast<size_t>(max_order + 1));
  for (int64_t context_len = 0; context_len <= max_order; ++context_len) {
    if (!dense_tables[static_cast<size_t>(context_len)].enabled) {
      tables[static_cast<size_t>(context_len)] = HashTable(buckets, slots, vocab_size);
    }
  }
  return tables;
}

const uint16_t *global_find(
    const std::vector<DenseTable> &dense_tables,
    const std::vector<HashTable> &hash_tables,
    const std::vector<uint64_t> &hash_keys,
    const std::vector<int64_t> &dense_keys,
    int64_t context_len) {
  const auto &dense = dense_tables[static_cast<size_t>(context_len)];
  if (dense.enabled) {
    return dense.row(dense_keys[static_cast<size_t>(context_len)]);
  }
  return hash_tables[static_cast<size_t>(context_len)].find(hash_keys[static_cast<size_t>(context_len)]);
}

uint16_t *global_find_or_insert(
    std::vector<DenseTable> &dense_tables,
    std::vector<HashTable> &hash_tables,
    const std::vector<uint64_t> &hash_keys,
    const std::vector<int64_t> &dense_keys,
    int64_t context_len) {
  auto &dense = dense_tables[static_cast<size_t>(context_len)];
  if (dense.enabled) {
    return dense.row(dense_keys[static_cast<size_t>(context_len)]);
  }
  return hash_tables[static_cast<size_t>(context_len)].find_or_insert(hash_keys[static_cast<size_t>(context_len)]);
}

double counts_to_distribution(
    const uint16_t *counts,
    int64_t vocab_size,
    double alpha,
    const std::vector<double> &uniform,
    std::vector<double> &output) {
  if (counts == nullptr) {
    output = uniform;
    return 0.0;
  }
  const double total = row_sum(counts, vocab_size);
  const double denominator = std::max(total + alpha * static_cast<double>(vocab_size), 1e-300);
  for (int64_t symbol = 0; symbol < vocab_size; ++symbol) {
    output[static_cast<size_t>(symbol)] = (static_cast<double>(counts[symbol]) + alpha) / denominator;
  }
  return total;
}

void advance_hash_keys(std::vector<uint64_t> &keys, int64_t target) {
  for (int64_t context_len = static_cast<int64_t>(keys.size()) - 1; context_len >= 1; --context_len) {
    keys[static_cast<size_t>(context_len)] = mix_context_key(keys[static_cast<size_t>(context_len - 1)], target);
  }
  keys[0] = 0;
}

void advance_dense_keys(std::vector<int64_t> &keys, int64_t target, int64_t vocab_size) {
  for (int64_t context_len = static_cast<int64_t>(keys.size()) - 1; context_len >= 1; --context_len) {
    keys[static_cast<size_t>(context_len)] = keys[static_cast<size_t>(context_len - 1)] * vocab_size + target;
  }
  keys[0] = 0;
}

py::dict weight_snapshot(int64_t depth, const std::vector<int64_t> &orders, const std::vector<double> &weights) {
  py::dict snapshot;
  snapshot["depth"] = depth;
  py::dict weight_dict;
  for (size_t index = 0; index < orders.size(); ++index) {
    weight_dict[py::str(std::to_string(orders[index]))] = weights[index];
  }
  snapshot["weights"] = weight_dict;
  return snapshot;
}

uint64_t bit_context_mask(int64_t context_len) {
  if (context_len <= 0) {
    return 0;
  }
  if (context_len >= 32) {
    return std::numeric_limits<uint64_t>::max();
  }
  return (uint64_t{1} << (2 * context_len)) - 1;
}

uint8_t complement_symbol(uint8_t symbol) {
  return static_cast<uint8_t>(3 - symbol);
}

void advance_bit_contexts(std::vector<uint64_t> &keys, uint8_t symbol, const std::vector<uint64_t> &masks) {
  for (int64_t context_len = static_cast<int64_t>(keys.size()) - 1; context_len >= 1; --context_len) {
    keys[static_cast<size_t>(context_len)] =
        ((keys[static_cast<size_t>(context_len - 1)] << 2) | static_cast<uint64_t>(symbol)) &
        masks[static_cast<size_t>(context_len)];
  }
  keys[0] = 0;
}

void advance_reverse_complement_contexts(
    std::vector<uint64_t> &keys,
    uint8_t symbol,
    const std::vector<uint64_t> &masks) {
  const uint64_t rc_symbol = static_cast<uint64_t>(complement_symbol(symbol));
  for (int64_t context_len = static_cast<int64_t>(keys.size()) - 1; context_len >= 1; --context_len) {
    const uint64_t high = rc_symbol << (2 * (context_len - 1));
    keys[static_cast<size_t>(context_len)] =
        (high | keys[static_cast<size_t>(context_len - 1)]) & masks[static_cast<size_t>(context_len)];
  }
  keys[0] = 0;
}

struct Geco2Expert {
  int64_t context_len = 0;
  double alpha_den = 1.0;
  double gamma = 0.9;
  int64_t hash_slots = 1;
  bool ir = false;
  bool edit = false;
  int64_t edit_threshold = 0;
  std::string label;
  DenseTable dense;
  HashTable hash;
  std::vector<std::vector<uint64_t>> keys;
  std::vector<std::vector<uint64_t>> rc_keys;
  std::vector<uint32_t> edit_masks;
  int64_t hits = 0;
  int64_t queries = 0;
};

std::vector<Geco2Expert> make_geco2_level10_experts(
    int64_t window_count,
    int64_t dense_max_entries,
    int64_t hash_bucket_count,
    bool disable_edit_experts,
    bool disable_ir) {
  struct Spec {
    int64_t ctx;
    double alpha_den;
    bool ir;
    int64_t hash_slots;
    double gamma;
    int64_t edits;
    double edit_alpha_den;
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
  std::vector<Geco2Expert> experts;
  for (const Spec &spec : specs) {
    Geco2Expert expert;
    expert.context_len = spec.ctx;
    expert.alpha_den = spec.alpha_den;
    expert.gamma = spec.gamma;
    expert.hash_slots = std::max<int64_t>(1, spec.hash_slots);
    expert.ir = spec.ir && !disable_ir;
    expert.edit = false;
    expert.label = "ctx" + std::to_string(spec.ctx);
    const int64_t dense_entries = spec.ctx >= 32 ? dense_max_entries + 1 : (int64_t{1} << (2 * spec.ctx));
    if (dense_entries <= dense_max_entries) {
      expert.dense = DenseTable(dense_entries, 4);
    } else {
      expert.hash = HashTable(hash_bucket_count, expert.hash_slots, 4);
    }
    expert.keys.assign(static_cast<size_t>(window_count), std::vector<uint64_t>(static_cast<size_t>(spec.ctx + 1), 0));
    if (expert.ir) {
      expert.rc_keys.assign(static_cast<size_t>(window_count), std::vector<uint64_t>(static_cast<size_t>(spec.ctx + 1), 0));
    }
    experts.push_back(std::move(expert));

    if (!disable_edit_experts && spec.edits > 0) {
      Geco2Expert edit;
      edit.context_len = spec.ctx;
      edit.alpha_den = spec.edit_alpha_den;
      edit.gamma = spec.edit_gamma;
      edit.hash_slots = std::max<int64_t>(1, spec.hash_slots);
      edit.ir = spec.ir && !disable_ir;
      edit.edit = true;
      edit.edit_threshold = spec.edits;
      edit.label = "ctx" + std::to_string(spec.ctx) + "_edit" + std::to_string(spec.edits);
      if (dense_entries <= dense_max_entries) {
        edit.dense = DenseTable(dense_entries, 4);
      } else {
        edit.hash = HashTable(hash_bucket_count, edit.hash_slots, 4);
      }
      edit.keys.assign(static_cast<size_t>(window_count), std::vector<uint64_t>(static_cast<size_t>(spec.ctx + 1), 0));
      if (edit.ir) {
        edit.rc_keys.assign(static_cast<size_t>(window_count), std::vector<uint64_t>(static_cast<size_t>(spec.ctx + 1), 0));
      }
      edit.edit_masks.assign(static_cast<size_t>(window_count), 0);
      experts.push_back(std::move(edit));
    }
  }
  return experts;
}

const uint16_t *expert_find(const Geco2Expert &expert, uint64_t key) {
  if (expert.dense.enabled) {
    return expert.dense.row(static_cast<int64_t>(key));
  }
  return expert.hash.find(key);
}

uint16_t *expert_find_or_insert(Geco2Expert &expert, uint64_t key) {
  if (expert.dense.enabled) {
    return expert.dense.row(static_cast<int64_t>(key));
  }
  return expert.hash.find_or_insert(key);
}

double geco2_distribution_from_counts(
    const uint16_t *counts,
    double alpha_den,
    const std::vector<double> &uniform,
    std::vector<double> &output) {
  if (counts == nullptr) {
    output = uniform;
    return 0.0;
  }
  double total = row_sum(counts, 4);
  const double denominator = 4.0 + alpha_den * total;
  for (int64_t symbol = 0; symbol < 4; ++symbol) {
    output[static_cast<size_t>(symbol)] = (1.0 + alpha_den * static_cast<double>(counts[symbol])) / denominator;
  }
  return total;
}

int64_t unique_best_symbol(const uint16_t *counts) {
  if (counts == nullptr) {
    return -1;
  }
  uint16_t best = counts[0];
  int64_t best_symbol = 0;
  int64_t ties = 1;
  uint32_t total = counts[0];
  for (int64_t symbol = 1; symbol < 4; ++symbol) {
    total += counts[symbol];
    if (counts[symbol] > best) {
      best = counts[symbol];
      best_symbol = symbol;
      ties = 1;
    } else if (counts[symbol] == best) {
      ++ties;
    }
  }
  if (total == 0 || ties != 1) {
    return -1;
  }
  return best_symbol;
}

py::dict geco2_weight_snapshot(int64_t depth, const std::vector<Geco2Expert> &experts, const std::vector<double> &weights) {
  py::dict snapshot;
  snapshot["depth"] = depth;
  py::dict weight_dict;
  for (size_t index = 0; index < experts.size(); ++index) {
    weight_dict[py::str(experts[index].label)] = weights[index];
  }
  snapshot["weights"] = weight_dict;
  return snapshot;
}

struct StrictHashTable {
  int64_t bucket_count = 0;
  int64_t slots = 0;
  std::vector<uint32_t> keys;
  std::vector<uint8_t> next_slot;
  std::vector<uint16_t> packed_counts;

  StrictHashTable() = default;

  StrictHashTable(int64_t buckets, int64_t slot_count)
      : bucket_count(std::max<int64_t>(1, buckets)),
        slots(std::max<int64_t>(1, slot_count)) {
    const size_t entry_count = static_cast<size_t>(bucket_count * slots);
    keys.assign(entry_count, 0);
    next_slot.assign(static_cast<size_t>(bucket_count), 0);
    packed_counts.assign(entry_count, 0);
  }

  int64_t bucket_for_hashed(uint64_t hashed_key) const {
    return static_cast<int64_t>(hashed_key % static_cast<uint64_t>(bucket_count));
  }

  int64_t find_index(uint64_t raw_key) const {
    const uint64_t hashed = zhash(raw_key);
    const int64_t bucket = bucket_for_hashed(hashed);
    const uint32_t low_key = static_cast<uint32_t>(hashed & 0xffffffffULL);
    const int64_t base = bucket * slots;
    for (int64_t slot = 0; slot < slots; ++slot) {
      const int64_t index = base + slot;
      if (keys[static_cast<size_t>(index)] == low_key) {
        return index;
      }
    }
    return -1;
  }

  uint32_t freqs(uint64_t raw_key, uint32_t alpha_den, std::array<uint32_t, 4> &out) const {
    const int64_t index = find_index(raw_key);
    if (index < 0) {
      out = {1, 1, 1, 1};
      return 4;
    }
    const uint16_t packed = packed_counts[static_cast<size_t>(index)];
    uint32_t sum = 0;
    for (int64_t symbol = 0; symbol < 4; ++symbol) {
      out[static_cast<size_t>(symbol)] =
          1U + alpha_den * static_cast<uint32_t>((packed >> (symbol << 2)) & 0x0fU);
      sum += out[static_cast<size_t>(symbol)];
    }
    return sum;
  }

  void update(uint64_t raw_key, uint8_t symbol) {
    const uint64_t hashed = zhash(raw_key);
    const int64_t bucket = bucket_for_hashed(hashed);
    const uint32_t low_key = static_cast<uint32_t>(hashed & 0xffffffffULL);
    const int64_t base = bucket * slots;
    for (int64_t slot = 0; slot < slots; ++slot) {
      const int64_t index = base + slot;
      if (keys[static_cast<size_t>(index)] == low_key) {
        uint16_t packed = packed_counts[static_cast<size_t>(index)];
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
        packed_counts[static_cast<size_t>(index)] = packed;
        return;
      }
    }

    uint8_t &cursor = next_slot[static_cast<size_t>(bucket)];
    ++cursor;
    if (cursor == slots) {
      cursor = 0;
    }
    const int64_t index = base + static_cast<int64_t>(cursor);
    keys[static_cast<size_t>(index)] = low_key;
    packed_counts[static_cast<size_t>(index)] = static_cast<uint16_t>(1U << (symbol << 2));
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
  const uint32_t limit_mask = context_len >= 32 ? std::numeric_limits<uint32_t>::max() : ((uint32_t{1} << context_len) - 1U);
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

template <typename scalar_t>
py::dict compute_current_nc_prefix_impl(
    const at::Tensor &symbols_tensor,
    int64_t window_bases,
    int64_t vocab_size,
    bool return_probabilities,
    int64_t hash_bucket_count,
    bool disable_edit_experts,
    bool disable_ir) {
  if (vocab_size != 4) {
    throw std::runtime_error("nc_prefix requires vocab_size=4 (ACGT)");
  }
  if (window_bases <= 0) {
    throw std::runtime_error("window_bases must be positive");
  }
  const scalar_t *symbols = symbols_tensor.data_ptr<scalar_t>();
  const int64_t n = symbols_tensor.size(0);
  if (n <= 0) {
    throw std::runtime_error("sequence must contain at least one base");
  }
  const auto started = Clock::now();
  const int64_t window_count = (n + window_bases - 1) / window_bases;
  const int64_t max_window_len = std::min<int64_t>(window_bases, n);
  std::vector<int64_t> window_starts(static_cast<size_t>(window_count));
  for (int64_t window = 0; window < window_count; ++window) {
    window_starts[static_cast<size_t>(window)] = window * window_bases;
  }

  at::Tensor bpb_tensor = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat64));
  at::Tensor target_tensor = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt64));
  at::Tensor emit_order_tensor = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt64));
  at::Tensor probabilities_tensor = return_probabilities
      ? torch::empty({n, 4}, torch::TensorOptions().dtype(torch::kFloat64))
      : torch::empty({0, 4}, torch::TensorOptions().dtype(torch::kFloat64));
  double *bpb = bpb_tensor.data_ptr<double>();
  int64_t *targets = target_tensor.data_ptr<int64_t>();
  int64_t *emit_order = emit_order_tensor.data_ptr<int64_t>();
  double *probabilities = return_probabilities ? probabilities_tensor.data_ptr<double>() : nullptr;
  for (int64_t pos = 0; pos < n; ++pos) {
    const int64_t symbol = static_cast<int64_t>(symbols[pos]);
    if (symbol < 0 || symbol >= 4) {
      throw std::runtime_error("nc_prefix symbols must be in ACGT alphabet");
    }
    targets[pos] = symbol;
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
      model.edit_symbols.assign(static_cast<size_t>(n), 0);
      model.edit_best_ids.assign(static_cast<size_t>(n), int8_t{-2});
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
  std::vector<std::array<uint32_t, 4>> predictor_freqs(predictors.size());
  std::vector<uint32_t> predictor_sums(predictors.size(), 4);
  std::vector<double> predictor_target_probs(predictors.size(), 0.25);
  std::vector<double> mixed_float(4, 0.0);
  std::array<uint32_t, 4> mx_freqs{1, 1, 1, 1};
  uint32_t mx_sum = 4;

  py::list weight_history;
  weight_history.append(strict_window_weight_snapshot(0, predictors, window_weights, window_count));
  const int64_t snapshot_stride = std::max<int64_t>(1, max_window_len / 32);
  int64_t emit_index = 0;
  int64_t depth_count = 0;
  double mixed_float_bits = 0.0;
  double quantized_bits = 0.0;

  for (int64_t depth = 0; depth < max_window_len; ++depth) {
    int64_t active_count = 0;

    for (int64_t window_id = 0; window_id < window_count; ++window_id) {
      const int64_t position = window_starts[static_cast<size_t>(window_id)] + depth;
      if (position >= n) {
        continue;
      }
      ++active_count;
      const uint8_t target = static_cast<uint8_t>(targets[position]);
      std::fill(mixed_float.begin(), mixed_float.end(), 0.0);
      double *weights = window_weights.data() + static_cast<size_t>(window_id) * predictor_count;

      for (size_t predictor_index = 0; predictor_index < predictors.size(); ++predictor_index) {
        StrictPredictor &predictor = predictors[predictor_index];
        StrictModel &model = models[static_cast<size_t>(predictor.model_index)];
        const uint64_t key = predictor.edit
            ? model.edit_pidx[static_cast<size_t>(window_id)]
            : model.pidx[static_cast<size_t>(window_id)];
        const uint32_t sum = strict_model_freqs(model, key, predictor.alpha_den, predictor_freqs[predictor_index]);
        predictor_sums[predictor_index] = sum;
        ++predictor.queries;
        if (sum > 4) {
          ++predictor.hits;
        }
        const double p_target =
            static_cast<double>(predictor_freqs[predictor_index][target]) / static_cast<double>(sum);
        predictor_target_probs[predictor_index] = std::max(p_target, 1e-300);
        if (predictor.edit) {
          model.edit_best_ids[static_cast<size_t>(position)] = static_cast<int8_t>(
              strict_best_id(predictor_freqs[predictor_index], sum));
        }
        const double factor = weights[predictor_index] / static_cast<double>(sum);
        for (int64_t symbol = 0; symbol < 4; ++symbol) {
          mixed_float[static_cast<size_t>(symbol)] +=
              static_cast<double>(predictor_freqs[predictor_index][static_cast<size_t>(symbol)]) * factor;
        }
      }

      mx_sum = 0;
      for (int64_t symbol = 0; symbol < 4; ++symbol) {
        const double probability = std::max(mixed_float[static_cast<size_t>(symbol)], 0.0);
        mx_freqs[static_cast<size_t>(symbol)] =
            1U + static_cast<uint32_t>(std::floor(probability * static_cast<double>(GECO2_MX_PMODEL)));
        mx_sum += mx_freqs[static_cast<size_t>(symbol)];
      }
      const double qprob = std::max(
          static_cast<double>(mx_freqs[static_cast<size_t>(target)]) / static_cast<double>(mx_sum),
          1e-300);
      const double fprob = std::max(mixed_float[static_cast<size_t>(target)], 1e-300);
      const double qbits = -std::log(qprob) / std::log(2.0);
      const double fbits = -std::log(fprob) / std::log(2.0);
      bpb[position] = qbits;
      quantized_bits += qbits;
      mixed_float_bits += fbits;
      if (return_probabilities) {
        double *row = probabilities + position * 4;
        for (int64_t symbol = 0; symbol < 4; ++symbol) {
          row[symbol] = static_cast<double>(mx_freqs[static_cast<size_t>(symbol)]) / static_cast<double>(mx_sum);
        }
      }
      emit_order[emit_index++] = position;

      double weight_total = 0.0;
      for (size_t predictor_index = 0; predictor_index < predictors.size(); ++predictor_index) {
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

    if (active_count == 0) {
      continue;
    }
    ++depth_count;

    for (int64_t window_id = 0; window_id < window_count; ++window_id) {
      const int64_t position = window_starts[static_cast<size_t>(window_id)] + depth;
      if (position >= n) {
        continue;
      }
      const uint8_t target = static_cast<uint8_t>(targets[position]);
      for (StrictModel &model : models) {
        const uint64_t key = model.pidx[static_cast<size_t>(window_id)];
        strict_model_update(model, key, target);
        if (model.ir) {
          const uint8_t old_symbol = depth >= model.context_len
              ? static_cast<uint8_t>(targets[position - model.context_len])
              : uint8_t{0};
          const uint8_t ir_symbol = complement_symbol(old_symbol);
          uint64_t &ir_key = model.pidx_ir[static_cast<size_t>(window_id)];
          ir_key = (ir_key >> 2) + (static_cast<uint64_t>(complement_symbol(target)) * model.multiplier);
          strict_model_update(model, ir_key, ir_symbol);
          ++model.ir_updates;
        }
      }

      for (StrictModel &model : models) {
        if (model.edits == 0) {
          continue;
        }
        const int64_t edit_predictor_index = edit_predictor_by_model[static_cast<size_t>(&model - models.data())];
        uint8_t edited_symbol = target;
        if (edit_predictor_index >= 0) {
          edited_symbol = strict_correct_edit_symbol(
              model,
              window_id,
              static_cast<int64_t>(model.edit_best_ids[static_cast<size_t>(position)]),
              target);
        }
        const uint8_t old_edit_symbol = depth >= model.context_len
            ? model.edit_symbols[static_cast<size_t>(position - model.context_len)]
            : uint8_t{0};
        uint64_t &edit_key = model.edit_pidx[static_cast<size_t>(window_id)];
        edit_key = ((edit_key - static_cast<uint64_t>(old_edit_symbol) * model.multiplier) << 2) + edited_symbol;
        edit_key &= (model.n_contexts - 1);
        model.edit_symbols[static_cast<size_t>(position)] = edited_symbol;
      }

      for (StrictModel &model : models) {
        const uint8_t old_symbol = depth >= model.context_len
            ? static_cast<uint8_t>(targets[position - model.context_len])
            : uint8_t{0};
        uint64_t &key = model.pidx[static_cast<size_t>(window_id)];
        key = ((key - static_cast<uint64_t>(old_symbol) * model.multiplier) << 2) + target;
        key &= (model.n_contexts - 1);
      }
    }

    if (depth + 1 == max_window_len || (depth + 1) % snapshot_stride == 0) {
      weight_history.append(strict_window_weight_snapshot(depth + 1, predictors, window_weights, window_count));
    }
  }

  if (emit_index != n) {
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
    item["hit_rate"] = predictor.queries > 0
        ? static_cast<double>(predictor.hits) / static_cast<double>(predictor.queries)
        : 0.0;
    predictor_list.append(item);
    final_weights[py::str(predictor.label)] = final_mean_weights[predictor_index];
    hit_rates[py::str(predictor.label)] = predictor.queries > 0
        ? static_cast<double>(predictor.hits) / static_cast<double>(predictor.queries)
        : 0.0;
  }

  py::dict metadata;
  metadata["backend"] = "fast_cpp";
  metadata["preset"] = "nc_prefix";
  metadata["algorithm"] = "geco2_level10_per_window_weights";
  metadata["geco2_level"] = 10;
  metadata["base_count"] = n;
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
  metadata["weight_scope"] = window_count == 1 ? "single_serial_stream" : "per_window_local_weights";
  metadata["weight_history_summary"] = "mean of per-window weights";
  metadata["context_key_mode"] = "geco2_acgt_2bit_full_context_zero_padded";
  metadata["reverse_complement_update"] = !disable_ir;
  metadata["edit_experts"] = !disable_edit_experts;
  metadata["edit_counter_sharing"] = "edit predictors query the owning base model counter table";
  metadata["dense_context_rule"] = "ctx < 15";
  metadata["dense_counter_mode"] = "uint16_default_max_count_halving";
  metadata["hash_bucket_count"] = effective_hash_buckets;
  metadata["hash_counter_mode"] = "geco2_4bit_packed_counter_halve_on_15";
  metadata["hash_replacement"] = "geco2_style_bucket_cursor_oldest_replacement";
  metadata["mixed_float_bits"] = mixed_float_bits;
  metadata["mixed_float_bits_per_base"] = mixed_float_bits / static_cast<double>(std::max<int64_t>(n, 1));
  metadata["geco2_quantized_bits"] = quantized_bits;
  metadata["geco2_quantized_bits_per_base"] = quantized_bits / static_cast<double>(std::max<int64_t>(n, 1));
  metadata["theoretical_bits"] = quantized_bits;
  metadata["theoretical_bits_per_base"] = quantized_bits / static_cast<double>(std::max<int64_t>(n, 1));
  metadata["final_order_weights"] = final_weights;
  metadata["expert_hit_rates"] = hit_rates;
  metadata["weight_history"] = weight_history;
  metadata["compute_seconds"] = elapsed_seconds(started, finished);

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
    int64_t preset_id,
    int64_t geco2_level,
    bool disable_edit_experts,
    bool disable_ir,
    int64_t dense_max_entries,
    int64_t hash_bucket_count,
    int64_t hash_slots) {
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
        hash_bucket_count,
        disable_edit_experts,
        disable_ir);
  }
  if (symbols_contig.scalar_type() == at::kShort) {
    return compute_current_nc_prefix_impl<int16_t>(
        symbols_contig,
        window_bases,
        vocab_size,
        return_probabilities,
        hash_bucket_count,
        disable_edit_experts,
        disable_ir);
  }
  if (symbols_contig.scalar_type() == at::kInt) {
    return compute_current_nc_prefix_impl<int32_t>(
        symbols_contig,
        window_bases,
        vocab_size,
        return_probabilities,
        hash_bucket_count,
        disable_edit_experts,
        disable_ir);
  }
  throw std::runtime_error("symbols must be int16, int32, or int64");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
      py::arg("preset_id") = NC_PREFIX_PRESET_GECO2_STRICT,
      py::arg("geco2_level") = 10,
      py::arg("disable_edit_experts") = false,
      py::arg("disable_ir") = false,
      py::arg("dense_max_entries") = DENSE_CONTEXT_MAX_ENTRIES,
      py::arg("hash_bucket_count") = DEFAULT_HASH_BUCKETS,
      py::arg("hash_slots") = DEFAULT_HASH_SLOTS);
}
