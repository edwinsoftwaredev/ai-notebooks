#include <arrow/io/api.h>
#include <arrow/scalar.h>
#include <arrow/table.h>
#include <arrow/type_fwd.h>
#include <iostream>
#include <memory>
#include <parquet/arrow/reader.h>

#include <FLAC++/decoder.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

/*
 * Set these to their max values.
 */
constexpr int BLOCK_SIZE = 4096;
constexpr int CHANNELS = 1;

namespace nb = nanobind;

using OutputFrame = std::array<int16_t, BLOCK_SIZE * CHANNELS>;
using OutputWave = std::vector<int16_t>; // note that reallocation takes O(n)

using AudioWaveBatch =
    nb::ndarray<int16_t, nb::numpy, nb::ndim<2>, nb::c_contig>;

class Decoder : public FLAC::Decoder::Stream {
public:
  explicit Decoder(const FLAC__byte *input, size_t size, OutputFrame &output,
                   size_t *output_end, OutputWave &output_wave)
      : input_(input), size_(size), output_(output), output_end_(output_end),
        output_wave_(output_wave) {}

protected:
  FLAC__StreamDecoderReadStatus read_callback(FLAC__byte buffer[],
                                              size_t *buffer_size) override {

    if (this->input_ == nullptr || this->size_ == 0) {
      *buffer_size = 0;
      return FLAC__STREAM_DECODER_READ_STATUS_ABORT;
    }

    /*
     * Copy from buffer containing the input binary data into the buffer
     * provided by the libFLAC API.
     */

    size_t actual_count = std::min(*buffer_size, this->size_ - this->position_);

    *buffer_size = actual_count;

    if (actual_count == 0)
      return FLAC__STREAM_DECODER_READ_STATUS_END_OF_STREAM;

    std::memcpy(buffer, this->input_ + this->position_, actual_count);

    this->position_ += actual_count;

    return FLAC__STREAM_DECODER_READ_STATUS_CONTINUE;
  }

  FLAC__StreamDecoderWriteStatus
  write_callback(const FLAC__Frame *frame,
                 const FLAC__int32 *const buffer[]) override {
    const unsigned blocksize = frame->header.blocksize;
    const unsigned channels = frame->header.channels;
    auto sample_rate = frame->header.sample_rate;
    auto bits_per_sample = frame->header.bits_per_sample;

    for (int i = 0; i < blocksize; ++i) {
      for (int j = 0; j < channels; ++j) {
        this->output_[i * channels + j] = static_cast<int16_t>(buffer[j][i]);
      }
    }

    this->output_wave_.insert(this->output_wave_.end(), this->output_.begin(),
                              this->output_.begin() + blocksize * channels);

    *this->output_end_ = blocksize * channels;

    return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
  }

  void error_callback(FLAC__StreamDecoderErrorStatus status) override {
    std::cout << "Got error callback: "
              << FLAC__StreamDecoderErrorStatusString[status];
  }

private:
  const FLAC__byte *input_;
  size_t size_;
  size_t position_ = 0;
  OutputFrame &output_;
  size_t *output_end_;
  OutputWave &output_wave_;
};

arrow::Result<std::shared_ptr<arrow::Table>> read_parquet() {
  const std::string path_to_file = "wave_renderer/dataset";
  arrow::MemoryPool *pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::io::RandomAccessFile> input;
  ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(path_to_file));

  // open parquet file reader
  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  ARROW_ASSIGN_OR_RAISE(arrow_reader, parquet::arrow::OpenFile(input, pool));

  std::shared_ptr<arrow::Table> table;
  ARROW_ASSIGN_OR_RAISE(table, arrow_reader->ReadTable());

  return table;
}

class FLACDatasetIter {
public:
  explicit FLACDatasetIter() : table(*read_parquet()) {}

  AudioWaveBatch next_batch() {
    auto *batch = new std::vector<uint16_t>();
    // note that nb::ndarray requires a storage with a contiguous layout
    // when the Order template argument is set to nb::c_contig

    const auto &[audio_path, audio_bin] = this->next_audio();
    const FLAC__byte *audio_bin_buff = audio_bin->value->data();
    auto audio_size = audio_bin->value->size();
    auto output_wave = std::make_unique<OutputWave>();
    size_t output_end = 0;

    Decoder decoder(audio_bin_buff, audio_size, this->output_frame, &output_end,
                    *output_wave);

    auto init_status = decoder.init();

    if (init_status != FLAC__STREAM_DECODER_INIT_STATUS_OK)
      throw -1;

    decoder.process_until_end_of_stream();

    decoder.finish();

    batch->insert(batch->end(), output_wave->begin(), output_wave->end());

    // Delete 'batch' when the 'owner' capsule expires
    nb::capsule owner(batch, [](void *p) noexcept {
      delete static_cast<std::vector<uint16_t> *>(p);
    });

    // increment this->cur_audio_idx

    return AudioWaveBatch(batch->data(), {1, output_wave->size()}, owner);
  }

private:
  long cur_audio_idx = 0;
  int batch_size = 1;
  std::shared_ptr<arrow::Table> table;
  OutputFrame output_frame;

  std::pair<std::string, std::shared_ptr<arrow::BinaryScalar>> next_audio() {
    auto audio_col = this->table->GetColumnByName("audio");
    auto struct_type =
        std::static_pointer_cast<arrow::StructType>(audio_col->type());

    int bytes_idx = struct_type->GetFieldIndex("bytes");
    int path_idx = struct_type->GetFieldIndex("path");

    auto audio_scalar = *audio_col->GetScalar(this->cur_audio_idx);
    auto audio_struct =
        std::static_pointer_cast<arrow::StructScalar>(audio_scalar);

    auto audio_path = *audio_struct->field(path_idx);
    auto audio_bytes = *audio_struct->field(bytes_idx);

    if (!audio_bytes->is_valid)
      throw audio_bytes->Validate().ToString();

    auto audio_bin = std::static_pointer_cast<arrow::BinaryScalar>(audio_bytes);

    return {audio_path->ToString(), audio_bin};
  }
};

NB_MODULE(decoder, m) {
  nb::class_<FLACDatasetIter>(m, "FLACDatasetIter")
      .def(nb::init())
      .def("next_batch", &FLACDatasetIter::next_batch);
}
