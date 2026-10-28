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

using AudioFrameArray =
    nb::ndarray<int16_t, nb::numpy, nb::ndim<1>, nb::f_contig>;

class Decoder : public FLAC::Decoder::Stream {
public:
  explicit Decoder(const FLAC__byte *input, size_t size, OutputFrame &output,
                   size_t *output_end)
      : input_(input), size_(size), output_(output), output_end_(output_end) {}

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

AudioFrameArray get_frame() {
  auto result = read_parquet();
  if (!result.ok())
    throw result.status().ToString();

  std::shared_ptr<arrow::Table> table = *result; // is this lazy?
  auto audio = table->GetColumnByName("audio");
  auto struct_type = std::static_pointer_cast<arrow::StructType>(audio->type());
  int bytes_idx = struct_type->GetFieldIndex("bytes");
  int path_idx = struct_type->GetFieldIndex("path");

  // TODO: process the entire chunked array
  auto audio_1 = *audio->GetScalar(0);
  auto struct_scalar = std::static_pointer_cast<arrow::StructScalar>(audio_1);
  auto audio_1_path = *struct_scalar->field(path_idx);
  std::cout << audio_1_path->ToString() << '\n';

  auto audio_1_bytes = *struct_scalar->field(bytes_idx);

  if (audio_1_bytes->is_valid) {
    auto binary_scalar =
        std::static_pointer_cast<arrow::BinaryScalar>(audio_1_bytes);

    const FLAC__byte *audio_binary =
        binary_scalar->value->data(); // note that a FLAC_byte is an uint8_t

    auto audio_size = binary_scalar->value->size();
    auto output = std::make_unique<OutputFrame>();
    size_t output_end = 0;

    Decoder decoder(audio_binary, audio_size, *output, &output_end);
    auto init_status = decoder.init();

    if (init_status == FLAC__STREAM_DECODER_INIT_STATUS_OK) {
      decoder.process_until_end_of_metadata();
      decoder.process_single();

      auto state = decoder.get_state();
      state = decoder.get_state();
      std::cout << state.as_cstring() << '\n';

      OutputFrame *output_ptr = output.release();
      nb::capsule owner(output_ptr, [](void *p) noexcept {
        delete static_cast<OutputFrame *>(p);
      });

      return AudioFrameArray(output_ptr->data(), {output_ptr->size()}, owner);
    }
  }

  throw -1;
}

NB_MODULE(decoder, m) { m.def("get_frame", &get_frame); }
