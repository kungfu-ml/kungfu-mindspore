#include "minddata/dataset/engine/datasetops/source/sampler/elastic_sampler.h"

#include "backend/kernel_compiler/cpu/kungfu/kungfu_common.h"
#include "backend/kernel_compiler/cpu/kungfu/kungfu_logger.h"

#include <algorithm>
#include <memory>

namespace mindspore
{
namespace dataset
{
ElasticSamplerRT::ElasticSamplerRT(int64_t num_samples, int64_t start_index,
                                   int64_t samples_per_buffer)
    : SamplerRT(num_samples, samples_per_buffer),
      current_id_(start_index),
      start_index_(start_index),
      id_count_(0)
{
    KF_LOG() << "ElasticSamplerRT"
             << "created at" << this          //
             << "num_samples" << num_samples  //
             << "start_index" << start_index  //
             << "samples_per_buffer" << samples_per_buffer;
}

Status ElasticSamplerRT::GetNextSample(std::unique_ptr<DataBuffer> *out_buffer)
{
    KF_LOG() << "ElasticSamplerRT::GetNextSample";

    if (id_count_ > num_samples_) {
        RETURN_STATUS_UNEXPECTED("ElasticSampler Internal Error");
    } else if (id_count_ == num_samples_) {
        (*out_buffer) =
            std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
    } else {
        if (HasChildSampler()) {
            RETURN_IF_NOT_OK(child_[0]->GetNextSample(&child_ids_));
        }

        (*out_buffer) =
            std::make_unique<DataBuffer>(current_id_, DataBuffer::kDeBFlagNone);
        std::shared_ptr<Tensor> sampleIds;

        // Compute how many ids are left to pack, and pack this amount into a
        // new buffer.  Respect the setting for samples per buffer though.
        int64_t remaining_ids = num_samples_ - id_count_;
        int64_t num_elements = std::min(remaining_ids, samples_per_buffer_);

        KF_LOG() << "remaining_ids" << remaining_ids;
        KF_LOG() << "num_elements" << num_elements;

        RETURN_IF_NOT_OK(CreateSamplerTensor(&sampleIds, num_elements));
        auto idPtr = sampleIds->begin<int64_t>();
        for (int64_t i = 0; i < num_elements; i++) {
            KF_LOG() << "ElasticSamplerRT::GetNextSample"
                     << "loop" << i;
            int64_t sampled_id = current_id_;
            if (HasChildSampler()) {
                RETURN_IF_NOT_OK(GetAssociatedChildId(&sampled_id, sampled_id));
            }

            *idPtr = sampled_id;
            current_id_++;  // Move the current id to the next one in the
                            // sequence
            idPtr++;
        }

        id_count_ += num_elements;  // Count the packed ids towards our overall
                                    // sample count

        TensorRow row(1, sampleIds);
        (*out_buffer)->set_tensor_table(std::make_unique<TensorQTable>(1, row));
    }
    return Status::OK();
}

Status ElasticSamplerRT::InitSampler()
{
    KF_LOG() << "ElasticSamplerRT" << ':' << __func__;
    if (_show_kungfu_debug_log) {
        KF_LOG() << "ElasticSamplerRT" << ':' << __func__;
    }
    CHECK_FAIL_RETURN_UNEXPECTED(start_index_ >= 0,
                                 "Invalid parameter, start_index must be "
                                 "greater than or equal to 0, but got " +
                                     std::to_string(start_index_) + ".\n");
    CHECK_FAIL_RETURN_UNEXPECTED(
        start_index_ < num_rows_,
        "Invalid parameter, start_index must be less than num_rows, but got "
        "start_index: " +
            std::to_string(start_index_) +
            ", num_rows: " + std::to_string(num_rows_) + ".\n");
    CHECK_FAIL_RETURN_UNEXPECTED(num_samples_ >= 0,
                                 "Invalid parameter, num_samples must be "
                                 "greater than or equal to 0, but got " +
                                     std::to_string(num_samples_) + ".\n");
    // Adjust the num_samples count based on the range of ids we are sequencing.
    // If num_samples is 0, we sample the entire set.  If it's non-zero, we will
    // implicitly cap the amount sampled based on available data.
    int64_t available_row_count = num_rows_ - start_index_;
    if (num_samples_ == 0 || num_samples_ > available_row_count) {
        num_samples_ = available_row_count;
    }
    CHECK_FAIL_RETURN_UNEXPECTED(num_samples_ > 0 && samples_per_buffer_ > 0,
                                 "Invalid parameter, samples_per_buffer must "
                                 "be greater than 0, but got " +
                                     std::to_string(samples_per_buffer_));
    samples_per_buffer_ =
        samples_per_buffer_ > num_samples_ ? num_samples_ : samples_per_buffer_;
    KF_LOG() << "after"
             << "ElasticSamplerRT" << __func__;
    KF_LOG() << "num_samples_" << num_samples_;
    return Status::OK();
}

Status ElasticSamplerRT::ResetSampler()
{
    if (_show_kungfu_debug_log) {
        KF_LOG() << "ElasticSamplerRT" << ':' << __func__;
    }
    CHECK_FAIL_RETURN_UNEXPECTED(id_count_ == num_samples_,
                                 "ERROR Reset() called early/late");
    current_id_ = start_index_;
    id_count_ = 0;

    if (HasChildSampler()) {
        RETURN_IF_NOT_OK(child_[0]->ResetSampler());
    }

    return Status::OK();
}

void ElasticSamplerRT::SamplerPrint(std::ostream &out, bool show_all) const
{
    if (_show_kungfu_debug_log) {
        KF_LOG() << "ElasticSamplerRT" << ':' << __func__;
    }
    out << "\nSampler: ElasticSampler";
    if (show_all) {
        // Call the super class for displaying any common detailed info
        SamplerRT::SamplerPrint(out, show_all);
        // Then add our own info
        out << "\nStart index: " << start_index_;
    }
}
}  // namespace dataset
}  // namespace mindspore
