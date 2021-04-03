#include "minddata/dataset/engine/datasetops/source/sampler/elastic_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"

#include <algorithm>
#include <string>

namespace mindspore
{
namespace dataset
{
ElasticSampler::ElasticSampler(int64_t num_samples, int64_t samples_per_buffer)
    : num_rows_(0),
      num_samples_(num_samples),
      samples_per_buffer_(samples_per_buffer),
      col_desc_(nullptr)
{
}

Status ElasticSampler::HandshakeRandomAccessOp(const RandomAccessOp *op)
{
    std::shared_ptr<ElasticSampler> child_sampler;
    if (HasChildSampler()) {
        child_sampler = std::dynamic_pointer_cast<ElasticSampler>(child_[0]);
        if (!child_sampler) {
            std::string err_msg(
                "Cannot handshake, child is not a sampler object.");
            RETURN_STATUS_UNEXPECTED(err_msg);
        }

        // Handshake and init child first.
        RETURN_IF_NOT_OK(child_sampler->HandshakeRandomAccessOp(op));
    }

    CHECK_FAIL_RETURN_UNEXPECTED(op != nullptr, "RandomAccessOp is nullptr\n");

    // If there's a child sampler, set the row count to be it's sample count
    if (HasChildSampler()) {
        num_rows_ = child_sampler->num_samples_;
    } else {
        RETURN_IF_NOT_OK(op->GetNumRowsInDataset(&num_rows_));
    }

    // It's up to the derived class to check the validity of the two args
    // Because some sampler only needs one of the arg (weighted_random_sampler)
    RETURN_IF_NOT_OK(InitSampler());  // init sampler after callback

    return Status::OK();
}

Status ElasticSampler::CreateSamplerTensor(std::shared_ptr<Tensor> *sample_ids,
                                           int64_t num_elements)
{
    if (num_elements == 0) {
        RETURN_STATUS_UNEXPECTED("Invalid data, num of elements cannot be 0.");
    }
    if (col_desc_ == nullptr) {
        // a ColDescriptor for Tensor that holds SampleIds
        col_desc_ = std::make_unique<ColDescriptor>(
            "sampleIds", DataType(DataType::DE_INT64), TensorImpl::kFlexible,
            1);
    }
    TensorShape shape(std::vector<dsize_t>(1, num_elements));
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(shape, col_desc_->type(), sample_ids));
    return Status::OK();
}

void ElasticSampler::SamplerPrint(std::ostream &out, bool show_all) const
{
    // Sampler printing is usually only called in the show_all mode.
    // Derived classes will display the name, then call back to this base
    // for common info.
    // No-op in the summary mode.
    if (show_all) {
        out << "\nnum_rows_: " << num_rows_
            << "\nnum_samples_: " << num_samples_;
    }
}

#ifdef ENABLE_PYTHON
Status ElasticSampler::GetAllIdsThenReset(py::array *data)
{
    std::unique_ptr<DataBuffer> db;
    std::shared_ptr<Tensor> sample_ids;
    TensorRow sample_row;

    // A call to derived class to get sample ids wrapped inside a buffer
    RETURN_IF_NOT_OK(GetNextSample(&db));
    // Get the only tensor inside the buffer that contains the actual SampleIds
    // for the entire epoch
    RETURN_IF_NOT_OK(db->GetRow(0, &sample_row));
    sample_ids = sample_row[0];

    // check this buffer is not a ctrl buffer
    CHECK_FAIL_RETURN_UNEXPECTED(db->buffer_flags() == DataBuffer::kDeBFlagNone,
                                 "ERROR ctrl buffer received");

    // perform error checking! Next buffer supposed to be EOE since last one
    // already contains all ids for current epoch
    RETURN_IF_NOT_OK(GetNextSample(&db));
    CHECK_FAIL_RETURN_UNEXPECTED(db->eoe(), "ERROR Non EOE received");
    // Reset Sampler since this is the end of the epoch
    RETURN_IF_NOT_OK(ResetSampler());

    {
        py::gil_scoped_acquire gil_acquire;
        if (Py_IsInitialized() == 0) {
            return Status(StatusCode::kPythonInterpreterFailure,
                          "Python Interpreter is finalized");
        }
        try {
            RETURN_IF_NOT_OK(sample_ids->GetDataAsNumpy(data));
        } catch (const std::runtime_error &e) {
            return Status(StatusCode::kPyFuncException, e.what());
        }
    }
    return Status::OK();
}
#endif

Status ElasticSampler::SetNumSamples(int64_t num_samples)
{
    CHECK_FAIL_RETURN_UNEXPECTED(
        num_samples >= 0,
        "Invalid parameter, num_samples must be greater than or equal to 0.");
    num_samples_ = num_samples;
    return Status::OK();
}

int64_t ElasticSampler::GetNumSamples()
{
    return num_samples_;
}

int64_t ElasticSampler::CalculateNumSamples(int64_t num_rows)
{
    int64_t childs = num_rows;
    if (!child_.empty()) {
        childs = child_[0]->CalculateNumSamples(num_rows);
    }

    return (num_samples_ > 0) ? std::min(childs, num_samples_) : childs;
}

Status ElasticSampler::SetNumRowsInDataset(int64_t num_rows)
{
    CHECK_FAIL_RETURN_UNEXPECTED(
        num_rows > 0,
        "Invalid data, data rows of input dataset must not be less than or "
        "equal to 0, please check the input dataset.");
    num_rows_ = num_rows;
    return Status::OK();
}

Status ElasticSampler::AddChild(std::shared_ptr<ElasticSampler> child)
{
    if (child == nullptr) {
        return Status::OK();
    }

    // Only samplers can be added, not any other DatasetOp.
    std::shared_ptr<ElasticSampler> sampler =
        std::dynamic_pointer_cast<ElasticSampler>(child);
    if (!sampler) {
        std::string err_msg("Cannot add child, child is not a sampler object.");
        RETURN_STATUS_UNEXPECTED(err_msg);
    }

    // Samplers can have at most 1 child.
    if (!child_.empty()) {
        std::string err_msg(
            "Cannot add child sampler, this sampler already has a child.");
        RETURN_STATUS_UNEXPECTED(err_msg);
    }

    child_.push_back(child);

    return Status::OK();
}

bool ElasticSampler::HasChildSampler()
{
    return !child_.empty();
}

Status ElasticSampler::GetAssociatedChildId(int64_t *out_associated_id,
                                            int64_t id)
{
    if (child_ids_ == nullptr) {
        RETURN_STATUS_UNEXPECTED(
            "Trying to get associated child id, but there are no child ids!");
    }

    TensorRow sample_row;
    RETURN_IF_NOT_OK(child_ids_->GetRow(0, &sample_row));
    std::shared_ptr<Tensor> sample_ids = sample_row[0];
    RETURN_IF_NOT_OK(sample_ids->GetItemAt<int64_t>(out_associated_id, {id}));
    return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
