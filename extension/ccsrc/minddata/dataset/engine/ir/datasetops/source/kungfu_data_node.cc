#include "minddata/dataset/engine/ir/datasetops/source/kungfu_data_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/mnist_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore
{
namespace dataset
{
KungfuDataNode::KungfuDataNode(std::string dataset_dir, std::string usage,
                               std::shared_ptr<SamplerObj> sampler,
                               std::shared_ptr<DatasetCache> cache)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      sampler_(sampler)
{
}

std::shared_ptr<DatasetNode> KungfuDataNode::Copy()
{
    std::shared_ptr<SamplerObj> sampler =
        (sampler_ == nullptr) ? nullptr : sampler_->Copy();
    auto node =
        std::make_shared<KungfuDataNode>(dataset_dir_, usage_, sampler, cache_);
    return node;
}

void KungfuDataNode::Print(std::ostream &out) const
{
    out << Name();
}

Status KungfuDataNode::ValidateParams()
{
    RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
    RETURN_IF_NOT_OK(ValidateDatasetDirParam("KungfuDataNode", dataset_dir_));

    RETURN_IF_NOT_OK(ValidateDatasetSampler("KungfuDataNode", sampler_));

    RETURN_IF_NOT_OK(ValidateStringValue("KungfuDataNode", usage_,
                                         {"train", "test", "all"}));

    return Status::OK();
}

Status KungfuDataNode::Build(std::vector<std::shared_ptr<DatasetOp>> *node_ops)
{
    // Do internal Schema generation.
    auto schema = std::make_unique<DataSchema>();
    RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor(
        "image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
    TensorShape scalar = TensorShape::CreateScalar();
    RETURN_IF_NOT_OK(
        schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32),
                                        TensorImpl::kFlexible, 0, &scalar)));
    RETURN_IF_NOT_OK(AddCacheOp(node_ops));

    node_ops->push_back(std::make_shared<MnistOp>(
        usage_, num_workers_, rows_per_buffer_, dataset_dir_,
        connector_que_size_, std::move(schema), std::move(sampler_->Build())));

    return Status::OK();
}

// Get the shard id of node
Status KungfuDataNode::GetShardId(int32_t *shard_id)
{
    *shard_id = sampler_->ShardId();

    return Status::OK();
}

// Get Dataset size
Status KungfuDataNode::GetDatasetSize(
    const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
    int64_t *dataset_size)
{
    if (dataset_size_ > 0) {
        *dataset_size = dataset_size_;
        return Status::OK();
    }
    int64_t num_rows, sample_size;
    RETURN_IF_NOT_OK(MnistOp::CountTotalRows(dataset_dir_, usage_, &num_rows));
    sample_size = sampler_->Build()->CalculateNumSamples(num_rows);
    *dataset_size = sample_size;
    dataset_size_ = *dataset_size;
    return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
