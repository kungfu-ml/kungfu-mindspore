#include "minddata/dataset/engine/ir/datasetops/source/kungfu_data_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backend/kernel_compiler/cpu/kungfu/kungfu_common.h"
#include "backend/kernel_compiler/cpu/kungfu/kungfu_logger.h"
#include "minddata/dataset/engine/datasetops/source/kungfu_data_op.h"

#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/status.h"

namespace mindspore::dataset
{
Status KungFuMappableSourceNode::Accept(IRNodePass *p, bool *modified)
{
    if (_show_kungfu_debug_log) {
        KF_LOG() << "KungFuMappableSourceNode:" << ':' << __func__;
    }
    return p->Visit(shared_from_base<KungFuMappableSourceNode>(), modified);
}

KungFuDataNode::KungFuDataNode(std::string dataset_dir, std::string usage,
                               std::shared_ptr<SamplerObj> sampler,
                               std::shared_ptr<DatasetCache> cache)
    : KungFuMappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      sampler_(sampler)
{
    if (_show_kungfu_debug_log) {
        KF_LOG() << __func__ << "created";
    }
}

std::shared_ptr<DatasetNode> KungFuDataNode::Copy()
{
    if (_show_kungfu_debug_log) {
        KF_LOG() << "KungFuDataNode:" << ':' << __func__;
    }

    std::shared_ptr<SamplerObj> sampler =
        (sampler_ == nullptr) ? nullptr : sampler_->Copy();
    auto node =
        std::make_shared<KungFuDataNode>(dataset_dir_, usage_, sampler, cache_);
    return node;
}

void KungFuDataNode::Print(std::ostream &out) const
{
    out << Name();
}

Status KungFuDataNode::ValidateParams()
{
    if (_show_kungfu_debug_log) {
        KF_LOG() << "KungFuDataNode:" << ':' << __func__;
    }

    RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
    RETURN_IF_NOT_OK(ValidateDatasetDirParam("KungFuDataNode", dataset_dir_));

    RETURN_IF_NOT_OK(ValidateDatasetSampler("KungFuDataNode", sampler_));

    RETURN_IF_NOT_OK(ValidateStringValue("KungFuDataNode", usage_,
                                         {"train", "test", "all"}));

    return Status::OK();
}

Status KungFuDataNode::Build(std::vector<std::shared_ptr<DatasetOp>> *node_ops)
{
    if (_show_kungfu_debug_log) {
        KF_LOG() << "KungFuDataNode:" << ':' << __func__;
    }

    // Do internal Schema generation.
    auto schema = std::make_unique<DataSchema>();
    RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor(
        "image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
    TensorShape scalar = TensorShape::CreateScalar();
    RETURN_IF_NOT_OK(
        schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32),
                                        TensorImpl::kFlexible, 0, &scalar)));
    // RETURN_IF_NOT_OK(schema->AddColumn(
    //     ColDescriptor("id", DataType(DataType::DE_INT64), TensorImpl::kCv,
    //     1)));
    RETURN_IF_NOT_OK(AddCacheOp(node_ops));

    node_ops->push_back(std::make_shared<KungFuDataOp>(
        usage_, num_workers_, rows_per_buffer_, dataset_dir_,
        connector_que_size_, std::move(schema), std::move(sampler_->Build())));

    if (_show_kungfu_debug_log) {
        KF_LOG() << "KungFuDataNode:" << ':' << __func__ << ':' << "OK";
    }

    return Status::OK();
}

// Get the shard id of node
Status KungFuDataNode::GetShardId(int32_t *shard_id)
{
    if (_show_kungfu_debug_log) {
        KF_LOG() << "KungFuDataNode:" << ':' << __func__;
    }

    *shard_id = sampler_->ShardId();

    return Status::OK();
}

// Get Dataset size
Status KungFuDataNode::GetDatasetSize(
    const std::shared_ptr<DatasetSizeGetter> &size_getter, bool _estimate,
    int64_t *dataset_size)
{
    if (_show_kungfu_debug_log) {
        KF_LOG() << "KungFuDataNode:" << ':' << __func__;
    }

    if (dataset_size_ > 0) {
        *dataset_size = dataset_size_;
        return Status::OK();
    }
    int64_t num_rows, sample_size;
    RETURN_IF_NOT_OK(
        KungFuDataOp::CountTotalRows(dataset_dir_, usage_, &num_rows));
    sample_size = sampler_->Build()->CalculateNumSamples(num_rows);
    *dataset_size = sample_size;
    dataset_size_ = *dataset_size;
    return Status::OK();
}
}  // namespace mindspore::dataset
