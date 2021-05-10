#pragma once
#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore
{
namespace dataset
{

// MappableSourceNode represents the leaf nodes that can be randomly accessed
// with indexes.
class KungFuMappableSourceNode : public DatasetNode
{  // forked from MappableSourceNode
  public:
    /// \brief Constructor
    KungFuMappableSourceNode() : DatasetNode()
    {
        mappable_ = kMappableSource;
    }

    /// \brief Constructor that initializes the cache
    /// \param dataset_cache DatasetCache
    explicit KungFuMappableSourceNode(
        const std::shared_ptr<DatasetCache> &dataset_cache)
        : DatasetNode(dataset_cache)
    {
        mappable_ = kMappableSource;
        // Initially set to false, and set to true by the optimizer when
        // conditions are met.
        descendant_of_cache_ = false;
    }

    Status Accept(IRNodePass *p, bool *modified) override;

    /// \brief Destructor
    ~KungFuMappableSourceNode() = default;

    /// \brief Node name getter
    /// \return Name of the current node
    virtual std::string Name() const = 0;
};

constexpr char kKungFuDataNode[] = "KungFuDataNode";

class KungFuDataNode : public KungFuMappableSourceNode
{
  public:
    /// \brief Constructor
    KungFuDataNode(std::string dataset_dir, std::string usage,
                   std::shared_ptr<SamplerObj> sampler,
                   std::shared_ptr<DatasetCache> cache);

    /// \brief Destructor
    ~KungFuDataNode() = default;

    /// \brief Node name getter
    /// \return Name of the current node
    std::string Name() const override
    {
        return kKungFuDataNode;
    }

    /// \brief Print the description
    /// \param out - The output stream to write output to
    void Print(std::ostream &out) const override;

    /// \brief Copy the node to a new object
    /// \return A shared pointer to the new copy
    std::shared_ptr<DatasetNode> Copy() override;

    /// \brief a base class override function to create the required runtime
    /// dataset op objects for this class \param node_ops - A vector containing
    /// shared pointer to the Dataset Ops that this object will create \return
    /// Status Status::OK() if build successfully
    Status Build(std::vector<std::shared_ptr<DatasetOp>> *node_ops) override;

    /// \brief Parameters validation
    /// \return Status Status::OK() if all the parameters are valid
    Status ValidateParams() override;

    /// \brief Get the shard id of node
    /// \return Status Status::OK() if get shard id successfully
    Status GetShardId(int32_t *shard_id) override;

    /// \brief Base-class override for GetDatasetSize
    /// \param[in] size_getter Shared pointer to DatasetSizeGetter
    /// \param[in] estimate This is only supported by some of the ops and it's
    /// used to speed up the process of getting
    ///     dataset size at the expense of accuracy.
    /// \param[out] dataset_size the size of the dataset
    /// \return Status of the function
    Status GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter,
                          bool estimate, int64_t *dataset_size) override;

  private:
    std::string dataset_dir_;
    std::string usage_;
    std::shared_ptr<SamplerObj> sampler_;
};

}  // namespace dataset
}  // namespace mindspore
