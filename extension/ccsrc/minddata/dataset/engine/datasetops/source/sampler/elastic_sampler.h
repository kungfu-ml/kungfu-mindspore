#pragma once
#include <limits>
#include <memory>

#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"

namespace mindspore
{
namespace dataset
{
class ElasticSamplerRT : public SamplerRT
{
  public:
    // Constructor
    // @param num_samples - The number of samples to draw. A value of 0
    // indicates the sampler should produce the
    //                      full amount of ids from the dataset
    // @param start_index - The starting index value
    // @param int64_t samplesPerBuffer - Num of Sampler Ids to fetch via 1
    // GetNextBuffer call
    ElasticSamplerRT(
        int64_t num_samples, int64_t start_index,
        int64_t samples_per_buffer = std::numeric_limits<int64_t>::max());

    // Destructor.
    ~ElasticSamplerRT() = default;

    // init sampler, called by python
    Status InitSampler() override;

    // for next epoch of sampleIds
    // @return Status The status code returned
    Status ResetSampler() override;

    // Op calls this to get next Buffer that contains all the sampleIds
    // @param std::unique_ptr<DataBuffer> pBuffer - Buffer to be returned to
    // corresponding Dataset Op
    // @param int32_t workerId - not meant to be used
    // @return Status The status code returned
    Status GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) override;

    // Printer for debugging purposes.
    // @param out - output stream to write to
    // @param show_all - bool to show detailed vs summary
    void SamplerPrint(std::ostream &out, bool show_all) const override;

  private:
    int64_t current_id_;  // The id sequencer.  Each new id increments from this
    int64_t start_index_;  // The starting id.  current_id_ begins from here.
    int64_t id_count_;     // An internal counter that tracks how many ids have
                           // been produced
};
}  // namespace dataset
}  // namespace mindspore
