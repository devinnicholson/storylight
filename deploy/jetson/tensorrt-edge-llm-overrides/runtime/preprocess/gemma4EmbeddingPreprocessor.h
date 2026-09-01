/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "common/mmapReader.h"
#include "common/tensor.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/tensorMap.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Runtime preprocessor for Gemma4 E-model per-layer embeddings (PLE).
//!
//! The upstream path loads the complete PLE table onto the GPU. Bookforge can
//! instead keep the exact FP16/BF16 table memory-mapped on NVMe and stage only
//! the rows referenced by the current token IDs. This makes Gemma4 E-models
//! viable on unified-memory devices whose RAM is smaller than the PLE sidecar.
class Gemma4EmbeddingPreprocessor
{
public:
    Gemma4EmbeddingPreprocessor(std::filesystem::path const& engineDir, LLMEngineConfig const& config,
        int32_t maxBatchSize, int32_t maxSeqLen, TensorMap& tensorMap, cudaStream_t stream,
        std::optional<Tensor> checkpointTable = std::nullopt);

    //! Gather PLE tensors for the current token-id tensor shape.
    void embed(Tensor const& tokenIds, cudaStream_t stream);

    //! Reshape already-bound output tensors for a CUDA-graph capture shape.
    void reshapeOutputs(int64_t batchSize, int64_t seqLen);

private:
    LLMEngineConfig mConfig{};
    Tensor mPleTable{};
    Tensor mPleOutputBuffer{}; //!< Unified owned backing buffer for all PLE layer outputs.
    //! Non-owned tensor views into mPleOutputBuffer. TensorMap stores pointers to these stable objects.
    std::vector<Tensor> mPleOutputViews{};

    bool mStorageBackedPle{false};
    int64_t mPleVocabSize{0};
    size_t mPleRowBytes{0};
    nvinfer1::DataType mPleDataType{nvinfer1::DataType::kHALF};
    std::unique_ptr<file_io::MmapReader> mPleMapping{};
    int8_t const* mPlePayload{nullptr};
    Tensor mHostTokenIds{};
    Tensor mHostPackedPle{};

    //! Construct a non-owned tensor view for one layer output inside mPleOutputBuffer.
    Tensor makeOutputViewForLayer(int32_t layerIdx, int64_t batchSize, int64_t seqLen);
};

} // namespace rt
} // namespace trt_edgellm
