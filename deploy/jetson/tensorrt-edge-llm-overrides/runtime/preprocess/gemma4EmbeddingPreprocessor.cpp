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

#include "runtime/preprocess/gemma4EmbeddingPreprocessor.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <sys/mman.h>

namespace trt_edgellm
{
namespace rt
{
namespace
{

bool storageBackedPleRequested()
{
    char const* value = std::getenv("EDGELLM_GEMMA4_PLE_STORAGE_BACKED");
    return value != nullptr && std::string_view(value) == "1";
}

} // namespace

Gemma4EmbeddingPreprocessor::Gemma4EmbeddingPreprocessor(std::filesystem::path const& engineDir,
    LLMEngineConfig const& config, int32_t maxBatchSize, int32_t maxSeqLen, TensorMap& tensorMap, cudaStream_t stream,
    std::optional<Tensor> checkpointTable)
    : mConfig(config)
{
    ELLM_CHECK(mConfig.pleEnabled, "Gemma4EmbeddingPreprocessor constructed while PLE is disabled");
    ELLM_CHECK(maxBatchSize > 0, "Gemma4EmbeddingPreprocessor requires positive max batch size");
    ELLM_CHECK(maxSeqLen > 0, "Gemma4EmbeddingPreprocessor requires positive max sequence length");

    Coords pleShape;
    if (storageBackedPleRequested() && !checkpointTable.has_value())
    {
        std::filesystem::path const plePath = engineDir / binding_names::kPleEmbeddingFileName;
        mPleMapping = std::make_unique<file_io::MmapReader>(plePath);
        safetensors::FileMetadata const metadata
            = safetensors::parseMetadata(mPleMapping->getData(), mPleMapping->getSize(), plePath.string());
        ELLM_CHECK(metadata.tensors.size() == 1,
            "ple_embedding.safetensors must contain exactly one tensor named weight");
        safetensors::TensorMetadata const& entry = metadata.tensors.front();
        ELLM_CHECK(entry.name == "weight", "ple_embedding.safetensors tensor must be named weight");
        ELLM_CHECK(entry.shape.getNumDims() == 2, "PLE table must be 2D [vocab, num_layers * hidden]");
        ELLM_CHECK(entry.dataType == nvinfer1::DataType::kHALF || entry.dataType == nvinfer1::DataType::kBF16,
            "PLE table must be FP16 or BF16");

        pleShape = entry.shape;
        mPleDataType = entry.dataType;
        mPleVocabSize = entry.shape[0];
        mPleRowBytes = static_cast<size_t>(entry.shape[1]) * utils::getTypeSize(entry.dataType);
        ELLM_CHECK(static_cast<size_t>(entry.shape[0]) * mPleRowBytes == entry.bytes,
            "PLE table byte count does not match its shape");
        mPlePayload = mPleMapping->getByteData() + metadata.dataOffset + entry.offset;

        // Random access suppresses multi-megabyte readahead when each prompt only
        // touches a small, sparse subset of the 4.4 GiB token table.
        (void) madvise(const_cast<int8_t*>(mPleMapping->getByteData()), mPleMapping->getSize(), MADV_RANDOM);
        mHostTokenIds = Tensor(
            {maxBatchSize, maxSeqLen}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "Gemma4PLE::hostTokenIds");
        mHostPackedPle = Tensor({mConfig.numPleInputs, maxBatchSize, maxSeqLen, mConfig.pleHiddenSize},
            DeviceType::kCPU, mPleDataType, "Gemma4PLE::hostPackedRows");
        mStorageBackedPle = true;
    }
    else
    {
        if (checkpointTable.has_value())
        {
            mPleTable = std::move(*checkpointTable);
        }
        else
        {
            std::filesystem::path const plePath = engineDir / binding_names::kPleEmbeddingFileName;
            std::vector<Tensor> pleTensors;
            ELLM_CHECK(safetensors::loadSafetensors(plePath, pleTensors, stream),
                "Failed to load " + std::string(binding_names::kPleEmbeddingFileName)
                    + " from model directory: " + engineDir.string());
            ELLM_CHECK(
                pleTensors.size() == 1, "ple_embedding.safetensors must contain exactly one tensor named weight");
            ELLM_CHECK(pleTensors[0].getName() == "weight", "ple_embedding.safetensors tensor must be named weight");
            mPleTable = std::move(pleTensors[0]);
        }
        pleShape = mPleTable.getShape();
        mPleDataType = mPleTable.getDataType();
        mPleVocabSize = pleShape[0];
    }

    ELLM_CHECK(pleShape.getNumDims() == 2, "PLE table must be 2D [vocab, num_layers * hidden]");
    ELLM_CHECK(pleShape[1] == static_cast<int64_t>(mConfig.numPleInputs) * mConfig.pleHiddenSize,
        "PLE table second dimension must equal num_ple_inputs * ple_hidden_size");
    ELLM_CHECK(mPleDataType == nvinfer1::DataType::kHALF || mPleDataType == nvinfer1::DataType::kBF16,
        "PLE table must be FP16 or BF16");

    mPleOutputBuffer = Tensor({mConfig.numPleInputs, maxBatchSize, maxSeqLen, mConfig.pleHiddenSize}, DeviceType::kGPU,
        mPleDataType, "Gemma4EmbeddingPreprocessor::mPleOutputBuffer");

    mPleOutputViews.reserve(mConfig.numPleInputs);
    for (int32_t idx = 0; idx < mConfig.numPleInputs; ++idx)
    {
        mPleOutputViews.emplace_back(makeOutputViewForLayer(idx, maxBatchSize, maxSeqLen));
        tensorMap.set(mPleOutputViews.back().getName(), mPleOutputViews.back());
    }

    LOG_INFO("Initialized Gemma4 PLE preprocessor: table=%s outputBuffer=%s numPleInputs=%d pleHiddenSize=%d "
             "storageBacked=%d",
        pleShape.formatString().c_str(), mPleOutputBuffer.getShape().formatString().c_str(), mConfig.numPleInputs,
        mConfig.pleHiddenSize, static_cast<int32_t>(mStorageBackedPle));
}

Tensor Gemma4EmbeddingPreprocessor::makeOutputViewForLayer(int32_t layerIdx, int64_t batchSize, int64_t seqLen)
{
    ELLM_CHECK(layerIdx >= 0 && layerIdx < mConfig.numPleInputs, "Gemma4 PLE layer index out of range");
    auto const outputShape = mPleOutputBuffer.getShape();
    ELLM_CHECK(batchSize > 0, "Gemma4 PLE batch size must be positive");
    ELLM_CHECK(seqLen > 0, "Gemma4 PLE sequence length must be positive");
    ELLM_CHECK(batchSize <= outputShape[1], "Gemma4 PLE batch size exceeds output buffer capacity");
    ELLM_CHECK(seqLen <= outputShape[2], "Gemma4 PLE sequence length exceeds output buffer capacity");

    int64_t const layerOutputCapacityBytes = outputShape[1] * outputShape[2] * mConfig.pleHiddenSize
        * static_cast<int64_t>(utils::getTypeSize(mPleDataType));
    void* const layerOutputPtr
        = static_cast<void*>(static_cast<char*>(mPleOutputBuffer.rawPointer()) + layerIdx * layerOutputCapacityBytes);
    return Tensor(layerOutputPtr, Coords{batchSize, seqLen, mConfig.pleHiddenSize}, DeviceType::kGPU, mPleDataType,
        binding_names::formatPleTokenEmbedsName(layerIdx));
}

void Gemma4EmbeddingPreprocessor::reshapeOutputs(int64_t batchSize, int64_t seqLen)
{
    for (int32_t idx = 0; idx < mConfig.numPleInputs; ++idx)
    {
        mPleOutputViews[idx] = makeOutputViewForLayer(idx, batchSize, seqLen);
    }
}

void Gemma4EmbeddingPreprocessor::embed(Tensor const& tokenIds, cudaStream_t stream)
{
    auto const tokenShape = tokenIds.getShape();
    ELLM_CHECK(tokenShape.getNumDims() == 2, "Gemma4 PLE token IDs must be [batch, seq_len]");
    reshapeOutputs(tokenShape[0], tokenShape[1]);
    if (!mStorageBackedPle)
    {
        kernel::gemma4PleGather(tokenIds, mPleTable, mPleOutputBuffer, mConfig.numPleInputs, mConfig.pleHiddenSize,
            mConfig.imageTokenId, mConfig.audioTokenId, stream);
        return;
    }

    int64_t const batchSize = tokenShape[0];
    int64_t const seqLen = tokenShape[1];
    int64_t const tokenCount = batchSize * seqLen;
    size_t const tokenBytes = static_cast<size_t>(tokenCount) * sizeof(int32_t);
    CUDA_CHECK(cudaMemcpyAsync(
        mHostTokenIds.rawPointer(), tokenIds.rawPointer(), tokenBytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const* hostTokenIds = mHostTokenIds.dataPointer<int32_t>();
    auto* packedRows = static_cast<int8_t*>(mHostPackedPle.rawPointer());
    size_t const hiddenBytes = static_cast<size_t>(mConfig.pleHiddenSize) * utils::getTypeSize(mPleDataType);
    size_t const activeLayerBytes = static_cast<size_t>(tokenCount) * hiddenBytes;
    for (int32_t layerIdx = 0; layerIdx < mConfig.numPleInputs; ++layerIdx)
    {
        for (int64_t tokenOffset = 0; tokenOffset < tokenCount; ++tokenOffset)
        {
            int32_t const tokenId = hostTokenIds[tokenOffset];
            int8_t* const destination = packedRows + static_cast<size_t>(layerIdx) * activeLayerBytes
                + static_cast<size_t>(tokenOffset) * hiddenBytes;
            bool const zeroFill = tokenId < 0 || tokenId >= mPleVocabSize
                || (mConfig.imageTokenId >= 0 && tokenId == mConfig.imageTokenId)
                || (mConfig.audioTokenId >= 0 && tokenId == mConfig.audioTokenId);
            if (zeroFill)
            {
                std::memset(destination, 0, hiddenBytes);
                continue;
            }
            int8_t const* const source = mPlePayload + static_cast<size_t>(tokenId) * mPleRowBytes
                + static_cast<size_t>(layerIdx) * hiddenBytes;
            std::memcpy(destination, source, hiddenBytes);
        }
    }

    auto const outputShape = mPleOutputBuffer.getShape();
    size_t const outputLayerPitch
        = static_cast<size_t>(outputShape[1] * outputShape[2]) * hiddenBytes;
    CUDA_CHECK(cudaMemcpy2DAsync(mPleOutputBuffer.rawPointer(), outputLayerPitch, packedRows, activeLayerBytes,
        activeLayerBytes, mConfig.numPleInputs, cudaMemcpyHostToDevice, stream));
}

} // namespace rt
} // namespace trt_edgellm
