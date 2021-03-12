/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 //!
 //! demo.cpp
 //! This file contains the implementation of the ONNX Model sample. It creates the network using
 //! the resnet18.onnx model.
 //! It can be run with the following command line:
 //! Command: ./OnnxTrtDemo [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
 //! [--useDLACore=<int>] [--int8 or --fp16]
 //!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include <opencv2/opencv.hpp>
#include "image.hpp"

#include "NvInfer.h"
#include "test_time.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>


const std::string DemoName = "TensorRT_Onnx.demo";

class OnnxTrtClassify
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
	static const int INPUT_H = 224;
	static const int INPUT_W = 224;
	static const int INPUT_C = 3;
	static const int OUTPUT_SIZE = 5;
	const char* INPUT_BLOB_NAME = "input";
	const char* OUTPUT_BLOB_NAME = "output";

public:
    OnnxTrtClassify(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }
    //! \brief Function builds the network engine
    bool build();

    //! \brief Runs the TensorRT inference engine for this sample
    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.
    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    //!
    //! \brief Parses an ONNX model  and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //! \brief Reads the input  and stores the result in a managed buffer
    bool processInput(const samplesCommon::BufferManager& buffers);

    //! \brief Classifies digits and verify result
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};


//! \brief Creates the network, configures the builder and creates the network engine
//! \return Returns true if the engine was created successfully and false otherwise
bool OnnxTrtClassify::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    //Build  Engine and save as plan file
    //The engine needs to be built for the first run, and the if statement is turned on. 
    //After the engine is built, the if statement can be closed and the code block that loads the plan file can be opened.
    if (false)
    {
		mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
			builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

		if (!mEngine)
		{
			return false;
		}
	    /*Save Serialize File*/
	    const char filename[] = "data/common/resnet18.plan";
	    nvinfer1::IHostMemory* trtModelStream = mEngine->serialize();
	    std::ofstream file;
	    file.open(filename, std::ios::binary | std::ios::out);
	    cout << "writing engine file..." << endl;
	    file.write((const char*)trtModelStream->data(), trtModelStream->size());
	    cout << "save engine file done" << endl;
	    file.close();

	    /*Verify that the engine is stored correctly */
	    std::fstream file_verify;
	    const std::string engineFile = "data/common/resnet18.plan";
	    file_verify.open(engineFile, std::ios::binary | std::ios::in);
	    file_verify.seekg(0, std::ios::end);
	    int length = file_verify.tellg();

	    std::cout << "length:" << length << std::endl;
	    file_verify.seekg(0, std::ios::beg);
	    std::unique_ptr<char[]> data(new char[length]);
	    file_verify.read(data.get(), length);

	    assert(trtModelStream->data == data);
	    assert(trtModelStream->size() == length);

	    /*Destroy modelstream*/
	    trtModelStream->destroy();
    }

   // After the engine is builtand serializedand stored, the plan file can be directly loadedand deserialized
    if (true)
    {
        /*Read plan File and deserializeCudaEngine*/
        const std::string engineFile = "data/common/resnet18.plan";
        std::fstream file;

        std::cout << "Loading Filename From:" << engineFile << std::endl;

        nvinfer1::IRuntime* trtRuntime;
        file.open(engineFile, std::ios::binary | std::ios::in);
        file.seekg(0, std::ios::end);
        int length = file.tellg();

        std::cout << "Length:" << length << std::endl;

        file.seekg(0, std::ios::beg);
        std::unique_ptr<char[]> data(new char[length]);
        file.read(data.get(), length);
        file.close();

        std::cout << "Load Engine Done" << std::endl;
        std::cout << "Deserializing" << std::endl;
        trtRuntime = createInferRuntime(sample::gLogger.getTRTLogger());
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(trtRuntime->deserializeCudaEngine(data.get(), length, nullptr)
        , samplesCommon::InferDeleter());

        std::cout << "Deserialize Done" << std::endl;
        assert(mEngine != nullptr);
        std::cout << "The engine in TensorRT.cpp is not nullptr" << std::endl;
        //DO NOT DESTORY! when verifying
        trtRuntime->destroy();

        ///*Verify the correctness of serialization and deserialization*/
        //nvinfer1::IHostMemory* trtModelStream{ nullptr };
        //trtModelStream = mEngine->serialize();
        //mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(trtRuntime->deserializeCudaEngine(trtModelStream->data(),
        //trtModelStream->size(), nullptr), samplesCommon::InferDeleter());
        //assert(mEngine != nullptr);
        //trtModelStream->destroy();
        //trtRuntime->destroy();
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    assert(mOutputDims.nbDims == 2);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the  Network and marks the output layers
//!
//! \param network Pointer to the network that will be populated with the  network
//!
//! \param builder Pointer to the engine builder
//!
bool OnnxTrtClassify::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }
    //setMaxWorkspaceSize
    config->setMaxWorkspaceSize(2048_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool OnnxTrtClassify::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    // Test infer times. Loop a thousand times
	  CSpendTime time;
	  time.Start();
	for (int i = 0; i < 1000; i++)
	{
		bool status = context->executeV2(buffers.getDeviceBindings().data());
		if (!status)
		{
			return false;
		}
	}
	  double dTime = time.End();
      sample::gLogInfo << " Time Used " << dTime / 1000 << std::endl;;

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool OnnxTrtClassify::processInput(const samplesCommon::BufferManager& buffers)
{

 	std::cout << "Loading..." << std::endl;
 	cv::Mat image = cv::imread(locateFile("turkish_coffee.jpg", mParams.dataDirs), cv::IMREAD_COLOR);
 	if (image.empty()) {
 		std::cout << "The input image is empty. Please check....." << std::endl;
 	}
 	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
 	cv::Mat dst = cv::Mat::zeros(INPUT_H, INPUT_W, CV_32FC3);
 	cv::resize(image, dst, dst.size());
 	float* data = normal(dst);
	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
	
    for (int i = 0; i < INPUT_H * INPUT_W * INPUT_C; i++)
	{
		hostDataBuffer[i] = data[i];
	}

    delete data;
    image.release();
    dst.release();
    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool OnnxTrtClassify::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = mOutputDims.d[1];
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0f};
    int idx{0};

    // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    sample::gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i])
        {
            idx = i;
        }
        sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
                         << " "
                         << "Class " << i << ": " << std::string(int(std::floor(output[i] *10)), '*')
                         << std::endl;
    }
    sample::gLogInfo <<" Result "<< idx<<" "<< output[idx] << std::endl;;

    return idx;
/*    return true;*/
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/common/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "resnet18.onnx";
    params.inputTensorNames.push_back("input");
    params.outputTensorNames.push_back("output");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./demo_onnx [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/common/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(DemoName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    OnnxTrtClassify sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx Network" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    system("pause");

    return sample::gLogger.reportPass(sampleTest);

}
