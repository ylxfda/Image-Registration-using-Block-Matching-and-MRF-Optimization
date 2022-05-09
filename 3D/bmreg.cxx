/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include <stdio.h>
#include <string.h>
#include <iostream>
#include "itkIndex.h"
#include "itkImage.h"
#include "itkRGBPixel.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkImageDuplicator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkLineIterator.h"
#include "itkMultiThreader.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkMaskFeaturePointSelectionFilter.h"
 //#include "itkBlockMatchingImageFilterMy.h"
#include "itkScalarToRGBColormapImageFilter.h"
#include "itkTranslationTransform.h"
#include "itkResampleImageFilter.h"

#include "itkBlockMatchingImageRegistrationFilter.h"

int main(int argc, char * argv[])
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << std::endl;
		std::cerr << " itkitkBlockMatchingImageFilterTest fixedImg movingImg warpedFix warpedMov fld2fix fld2mov gridSize numLevel reguFactor sampRate blurSize thread numSteps simMetric" << std::endl;
		//                                                   1        2         3         4         5        6       7       8         9          10      11       12     13          14
		return EXIT_FAILURE;
	}

	// set max num of threads
	itk::MultiThreader::SetGlobalMaximumNumberOfThreads(atoi(argv[12]));
	itk::MultiThreader::SetGlobalDefaultNumberOfThreads(atoi(argv[12]));

	//const double selectFraction = 0.01;

	typedef unsigned short                  InputPixelType;
	typedef unsigned short				  OutputPixelType;
	static const unsigned int Dimension = 3;

	typedef itk::Image< InputPixelType, Dimension >  InputImageType;
	typedef itk::Image< OutputPixelType, Dimension >  OutputImageType;

	typedef itk::ImageFileReader< InputImageType >  ReaderType;

	//Set up the reader
	ReaderType::Pointer fixedreader = ReaderType::New();
	fixedreader->SetFileName(argv[1]);
	try
	{
		fixedreader->Update();
	}
	catch (itk::ExceptionObject & e)
	{
		std::cerr << "Error in reading the input image: " << e << std::endl;
		return EXIT_FAILURE;
	}

	ReaderType::Pointer movingreader = ReaderType::New();
	movingreader->SetFileName(argv[2]);
	try
	{
		movingreader->Update();
	}
	catch (itk::ExceptionObject & e)
	{
		std::cerr << "Error in reading the input image: " << e << std::endl;
		return EXIT_FAILURE;
	}

	typedef itk::BlockMatchingImageRegistrationFilter< InputImageType > BMRegistrationType;
	BMRegistrationType::Pointer bmreg = BMRegistrationType::New();

	bmreg->SetFixedImage(fixedreader->GetOutput());
	bmreg->SetMovingImage(movingreader->GetOutput());
	bmreg->SetGridSize(atoi(argv[7]));
	bmreg->SetGaussianBlurFactor(atoi(argv[11]));
	bmreg->SetNumOfStep(atoi(argv[13]));
	bmreg->SetSamplingRate(atof(argv[10]));
	bmreg->SetNumOfLevels(atoi(argv[8]));
	bmreg->SetSmoothness(atof(argv[9]));
	bmreg->SetMetricSelected(atoi(argv[14]));

	bmreg->Update();

	bmreg->outputResults(argv[3], argv[4], argv[5], argv[6]);

	return EXIT_SUCCESS;
}
