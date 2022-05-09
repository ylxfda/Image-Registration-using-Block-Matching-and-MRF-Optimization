#ifndef __itkBlockMatchingImageRegistrationFilter_hxx
#define __itkBlockMatchingImageRegistrationFilter_hxx

#include "itkBlockMatchingImageRegistrationFilter.h"
#include "GC.h"
#include "graph.h"
#include "itkObjectFactory.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkExtractImageFilter.h"
#include "itkConstantPadImageFilter.h"
#include <itkDiscreteGaussianImageFilter.h>
#include "itkComposeImageFilter.h"
#include "itkVectorImage.h"
#include "itkIdentityTransform.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include <itkWarpImageFilter.h>
#include "itkCropImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkImageAdaptor.h"
#include "itkImageRegionIteratorWithIndex.h"
#include <itkComposeDisplacementFieldsImageFilter.h>
#include "itkInvertDisplacementFieldImageFilter.h"
#include "itkVectorIndexSelectionCastImageFilter.h"
//#include "itkVectorScaleImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkGaussianDistribution.h"
#include <omp.h>
#include <time.h>
//#include "mpi.h"
#include <sys/time.h>
#include <itkMattesMutualInformationImageToImageMetric.h>
#include <itkNormalizedCorrelationImageToImageMetric.h>

namespace itk {

    template< typename TImage>
    BlockMatchingImageRegistrationFilter<TImage>::BlockMatchingImageRegistrationFilter() {

        this->SetNumberOfRequiredInputs(2);
        this->SetSmoothness(0.1);
        this->SetMetricFactor(1.5);
        this->SetEdgeFactor(2.5);

        m_numBinFix = 64;
        m_numBinMov = 64;
        m_binMinimumFix = -0.0;
        m_binMaximumFix = 600.5;
        m_binMinimumMov = -0.0;
        m_binMaximumMov = 600.5;

        // set the histogram filter
        this->histogramFilterJointFix = HistogramFilterType::New();
        this->histogramFilterJointMov = HistogramFilterType::New();

        this->probabilityJointFix = Allocate2DArray<float>(m_numBinFix, m_numBinMov);
        this->probabilityJointMov = Allocate2DArray<float>(m_numBinMov, m_numBinFix);

        this->probabilityMov = new float[m_numBinMov];
        this->probabilityFix = new float[m_numBinFix];
        this->probabilityMovWarped = new float[m_numBinMov];
        this->probabilityFixWarped = new float[m_numBinFix];

    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::SetFixedImage(const TImage* image) {
        this->SetNthInput(0, const_cast<TImage*> (image));
    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::SetMovingImage(const TImage* image) {
        this->SetNthInput(1, const_cast<TImage*> (image));
    }

    template< class TImage >
    typename TImage::ConstPointer BlockMatchingImageRegistrationFilter<TImage>::GetFixedImage() {
        return static_cast<const TImage *>
                (this->ProcessObject::GetInput(0));
    }

    template< class TImage >
    typename TImage::ConstPointer BlockMatchingImageRegistrationFilter<TImage>::GetMovingImage() {
        return static_cast<const TImage *>
                (this->ProcessObject::GetInput(1));
    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::GenerateData() {

        m_inputImgSpacing = this->GetFixedImage()->GetSpacing();

        for (int l = 0; l < this->GetNumOfLevels(); l++) {

            std::cout << "==================== Level " << l << "====================" << std::endl;
            //this->SetGaussianBlurVariance(this->GetGridSize() / this->GetGaussianBlurFactor() / this->GetNumOfStep());
            float variance = this->GetGridSize() * m_inputImgSpacing[0] / this->GetNumOfStep() * this->GetGaussianBlurFactor();
            this->SetGaussianBlurVariance(variance);
            //this->SetGaussianBlurVariance(0.5); // test not using Gau bluring
            //if (this->GetGridSize() / 2 / this->GetNumOfStep() < 1.0) { // when step size < 1.0
            //	this->SetNumOfStep(this->GetGridSize()); // use 0.5 as step size
            //}
            //this->SetSmoothness(this->smoothness[l]);
            this->m_level = l;

            //if (l==this->GetNumOfLevels()-1) {
            this->SetNumOfStep(5);

            ComputeDfmfld(l);

            //this->SetGridSize(this->GetGridSize() / 1.2);
            this->SetGridSize(this->GetGridSize() - 6);
            //this->SetNumOfStep(this->GetNumOfStep() / 2.0);
            this->SetSmoothness(this->GetSmoothness() * 0.98);
            this->SetMetricFactor(this->GetMetricFactor() * 0.8);
            this->SetEdgeFactor(this->GetEdgeFactor() * 0.8);

            /*int NumOfStepNext = floor(this->GetNumOfStep() / 1.2);
            if (NumOfStepNext > 3) {
                    this->SetNumOfStep(NumOfStepNext);
            }
            else {
                    this->SetNumOfStep(3);
            }*/

            std::cout << std::endl;
        }

    }

    template< typename TImage >
    void BlockMatchingImageRegistrationFilter<TImage>::ComputeDfmfld(int level) {

        // the real work start from here
        // prepare the image, padding and smoothing etc
        //clock_t tStart;
        struct timeval tpstart, tpend;
        double timeuse;
        int iter = 3;

        for (int it = 0; it < iter; it++) {

            printf("-----------iter %d------------\n", it);
            /* Do your stuff here */
            gettimeofday(&tpstart, NULL);
            printf("Image preparation ... \n");
            this->PrepareImageForMatch(level, it);
            gettimeofday(&tpend, NULL);
            timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
            timeuse /= 1000000;
            printf("Preparation done %.2fs\n", timeuse);

            // make the objective function
            int grid_num_x = (int) floor(float(this->internalImgSize[1] / this->GetGridSize())); // the number of grids in each direction
            int grid_num_y = (int) floor(float(this->internalImgSize[0] / this->GetGridSize()));
            int grid_num_z = (int) floor(float(this->internalImgSize[2] / this->GetGridSize()));

            int contPt_num_x = grid_num_x + 1; // +1 to have the number of control point
            int contPt_num_y = grid_num_y + 1; //
            int contPt_num_z = grid_num_z + 1;

            int numPoints = contPt_num_x * contPt_num_y*contPt_num_z; // the num of control points on the grid
            int numLabels = (int) powf((this->GetNumOfStep() * 2 + 1), 3.0); // 3D image

            // count the number of edges in the grid cube
            unsigned int numPairs = 0;
            for (int i = 0; i <= contPt_num_z - 1; i++) {
                for (int j = 0; j <= contPt_num_x - 1; j++) {
                    for (int k = 0; k <= contPt_num_y - 1; k++) {

                        if ((i - 1) >= 0) numPairs++;
                        if ((i + 1) <= contPt_num_z - 1) numPairs++;

                        if ((j - 1) >= 0) numPairs++;
                        if ((j + 1) <= contPt_num_x - 1) numPairs++;

                        if ((k - 1) >= 0) numPairs++;
                        if ((k + 1) <= contPt_num_y - 1) numPairs++;
                    }
                }
            }

            numPairs = numPairs / 2; // every edge were counted for twice

            //int numPairs = numPoints*6 - 8*3 - (contPt_num_row+contPt_num_col-4)*4*2 - ; //assuming all have 6 neighbors than minus the surface points
            int max_iters = 1000;

            // puts some information
            std::cout << "grid size: " << this->GetGridSize() << " " << grid_num_x << "x" << grid_num_y << "x" << grid_num_z << std::endl;
            std::cout << "number of steps: " << this->GetNumOfStep() << std::endl;
            std::cout << "number of labels: " << numLabels << std::endl;
            std::cout << "number of control points: " << numPoints << std::endl;
            std::cout << "number of pairs: " << numPairs << std::endl;
            std::cout << "the smoothness factor: " << this->GetSmoothness() << std::endl;

            float *labelCosts = new float[numPoints * numLabels];
            int *edges = new int[numPairs * 2];
            //float  *dist  = new float[numlabels*numlabels];
            float *wCosts = new float[numPairs];
            //float weightUnary = float(numPairs / numPoints);
            float weightUnary = 1.0;

            for (unsigned int i = 0; i < numPairs; i++) {
                wCosts[i] = this->GetSmoothness();
            }

            //this->CreateLabelCostTabel(this->GetNumOfStep(), (float) this->GetGridSize() / (float) this->GetNumOfStep() / 4.0, 1.4, 0.8);
            this->CreateLabelCostTabel(this->GetNumOfStep(), 1.0, 1.0, 0.8);
            //this->CreateLabelCostTabel(this->GetNumOfStep(), (float)this->GetGridSize() / (float)this->GetNumOfStep(), 1.0);

            // define the sampling pattern
            int** offset; // an array to hold the random sampling pattern
            float* weights; // an array to hold the random sampling weight

            int numOfSample = 0;
            int minimum = 100;

            int numVoxInGrid = pow(this->GetGridSize(), 3);

            if (numVoxInGrid * this->GetSamplingRate() > minimum) {
                numOfSample = numVoxInGrid * this->GetSamplingRate();
                offset = Allocate2DArray<int>(numOfSample, 3);
                weights = new float[numOfSample];
                RandomSampling(floor((float) this->GetGridSize() / 3), floor((float) this->GetGridSize() / 3), floor((float) this->GetGridSize() / 3), offset, weights, numOfSample);
                std::cout << "sampling at rate: " << this->GetSamplingRate() << std::endl;
            } else {

                numOfSample = minimum;
                offset = Allocate2DArray<int>(numOfSample, 3);
                weights = new float[numOfSample];
                RandomSampling(floor((float) this->GetGridSize() / 3), floor((float) this->GetGridSize() / 3), floor((float) this->GetGridSize() / 3), offset, weights, numOfSample);
                std::cout << "sampling at minimum: " << numOfSample << std::endl;
            }


            std::cout << "The number of samples in each grid: " << numOfSample << std::endl;

            // define the pairs
            bool onBorder = false;
            int center_x, center_y, center_z, p;
            int count = 0;
            float ncc = 0.0;
            float flexibility = 1.0;
            float boarder_edge_weight = this->GetSmoothness();

            for (int i = 0; i <= contPt_num_z - 1; i++) {
                for (int j = 0; j <= contPt_num_x - 1; j++) {
                    for (int k = 0; k <= contPt_num_y - 1; k++) {

                        //std::cout << " i j k: " << i <<" "<<j<<" "<<k<< std::endl;

                        center_x = this->GetGridSize() * j;
                        center_y = this->GetGridSize() * k;
                        center_z = this->GetGridSize() * i;

                        if ((j <= 0) || (j + 1 >= contPt_num_x - 1) ||
                                (k <= 0) || (k + 1 >= contPt_num_y - 1) ||
                                (i <= 0) || (i + 1 >= contPt_num_z - 1)) {
                            onBorder = true;
                        } else {
                            onBorder = false;
                        }

                        /*if ((i - 1) >= 0) {
                        edges[count * 2] = k + j*contPt_num_y + i*contPt_num_x*contPt_num_y;
                        edges[count * 2 + 1] = k + j*contPt_num_x + (i - 1)*contPt_num_x*contPt_num_y;
                        count++;
                  }*/
                        if ((i + 1) <= contPt_num_z - 1) {
                            edges[count * 2] = k + j * contPt_num_y + i * contPt_num_x*contPt_num_y;
                            edges[count * 2 + 1] = k + j * contPt_num_y + (i + 1) * contPt_num_x*contPt_num_y;

                            ncc = NCC(fixedImg4Match, fixedImg4Match, center_x, center_y, center_z, 0, 0, this->GetGridSize(), offset, weights, numOfSample, onBorder);
                            wCosts[count] = this->GetSmoothness() * powf(flexibility, powf(ncc, 2)) / weightUnary;

                            if ((i == 0) || (i + 1) == contPt_num_z - 1) wCosts[count] = boarder_edge_weight;
                            //std::cout << "weight: " << wCosts[count] << std::endl;

                            count++;
                        }


                        /*if ((j - 1) >= 0) {
                        edges[count * 2] = k + j*contPt_num_y + i*contPt_num_x*contPt_num_y;
                        edges[count * 2 + 1] = k + (j - 1)*contPt_num_x + i*contPt_num_x*contPt_num_y;
                        count++;
                  }*/
                        if ((j + 1) <= contPt_num_x - 1) {
                            edges[count * 2] = k + j * contPt_num_y + i * contPt_num_x*contPt_num_y;
                            edges[count * 2 + 1] = k + (j + 1) * contPt_num_y + i * contPt_num_x*contPt_num_y;

                            ncc = NCC(fixedImg4Match, fixedImg4Match, center_x, center_y, center_z, this->GetGridSize(), 0, 0, offset, weights, numOfSample, onBorder);
                            wCosts[count] = this->GetSmoothness() * powf(flexibility, powf(ncc, 2)) / weightUnary;

                            if ((j == 0) || (j + 1) == contPt_num_x - 1) wCosts[count] = boarder_edge_weight;
                            //std::cout << "weight: " << wCosts[count] << std::endl;

                            count++;
                        }

                        /*if ((k - 1) >= 0) {
                        edges[count * 2] = k + j*contPt_num_y + i*contPt_num_x*contPt_num_y;
                        edges[count * 2 + 1] = k - 1 + j*contPt_num_x + i*contPt_num_x*contPt_num_y;
                        count++;
                    }*/
                        if ((k + 1) <= contPt_num_y - 1) {
                            edges[count * 2] = k + j * contPt_num_y + i * contPt_num_x*contPt_num_y;
                            edges[count * 2 + 1] = k + 1 + j * contPt_num_y + i * contPt_num_x*contPt_num_y;

                            ncc = NCC(fixedImg4Match, fixedImg4Match, center_x, center_y, center_z, 0, this->GetGridSize(), 0, offset, weights, numOfSample, onBorder);
                            wCosts[count] = this->GetSmoothness() * powf(flexibility, powf(ncc, 2)) / weightUnary;

                            if ((k == 0) || (k + 1) == contPt_num_y - 1) wCosts[count] = boarder_edge_weight;
                            //std::cout << "weight: " << wCosts[count] << std::endl;

                            count++;
                        }
                    }
                }
            }


            std::cout << "number of edges added: " << count << std::endl;

            //exit(0);
            //debug only
            //std::cout << "label2displacement[9000][1]: " << label2displacement[9000][1] << std::endl;

            //RandomSampling(floor((float)this->GetGridSize() / 6), floor((float)this->GetGridSize() / 6), floor((float)this->GetGridSize() / 6), offset, numOfSample);

            // compute the similarity cost (takes most of the time, can be paralelled)
            printf("Metric computing ...");
            gettimeofday(&tpstart, NULL);
            omp_set_dynamic(0); // Explicitly disable dynamic teams
            omp_set_num_threads(18); // Use 4 threads for all consecutive parallel regions                
#pragma omp parallel for private(center_x, center_y, center_z, p, onBorder)
            for (int z = 0; z <= contPt_num_z - 1; z++) {
                for (int x = 0; x <= contPt_num_x - 1; x++) {
                    for (int y = 0; y <= contPt_num_y - 1; y++) {
                        //get the position of this point
                        center_x = this->GetGridSize() * x;
                        center_y = this->GetGridSize() * y;
                        center_z = this->GetGridSize() * z;
                        p = z * contPt_num_y * contPt_num_x + x * contPt_num_y + y;

                        if ((x - 1 <= 0) || (x + 1 >= contPt_num_x - 1) ||
                                (y - 1 <= 0) || (y + 1 >= contPt_num_y - 1) ||
                                (z - 1 <= 0) || (z + 1 >= contPt_num_z - 1)) {
                            onBorder = true;
                        } else {
                            onBorder = false;
                        }

                        //std::cout<<"computing the cost for control points #"<<p<<" position: "<<center_y<<" "<<center_x<< " " << center_x<<std::endl;
                        for (int l = 0; l < numLabels; l++) {
                            if (this->m_MetricSelected == 0) {
                                //float cost = NCC(fixedImg4Match, movingImg4Match, center_x, center_y, this->label2displacement[l][1], this->label2displacement[l][0], offset, numOfSample);
                                labelCosts[l * numPoints + p] = NCC(movingImgTemplate, fixedImg4Match, center_x, center_y, center_z, this->label2displacement[l][1], this->label2displacement[l][0], this->label2displacement[l][2], offset, weights, numOfSample, onBorder);
                                continue;
                            } else if (this->m_MetricSelected == 1) {
                                labelCosts[l * numPoints + p] = PWMI(movingImgTemplate, fixedImg4Match, center_x, center_y, center_z,
                                        this->label2displacement[l][1], this->label2displacement[l][0], this->label2displacement[l][2],
                                        offset, numOfSample, onBorder, probabilityJointMov, probabilityMov, probabilityFixWarped);
                                continue;

                                //count++;
                                //lcosts[ p*numlabels + l ] = cost;
                                //std::cout << "label #" << l << " move: " << this->label2displacement[l][1] << " " << this->label2displacement[l][0] << " " << this->label2displacement[l][2]
                                //        << " cost: " << labelCosts[l * numPoints + p] << std::endl;
                            }
                        }
                    }
                }
            }

            gettimeofday(&tpend, NULL);
            timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
            timeuse /= 1000000;
            printf(" Done %.2fs\n", timeuse);
            //std::cout<<"total # probes"<<count<<std::endl;
            /*for (int i=0; i<numLabels*numLabels; i++) {
            std::cout<<label2cost[i]<<" ";
      }*/

            /*count=0;
            for (int i=0; i<numLabels*numPoints; i++) {
            std::cout<<labelCosts[i]<<" ";
            count++;
            }
            std::cout<<"total # probes"<<count<<std::endl;*/

            GCLib::GC pd_fix2mov(numPoints, numLabels, labelCosts, numPairs, edges, labelDiffCost, max_iters, wCosts);
            pd_fix2mov.run();

            // display the movement
            //for (int r=0; r<grid_row; r++) {
            //	for (int c=0; c<grid_col; c++) {
            //		int index = r*grid_col+c;
            //		center_y = this->GetGridSize()*(r+1);
            //		center_x = this->GetGridSize()*(c+1);
            //		//std::cout<<pd._pinfo[index].label<<std::endl;
            //		std::cout<<"control points #"<<index<<" position: "<<center_x<<" "<<center_y<<" move: "<<label2displacement[pd._pinfo[index].label][1]<<" "<<label2displacement[pd._pinfo[index].label][0]<<std::endl;
            //	}
            //}

            // create the grid image first
            typename InternalImageType::Pointer dfmgrid_fix2mov_x = InternalImageType::New();
            typename InternalImageType::Pointer dfmgrid_fix2mov_y = InternalImageType::New();
            typename InternalImageType::Pointer dfmgrid_fix2mov_z = InternalImageType::New();

            typename InternalImageType::SizeType size;
            typename InternalImageType::SpacingType spacing;
            //size[0] = grid_col+2;
            //size[1] = grid_row+2;
            size[0] = contPt_num_y;
            size[1] = contPt_num_x;
            size[2] = contPt_num_z;

            spacing[0] = this->GetGridSize() * m_inputImgSpacing[0];
            spacing[1] = this->GetGridSize() * m_inputImgSpacing[1];
            spacing[2] = this->GetGridSize() * m_inputImgSpacing[2];

            typename InternalImageType::RegionType region;
            typename InternalImageType::IndexType start;

            start[0] = 0;
            start[1] = 0;
            start[2] = 0;

            region.SetSize(size);
            region.SetIndex(start);

            dfmgrid_fix2mov_x->SetRegions(region);
            dfmgrid_fix2mov_y->SetRegions(region);
            dfmgrid_fix2mov_z->SetRegions(region);

            dfmgrid_fix2mov_x->Allocate();
            dfmgrid_fix2mov_y->Allocate();
            dfmgrid_fix2mov_z->Allocate();

            dfmgrid_fix2mov_x->SetSpacing(spacing);
            dfmgrid_fix2mov_y->SetSpacing(spacing);
            dfmgrid_fix2mov_z->SetSpacing(spacing);


            typename InternalImageType::IndexType Index;
            for (int z = 0; z <= contPt_num_z - 1; z++) {
                for (int x = 0; x <= contPt_num_x - 1; x++) {
                    for (int y = 0; y <= contPt_num_y - 1; y++) {

                        Index[2] = z;
                        Index[1] = x;
                        Index[0] = y;

                        //get the position of this point
                        p = z * contPt_num_y * contPt_num_x + x * contPt_num_y + y;

                        dfmgrid_fix2mov_z->SetPixel(Index, label2displacement[pd_fix2mov._pinfo[p].label][2] * m_inputImgSpacing[2]);
                        dfmgrid_fix2mov_x->SetPixel(Index, label2displacement[pd_fix2mov._pinfo[p].label][1] * m_inputImgSpacing[1]);
                        dfmgrid_fix2mov_y->SetPixel(Index, label2displacement[pd_fix2mov._pinfo[p].label][0] * m_inputImgSpacing[0]);
                    }
                }
            }

            //pd_fix2mov.~GC();

            WriteImage<InternalImageType>("dfmgrid_fix2mov_x.mhd", dfmgrid_fix2mov_x);
            WriteImage<InternalImageType>("dfmgrid_fix2mov_y.mhd", dfmgrid_fix2mov_y);
            WriteImage<InternalImageType>("dfmgrid_fix2mov_z.mhd", dfmgrid_fix2mov_z);
            std::cout << "fix2mov finished" << std::endl;

            // make the dfmfld in the other direction --------------------------------------------------------------
            //re-prepare the wCost etc.
            count = 0;
            for (int i = 0; i <= contPt_num_z - 1; i++) {
                for (int j = 0; j <= contPt_num_x - 1; j++) {
                    for (int k = 0; k <= contPt_num_y - 1; k++) {

                        //std::cout << " i j k: " << i <<" "<<j<<" "<<k<< std::endl;

                        center_x = this->GetGridSize() * j;
                        center_y = this->GetGridSize() * k;
                        center_z = this->GetGridSize() * i;

                        if ((j <= 0) || (j + 1 >= contPt_num_x - 1) ||
                                (k <= 0) || (k + 1 >= contPt_num_y - 1) ||
                                (i <= 0) || (i + 1 >= contPt_num_z - 1)) {
                            onBorder = true;
                        } else {
                            onBorder = false;
                        }

                        /*if ((i - 1) >= 0) {
                        edges[count * 2] = k + j*contPt_num_y + i*contPt_num_x*contPt_num_y;
                        edges[count * 2 + 1] = k + j*contPt_num_x + (i - 1)*contPt_num_x*contPt_num_y;
                        count++;
                  }*/
                        if ((i + 1) <= contPt_num_z - 1) {
                            edges[count * 2] = k + j * contPt_num_y + i * contPt_num_x*contPt_num_y;
                            edges[count * 2 + 1] = k + j * contPt_num_y + (i + 1) * contPt_num_x*contPt_num_y;

                            ncc = NCC(movingImg4Match, movingImg4Match, center_x, center_y, center_z, 0, 0, this->GetGridSize(), offset, weights, numOfSample, onBorder);
                            wCosts[count] = this->GetSmoothness() * powf(flexibility, powf(ncc, 2)) / weightUnary;

                            if ((i == 0) || (i + 1) == contPt_num_z - 1) wCosts[count] = boarder_edge_weight;
                            //std::cout << "weight: " << wCosts[count] << std::endl;

                            count++;
                        }


                        /*if ((j - 1) >= 0) {
                        edges[count * 2] = k + j*contPt_num_y + i*contPt_num_x*contPt_num_y;
                        edges[count * 2 + 1] = k + (j - 1)*contPt_num_x + i*contPt_num_x*contPt_num_y;
                        count++;
                  }*/
                        if ((j + 1) <= contPt_num_x - 1) {
                            edges[count * 2] = k + j * contPt_num_y + i * contPt_num_x*contPt_num_y;
                            edges[count * 2 + 1] = k + (j + 1) * contPt_num_y + i * contPt_num_x*contPt_num_y;

                            ncc = NCC(movingImg4Match, movingImg4Match, center_x, center_y, center_z, this->GetGridSize(), 0, 0, offset, weights, numOfSample, onBorder);
                            wCosts[count] = this->GetSmoothness() * powf(flexibility, powf(ncc, 2)) / weightUnary;

                            if ((j == 0) || (j + 1) == contPt_num_x - 1) wCosts[count] = boarder_edge_weight;
                            //std::cout << "weight: " << wCosts[count] << std::endl;

                            count++;
                        }

                        /*if ((k - 1) >= 0) {
                        edges[count * 2] = k + j*contPt_num_y + i*contPt_num_x*contPt_num_y;
                        edges[count * 2 + 1] = k - 1 + j*contPt_num_x + i*contPt_num_x*contPt_num_y;
                        count++;
                  }*/
                        if ((k + 1) <= contPt_num_y - 1) {
                            edges[count * 2] = k + j * contPt_num_y + i * contPt_num_x*contPt_num_y;
                            edges[count * 2 + 1] = k + 1 + j * contPt_num_y + i * contPt_num_x*contPt_num_y;

                            ncc = NCC(movingImg4Match, movingImg4Match, center_x, center_y, center_z, 0, this->GetGridSize(), 0, offset, weights, numOfSample, onBorder);
                            wCosts[count] = this->GetSmoothness() * powf(flexibility, powf(ncc, 2)) / weightUnary;

                            if ((k == 0) || (k + 1) == contPt_num_y - 1) wCosts[count] = boarder_edge_weight;
                            //std::cout << "weight: " << wCosts[count] << std::endl;

                            count++;
                        }
                    }
                }
            }
            std::cout << "Metric computing ...";
            gettimeofday(&tpstart, NULL);
            omp_set_dynamic(0); // Explicitly disable dynamic teams
            omp_set_num_threads(18); // Use 4 threads for all consecutive parallel regions
#pragma omp parallel for private(center_x, center_y, center_z, p, onBorder)                
            for (int z = 0; z <= contPt_num_z - 1; z++) {
                for (int x = 0; x <= contPt_num_x - 1; x++) {
                    for (int y = 0; y <= contPt_num_y - 1; y++) {
                        //get the position of this point
                        center_x = this->GetGridSize() * x;
                        center_y = this->GetGridSize() * y;
                        center_z = this->GetGridSize() * z;
                        p = z * contPt_num_y * contPt_num_x + x * contPt_num_y + y;

                        if ((x - 1 <= 0) || (x + 1 >= contPt_num_x - 1) ||
                                (y - 1 <= 0) || (y + 1 >= contPt_num_y - 1) ||
                                (z - 1 <= 0) || (z + 1 >= contPt_num_z - 1)) {
                            onBorder = true;
                        } else {
                            onBorder = false;
                        }

                        //std::cout<<"computing the cost for control points #"<<p<<" position: "<<center_y<<" "<<center_x<<std::endl;
                        for (int l = 0; l < numLabels; l++) {
                            if (this->m_MetricSelected == 0) {
                                //float cost = NCC(fixedImg4Match, movingImg4Match, center_x, center_y, this->label2displacement[l][1], this->label2displacement[l][0], offset, numOfSample);
                                labelCosts[l * numPoints + p] = NCC(fixedImgTemplate, movingImg4Match, center_x, center_y, center_z, this->label2displacement[l][1], this->label2displacement[l][0], this->label2displacement[l][2], offset, weights, numOfSample, onBorder);
                                continue;
                            } else if (this->m_MetricSelected == 1) {
                                labelCosts[l * numPoints + p] = PWMI(fixedImgTemplate, movingImg4Match, center_x, center_y, center_z,
                                        this->label2displacement[l][1], this->label2displacement[l][0], this->label2displacement[l][2],
                                        offset, numOfSample, onBorder, probabilityJointFix, probabilityFix, probabilityMovWarped);
                                continue;
                                //count++;
                                //lcosts[ p*numlabels + l ] = cost;
                                //std::cout<<"label #"<<l<<" move: "<<this->label2displacement[l][1]<<" "<<this->label2displacement[l][0]<<" cost: "<<cost<<std::endl;
                            }
                        }
                    }
                }
            }
            gettimeofday(&tpend, NULL);
            timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
            timeuse /= 1000000;
            printf(" Done %.2fs\n", timeuse);
            GCLib::GC pd_mov2fix(numPoints, numLabels, labelCosts, numPairs, edges, labelDiffCost, max_iters, wCosts);
            pd_mov2fix.run();

            typename InternalImageType::Pointer dfmgrid_mov2fix_x = InternalImageType::New();
            typename InternalImageType::Pointer dfmgrid_mov2fix_y = InternalImageType::New();
            typename InternalImageType::Pointer dfmgrid_mov2fix_z = InternalImageType::New();

            dfmgrid_mov2fix_x->SetRegions(region);
            dfmgrid_mov2fix_y->SetRegions(region);
            dfmgrid_mov2fix_z->SetRegions(region);

            dfmgrid_mov2fix_x->SetSpacing(spacing);
            dfmgrid_mov2fix_y->SetSpacing(spacing);
            dfmgrid_mov2fix_z->SetSpacing(spacing);

            dfmgrid_mov2fix_x->Allocate();
            dfmgrid_mov2fix_y->Allocate();
            dfmgrid_mov2fix_z->Allocate();

            for (int z = 0; z <= contPt_num_z - 1; z++) {
                for (int x = 0; x <= contPt_num_x - 1; x++) {
                    for (int y = 0; y <= contPt_num_y - 1; y++) {

                        Index[2] = z;
                        Index[1] = x;
                        Index[0] = y;

                        //get the position of this point
                        p = z * contPt_num_y * contPt_num_x + x * contPt_num_y + y;

                        dfmgrid_mov2fix_z->SetPixel(Index, label2displacement[pd_mov2fix._pinfo[p].label][2] * m_inputImgSpacing[2]);
                        dfmgrid_mov2fix_x->SetPixel(Index, label2displacement[pd_mov2fix._pinfo[p].label][1] * m_inputImgSpacing[1]);
                        dfmgrid_mov2fix_y->SetPixel(Index, label2displacement[pd_mov2fix._pinfo[p].label][0] * m_inputImgSpacing[0]);

                    }
                }
            }
            std::cout << "mov2fix finished" << std::endl;
            //pd_mov2fix.~GC();

            // make the deformation field by bspline inter-polation
            std::cout << "dfmfld interpolation ...";
            gettimeofday(&tpstart, NULL);
            typename InternalImageType::Pointer dfmfld_x_fix2mov = InternalImageType::New();
            typename InternalImageType::Pointer dfmfld_y_fix2mov = InternalImageType::New();
            typename InternalImageType::Pointer dfmfld_z_fix2mov = InternalImageType::New();

            typename InternalImageType::Pointer dfmfld_x_mov2fix = InternalImageType::New();
            typename InternalImageType::Pointer dfmfld_y_mov2fix = InternalImageType::New();
            typename InternalImageType::Pointer dfmfld_z_mov2fix = InternalImageType::New();

            InterpImageBySpline(dfmgrid_fix2mov_x, dfmfld_x_fix2mov, contPt_num_x, contPt_num_y, contPt_num_z, this->inputImgSize[1], this->inputImgSize[0], this->inputImgSize[2]);
            InterpImageBySpline(dfmgrid_fix2mov_y, dfmfld_y_fix2mov, contPt_num_x, contPt_num_y, contPt_num_z, this->inputImgSize[1], this->inputImgSize[0], this->inputImgSize[2]);
            InterpImageBySpline(dfmgrid_fix2mov_z, dfmfld_z_fix2mov, contPt_num_x, contPt_num_y, contPt_num_z, this->inputImgSize[1], this->inputImgSize[0], this->inputImgSize[2]);

            InterpImageBySpline(dfmgrid_mov2fix_x, dfmfld_x_mov2fix, contPt_num_x, contPt_num_y, contPt_num_z, this->inputImgSize[1], this->inputImgSize[0], this->inputImgSize[2]);
            InterpImageBySpline(dfmgrid_mov2fix_y, dfmfld_y_mov2fix, contPt_num_x, contPt_num_y, contPt_num_z, this->inputImgSize[1], this->inputImgSize[0], this->inputImgSize[2]);
            InterpImageBySpline(dfmgrid_mov2fix_z, dfmfld_z_mov2fix, contPt_num_x, contPt_num_y, contPt_num_z, this->inputImgSize[1], this->inputImgSize[0], this->inputImgSize[2]);
            gettimeofday(&tpend, NULL);
            timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
            timeuse /= 1000000;
            printf(" Done %.2fs\n", timeuse);

            // debug only
            char dfmfld_x[100];
            char dfmfld_y[100];
            char dfmfld_z[100];
            sprintf(dfmfld_x, "dfmfld_x_fix2mov%d%d.mhd", level,it);
            sprintf(dfmfld_y, "dfmfld_y_fix2mov%d%d.mhd", level,it);
            sprintf(dfmfld_z, "dfmfld_z_fix2mov%d%d.mhd", level,it);
            WriteImage<InternalImageType>(dfmfld_x, dfmfld_x_fix2mov);
            WriteImage<InternalImageType>(dfmfld_y, dfmfld_y_fix2mov);
            WriteImage<InternalImageType>(dfmfld_z, dfmfld_z_fix2mov);

            // combine the component to have the vector dfmfld
            std::cout << "composing the dfmfld ...";
            gettimeofday(&tpstart, NULL);
            typedef itk::ComposeImageFilter<InternalImageType, DeformationFieldType> ImageToVectorImageFilterType;
            typename ImageToVectorImageFilterType::Pointer imageToVectorImageFilter_mov2fix = ImageToVectorImageFilterType::New();
            typename ImageToVectorImageFilterType::Pointer imageToVectorImageFilter_fix2mov = ImageToVectorImageFilterType::New();
            imageToVectorImageFilter_mov2fix->SetInput(2, dfmfld_z_mov2fix);
            imageToVectorImageFilter_mov2fix->SetInput(1, dfmfld_x_mov2fix);
            imageToVectorImageFilter_mov2fix->SetInput(0, dfmfld_y_mov2fix);
            imageToVectorImageFilter_mov2fix->Update();

            imageToVectorImageFilter_fix2mov->SetInput(2, dfmfld_z_fix2mov);
            imageToVectorImageFilter_fix2mov->SetInput(1, dfmfld_x_fix2mov);
            imageToVectorImageFilter_fix2mov->SetInput(0, dfmfld_y_fix2mov);
            imageToVectorImageFilter_fix2mov->Update();

            // update the dfmfld by composing the fld from last level
            typedef itk::ComposeDisplacementFieldsImageFilter<DeformationFieldType> DfmfldComposerType;
            if (level == 0) {
                this->dfmfld_fix2mov = imageToVectorImageFilter_fix2mov->GetOutput();
                this->dfmfld_mov2fix = imageToVectorImageFilter_mov2fix->GetOutput();
            } else {
                typename DfmfldComposerType::Pointer composer_fix2mov = DfmfldComposerType::New();
                typename DfmfldComposerType::Pointer composer_mov2fix = DfmfldComposerType::New();

                composer_fix2mov->SetWarpingField(imageToVectorImageFilter_fix2mov->GetOutput());
                composer_fix2mov->SetDisplacementField(this->dfmfld_fix2mov);
                composer_fix2mov->Update();
                this->dfmfld_fix2mov = composer_fix2mov->GetOutput();

                composer_mov2fix->SetWarpingField(imageToVectorImageFilter_mov2fix->GetOutput());
                composer_mov2fix->SetDisplacementField(this->dfmfld_mov2fix);
                composer_mov2fix->Update();
                this->dfmfld_mov2fix = composer_mov2fix->GetOutput();

                // output the composed dfmfld, debug only
                //WriteImage<DeformationFieldType>("dfmfld_fix2mov.nii.gz", composer_fix2mov->GetOutput());
                //WriteImage<DeformationFieldType>("dfmfld_mov2fix.nii.gz", composer_mov2fix->GetOutput());
            }
            gettimeofday(&tpend, NULL);
            timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
            timeuse /= 1000000;
            printf(" Done %.2fs\n", timeuse);
            // crop the deformation field
            /*InternalImageType::Pointer dfmfld_x_crop_fix2mov = InternalImageType::New();
            typename InternalImageType::Pointer dfmfld_y_crop_fix2mov = InternalImageType::New();
            typename InternalImageType::Pointer dfmfld_x_crop_mov2fix = InternalImageType::New();
            typename InternalImageType::Pointer dfmfld_y_crop_mov2fix = InternalImageType::New();

            CropImage(dfmfld_x_fix2mov, dfmfld_x_crop_fix2mov, this->GetGridSize());
            CropImage(dfmfld_y_fix2mov, dfmfld_y_crop_fix2mov, this->GetGridSize());

            CropImage(dfmfld_x_mov2fix, dfmfld_x_crop_mov2fix, this->GetGridSize());
            CropImage(dfmfld_y_mov2fix, dfmfld_y_crop_mov2fix, this->GetGridSize());	*/

            std::cout << "dfmfld modulation ...";
            gettimeofday(&tpstart, NULL);
            symetricDfmfld();
            gettimeofday(&tpend, NULL);
            timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
            timeuse /= 1000000;
            printf(" Done %.2fs\n", timeuse);

            // warp the image
            printf("Similarity measure before warping: %f\n",m_similarity);
            this->warpImage();
            this->m_similarity = Compute_Similarity(this->GetFixedImage(),this->warpedMovImage); 
            printf("Similarity measure after warping: %f\n",m_similarity);

            //debug only
            char warpedFixFile[100];
            char warpedMovFile[100];
            sprintf(warpedFixFile, "warpedFix%d%d.nii.gz", level, it);
            sprintf(warpedMovFile, "warpedMov%d%d.nii.gz", level, it);
            WriteImage<InputImageType>(warpedMovFile, this->warpedMovImage);
            WriteImage<InputImageType>(warpedFixFile, this->warpedFixImage);
            //WriteImage<DeformationFieldType>("dfmfld_fix2mov.nii.gz", this->dfmfld_fix2mov);
            //WriteImage<DeformationFieldType>("dfmfld_mov2fix.nii.gz", this->dfmfld_mov2fix);
            //WriteImage<InternalImageType>( "dfmfld_x.nii.gz", dfmfld_x_crop );
            //WriteImage<InternalImageType>( "dfmfld_y.nii.gz", dfmfld_y_crop );

            // release the image not needed anymore
            Release3DArray<typename InternalImageType::PixelType > (this->internalImgSize[1], this->internalImgSize[0], this->internalImgSize[2], this->fixedImg4Match);
            Release3DArray<typename InternalImageType::PixelType > (this->internalImgSize[1], this->internalImgSize[0], this->internalImgSize[2], this->movingImg4Match);

            Release3DArray<typename InternalImageType::PixelType > (this->internalImgSize[1], this->internalImgSize[0], this->internalImgSize[2], this->fixedImgTemplate);
            Release3DArray<typename InternalImageType::PixelType > (this->internalImgSize[1], this->internalImgSize[0], this->internalImgSize[2], this->movingImgTemplate);

            Release2DArray<int >(numOfSample, 3, offset);

            Release2DArray<float>(numLabels, 3, this->label2displacement);

            delete[] labelCosts;
            delete[] edges;
            delete[] labelDiffCost;
            delete[] wCosts;
            delete[] weights;

        }
    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::CropImage(typename InternalImageType::Pointer& inputImg, typename InternalImageType::Pointer& croppedImg, int cropSize) {
        // not using the cropimagefilter as it changes the offset

        const typename InternalImageType::SizeType sizeOfImage = inputImg->GetLargestPossibleRegion().GetSize();

        typename InternalImageType::SizeType size;
        typename InternalImageType::SpacingType spacing;
        for (int i = 0; i < ImageDimension; i++)
            size[i] = sizeOfImage[i] - cropSize * 2;

        spacing[0] = 1;
        spacing[1] = 1;

        typename InternalImageType::RegionType region;
        typename InternalImageType::IndexType start;

        start[0] = 0;
        start[1] = 0;

        region.SetSize(size);
        region.SetIndex(start);

        croppedImg->SetRegions(region);
        croppedImg->Allocate();

        typename InternalImageType::IndexType IndexIn, IndexOut;
        for (long i = cropSize; i < sizeOfImage[1] - cropSize; i++) {
            for (long j = cropSize; j < sizeOfImage[0] - cropSize; j++) {

                IndexIn[0] = j;
                IndexIn[1] = i;
                IndexOut[0] = j - cropSize;
                IndexOut[1] = i - cropSize;

                croppedImg->SetPixel(IndexOut, inputImg->GetPixel(IndexIn));

            }
        }

    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::PadImage(const TImage* inputImg, typename TImage::Pointer& paddedImg, int padSize) {
        // pad the image with the highest level grid size (in pixels)
        typedef ConstantPadImageFilter <TImage, TImage> ConstantPadImageFilterType;

        typename TImage::SizeType lowerExtendRegion;
        for (int i = 0; i < ImageDimension; i++)
            lowerExtendRegion[i] = padSize;

        typename TImage::SizeType upperExtendRegion;
        for (int i = 0; i < ImageDimension; i++)
            upperExtendRegion[i] = padSize;

        typename TImage::PixelType constantPixel = 0;

        typename ConstantPadImageFilterType::Pointer padFilter = ConstantPadImageFilterType::New();
        padFilter->SetInput(inputImg);
        padFilter->SetPadLowerBound(lowerExtendRegion);
        padFilter->SetPadUpperBound(upperExtendRegion);
        padFilter->SetConstant(constantPixel);
        padFilter->Update();

        paddedImg = padFilter->GetOutput();

    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::GaussianSmoothing(const TImage* inputImg, typename InternalImageType::Pointer& smoothedImg, float sigma) {

        typedef DiscreteGaussianImageFilter< TImage, InternalImageType > GaussianFilterType;

        typename GaussianFilterType::Pointer BlurImgFilter = GaussianFilterType::New();

        BlurImgFilter->SetInput(inputImg);
        BlurImgFilter->SetVariance(sigma);

        BlurImgFilter->Update();

        smoothedImg = BlurImgFilter->GetOutput();

    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::PrepareImageForMatch(int level, int iter) {

        int gridSize = this->GetGridSize();
        float variance = this->GetGaussianBlurVariance();
        std::cout << "blur kernal size: " << variance << std::endl;

        // pad the image
        /*typename TImage::Pointer paddedFixed, paddedMoving, paddedFixedWarped, paddedMovingWarped;
        if (level == 0) {
        this->PadImage(this->GetMovingImage(), paddedMoving, gridSize);
        this->PadImage(this->GetFixedImage(), paddedFixed, gridSize);
        //this->PadImage(this->GetMovingImage(), paddedMovingWarped, gridSize);
        //this->PadImage(this->GetFixedImage(), paddedFixedWarped, gridSize);
  }
  else {
  this->PadImage(this->warpedMovImage, paddedMovingWarped, gridSize);
  this->PadImage(this->warpedFixImage, paddedFixedWarped, gridSize);
  //this->PadImage(this->GetMovingImage(), paddedMoving, gridSize);
  //this->PadImage(this->GetFixedImage(), paddedFixed, gridSize);
  }*/

        // blur the image
        typename InternalImageType::Pointer smoothedFixed, smoothedMoving, smoothedWarpedFixed, smoothedWarpedMoving;

        this->GaussianSmoothing(this->GetFixedImage(), smoothedFixed, variance);
        this->GaussianSmoothing(this->GetMovingImage(), smoothedMoving, variance);

        if (!((level == 0)&&(iter == 0))) {
            this->GaussianSmoothing(warpedFixImage, smoothedWarpedFixed, variance);
            this->GaussianSmoothing(warpedMovImage, smoothedWarpedMoving, variance);
        }

        if ((level == 0)&&(iter == 0)) {
            this->m_similarity = this->Compute_Similarity(this->GetFixedImage(),this->GetMovingImage());
            printf("Similarity measure at this moment: %f\n",m_similarity);
        }

        // allocate the space for image array used in real matching
        this->internalImgSize = smoothedFixed->GetLargestPossibleRegion().GetSize();
        this->inputImgSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize();

        // debug only
        WriteImage<InternalImageType>("internalMov.mhd", smoothedMoving);
        WriteImage<InternalImageType>("internalFix.mhd", smoothedFixed);

        // info cout
        std::cout << "input size: " << this->inputImgSize[2] << "x" << this->inputImgSize[1] << "x" << this->inputImgSize[0] << std::endl;
        std::cout << "internal size: " << this->internalImgSize[2] << "x" << this->internalImgSize[1] << "x" << this->internalImgSize[0] << std::endl;

        this->fixedImg4Match = Allocate3DArray<typename InternalImageType::PixelType > (internalImgSize[1], internalImgSize[0], this->internalImgSize[2]);
        this->movingImg4Match = Allocate3DArray<typename InternalImageType::PixelType > (internalImgSize[1], internalImgSize[0], this->internalImgSize[2]);

        this->fixedImgTemplate = Allocate3DArray<typename InternalImageType::PixelType > (internalImgSize[1], internalImgSize[0], this->internalImgSize[2]);
        this->movingImgTemplate = Allocate3DArray<typename InternalImageType::PixelType > (internalImgSize[1], internalImgSize[0], this->internalImgSize[2]);

        //typedef itk::ImageRegionIteratorWithIndex< InternalImageType > IteratorType;
        //IteratorType itFix(smoothedFixed, smoothedFixed->GetRequestedRegion());
        //IteratorType itMov(smoothedMoving, smoothedMoving->GetRequestedRegion());
        //IteratorType itFixWarped(smoothedWarpedFixed, smoothedWarpedFixed->GetRequestedRegion());
        //IteratorType itMovWarped(smoothedWarpedMoving, smoothedWarpedMoving->GetRequestedRegion());
        typename InternalImageType::IndexType idx;

        for (unsigned int z = 0; z <= internalImgSize[2] - 1; z++) {
            for (unsigned int x = 0; x <= internalImgSize[1] - 1; x++) {
                for (unsigned int y = 0; y <= internalImgSize[0] - 1; y++) {
                    idx[2] = z;
                    idx[1] = x;
                    idx[0] = y;

                    fixedImgTemplate[z][x][y] = smoothedFixed->GetPixel(idx);
                    movingImgTemplate[z][x][y] = smoothedMoving->GetPixel(idx);

                    if (!((level == 0)&&(iter == 0))) {
                        fixedImg4Match[z][x][y] = smoothedWarpedFixed->GetPixel(idx);
                        movingImg4Match[z][x][y] = smoothedWarpedMoving->GetPixel(idx);
                    } else {
                        fixedImg4Match[z][x][y] = smoothedFixed->GetPixel(idx);
                        movingImg4Match[z][x][y] = smoothedMoving->GetPixel(idx);
                    }
                }
            }
        }

        // compute the histogram if using MI as metric
        if (this->m_MetricSelected == 1) {
            if ((level == 0)&&(iter == 0)) {
                this->updateHistogram(smoothedFixed, smoothedMoving, smoothedFixed, smoothedMoving);
            } else {
                this->updateHistogram(smoothedFixed, smoothedMoving, smoothedWarpedFixed, smoothedWarpedMoving);
            }
        }
    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::CreateLabelCostTabel(int numOfSteps, float stepSize, float weight, float thresh) {
        // compute number of lablels
        int numOfLabels = (int) powf((this->GetNumOfStep() * 2 + 1), 3.0);

        std::cout << "constructing the label to diplacement table ..." << std::endl;
        std::cout << "step size: " << stepSize << std::endl;

        // Table label2displacement
        this->label2displacement = Allocate2DArray<float>(numOfLabels, 3);
        int index = 0;
        for (int k = -numOfSteps; k <= numOfSteps; k++) {
            for (int j = -numOfSteps; j <= numOfSteps; j++) {
                for (int i = -numOfSteps; i <= numOfSteps; i++) {
                    // assume isotropic size, otherwise need to consider the spacing
                    this->label2displacement[index][2] = (float) k*stepSize;
                    this->label2displacement[index][1] = (float) j*stepSize;
                    this->label2displacement[index][0] = (float) i*stepSize;

                    //std::cout<<"label #"<<index<<" move: "<<label2displacement[index][1]<<" "<<label2displacement[index][0]<<std::endl;
                    index++;
                }
            }
        }

        // Table label2cost
        //Allocate2DArray<float>(numOfLabels, numOfLabels, this->label2cost);
        float cost = 0.0;
        this->labelDiffCost = new float[numOfLabels * numOfLabels];
        for (long i = 0; i < numOfLabels; i++) {
            for (long j = 0; j < numOfLabels; j++) {

                /*this->labelDiffCost[i*numOfLabels + j] = powf((powf(float(label2displacement[i][0] - label2displacement[j][0]), 2.0)
                + powf(float(label2displacement[i][1] - label2displacement[j][1]), 2.0)
                + powf(float(label2displacement[i][2] - label2displacement[j][2]), 2.0)) / powf(this->GetGridSize(), 2.0), weight);  //range(0,1.0) */

                /*this->labelDiffCost[i*numOfLabels + j] = (abs(float(label2displacement[i][0] - label2displacement[j][0]))
                + abs(float(label2displacement[i][1] - label2displacement[j][1]))
                + abs(float(label2displacement[i][2] - label2displacement[j][2]))) / this->GetGridSize() / 3;*/ //range(0,1.0)

                /*this->labelDiffCost[i*numOfLabels + j] = powf((powf(float(label2displacement[i][0] - label2displacement[j][0]), 2.0)
                + powf(float(label2displacement[i][1] - label2displacement[j][1]), 2.0)
                + powf(float(label2displacement[i][2] - label2displacement[j][2]), 2.0)), 0.5) / 80.0;*/ //range(0,1.0), without normalization

                //std::cout << "pair cost: " << this->labelDiffCost[i*numOfLabels + j] << std::endl;

                /*this->labelDiffCost[i*numOfLabels + j] = pow(pow(float(label2displacement[i][0] - label2displacement[j][0]), 2)
                + pow(float(label2displacement[i][1] - label2displacement[j][1]), 2)
                + pow(float(label2displacement[i][2] - label2displacement[j][2]), 2), 2) / (numOfSteps*stepSize) * weight;*/

                cost = powf(powf(float(label2displacement[i][0] - label2displacement[j][0]) / (numOfSteps * stepSize * 2), 2)
                        + powf(float(label2displacement[i][1] - label2displacement[j][1]) / (numOfSteps * stepSize * 2), 2)
                        + powf(float(label2displacement[i][2] - label2displacement[j][2]) / (numOfSteps * stepSize * 2), 2), 1) / 3 / weight;

                /*this->labelDiffCost[i*numOfLabels + j] = (abs(float(label2displacement[i][0] - label2displacement[j][0]))
                + abs(float(label2displacement[i][1] - label2displacement[j][1]))
                + abs(float(label2displacement[i][2] - label2displacement[j][2]))) / (numOfSteps*stepSize) * weight;*/

                /*this->labelDiffCost[i*numOfLabels+j] = pow( pow(float(label2displacement[i][0] - label2displacement[j][0]), 2)
                +pow(float(label2displacement[i][1] - label2displacement[j][1]), 2)
                +pow(float(label2displacement[i][2] - label2displacement[j][2]), 2), 6) * weight;*/

                if (cost > thresh) {
                    //std::cout<<"truncated from "<<cost<<" to "<<thresh<<std::endl;
                    this->labelDiffCost[i * numOfLabels + j] = thresh;
                } else {
                    this->labelDiffCost[i * numOfLabels + j] = cost;
                }
            } // note: it is normalized by GridSize;
        }
    }

    template< typename TImage>
    template<typename T> T** BlockMatchingImageRegistrationFilter<TImage>::Allocate2DArray(int x, int y) {

        //array = (T**)malloc(x * sizeof(T*));
        T** array = new T*[x];

        /*if (array == NULL) {
        fprintf(stderr, "out of memory\n");
        exit(0);
  }*/

        for (int i = 0; i < x; i++) {
            //array[i] = (T*)malloc(y * sizeof(T));
            array[i] = new T[y];
            /*if (array[i] == NULL) {
            fprintf(stderr, "out of memory\n");
            exit(0);
      }*/
        }

        // initialize with 0
        for (int j = 0; j < x; j++) {
            for (int i = 0; i < y; i++) {
                array[j][i] = 0.0;
            }
        }

        return (array);
    }

    /*
    template< typename TImage>
    template<typename T> void BlockMatchingImageRegistrationFilter<TImage>::Allocate3DArray(int x, int y, int z, T***& array) {

    array = (T***)malloc(z * sizeof(T**));

    if (array == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(0);
    }

    for (int k = 0; k < z; k++) {
    array[k] = (T**)malloc(x * sizeof(T*));
    if (array[k] == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(0);
    }

    for (int i = 0; i < x; i++) {
    array[k][i] = (T*)malloc(y * sizeof(T));
    if (array[k][i] == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(0);
    }
    }
    }

    //initialize with 0
    for (int k = 0; k < z; k++) {
    for (int j = 0; j < x; j++) {
    for (int i = 0; i < y; i++) {
    array[k][j][i] = 0;
    }
    }
    }
    }*/

    template< typename TImage>
    template<typename T> T*** BlockMatchingImageRegistrationFilter<TImage>::Allocate3DArray(int x, int y, int z) {
        T*** array;
        int i, j, k;

        //array = (T***)malloc(z * sizeof(T **));
        array = new T**[z];

        for (i = 0; i < z; i++) {
            //array[i] = (T**)malloc(x * sizeof(T*));
            array[i] = new T*[x];
            for (j = 0; j < x; j++) {
                //array[i][j] = (T*)malloc(y * sizeof(T));
                array[i][j] = new T[y];
            }
        }

        //initialize with 0
        for (k = 0; k < z; k++) {
            for (j = 0; j < x; j++) {
                for (i = 0; i < y; i++) {
                    array[k][j][i] = 0;
                }
            }
        }

        return (array);
    }

    template< typename TImage>
    template<typename T> void BlockMatchingImageRegistrationFilter<TImage>::Release2DArray(int x, int y, T** array) {
        for (int i = 0; i < x; i++) {
            //free(array[i]);
            delete[] array[i];
            //std::cout << i << std::endl;
        }
        //free(array);
        delete[] array;
    }

    template< typename TImage>
    template<typename T> void BlockMatchingImageRegistrationFilter<TImage>::Release3DArray(int x, int y, int z, T*** array) {
        for (int j = 0; j < z; j++) {
            for (int i = 0; i < x; i++) {
                //free(array[j][i]);
                delete[] array[j][i];
            }
            //free(array[j]);
            delete[] array[j];
        }
        //free(array);
        delete[] array;
    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::RandomSampling(int radius_x, int radius_y, int radius_z, int** offsets, float* weights, int numOfSample) {

        itk::Statistics::GaussianDistribution::Pointer gaussian = itk::Statistics::GaussianDistribution::New();
        gaussian->SetMean(0.0);
        gaussian->SetVariance(5.0);
        float ref = gaussian->EvaluatePDF(3.0);
        //std::cout<<"ref: "<<ref<<std::endl;
        float dist = 0.0;
        float sum = 0.0;

        for (int i = 0; i < numOfSample; i++) {
            offsets[i][2] = rand() % (radius_z * 2) - radius_z;
            offsets[i][1] = rand() % (radius_x * 2) - radius_x;
            offsets[i][0] = rand() % (radius_y * 2) - radius_y;

            dist = powf(powf((float) offsets[i][0] / (float) radius_y, 2) + powf((float) offsets[i][1] / (float) radius_x, 2) + powf((float) offsets[i][2] / (float) radius_z, 2), 0.5);
            weights[i] = gaussian->EvaluatePDF(dist / 1.44 * 5) / ref;
            //std::cout<<offsets[i][0]<<" "<<offsets[i][1]<<" "<<offsets[i][2]<<std::endl;
            //std::cout<<dist/1.44*3<<" "<<weights[i]<<std::endl;
            sum += weights[i];
        }

        float scale = sum / numOfSample;
        //std::cout<<"sum: "<<sum<<" scale: "<<scale<<std::endl;

        for (int i = 0; i < numOfSample; i++) {
            weights[i] = weights[i] / scale;
        }
    }

    template< typename TImage>
    float BlockMatchingImageRegistrationFilter<TImage>::
    NCC(typename InternalImageType::PixelType***& fixed, typename InternalImageType::PixelType***& moving, int center_x, int center_y, int center_z, float translate_x, float translate_y, float translate_z, int** offsets, float* weights, int numOfSample, bool checkBorder) {
        /*
        center_x, center_y: center of the block to be moved
        translate_x, translate_y: the translation of the block in moving image

        center_x+translate_x is the corresponding position in fixed image
         */

        float sumSquareFixed = 0.0;
        float sumSquareMoving = 0.0;
        float sumProduct = 0.0;
        float nonzero = 1e-9; // very small value to prevent divide by zero

        typename InternalImageType::PixelType interpFixedPixel;
        typename InternalImageType::PixelType movingPixel;

        //std::cout << " translation: " << translate_x << " "<< translate_y << " " << translate_z << std::endl;

        for (int i = 0; i < numOfSample; i++) {

            if (checkBorder) {
                if ((center_x + translate_x + offsets[i][1] < 0) || (center_x + translate_x + offsets[i][1] > int(internalImgSize[1] - 1)) ||
                        (center_y + translate_y + offsets[i][0] < 0) || (center_y + translate_y + offsets[i][0] > int(internalImgSize[0] - 1)) ||
                        (center_z + translate_z + offsets[i][2] < 0) || (center_z + translate_z + offsets[i][2] > int(internalImgSize[2] - 1))) {

                    interpFixedPixel = 0.0;
                } else {

                    interpFixedPixel = LinearInterpImage<typename InternalImageType::PixelType > (fixed, center_x + translate_x + offsets[i][1],
                            center_y + translate_y + offsets[i][0], center_z + translate_z + offsets[i][2]);
                }

                if ((center_x + offsets[i][1] < 0) || (center_x + offsets[i][1] > int(internalImgSize[1] - 1)) ||
                        (center_y + offsets[i][0] < 0) || (center_y + offsets[i][0] > int(internalImgSize[0] - 1)) ||
                        (center_z + offsets[i][2] < 0) || (center_z + offsets[i][2] > int(internalImgSize[2] - 1))) {

                    movingPixel = 0.0;
                } else {
                    //std::cout << center_x + offsets[i][1] << " "<<center_y + offsets[i][1] << " " << center_z + offsets[i][2] << std::endl;
                    //std::cout << "value: " << moving[center_z + offsets[i][2]][center_x + offsets[i][1]][center_y + offsets[i][0]] << std::endl;
                    movingPixel = moving[center_z + offsets[i][2]][center_x + offsets[i][1]][center_y + offsets[i][0]];
                }

            } else {

                interpFixedPixel = LinearInterpImage<typename InternalImageType::PixelType > (fixed, center_x + translate_x + offsets[i][1],
                        center_y + translate_y + offsets[i][0], center_z + translate_z + offsets[i][2]);
                movingPixel = moving[center_z + offsets[i][2]][center_x + offsets[i][1]][center_y + offsets[i][0]];

                //std::cout<<"fixed: "<<center_x + translate_x + offsets[i][1]<<" "<<center_y + translate_y + offsets[i][0]<<std::endl;
                //std::cout<<"moving: "<<center_x + offsets[i][1]<<" "<<center_y + offsets[i][0]<<std::endl;
            }

            sumSquareFixed += powf(interpFixedPixel + nonzero, 2.0) * weights[i];
            sumSquareMoving += powf(movingPixel + nonzero, 2.0) * weights[i];
            sumProduct += (interpFixedPixel + nonzero)*(movingPixel + nonzero) * weights[i];
        }

        //float norm = sumSquareFixed*sumSquareMoving;

        if ((sumSquareFixed == 0.0) || (sumSquareMoving == 0.0)) {
            std::cout << "divide by zero!!!!!!!!" << std::endl;
            exit(-1);
        } else {
            //std::cout<<"NCC="<<sumProduct*sumProduct / sumSquareFixed / sumSquareMoving<<std::endl;
            //std::cout<<"NCC="<<1 - powf(sumProduct*sumProduct / sumSquareFixed / sumSquareMoving, 0.5)<<std::endl;
            return (1 - powf(sumProduct * sumProduct / sumSquareFixed / sumSquareMoving, 2.0)); // range(0,1)
            //return powf(4.0,(1 - powf(sumProduct*sumProduct / sumSquareFixed / sumSquareMoving, 0.5)))-1.0; //range(0,3)
        }

    }

    template< typename TImage>
    float BlockMatchingImageRegistrationFilter<TImage>::
    PWMI(typename InternalImageType::PixelType***& fixed, typename InternalImageType::PixelType***& moving,
            int center_x, int center_y, int center_z, float translate_x, float translate_y, float translate_z,
            int** offsets, int numOfSample, bool checkBorder,
            float** jointProbability, float* fixProbability, float* movProbability) {
        /*
        center_x, center_y: center of the block to be moved
        translate_x, translate_y: the translation of the block in moving image

        center_x+translate_x is the corresponding position in fixed image
         */

        float avgMI = 0.0;
        float binWidthFix = (m_binMaximumFix - m_binMinimumFix) / m_numBinFix;
        float binWidthMov = (m_binMaximumMov - m_binMinimumMov) / m_numBinMov;

        typename InternalImageType::PixelType interpFixedPixel;
        typename InternalImageType::PixelType movingPixel;
        int indFix, indMov;

        //std::cout << " translation: " << translate_x << " "<< translate_y << " " << translate_z << std::endl;

        for (int i = 0; i < numOfSample; i++) {

            if (checkBorder) {
                if ((center_x + translate_x + offsets[i][1] < 0) || (center_x + translate_x + offsets[i][1] > int(internalImgSize[1] - 1)) ||
                        (center_y + translate_y + offsets[i][0] < 0) || (center_y + translate_y + offsets[i][0] > int(internalImgSize[0] - 1)) ||
                        (center_z + translate_z + offsets[i][2] < 0) || (center_z + translate_z + offsets[i][2] > int(internalImgSize[2] - 1))) {

                    interpFixedPixel = 0.0;
                } else {

                    interpFixedPixel = LinearInterpImage<typename InternalImageType::PixelType > (fixed, center_x + translate_x + offsets[i][1],
                            center_y + translate_y + offsets[i][0], center_z + translate_z + offsets[i][2]);
                }

                if ((center_x + offsets[i][1] < 0) || (center_x + offsets[i][1] > int(internalImgSize[1] - 1)) ||
                        (center_y + offsets[i][0] < 0) || (center_y + offsets[i][0] > int(internalImgSize[0] - 1)) ||
                        (center_z + offsets[i][2] < 0) || (center_z + offsets[i][2] > int(internalImgSize[2] - 1))) {

                    movingPixel = 0.0;
                } else {
                    //std::cout << center_x + offsets[i][1] << " "<<center_y + offsets[i][1] << " " << center_z + offsets[i][2] << std::endl;
                    //std::cout << "value: " << moving[center_z + offsets[i][2]][center_x + offsets[i][1]][center_y + offsets[i][0]] << std::endl;
                    movingPixel = moving[center_z + offsets[i][2]][center_x + offsets[i][1]][center_y + offsets[i][0]];
                }

            } else {

                interpFixedPixel = LinearInterpImage<typename InternalImageType::PixelType > (fixed, center_x + translate_x + offsets[i][1],
                        center_y + translate_y + offsets[i][0], center_z + translate_z + offsets[i][2]);
                movingPixel = moving[center_z + offsets[i][2]][center_x + offsets[i][1]][center_y + offsets[i][0]];

                //std::cout<<"fixed: "<<center_x + translate_x + offsets[i][1]<<" "<<center_y + translate_y + offsets[i][0]<<std::endl;
                //std::cout<<"moving: "<<center_x + offsets[i][1]<<" "<<center_y + offsets[i][0]<<std::endl;
            }

            indFix = floor(interpFixedPixel / binWidthFix);
            indMov = floor(movingPixel / binWidthMov);

            /*if (interpFixedPixel >= m_binMaximumFix) {
            indFix = m_numBinFix-1;
      }
      else {
      indFix = ceil(interpFixedPixel / binWidthFix);
    }

    if (movingPixel >= m_binMaximumMov) {
    indMov = m_numBinMov - 1;
            }
            else {
            indMov = ceil(movingPixel / binWidthMov);
            }*/

            if (indFix > m_numBinFix - 1) indFix = m_numBinFix - 1;
            if (indMov > m_numBinMov - 1) indMov = m_numBinMov - 1;

            /*if ((indFix >= m_numBinFix) || (indMov >= m_numBinMov)) {
            std::cout << indFix << " " << indMov << std::endl;
      }*/

            avgMI += jointProbability[indFix][indMov] / movProbability[indMov] / fixProbability[indFix];
        }

        //std::cout << "pwMI: " << avgMI << std::endl;
        return (1 - avgMI / numOfSample / 1000);
    }

    template< typename TImage>
    template<typename T> T BlockMatchingImageRegistrationFilter<TImage>::LinearInterpImage(T*** Input, float x, float y, float z) {

        unsigned int low_x, up_x, low_y, up_y, low_z, up_z;

        unsigned int lim_x = this->internalImgSize[1] - 1;
        unsigned int lim_y = this->internalImgSize[0] - 1;
        unsigned int lim_z = this->internalImgSize[2] - 1;

        low_x = floor(x);
        up_x = floor(x + 1);

        low_y = floor(y);
        up_y = floor(y + 1);

        low_z = floor(z);
        up_z = floor(z + 1);

        if (up_x > lim_x) up_x = lim_x;
        if (up_y > lim_y) up_y = lim_y;
        if (up_z > lim_z) up_z = lim_z;

        return interp3d(x, y, z, Input[low_z][low_x][low_y], Input[low_z][low_x][up_y], Input[low_z][up_x][low_y],
                Input[low_z][up_x][up_y], Input[up_z][low_x][low_y], Input[up_z][low_x][up_y], Input[up_z][up_x][low_y],
                Input[up_z][up_x][up_y], low_x, up_x, low_y, up_y, low_z, up_z);

        /*
        double input, fractpart, intpart;

        fractpart = modf (x , &intpart);
        unsigned int low_x = (unsigned int)(intpart);
        unsigned int up_x = (unsigned int)ceil(intpart+fractpart);

        fractpart = modf (y , &intpart);
        unsigned int low_y = (unsigned int)(intpart);
        unsigned int up_y = (unsigned int)ceil(intpart+fractpart);
         */
        //return bilinear<float>( x, y, Input[low_x][low_y], Input[up_x][low_y], Input[low_x][up_y], Input[up_x][up_y]);
        //if ((x>2999)||(y>2999)) {
        //std::cout<<"interp: "<<low_x<<" "<<low_y<<" "<<up_x<<" "<<up_y<<std::endl;
        //}
        //return ( (Input[low_x][low_y]*(x - low_x) + Input[up_x][low_y]*(up_x - x))*(y - low_y) + (Input[low_x][up_y]*(x - low_x) + Input[up_x][up_y]*(up_x - x))*(up_y - y) );
        //return ( (Input[low_x][low_y]*(up_x - x) + Input[up_x][low_y]*(x - low_x))*(up_y - y) + (Input[low_x][up_y]*(up_x - x) + Input[up_x][up_y]*(x - low_x))*(y - low_y) );
    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::InterpImageBySpline(const typename InternalImageType::Pointer& InputImg, typename InternalImageType::Pointer& OutputImg, int in_x, int in_y, int in_z, int out_x, int out_y, int out_z) {

        //do the interpolation using cubic spline
        typedef itk::IdentityTransform<double, 3> TransformType;
        typedef itk::BSplineInterpolateImageFunction<InternalImageType, double, double> InterpolatorType;
        typedef itk::ResampleImageFilter<InternalImageType, InternalImageType> ResampleFilterType;

        typename TransformType::Pointer transform = TransformType::New();
        transform->SetIdentity();

        // Instantiate the b-spline interpolator and set it as the third order for bicubic.
        typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
        interpolator->SetSplineOrder(3);

        // Instantiate the resampler. Wire in the transform and the interpolator.
        typename ResampleFilterType::Pointer ResizeFilter = ResampleFilterType::New();
        ResizeFilter->SetTransform(transform);
        ResizeFilter->SetInterpolator(interpolator);

        // Set the output origin.
        const double vfOutputOrigin[3] = {0.0, 0.0, 0.0};
        ResizeFilter->SetOutputOrigin(vfOutputOrigin);

        double vfOutputSpacing[3];
        vfOutputSpacing[0] = m_inputImgSpacing[0];
        vfOutputSpacing[1] = m_inputImgSpacing[1];
        vfOutputSpacing[2] = m_inputImgSpacing[2];

        // Set the output spacing.
        ResizeFilter->SetOutputSpacing(vfOutputSpacing);

        // Set the output size
        typename InternalImageType::SizeType vnOutputSize;
        vnOutputSize[0] = out_y;
        vnOutputSize[1] = out_x;
        vnOutputSize[2] = out_z;

        ResizeFilter->SetSize(vnOutputSize);
        ResizeFilter->SetInput(InputImg);
        ResizeFilter->Update();

        // do the interpolation
        OutputImg = ResizeFilter->GetOutput();

    }

    template< typename TImage>
    template <typename T> void BlockMatchingImageRegistrationFilter<TImage>::WriteImage(const char * filename, const T* image) {
        typedef ImageFileWriter<T> WriterType;
        typename WriterType::Pointer writer = WriterType::New();
        writer->SetFileName(filename);
        writer->SetInput(image);

        try {
            writer->Update();
        } catch (ExceptionObject & excp) {
            throw excp;
        } catch (...) {
            ExceptionObject e(__FILE__, __LINE__,
                    "Caught unknown exception", ITK_LOCATION);
            throw e;
        }
    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::symetricDfmfld(void) {

        //WriteImage<DeformationFieldType>( "dfmfld_in.nii.gz", this->dfmfld_fix2mov );

        // touchup
        int expstep = 4;
        float factor = (float) this->GetGridSize() / 2.0;

        typedef itk::ImageRegionIterator< DeformationFieldType> IteratorType;
        IteratorType fix2movIt(this->dfmfld_fix2mov, this->dfmfld_fix2mov->GetLargestPossibleRegion());
        IteratorType mov2fixIt(this->dfmfld_mov2fix, this->dfmfld_mov2fix->GetLargestPossibleRegion());

        float factorF = 1.0 / factor;
        float coeff = 1.0 / (float) powf(2.0, expstep);

        fix2movIt.GoToBegin();
        mov2fixIt.GoToBegin();

        while (!fix2movIt.IsAtEnd()) {
            typename DeformationFieldType::PixelType vector = fix2movIt.Get();
            vector[2] = coeff * vector[2] * factorF;
            vector[1] = coeff * vector[1] * factorF;
            vector[0] = coeff * vector[0] * factorF;
            fix2movIt.Set(vector);
            ++fix2movIt;
        }

        while (!mov2fixIt.IsAtEnd()) {
            typename DeformationFieldType::PixelType vector = mov2fixIt.Get();
            vector[2] = coeff * vector[2] * factorF;
            vector[1] = coeff * vector[1] * factorF;
            vector[0] = coeff * vector[0] * factorF;
            mov2fixIt.Set(vector);
            ++mov2fixIt;
        }

        //WriteImage<DeformationFieldType>( "dfmfld_beforeCompose.nii.gz", this->dfmfld_fix2mov );

        typedef itk::ComposeDisplacementFieldsImageFilter<DeformationFieldType> DfmfldComposerType;
        //DeformationFieldType::Pointer tempMov2Fix = this->dfmfld_mov2fix;
        //DeformationFieldType::Pointer tempFix2Mov = this->dfmfld_fix2mov;

        for (int it = 0; it < expstep; it++) {
            typename DfmfldComposerType::Pointer composer_fix2mov = DfmfldComposerType::New();
            typename DfmfldComposerType::Pointer composer_mov2fix = DfmfldComposerType::New();

            composer_fix2mov->SetWarpingField(this->dfmfld_fix2mov);
            composer_fix2mov->SetDisplacementField(this->dfmfld_fix2mov);
            composer_fix2mov->Update();
            this->dfmfld_fix2mov = composer_fix2mov->GetOutput();

            composer_mov2fix->SetWarpingField(this->dfmfld_mov2fix);
            composer_mov2fix->SetDisplacementField(this->dfmfld_mov2fix);
            composer_mov2fix->Update();
            this->dfmfld_mov2fix = composer_mov2fix->GetOutput();

        }

        //WriteImage<DeformationFieldType>( "dfmfld_afterCompose.nii.gz", this->dfmfld_fix2mov );

        IteratorType fix2movIt2(this->dfmfld_fix2mov, this->dfmfld_fix2mov->GetLargestPossibleRegion());
        IteratorType mov2fixIt2(this->dfmfld_mov2fix, this->dfmfld_mov2fix->GetLargestPossibleRegion());
        fix2movIt2.GoToBegin();
        mov2fixIt2.GoToBegin();

        while (!fix2movIt2.IsAtEnd()) {
            typename DeformationFieldType::PixelType vector = fix2movIt2.Get();

            //std::cout<<"before: "<<vector[1]<<std::endl;
            vector[2] = vector[2] * factor * 0.5;
            vector[1] = vector[1] * factor * 0.5;
            vector[0] = vector[0] * factor * 0.5;
            fix2movIt2.Set(vector);
            //std::cout<<"after: "<<vector[1]<<std::endl;

            ++fix2movIt2;
        }

        while (!mov2fixIt2.IsAtEnd()) {
            typename DeformationFieldType::PixelType vector = mov2fixIt2.Get();

            vector[2] = vector[2] * factor * 0.5;
            vector[1] = vector[1] * factor * 0.5;
            vector[0] = vector[0] * factor * 0.5;
            mov2fixIt2.Set(vector);

            ++mov2fixIt2;
        }

        //WriteImage<DeformationFieldType>( "dfmfld_afterTouchup.nii.gz", this->dfmfld_fix2mov );

        // caculate inverse for both direction
        typedef itk::InvertDisplacementFieldImageFilter< DeformationFieldType, DeformationFieldType > DfmfldInverterType;
        typename DfmfldInverterType::Pointer invert_mov2fix = DfmfldInverterType::New();
        typename DfmfldInverterType::Pointer invert_fix2mov = DfmfldInverterType::New();

        invert_mov2fix->SetInput(this->dfmfld_mov2fix);
        invert_mov2fix->SetMaximumNumberOfIterations(10);
        invert_mov2fix->SetMeanErrorToleranceThreshold(0.001);
        invert_mov2fix->SetMaxErrorToleranceThreshold(0.1);
        invert_mov2fix->Update();

        invert_fix2mov->SetInput(this->dfmfld_fix2mov);
        invert_fix2mov->SetMaximumNumberOfIterations(10);
        invert_fix2mov->SetMeanErrorToleranceThreshold(0.001);
        invert_fix2mov->SetMaxErrorToleranceThreshold(0.1);
        invert_fix2mov->Update();

        try {
            invert_fix2mov->UpdateLargestPossibleRegion();
            invert_mov2fix->UpdateLargestPossibleRegion();
        } catch (itk::ExceptionObject & excp) {
            std::cerr << "Exception thrown while inverting dfmfld" << std::endl;
            std::cerr << excp << std::endl;
        }

        // combine the two directions to be symetric
        typename DfmfldComposerType::Pointer composer_fix2mov = DfmfldComposerType::New();
        typename DfmfldComposerType::Pointer composer_mov2fix = DfmfldComposerType::New();

        composer_fix2mov->SetWarpingField(invert_mov2fix->GetOutput());
        composer_fix2mov->SetDisplacementField(this->dfmfld_fix2mov);
        composer_fix2mov->Update();
        this->dfmfld_fix2mov = composer_fix2mov->GetOutput();

        composer_mov2fix->SetWarpingField(invert_fix2mov->GetOutput());
        composer_mov2fix->SetDisplacementField(this->dfmfld_mov2fix);
        composer_mov2fix->Update();
        this->dfmfld_mov2fix = composer_mov2fix->GetOutput();

    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::warpImage(void) {
        typedef itk::WarpImageFilter< InputImageType, InputImageType, DeformationFieldType > WarpImageFilterType;
        typename WarpImageFilterType::Pointer warpImageFilterMov = WarpImageFilterType::New();
        typename WarpImageFilterType::Pointer warpImageFilterFix = WarpImageFilterType::New();

        typedef itk::LinearInterpolateImageFunction< InputImageType, double > LinearInterpolatorType;
        typename LinearInterpolatorType::Pointer interpolatorLinearMov = LinearInterpolatorType::New();
        typename LinearInterpolatorType::Pointer interpolatorLinearFix = LinearInterpolatorType::New();

        warpImageFilterMov->SetInterpolator(interpolatorLinearMov);
        warpImageFilterMov->SetOutputSpacing(this->GetFixedImage()->GetSpacing());
        warpImageFilterMov->SetOutputOrigin(this->GetFixedImage()->GetOrigin());
        warpImageFilterMov->SetDisplacementField(this->dfmfld_fix2mov);
        warpImageFilterMov->SetInput(this->GetMovingImage());
        warpImageFilterMov->Update();
        this->warpedMovImage = warpImageFilterMov->GetOutput();

        warpImageFilterFix->SetInterpolator(interpolatorLinearFix);
        warpImageFilterFix->SetOutputSpacing(this->GetMovingImage()->GetSpacing());
        warpImageFilterFix->SetOutputOrigin(this->GetMovingImage()->GetOrigin());
        warpImageFilterFix->SetDisplacementField(this->dfmfld_mov2fix);
        warpImageFilterFix->SetInput(this->GetFixedImage());
        warpImageFilterFix->Update();
        this->warpedFixImage = warpImageFilterFix->GetOutput();
    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::outputResults(char* warpedFix, char* warpedMov, char* fld2Fix, char* fld2Mov) {
        WriteImage<InputImageType>(warpedMov, this->warpedMovImage);
        WriteImage<InputImageType>(warpedFix, this->warpedFixImage);
        WriteImage<DeformationFieldType>(fld2Mov, this->dfmfld_fix2mov);
        WriteImage<DeformationFieldType>(fld2Fix, this->dfmfld_mov2fix);
    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::updateHistogram(typename InternalImageType::Pointer& smoothedFixed, typename InternalImageType::Pointer& smoothedMoving,
            typename InternalImageType::Pointer& smoothedWarpedFixed, typename InternalImageType::Pointer& smoothedWarpedMoving) {

        std::cout << "computing histograms ..." << std::endl;

        // joint histogram fixed+warped moving
        typename JoinImageFilterType::Pointer joinFilterFix = JoinImageFilterType::New();
        joinFilterFix->SetInput1(smoothedFixed);
        joinFilterFix->SetInput2(smoothedWarpedMoving);
        joinFilterFix->Update();

        // joint histogram moving+warped fixed
        typename JoinImageFilterType::Pointer joinFilterMov = JoinImageFilterType::New();
        joinFilterMov->SetInput1(smoothedMoving);
        joinFilterMov->SetInput2(smoothedWarpedFixed);
        joinFilterMov->Update();

        // compute histogram
        histogramFilterJointFix->SetInput(joinFilterFix->GetOutput());
        histogramFilterJointMov->SetInput(joinFilterMov->GetOutput());
        histogramFilterJointFix->SetMarginalScale(10.0);
        histogramFilterJointMov->SetMarginalScale(10.0);

        typedef typename HistogramFilterType::HistogramSizeType HistogramSizeType;
        HistogramSizeType sizeFix(2);
        HistogramSizeType sizeMov(2);
        sizeFix[0] = m_numBinFix; // number of bins for the first  channel
        sizeFix[1] = m_numBinMov; // number of bins for the second channel
        sizeMov[0] = m_numBinMov; // number of bins for the first  channel
        sizeMov[1] = m_numBinFix; // number of bins for the second channel
        histogramFilterJointFix->SetHistogramSize(sizeFix);
        histogramFilterJointMov->SetHistogramSize(sizeMov);

        typedef typename HistogramFilterType::HistogramMeasurementVectorType HistogramMeasurementVectorType;
        HistogramMeasurementVectorType binMinimumFix(2);
        HistogramMeasurementVectorType binMaximumFix(2);
        binMinimumFix[0] = m_binMinimumFix;
        binMinimumFix[1] = m_binMinimumMov;
        binMaximumFix[0] = m_binMaximumFix;
        binMaximumFix[1] = m_binMaximumMov;
        histogramFilterJointFix->SetHistogramBinMinimum(binMinimumFix);
        histogramFilterJointFix->SetHistogramBinMaximum(binMaximumFix);
        histogramFilterJointFix->Update();

        HistogramMeasurementVectorType binMinimumMov(2);
        HistogramMeasurementVectorType binMaximumMov(2);
        binMinimumMov[0] = m_binMinimumMov;
        binMinimumMov[1] = m_binMinimumFix;
        binMaximumMov[0] = m_binMaximumMov;
        binMaximumMov[1] = m_binMaximumFix;
        histogramFilterJointMov->SetHistogramBinMinimum(binMinimumMov);
        histogramFilterJointMov->SetHistogramBinMaximum(binMaximumMov);
        histogramFilterJointMov->Update();

        typedef typename HistogramFilterType::HistogramType HistogramType;
        const HistogramType* histogramFix = histogramFilterJointFix->GetOutput();
        const HistogramType* histogramMov = histogramFilterJointMov->GetOutput();

        // compute the joint probability fix as template
        typename HistogramType::ConstIterator itr = histogramFix->Begin();
        typename HistogramType::ConstIterator end = histogramFix->End();
        double sumFix = histogramFix->GetTotalFrequency();

        while (itr != end) {
            const double count = itr.GetFrequency();
            const typename HistogramType::IndexType ind = itr.GetIndex();

            const double probability = count / sumFix;
            probabilityJointFix[ind[1]][ind[0]] = probability;

            //std::cout << "jointFix[" << ind[1] << "]" << "[" << ind[0] << "]:" << probability << std::endl;

            ++itr;
        }

        // compute the joint probability mov as template
        itr = histogramMov->Begin();
        end = histogramMov->End();
        double sumMov = histogramMov->GetTotalFrequency();

        while (itr != end) {
            const double count = itr.GetFrequency();
            const typename HistogramType::IndexType ind = itr.GetIndex();

            const double probability = count / sumMov;
            probabilityJointMov[ind[1]][ind[0]] = probability;

            //std::cout << "jointMov[" << ind[1] << "]" << "[" << ind[0] << "]:" << probability << std::endl;

            ++itr;
        }

        // compute the unary probability of warped fix
        sizeMov[0] = 1; // number of bins for the first channel
        sizeMov[1] = m_numBinFix; // number of bins for the second channel
        histogramFilterJointMov->SetHistogramSize(sizeMov);
        histogramFilterJointMov->Update();
        histogramMov = histogramFilterJointMov->GetOutput();

        itr = histogramMov->Begin();
        end = histogramMov->End();
        sumFix = histogramMov->GetTotalFrequency();

        while (itr != end) {
            const double count = itr.GetFrequency();
            const typename HistogramType::IndexType ind = itr.GetIndex();

            const double probability = count / sumFix;
            probabilityFixWarped[ind[1]] = probability;

            //std::cout << "warpedFix[" << ind[1] << "]" << "[" << ind[0] << "]:" << probability << std::endl;

            ++itr;
        }

        // compute the unary probability of original mov
        sizeMov[0] = m_numBinMov; // number of bins for the first channel
        sizeMov[1] = 1; // number of bins for the second channel
        histogramFilterJointMov->SetHistogramSize(sizeMov);
        histogramFilterJointMov->Update();
        histogramMov = histogramFilterJointMov->GetOutput();

        itr = histogramMov->Begin();
        end = histogramMov->End();
        sumMov = histogramMov->GetTotalFrequency();

        while (itr != end) {
            const double count = itr.GetFrequency();
            const typename HistogramType::IndexType ind = itr.GetIndex();

            const double probability = count / sumMov;
            probabilityMov[ind[0]] = probability;

            //std::cout << "mov[" << ind[1] << "]" << "[" << ind[0] << "]:" << probability << std::endl;

            ++itr;
        }

        // compute the unary probability of warped mov
        sizeFix[0] = 1; // number of bins for the first channel
        sizeFix[1] = m_numBinMov; // number of bins for the second channel
        histogramFilterJointFix->SetHistogramSize(sizeFix);
        histogramFilterJointFix->Update();
        histogramFix = histogramFilterJointFix->GetOutput();

        itr = histogramFix->Begin();
        end = histogramFix->End();
        sumMov = histogramFix->GetTotalFrequency();

        while (itr != end) {
            const double count = itr.GetFrequency();
            const typename HistogramType::IndexType ind = itr.GetIndex();

            const double probability = count / sumMov;
            probabilityMovWarped[ind[1]] = probability;

            //std::cout << "warpedMov[" << ind[1] << "]" << "[" << ind[0] << "]:" << probability << std::endl;

            ++itr;
        }

        // compute the unary probability of original fix
        sizeFix[0] = m_numBinFix; // number of bins for the first channel
        sizeFix[1] = 1; // number of bins for the second channel
        histogramFilterJointFix->SetHistogramSize(sizeFix);
        histogramFilterJointFix->Update();
        histogramFix = histogramFilterJointFix->GetOutput();

        itr = histogramFix->Begin();
        end = histogramFix->End();
        sumFix = histogramFix->GetTotalFrequency();

        while (itr != end) {
            const double count = itr.GetFrequency();
            const typename HistogramType::IndexType ind = itr.GetIndex();

            const double probability = count / sumFix;
            probabilityFix[ind[0]] = probability;

            //std::cout << "fix[" << ind[1] << "]" << "[" << ind[0] << "]:" << probability << std::endl;

            ++itr;
        }
    }

    template< typename TImage>
    double BlockMatchingImageRegistrationFilter<TImage>::Compute_Similarity(const TImage* fixed, const TImage* moving) {

        double simi = 0.0;

        /**  Type of the metric. */
        typedef itk::MattesMutualInformationImageToImageMetric< InputImageType, InputImageType > MIMetricType;
        typedef itk::NormalizedCorrelationImageToImageMetric< InputImageType, InputImageType > CCMetricType;

        typedef itk::LinearInterpolateImageFunction< InputImageType, double > InterpolatorType;
        typedef itk::IdentityTransform< double, 3 > TransformType;

        /**  Type of the Transform . */
        typename TransformType::Pointer identityTran = TransformType::New();
        typename InterpolatorType::Pointer interpolator = InterpolatorType::New();

        //if (this->m_MetricSelected == 0) {
        if (0 == 0) {
            typename MIMetricType::Pointer metric = MIMetricType::New();
            //typename MIMetricType::ParametersType ParametersType;

            metric->SetMovingImage(moving);
            metric->SetFixedImage(fixed);
            metric->SetTransform(identityTran);
            metric->SetInterpolator(interpolator);
            metric->SetFixedImageRegion(fixed->GetLargestPossibleRegion());
            
            metric->SetNumberOfHistogramBins( 128 );
            metric->SetNumberOfSpatialSamples( 50000 );
            metric->ReinitializeSeed( 76926294 );

            const unsigned int dummyspaceDimension = metric->GetNumberOfParameters();
            typename MIMetricType::ParametersType dummyPosition(dummyspaceDimension);
            for (unsigned int i = 0; i < dummyspaceDimension; i++) {
                dummyPosition[i] = 0;
            }

            metric->Initialize();

            simi = metric->GetValue(dummyPosition);
        }
        
        if (this->m_MetricSelected == 8) {
            typename CCMetricType::Pointer metric = CCMetricType::New();
            //typename MIMetricType::ParametersType ParametersType;

            metric->SetMovingImage(moving);
            metric->SetFixedImage(fixed);
            metric->SetTransform(identityTran);
            metric->SetInterpolator(interpolator);
            metric->SetFixedImageRegion(fixed->GetLargestPossibleRegion());

            const unsigned int dummyspaceDimension = metric->GetNumberOfParameters();
            typename CCMetricType::ParametersType dummyPosition(dummyspaceDimension);
            for (unsigned int i = 0; i < dummyspaceDimension; i++) {
                dummyPosition[i] = 0;
            }

            metric->Initialize();

            simi = metric->GetValue(dummyPosition);
        }

        return simi;
    }

}// end namespace
#endif
