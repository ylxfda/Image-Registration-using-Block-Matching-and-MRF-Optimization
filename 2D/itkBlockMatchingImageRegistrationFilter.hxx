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

namespace itk {

    template< typename TImage>
    BlockMatchingImageRegistrationFilter<TImage>::BlockMatchingImageRegistrationFilter() {
        
        this->SetNumberOfRequiredInputs(2);
		this->SetSmoothness(0.1);
                
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

		for (int l=0; l<this->GetNumOfLevels(); l++) {
			
			this->SetGaussianBlurVariance( this->GetGridSize()/this->GetGaussianBlurFactor()/this->GetNumOfStep() );
			if ( this->GetGridSize()/2/this->GetNumOfStep() < 1.0 ) {
				this->SetNumOfStep(this->GetGridSize()/2);
			}
			ComputeDfmfld( l );
			this->SetGridSize(this->GetGridSize()/1.98);			
		}
        
    }

	template< typename TImage >
	void BlockMatchingImageRegistrationFilter<TImage>::ComputeDfmfld( int level ) {

		// the real work start from here
        // prepare the image, padding and smoothing etc
        this->PrepareImageForMatch( level );

        // make the objective function
        int grid_row = (int)floor(float(this->internalImgSize[1]/this->GetGridSize()))-1; // -2 to prevent from moving out of the image
        int grid_col = (int)floor(float(this->internalImgSize[0]/this->GetGridSize()))-1;
        int numPoints = grid_row*grid_col; // the num of control points on the grid
		int numLabels = (int)powf((this->GetNumOfStep() * 2 + 1), 2.0); 
		int numPairs = (grid_row-1)*grid_col+grid_row*(grid_col-1);
		int max_iters = 1000;

		// puts some information
		std::cout<<"grid size: "<<this->GetGridSize()<<" "<<grid_row<<"x"<<grid_col<<std::endl;
		std::cout<<"searching steps: "<<this->GetNumOfStep()<<std::endl;
		std::cout<<"number of labels: "<<numLabels<<std::endl;
		std::cout<<"number of control points: "<<numPoints<<std::endl;
		std::cout<<"number of edges: "<<numPairs<<std::endl;

        float  *labelCosts = new float[numPoints*numLabels];
		int *edges  = new int [numPairs*2];
		//float  *dist  = new float[numlabels*numlabels];
		float  *wCosts = new float[numPairs];
		for (int i=0; i<numPairs; i++) {
			wCosts[i] = this->GetSmoothness();
		}
        
        this->CreateLabelCostTabel(this->GetNumOfStep(), (float)this->GetGridSize()/(float)this->GetNumOfStep()/2.0, 1.0);

        // define the pairs
        int count = 0;
        for (int row = 0; row<grid_row-1; row++) {
            for (int col = 0; col<grid_col-1; col++) {
                edges[count*2] = row*grid_col+col;
                edges[count*2+1] = row*grid_col+col+1;
				//std::cout<<"adding "<<edges[count*2]<<" and "<<edges[count*2+1]<<std::endl;
                count++;
                
                edges[count*2] = row*grid_col+col;
                edges[count*2+1] = (row+1)*grid_col+col;
				//std::cout<<"adding "<<edges[count*2]<<" and "<<edges[count*2+1]<<std::endl;
                count++;
            }
        }
        
        for (int col=0; col<grid_col-1; col++) {
            edges[count*2] = (grid_row-1)*grid_col+col;
            edges[count*2+1] = (grid_row-1)*grid_col+col+1;
			//std::cout<<"adding "<<edges[count*2]<<" and "<<edges[count*2+1]<<std::endl;
            count++;
        }
        
        for (int row=0; row<grid_row-1; row++) {
            edges[count*2] = row*grid_col+grid_col-1;
            edges[count*2+1] = (row+1)*grid_col+grid_col-1;
			//std::cout<<"adding "<<edges[count*2]<<" and "<<edges[count*2+1]<<std::endl;
            count++;
        }

		std::cout<<"number of edges added: "<<count<<std::endl;

		// define the sampling pattern
		int** offset;
		int numOfSample = 0;
		int minimum = 100;

		if ( this->GetGridSize()*this->GetGridSize()*this->GetSamplingRate() > minimum ) {
			numOfSample = this->GetGridSize()*this->GetGridSize()*this->GetSamplingRate();
			Allocate2DArray<int> (numOfSample, 2, offset);
			RandomSampling( floor((float)this->GetGridSize()/2), floor((float)this->GetGridSize()/2), offset, numOfSample);		
			std::cout<<" sampling at rate "<<std::endl;
		} else if ( this->GetGridSize()*this->GetGridSize() > minimum*2 ) {
			numOfSample = minimum*2;
			Allocate2DArray<int> (numOfSample, 2, offset);		
			RandomSampling( floor((float)this->GetGridSize()/2), floor((float)this->GetGridSize()/2), offset, numOfSample);	
			std::cout<<" sampling at preset minimum "<<std::endl;			
		} else {
			int half_size = floor((float)this->GetGridSize()/2);
		    numOfSample = (half_size*2+1)*(half_size*2+1);
			Allocate2DArray<int> (numOfSample, 2, offset);
			int count = 0;
			for (int i = -half_size; i <= half_size; i++) {
				for (int j = -half_size; j <= half_size; j++ ) {
					offset[count][1] = i;
					offset[count][0] = j;
					count++;
				}
			}
			std::cout<<" sampling all "<<std::endl;
		}
		
		std::cout<<"The number of samples in each grid: "<<numOfSample<<std::endl;		
	
		// compute the similarity cost (takes most of the time, can be paralelled)
		//count = 0;
		int center_x, center_y, p;
		for (int r=1; r<=grid_row; r++) {
			for (int c=1; c<=grid_col; c++) {
				//get the position of this point
				center_x = this->GetGridSize()*r;
				center_y = this->GetGridSize()*c;
				p = (r-1)*grid_col+c-1;
				//std::cout<<"computing the cost for control points #"<<p<<" position: "<<center_y<<" "<<center_x<<std::endl;
				for (int l=0; l<numLabels; l++) {
					//float cost = NCC(fixedImg4Match, movingImg4Match, center_x, center_y, this->label2displacement[l][1], this->label2displacement[l][0], offset, numOfSample);
					labelCosts[ l*numPoints + p ] = NCC(warpedMovImg4Match, fixedImg4Match, center_x, center_y, this->label2displacement[l][1], this->label2displacement[l][0], offset, numOfSample);		
					//count++;
					//lcosts[ p*numlabels + l ] = cost;
					//std::cout<<"label #"<<l<<" move: "<<this->label2displacement[l][1]<<" "<<this->label2displacement[l][0]<<" cost: "<<cost<<std::endl;
				}
			 }
		}	
 
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
	
       GCLib::GC pd_fix2mov( numPoints, numLabels, labelCosts, numPairs, edges, label2cost, max_iters, wCosts );
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

	   typename InternalImageType::SizeType size;
       typename InternalImageType::SpacingType spacing;
       //size[0] = grid_col+2;
       //size[1] = grid_row+2;
	   size[0] = grid_col;
       size[1] = grid_row;
       
       spacing[0] = this->GetGridSize();
       spacing[1] = this->GetGridSize();

       typename InternalImageType::RegionType region;
       typename InternalImageType::IndexType start;

       start[0] = 0;
       start[1] = 0;
            
	   region.SetSize(size);
       region.SetIndex(start);
	   			
       dfmgrid_fix2mov_x->SetRegions(region); 
	   dfmgrid_fix2mov_y->SetRegions(region); 
	   dfmgrid_fix2mov_x->SetSpacing(spacing);
	   dfmgrid_fix2mov_y->SetSpacing(spacing);
       dfmgrid_fix2mov_x->Allocate(); 
	   dfmgrid_fix2mov_y->Allocate();
			
       typename InternalImageType::IndexType Index;
		for (int r=0; r<grid_row; r++) {
			for (int c=0; c<grid_col; c++) {

				Index[1]=r;
				Index[0]=c;
								
				//get the position of this point
				p = r*grid_col+c;
				dfmgrid_fix2mov_x->SetPixel(Index, label2displacement[pd_fix2mov._pinfo[p].label][1]);
				dfmgrid_fix2mov_y->SetPixel(Index, label2displacement[pd_fix2mov._pinfo[p].label][0]);

			 }
		}	

		//pd_fix2mov.~GC();

		//WriteImage<InternalImageType>( "dfmgrid_fix2mov_x.mhd", dfmgrid_fix2mov_x );
		//WriteImage<InternalImageType>( "dfmgrid_fix2mov_y.mhd", dfmgrid_fix2mov_y );

		// make the dfmfld in the other direction --------------------------------------------------------------
		for (int r=1; r<=grid_row; r++) {
			for (int c=1; c<=grid_col; c++) {
				//get the position of this point
				center_x = this->GetGridSize()*r;
				center_y = this->GetGridSize()*c;
				p = (r-1)*grid_col+c-1;
				//std::cout<<"computing the cost for control points #"<<p<<" position: "<<center_y<<" "<<center_x<<std::endl;
				for (int l=0; l<numLabels; l++) {
					//float cost = NCC(fixedImg4Match, movingImg4Match, center_x, center_y, this->label2displacement[l][1], this->label2displacement[l][0], offset, numOfSample);
					labelCosts[ l*numPoints + p ] = NCC(warpedFixImg4Match, movingImg4Match, center_x, center_y, this->label2displacement[l][1], this->label2displacement[l][0], offset, numOfSample);		
					//count++;
					//lcosts[ p*numlabels + l ] = cost;
					//std::cout<<"label #"<<l<<" move: "<<this->label2displacement[l][1]<<" "<<this->label2displacement[l][0]<<" cost: "<<cost<<std::endl;
				}
			 }
		}

		GCLib::GC pd_mov2fix( numPoints, numLabels, labelCosts, numPairs, edges, label2cost, max_iters, wCosts );
		pd_mov2fix.run();

	   typename InternalImageType::Pointer dfmgrid_mov2fix_x = InternalImageType::New();
	   typename InternalImageType::Pointer dfmgrid_mov2fix_y = InternalImageType::New();
	   			
       dfmgrid_mov2fix_x->SetRegions(region); 
	   dfmgrid_mov2fix_y->SetRegions(region); 
	   dfmgrid_mov2fix_x->SetSpacing(spacing);
	   dfmgrid_mov2fix_y->SetSpacing(spacing);
       dfmgrid_mov2fix_x->Allocate(); 
	   dfmgrid_mov2fix_y->Allocate();
			
        for (int r=0; r<grid_row; r++) {
			for (int c=0; c<grid_col; c++) {

				Index[1]=r;
				Index[0]=c;
								
				//get the position of this point
				p = r*grid_col+c;
				dfmgrid_mov2fix_x->SetPixel(Index, label2displacement[pd_mov2fix._pinfo[p].label][1]);
				dfmgrid_mov2fix_y->SetPixel(Index, label2displacement[pd_mov2fix._pinfo[p].label][0]);

			 }
		}

		//pd_mov2fix.~GC();

		//WriteImage<InternalImageType>( "dfmgrid_mov2fix_x.mhd", dfmgrid_mov2fix_x );
		//WriteImage<InternalImageType>( "dfmgrid_mov2fix_y.mhd", dfmgrid_mov2fix_y );

		// make the deformation field by bspline inter-polation
		typename InternalImageType::Pointer dfmfld_x_fix2mov = InternalImageType::New();
		typename InternalImageType::Pointer dfmfld_y_fix2mov = InternalImageType::New();
		typename InternalImageType::Pointer dfmfld_x_mov2fix = InternalImageType::New();
		typename InternalImageType::Pointer dfmfld_y_mov2fix = InternalImageType::New();

		InterpImageBySpline( dfmgrid_fix2mov_x, dfmfld_x_fix2mov, grid_row, grid_col, this->inputImgSize[1], this->inputImgSize[0]);
		InterpImageBySpline( dfmgrid_fix2mov_y, dfmfld_y_fix2mov, grid_row, grid_col, this->inputImgSize[1], this->inputImgSize[0]);
		InterpImageBySpline( dfmgrid_mov2fix_x, dfmfld_x_mov2fix, grid_row, grid_col, this->inputImgSize[1], this->inputImgSize[0]);
		InterpImageBySpline( dfmgrid_mov2fix_y, dfmfld_y_mov2fix, grid_row, grid_col, this->inputImgSize[1], this->inputImgSize[0]);

		//WriteImage<InternalImageType>( "dfmfld_mov2fix_x.mhd", dfmfld_x_mov2fix );
		//WriteImage<InternalImageType>( "dfmfld_mov2fix_y.mhd", dfmfld_y_mov2fix );

		// combine the component to have the vector dfmfld
		typedef itk::ComposeImageFilter<InternalImageType, DeformationFieldType> ImageToVectorImageFilterType;
		typename ImageToVectorImageFilterType::Pointer imageToVectorImageFilter_mov2fix = ImageToVectorImageFilterType::New();
		typename ImageToVectorImageFilterType::Pointer imageToVectorImageFilter_fix2mov = ImageToVectorImageFilterType::New();
		imageToVectorImageFilter_mov2fix->SetInput(1, dfmfld_x_mov2fix);
		imageToVectorImageFilter_mov2fix->SetInput(0, dfmfld_y_mov2fix);
		imageToVectorImageFilter_mov2fix->Update();
		imageToVectorImageFilter_fix2mov->SetInput(1, dfmfld_x_fix2mov);
		imageToVectorImageFilter_fix2mov->SetInput(0, dfmfld_y_fix2mov);
		imageToVectorImageFilter_fix2mov->Update();

		// update the dfmfld by composing the fld from last level     
		typedef itk::ComposeDisplacementFieldsImageFilter<DeformationFieldType> DfmfldComposerType;   
	    if (level == 0) {
			 this->dfmfld_fix2mov =  imageToVectorImageFilter_fix2mov->GetOutput();
			 this->dfmfld_mov2fix =  imageToVectorImageFilter_mov2fix->GetOutput();
	    } else {
			 typename DfmfldComposerType::Pointer composer_fix2mov = DfmfldComposerType::New();
			 typename DfmfldComposerType::Pointer composer_mov2fix = DfmfldComposerType::New();

			 composer_fix2mov->SetWarpingField( imageToVectorImageFilter_fix2mov->GetOutput() );
			 composer_fix2mov->SetDisplacementField( this->dfmfld_fix2mov );
			 composer_fix2mov->Update();		
			 this->dfmfld_fix2mov = composer_fix2mov->GetOutput();

			 composer_mov2fix->SetWarpingField( imageToVectorImageFilter_mov2fix->GetOutput() );
			 composer_mov2fix->SetDisplacementField( this->dfmfld_mov2fix );
			 composer_mov2fix->Update();		
			 this->dfmfld_mov2fix = composer_mov2fix->GetOutput();
	    }

		// crop the deformation field
		/*InternalImageType::Pointer dfmfld_x_crop_fix2mov = InternalImageType::New();
		typename InternalImageType::Pointer dfmfld_y_crop_fix2mov = InternalImageType::New();
		typename InternalImageType::Pointer dfmfld_x_crop_mov2fix = InternalImageType::New();
		typename InternalImageType::Pointer dfmfld_y_crop_mov2fix = InternalImageType::New();

		CropImage(dfmfld_x_fix2mov, dfmfld_x_crop_fix2mov, this->GetGridSize());
		CropImage(dfmfld_y_fix2mov, dfmfld_y_crop_fix2mov, this->GetGridSize());		 

		CropImage(dfmfld_x_mov2fix, dfmfld_x_crop_mov2fix, this->GetGridSize());
		CropImage(dfmfld_y_mov2fix, dfmfld_y_crop_mov2fix, this->GetGridSize());	*/


   //WriteImage<DeformationFieldType>( "dfmfld_fix2mov.nii.gz", composer_fix2mov->GetOutput() );
   //WriteImage<DeformationFieldType>( "dfmfld_mov2fix.nii.gz", composer_mov2fix->GetOutput() );

		symetricDfmfld();	

	  // warp the image
		this->warpImage();

		//WriteImage<InputImageType>( "warpedMov.nii.gz", this->warpedMovImage );
		//WriteImage<InputImageType>( "warpedFix.nii.gz", this->warpedFixImage );
		//WriteImage<DeformationFieldType>( "dfmfld_fix2mov.nii.gz", this->dfmfld_fix2mov );
		//WriteImage<DeformationFieldType>( "dfmfld_mov2fix.nii.gz", this->dfmfld_mov2fix );
		//WriteImage<InternalImageType>( "dfmfld_x.nii.gz", dfmfld_x_crop );
		//WriteImage<InternalImageType>( "dfmfld_y.nii.gz", dfmfld_y_crop );

        // release the image not needed anymore
        Release2DArray<typename InternalImageType::PixelType > (this->internalImgSize[1], this->internalImgSize[0], this->fixedImg4Match);
        Release2DArray<typename InternalImageType::PixelType > (this->internalImgSize[1], this->internalImgSize[0], this->movingImg4Match);

		Release2DArray<typename InternalImageType::PixelType > (this->internalImgSize[1], this->internalImgSize[0], this->warpedFixImg4Match);
        Release2DArray<typename InternalImageType::PixelType > (this->internalImgSize[1], this->internalImgSize[0], this->warpedMovImg4Match);

		Release2DArray<typename InternalImageType::PixelType>(numLabels, 2, this->label2displacement);

		Release2DArray<int > (numOfSample, 2, offset);
        
        delete[] labelCosts, edges, label2cost, wCosts;

	}

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::CropImage( typename InternalImageType::Pointer& inputImg, typename InternalImageType::Pointer& croppedImg, int cropSize) {
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
            for (long i = cropSize; i < sizeOfImage[1]-cropSize; i++) {
                for (long j = cropSize; j < sizeOfImage[0]-cropSize; j++) {
                   
						IndexIn[0]=j;
						IndexIn[1]=i;
						IndexOut[0]=j - cropSize;
						IndexOut[1]=i - cropSize;

						croppedImg->SetPixel(IndexOut, inputImg->GetPixel(IndexIn));
                    
                }
            }

    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::PadImage(const TImage* inputImg, typename TImage::Pointer& paddedImg, int padSize) {
        // pad the image with the highest level grid size
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
    void BlockMatchingImageRegistrationFilter<TImage>::GaussianSmoothing(const typename TImage::Pointer inputImg, typename InternalImageType::Pointer& smoothedImg, float sigma) {

		typedef DiscreteGaussianImageFilter< TImage, InternalImageType > GaussianFilterType;

        typename GaussianFilterType::Pointer BlurImgFilter = GaussianFilterType::New();

        BlurImgFilter->SetInput(inputImg);
        BlurImgFilter->SetVariance(sigma);

        BlurImgFilter->Update();

        smoothedImg = BlurImgFilter->GetOutput();

    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::PrepareImageForMatch( int level ) {
        
		int gridSize = this->GetGridSize();
		int variance = this->GetGaussianBlurVariance();

		// pad the image 
        typename TImage::Pointer paddedFixed, paddedMoving, paddedFixedWarped, paddedMovingWarped;     
		if (level == 0) {
			this->PadImage(this->GetMovingImage(), paddedMoving, gridSize );
		    this->PadImage(this->GetFixedImage(), paddedFixed, gridSize );
			this->PadImage(this->GetMovingImage(), paddedMovingWarped, gridSize );
		    this->PadImage(this->GetFixedImage(), paddedFixedWarped, gridSize );
		}	else {
			this->PadImage(this->warpedMovImage, paddedMovingWarped, gridSize );
			this->PadImage(this->warpedFixImage, paddedFixedWarped, gridSize );
			this->PadImage(this->GetMovingImage(), paddedMoving, gridSize );
			this->PadImage(this->GetFixedImage(), paddedFixed, gridSize );
		}

        // blur the image
        typename InternalImageType::Pointer smoothedFixed, smoothedMoving, smoothedWarpedFixed, smoothedWarpedMoving;
        this->GaussianSmoothing(paddedFixed, smoothedFixed, variance);
        this->GaussianSmoothing(paddedMoving, smoothedMoving, variance);	
		 this->GaussianSmoothing(paddedFixedWarped, smoothedWarpedFixed, variance);
        this->GaussianSmoothing(paddedMovingWarped, smoothedWarpedMoving, variance);	

        // allocate the space for image array used in real matching
        this->internalImgSize = smoothedFixed->GetLargestPossibleRegion().GetSize();
		this->inputImgSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize();
        
		// debug only
		//WriteImage<InternalImageType>( "internalMov.mhd", smoothedMoving );
		//WriteImage<InternalImageType>( "internalFix.mhd", smoothedFixed );

		std::cout<<"input size: "<<this->inputImgSize[1]<<"x"<<this->inputImgSize[0]<<std::endl;
		std::cout<<"internal size: "<<this->internalImgSize[1]<<"x"<<this->internalImgSize[0]<<std::endl;
        
        Allocate2DArray<typename InternalImageType::PixelType > (internalImgSize[1], internalImgSize[0], this->fixedImg4Match);
        Allocate2DArray<typename InternalImageType::PixelType > (internalImgSize[1], internalImgSize[0], this->movingImg4Match);
		Allocate2DArray<typename InternalImageType::PixelType > (internalImgSize[1], internalImgSize[0], this->warpedFixImg4Match);
        Allocate2DArray<typename InternalImageType::PixelType > (internalImgSize[1], internalImgSize[0], this->warpedMovImg4Match);

		typedef itk::ImageRegionIteratorWithIndex< InternalImageType > IteratorType;
		IteratorType itFix( smoothedFixed, smoothedFixed->GetRequestedRegion() );
		IteratorType itMov( smoothedMoving, smoothedMoving->GetRequestedRegion() );
		IteratorType itFixWarped( smoothedWarpedFixed, smoothedWarpedFixed->GetRequestedRegion() );
		IteratorType itMovWarped( smoothedWarpedMoving, smoothedWarpedMoving->GetRequestedRegion() );

		for ( itFix.GoToBegin(); !itFix.IsAtEnd(); ++itFix) {
			typename InternalImageType::IndexType idx = itFix.GetIndex();
			//std::cout<<"index: "<<idx[1]<<" "<<idx[0]<<std::endl;
			fixedImg4Match[ idx[1]+this->GetGridSize() ][ idx[0]+this->GetGridSize() ] = smoothedFixed->GetPixel(idx);
			movingImg4Match[ idx[1]+this->GetGridSize() ][ idx[0]+this->GetGridSize() ] = smoothedMoving->GetPixel(idx);
			warpedFixImg4Match[ idx[1]+this->GetGridSize() ][ idx[0]+this->GetGridSize() ] = smoothedWarpedFixed->GetPixel(idx);
			warpedMovImg4Match[ idx[1]+this->GetGridSize() ][ idx[0]+this->GetGridSize() ] = smoothedWarpedMoving->GetPixel(idx);
		}

    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::CreateLabelCostTabel(int numOfSteps, float stepSize, float weight) {
        // compute number of lablels
        int numOfLabels = (int)powf((this->GetNumOfStep() * 2 + 1), 2.0);

		std::cout<<"constructing the label to movement table ..."<<std::endl;
        // Table label2displacement
        Allocate2DArray<float>(numOfLabels, 2, this->label2displacement);
        int index = 0;
        for (int i = -numOfSteps; i <= numOfSteps; i++) {
            for (int j = -numOfSteps; j <= numOfSteps; j++) {
                this->label2displacement[index][1] = (float)i*stepSize;
                this->label2displacement[index][0] = (float)j*stepSize;
                //std::cout<<"label #"<<index<<" move: "<<label2displacement[index][1]<<" "<<label2displacement[index][0]<<std::endl;
				index++;
            }
        }

        // Table label2cost
        //Allocate2DArray<float>(numOfLabels, numOfLabels, this->label2cost);
        this->label2cost = new float[numOfLabels*numOfLabels];
        for (long i = 0; i < numOfLabels; i++) {
            for (long j = 0; j < numOfLabels; j++) {
                this->label2cost[i*numOfLabels+j] = sqrt(pow(float(label2displacement[i][0] - label2displacement[j][0]), 2) + pow(float(label2displacement[i][1] - label2displacement[j][1]), 2)) / this->GetGridSize() * weight;
				//this->label2cost[i*numOfLabels+j] = sqrt(pow(float(label2displacement[i][0] - label2displacement[j][0]), 2) + pow(float(label2displacement[i][1] - label2displacement[j][1]), 2)) * weight;
            } // note: it is normalized by GridSize;
        }

    }

    template< typename TImage>
    template<typename T> void BlockMatchingImageRegistrationFilter<TImage>::Allocate2DArray(int x, int y, T**& array) {
        
		array = (T**)malloc(x * sizeof (T*));

        if (array == NULL) {
            fprintf(stderr, "out of memory\n");
            exit (0);
        }

        for (int i = 0; i < x; i++) {
            array[i] = (T*)malloc(y * sizeof (T));
            if (array[i] == NULL) {
                fprintf(stderr, "out of memory\n");
                exit (0);
            }
        }
    }

    template< typename TImage>
    template<typename T> void BlockMatchingImageRegistrationFilter<TImage>::Release2DArray(int x, int y, T**& array) {
        for (int i = 0; i < x; i++) {
            free(array[i]);
        }
        free(array);
    }

    template< typename TImage>
    void BlockMatchingImageRegistrationFilter<TImage>::RandomSampling(int radius_x, int radius_y, int** offsets, int numOfSample) {
        for (int i = 0; i < numOfSample; i++) {
            offsets[i][1] = rand() % (radius_x * 2) - radius_x;
            offsets[i][0] = rand() % (radius_y * 2) - radius_y;
        }
    }

    template< typename TImage>
    float BlockMatchingImageRegistrationFilter<TImage>::
    NCC(typename InternalImageType::PixelType**& fixed, typename InternalImageType::PixelType**& moving, int center_x, int center_y, float translate_x, float translate_y, int** offsets, int numOfSample) {
        /*  
         center_x, center_y: center of the block to be moved 
         translate_x, translate_y: the translation of the block in moving image
         
         center_x+translate_x is the corresponding position in fixed image
         */

        float sumSquareFixed = 0.0;
        float sumSquareMoving = 0.0;
        float sumProduct = 0.0;

        typename InternalImageType::PixelType interpFixedPixel;
        typename InternalImageType::PixelType movingPixel;

        for (int i = 0; i < numOfSample; i++) {

            interpFixedPixel = LinearInterpImage<typename InternalImageType::PixelType > ( fixed, center_x + translate_x + offsets[i][1], center_y + translate_y + offsets[i][0] );
            movingPixel = moving[center_x + offsets[i][1]][center_y + offsets[i][0]];

			//std::cout<<"fixed: "<<center_x + translate_x + offsets[i][1]<<" "<<center_y + translate_y + offsets[i][0]<<std::endl;
			//std::cout<<"moving: "<<center_x + offsets[i][1]<<" "<<center_y + offsets[i][0]<<std::endl;

            sumSquareFixed += powf(interpFixedPixel, 2.0);
            sumSquareMoving += powf(movingPixel, 2.0);
			sumProduct += interpFixedPixel*movingPixel;
        }

        //float norm = sumSquareFixed*sumSquareMoving;

		if ( (sumSquareFixed == 0.0) && (sumSquareMoving == 0.0) ) {
			return (0.0);
		}
        else if ( (sumSquareMoving != 0.0) && (sumSquareFixed != 0.0) ) {
            return (1-sumProduct*sumProduct/sumSquareFixed/sumSquareMoving);
        } 
		else {
			return (2.0);
		}	
    }

    template< typename TImage>
    template<typename T> T BlockMatchingImageRegistrationFilter<TImage>::LinearInterpImage(T** Input, float x, float y) {

        float low_x, up_x, low_y, up_y;
        
        low_x = floor(x);
        if (low_x==x)
            up_x = low_x;
        else
            up_x = floor(x+1);
        
        low_y = floor(y);
        if (low_y==y)
            up_y = low_y;
        else
            up_y = floor(y+1);
        
        return ( (Input[(int)low_x][(int)low_y]*(x - low_x) + Input[(int)up_x][(int)low_y]*(up_x - x))*(y - low_y) + (Input[(int)low_x][(int)up_y]*(x - low_x) + Input[(int)up_x][(int)up_y]*(up_x - x))*(up_y - y) );

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
	void BlockMatchingImageRegistrationFilter<TImage>::InterpImageBySpline( const typename InternalImageType::Pointer& InputImg, typename InternalImageType::Pointer& OutputImg, int in_x, int in_y, int out_x, int out_y) {
				
		//do the interpolation using cubic spline
		typedef itk::IdentityTransform<double, 2>    TransformType;
		typedef itk::BSplineInterpolateImageFunction<InternalImageType, double, double>    InterpolatorType;		
		typedef itk::ResampleImageFilter<InternalImageType, InternalImageType>    ResampleFilterType;
		
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
		const double vfOutputOrigin[2]  = { 0.0, 0.0 };
		ResizeFilter->SetOutputOrigin(vfOutputOrigin);
		
		  double vfOutputSpacing[2];
		  vfOutputSpacing[0] = 1;
		  vfOutputSpacing[1] = 1;
		 
		  // Set the output spacing. 
		  ResizeFilter->SetOutputSpacing(vfOutputSpacing);
		  
		  // Set the output size
		  typename InternalImageType::SizeType vnOutputSize;
		  vnOutputSize[0] = out_y;
		  vnOutputSize[1] = out_x;
		  
		  ResizeFilter->SetSize(vnOutputSize);		 
		  ResizeFilter->SetInput(InputImg);		  
		  ResizeFilter->Update();
		  
		  // do the interpolation 
		 OutputImg = ResizeFilter->GetOutput();

	}

	template< typename TImage>
	template <typename T> void BlockMatchingImageRegistrationFilter<TImage>::WriteImage( const char * filename, const T* image ) 
	{
	  typedef ImageFileWriter<T> WriterType;
	  typename WriterType::Pointer writer = WriterType::New();
	  writer->SetFileName( filename );
	  writer->SetInput( image );

	  try
		{
		writer->Update();
		}
	  catch( ExceptionObject & excp )
		{
		throw excp;
		}
	  catch( ... )
		{
		ExceptionObject e( __FILE__, __LINE__, 
								  "Caught unknown exception", ITK_LOCATION );
		throw e;
		}
	}

	template< typename TImage>
	void BlockMatchingImageRegistrationFilter<TImage>::symetricDfmfld(void) {

		//WriteImage<DeformationFieldType>( "dfmfld_in.nii.gz", this->dfmfld_fix2mov );

		// touchup
		int expstep = 4;
		float factor = (float)this->GetGridSize()/2.0;

		typedef itk::ImageRegionIterator< DeformationFieldType> IteratorType;
		IteratorType fix2movIt( this->dfmfld_fix2mov, this->dfmfld_fix2mov->GetLargestPossibleRegion() );
		IteratorType mov2fixIt( this->dfmfld_mov2fix, this->dfmfld_mov2fix->GetLargestPossibleRegion() );

		float factorF=1.0/factor;
		float coeff=1.0/(float)powf(2.0,expstep);

		fix2movIt.GoToBegin();
		mov2fixIt.GoToBegin();

		while( !fix2movIt.IsAtEnd() )
		{
			typename DeformationFieldType::PixelType vector = fix2movIt.Get();
			vector[1] = coeff*vector[1]*factorF;
			vector[0] = coeff*vector[0]*factorF;
			fix2movIt.Set( vector );
			++fix2movIt;
		}

		while( !mov2fixIt.IsAtEnd() )
		{
			typename DeformationFieldType::PixelType vector = mov2fixIt.Get();
			vector[1] = coeff*vector[1]*factorF;
			vector[0] = coeff*vector[0]*factorF;
			mov2fixIt.Set( vector );
			++mov2fixIt;
		}
		 
		//WriteImage<DeformationFieldType>( "dfmfld_beforeCompose.nii.gz", this->dfmfld_fix2mov );

		typedef itk::ComposeDisplacementFieldsImageFilter<DeformationFieldType> DfmfldComposerType;	   
		//DeformationFieldType::Pointer tempMov2Fix = this->dfmfld_mov2fix;
		//DeformationFieldType::Pointer tempFix2Mov = this->dfmfld_fix2mov;

		for (int it=0; it<expstep; it++) {
			typename DfmfldComposerType::Pointer composer_fix2mov = DfmfldComposerType::New();
			typename DfmfldComposerType::Pointer composer_mov2fix = DfmfldComposerType::New();

			composer_fix2mov->SetWarpingField( this->dfmfld_fix2mov );
			composer_fix2mov->SetDisplacementField( this->dfmfld_fix2mov );
			composer_fix2mov->Update();		
			this->dfmfld_fix2mov = composer_fix2mov->GetOutput();

			composer_mov2fix->SetWarpingField( this->dfmfld_mov2fix );
			composer_mov2fix->SetDisplacementField( this->dfmfld_mov2fix );
			composer_mov2fix->Update();		
			this->dfmfld_mov2fix = composer_mov2fix->GetOutput();
			
	    }	

		//WriteImage<DeformationFieldType>( "dfmfld_afterCompose.nii.gz", this->dfmfld_fix2mov );

		IteratorType fix2movIt2( this->dfmfld_fix2mov, this->dfmfld_fix2mov->GetLargestPossibleRegion() );
		IteratorType mov2fixIt2( this->dfmfld_mov2fix, this->dfmfld_mov2fix->GetLargestPossibleRegion() );
		fix2movIt2.GoToBegin();
		mov2fixIt2.GoToBegin();

		while( !fix2movIt2.IsAtEnd() )
		{
			typename DeformationFieldType::PixelType vector = fix2movIt2.Get();
			
			//std::cout<<"before: "<<vector[1]<<std::endl;
			vector[1] = vector[1]*factor*0.5;
			vector[0] = vector[0]*factor*0.5;
			fix2movIt2.Set( vector );
			//std::cout<<"after: "<<vector[1]<<std::endl;

			++fix2movIt2;			
		}

		while( !mov2fixIt2.IsAtEnd() )
		{
			typename DeformationFieldType::PixelType vector = mov2fixIt2.Get();

			vector[1] = vector[1]*factor*0.5;
			vector[0] = vector[0]*factor*0.5;
			mov2fixIt2.Set( vector );

			++mov2fixIt2;
		}

		//WriteImage<DeformationFieldType>( "dfmfld_afterTouchup.nii.gz", this->dfmfld_fix2mov );

		// caculate inverse for both direction
		typedef   itk::InvertDisplacementFieldImageFilter< DeformationFieldType, DeformationFieldType >  DfmfldInverterType;
		typename DfmfldInverterType::Pointer invert_mov2fix = DfmfldInverterType::New();
		typename DfmfldInverterType::Pointer invert_fix2mov = DfmfldInverterType::New();
		
       invert_mov2fix->SetInput( this->dfmfld_mov2fix );	
	   invert_mov2fix->SetMaximumNumberOfIterations( 10 );
	   invert_mov2fix->SetMeanErrorToleranceThreshold( 0.001 );
       invert_mov2fix->SetMaxErrorToleranceThreshold( 0.1 );
       invert_mov2fix->Update();
		
	   invert_fix2mov->SetInput( this->dfmfld_fix2mov );	
	   invert_fix2mov->SetMaximumNumberOfIterations( 10 );
	   invert_fix2mov->SetMeanErrorToleranceThreshold( 0.001 );
       invert_fix2mov->SetMaxErrorToleranceThreshold( 0.1 );
       invert_fix2mov->Update();

	  try
		{
		invert_fix2mov->UpdateLargestPossibleRegion();
		invert_mov2fix->UpdateLargestPossibleRegion();
		}
	  catch( itk::ExceptionObject & excp )
		{
		std::cerr << "Exception thrown while inverting dfmfld" << std::endl;
		std::cerr << excp << std::endl;
		}
			
	   // combine the two directions to be symetric
	   typename DfmfldComposerType::Pointer composer_fix2mov = DfmfldComposerType::New();
	   typename DfmfldComposerType::Pointer composer_mov2fix = DfmfldComposerType::New();

		 composer_fix2mov->SetWarpingField( invert_mov2fix->GetOutput() );
		 composer_fix2mov->SetDisplacementField( this->dfmfld_fix2mov );
		 composer_fix2mov->Update();		
		 this->dfmfld_fix2mov = composer_fix2mov->GetOutput();

		 composer_mov2fix->SetWarpingField( invert_fix2mov->GetOutput() );
		 composer_mov2fix->SetDisplacementField( this->dfmfld_mov2fix );
		 composer_mov2fix->Update();		
		 this->dfmfld_mov2fix = composer_mov2fix->GetOutput();

	}

	template< typename TImage>
	void BlockMatchingImageRegistrationFilter<TImage>::warpImage( void )
	{  
		  typedef itk::WarpImageFilter< InputImageType, InputImageType, DeformationFieldType  >  WarpImageFilterType; 
		  typename WarpImageFilterType::Pointer warpImageFilterMov = WarpImageFilterType::New();
		  typename WarpImageFilterType::Pointer warpImageFilterFix = WarpImageFilterType::New();
		 
		  typedef itk::LinearInterpolateImageFunction< InputImageType, double >  LinearInterpolatorType;		  
		  typename LinearInterpolatorType::Pointer interpolatorLinearMov =  LinearInterpolatorType::New();
		  typename LinearInterpolatorType::Pointer interpolatorLinearFix =  LinearInterpolatorType::New();
		  
		  warpImageFilterMov->SetInterpolator( interpolatorLinearMov );		  	
		  warpImageFilterMov->SetOutputSpacing( this->GetFixedImage()->GetSpacing() );
		  warpImageFilterMov->SetOutputOrigin(  this->GetFixedImage()->GetOrigin() );
		  warpImageFilterMov->SetDisplacementField( this->dfmfld_fix2mov );
		  warpImageFilterMov->SetInput( this->GetMovingImage() );
		  warpImageFilterMov->Update(); 
		  this->warpedMovImage = warpImageFilterMov->GetOutput(); 

		  warpImageFilterFix->SetInterpolator( interpolatorLinearFix );		  	
		  warpImageFilterFix->SetOutputSpacing( this->GetMovingImage()->GetSpacing() );
		  warpImageFilterFix->SetOutputOrigin(  this->GetMovingImage()->GetOrigin() );
		  warpImageFilterFix->SetDisplacementField( this->dfmfld_mov2fix );
		  warpImageFilterFix->SetInput( this->GetFixedImage() );
		  warpImageFilterFix->Update(); 
		  this->warpedFixImage = warpImageFilterFix->GetOutput(); 
	}

	template< typename TImage>
	void BlockMatchingImageRegistrationFilter<TImage>::outputResults(char* warpedFix, char* warpedMov, char* fld2Fix, char* fld2Mov) {
		WriteImage<InputImageType>( warpedMov, this->warpedMovImage );
		WriteImage<InputImageType>( warpedFix, this->warpedFixImage );
		WriteImage<DeformationFieldType>( fld2Mov, this->dfmfld_fix2mov );
		WriteImage<DeformationFieldType>( fld2Fix, this->dfmfld_mov2fix );
	}

}// end namespace

#endif 
