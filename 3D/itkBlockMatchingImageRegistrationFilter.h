#ifndef __itkBlockMatchingImageRegistrationFilter_h
#define __itkBlockMatchingImageRegistrationFilter_h

#include "itkImageToImageFilter.h"
#include <math.h>
#include "itkJoinImageFilter.h"
#include "itkImageToHistogramFilter.h"

float interp1d(float x, float x1, float x2, float q00, float q01) {
	/*float val = ((x2 - x) / (x2 - x1)) * q00 + ((x - x1) / (x2 - x1)) * q01;
	if (isnan(val)!=0) {
		std::cout << "not a number" << std::endl;
	}*/
	if ((x2 - x1) == 0) {
		return 0.0;
	}
	else {
		return ((x2 - x) / (x2 - x1)) * q00 + ((x - x1) / (x2 - x1)) * q01;
	}
}

float interp2d(float x, float y, float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2) {
	float r1 = interp1d(x, x1, x2, q11, q21);
	float r2 = interp1d(x, x1, x2, q12, q22);

	return interp1d(y, y1, y2, r1, r2);
}

float interp3d(float x, float y, float z, float q000, float q001, float q010, float q011, float q100, float q101, float q110, float q111, float x1, float x2, float y1, float y2, float z1, float z2) {
	float x00 = interp1d(x, x1, x2, q000, q100);
	float x10 = interp1d(x, x1, x2, q010, q110);
	float x01 = interp1d(x, x1, x2, q001, q101);
	float x11 = interp1d(x, x1, x2, q011, q111);
	float r0 = interp1d(y, y1, y2, x00, x01);
	float r1 = interp1d(y, y1, y2, x10, x11);

	return interp1d(z, z1, z2, r0, r1);
}

namespace itk
{
	template< typename TImage>
	class BlockMatchingImageRegistrationFilter : public ImageToImageFilter< TImage, TImage >
	{
	public:
		itkStaticConstMacro(ImageDimension, unsigned, TImage::ImageDimension);

		/** Standard class typedefs. */
		typedef BlockMatchingImageRegistrationFilter             Self;
		typedef ImageToImageFilter< TImage, TImage > Superclass;
		typedef SmartPointer< Self >        Pointer;

		/** typedefs */
		typedef TImage                            InputImageType;
		typedef Image<float, ImageDimension>       InternalImageType;
		//typedef InputImageType                    InternalImageType;
		typedef   itk::Vector< float, ImageDimension >    VectorType;
		typedef   itk::Image< VectorType, ImageDimension >   DeformationFieldType;
		//typedef Image<float, ImageDimension>       DfmfldImageType;
		typedef itk::JoinImageFilter< InternalImageType, InternalImageType >  JoinImageFilterType;
		typedef typename JoinImageFilterType::OutputImageType               OverlapImageType;
		typedef typename itk::Statistics::ImageToHistogramFilter< OverlapImageType >  HistogramFilterType;

		/** Method for creation through the object factory. */
		itkNewMacro(Self);

		/** Run-time type information (and related methods). */
		itkTypeMacro(BlockMatchingImageRegistrationFilter, ImageToImageFilter);

		/** The image to be inpainted in regions where the mask is white.*/
		void SetFixedImage(const TImage* image);

		/** The mask to be inpainted. White pixels will be inpainted, black pixels will be passed through to the output.*/
		void SetMovingImage(const TImage* image);

		/** set/get gaussian blur size */
		itkSetMacro(GaussianBlurVariance, float);
		itkGetConstMacro(GaussianBlurVariance, float);

		/** set/get sampling rate */
		itkSetMacro(SamplingRate, float);
		itkGetConstMacro(SamplingRate, float);

		/** set/get smoothness */
		itkSetMacro(Smoothness, float);
		itkGetConstMacro(Smoothness, float);

		/** set/get grid size */
		itkSetMacro(GridSize, int);
		itkGetConstMacro(GridSize, int);

		/** set/get gaussian blur factor */
		itkSetMacro(GaussianBlurFactor, float);
		itkGetConstMacro(GaussianBlurFactor, float);

		/** set/get # of levels */
		itkSetMacro(NumOfLevels, int);
		itkGetConstMacro(NumOfLevels, int);

		/** set/get num of steps */
		itkSetMacro(NumOfStep, int);
		itkGetConstMacro(NumOfStep, int);

		/** set/get m_MetricFactor */
		itkSetMacro(MetricFactor, float);
		itkGetConstMacro(MetricFactor, float);

		/** set/get m_EdgeFactor */
		itkSetMacro(EdgeFactor, float);
		itkGetConstMacro(EdgeFactor, float);

		/** set/get similarity metric */
		itkSetMacro(MetricSelected, int);
		itkGetConstMacro(MetricSelected, int);

		void outputResults(char* warpedFix, char* warpedMov, char* fld2Fix, char* Fld2Mov);  // output the results

	protected:
		BlockMatchingImageRegistrationFilter();
		~BlockMatchingImageRegistrationFilter() {

			delete[] probabilityMov;
			delete[] probabilityFix;
			delete[] probabilityMovWarped;
			delete[] probabilityFixWarped;

			Release2DArray<float>(m_numBinFix, m_numBinMov, this->probabilityJointFix);
			Release2DArray<float>(m_numBinMov, m_numBinFix, this->probabilityJointMov);
		}

		/** some internal functions */
		typename TImage::ConstPointer GetFixedImage();
		typename TImage::ConstPointer GetMovingImage();

		/** Does the real work. */
		virtual void GenerateData();

	private:
		BlockMatchingImageRegistrationFilter(const Self &); //purposely not implemented
		void operator=(const Self &);  //purposely not implemented

		// functions
		void GaussianSmoothing(const TImage* inputImg, typename InternalImageType::Pointer& smoothedImg, float sigma); //do gaussian smoothing of the image
		void PadImage(const TImage* inputImg, typename TImage::Pointer& paddedImg, int padSize);          //pad the image with grid size
		void CropImage(typename InternalImageType::Pointer& inputImg, typename InternalImageType::Pointer& croppedImg, int cropSize);          //pad the image with grid size
		//void BlockMatching();     //do block matching to build the graph model
		void CreateLabelCostTabel(int numOfSteps, float stepSize, float weight, float thresh);  //create a table to map from labels to movements its cost
		void ComputeDfmfld(int level);      // caculate the deformation field at certain level (start from '1')
		void PrepareImageForMatch(int level, int iter);
		void RandomSampling(int radius_x, int radius_y, int radius_z, int** offsets, float* weights, int numOfSample);     // generate a random sampling scheme
		// function that compute the NCC metric. bool check border is used when this's possiblity sampling position is out of image
		float NCC(typename InternalImageType::PixelType***& fixed, typename InternalImageType::PixelType***& moving, int center_x, int center_y, int center_z, float translate_x, float translate_y, float translate_z, int** offsets, float* weights, int numOfSample, bool checkBorder);
		float PWMI(typename InternalImageType::PixelType***& fixed, typename InternalImageType::PixelType***& moving,
			int center_x, int center_y, int center_z, float translate_x, float translate_y, float translate_z,
			int** offsets, int numOfSample, bool checkBorder,
			float** jointProbability, float* fixProbability, float* movProbability);
		void InterpImageBySpline(const typename InternalImageType::Pointer& InputImg, typename InternalImageType::Pointer& OutputImg, int in_x, int in_y, int in_z, int out_x, int out_y, int out_z);
		template<typename T> T LinearInterpImage(T*** Input, float x, float y, float z);

		template<typename T> T** Allocate2DArray(int x, int y);   // return array[x][y]
		template<typename T> void Release2DArray(int x, int y, T** array);

		template<typename T> T*** Allocate3DArray(int x, int y, int z);   // return array[z][x][y]
		template<typename T> void Release3DArray(int x, int y, int z, T*** array);

		template <typename T> void WriteImage(const char * filename, const T* image);
		void warpImage(void);  //warp the image using current dfmfld
		void symetricDfmfld(void);  // compute the symetric deformation field
		void updateHistogram(typename InternalImageType::Pointer& smoothedFixed, typename InternalImageType::Pointer& smoothedMoving,
			typename InternalImageType::Pointer& smoothedWarpedFixed, typename InternalImageType::Pointer& smoothedWarpedMoving);  // update the histogram
                double Compute_Similarity(const TImage* fixed, const TImage* moving);

			// parameters
		float         m_GaussianBlurVariance;   //the size of gaussian blur
		float         m_SamplingRate;   //the sampling rate in the block
		float         m_Smoothness;   //the sampling rate in the block
		int            m_GridSize;       //the size of the grid
		int            m_NumOfStep;      //the number of steps when move the grid
		int            m_NumOfLevels;    //the number of levels of image downsampling
		float          m_GaussianBlurFactor;      //the ratio between gaussian variance and the step size
		typename InternalImageType::SizeType internalImgSize;
		typename InputImageType::SizeType inputImgSize;

		float m_MetricFactor;
		float m_EdgeFactor;
		int m_level;
		int m_MetricSelected; // 0 for ncc and 1 for MI

	  //float smoothness[3] = {8.0,4.0,2.0};
	  //float edgeWeights[3] = {1.0,1.0,1.0};
	  //float metricWeights[3] = {2.0,2.0,2.0};

		typename InputImageType::SpacingType m_inputImgSpacing;

		// images needed for computation
		typename InternalImageType::PixelType*** fixedImg4Match; // the image that is warped and smoothed used for the block matching
		typename InternalImageType::PixelType*** movingImg4Match;
		typename InternalImageType::PixelType*** fixedImgTemplate; // the image that is used as the template for alignment in each direction, also smoothed
		typename InternalImageType::PixelType*** movingImgTemplate;
		//typename InternalImageType::PixelType*** warpedMovImg4Match;
		//typename InternalImageType::PixelType*** warpedFixImg4Match;
		//typename InternalImageType::Pointer smoothedFixedImg;
		//typename InternalImageType::Pointer smoothedMovingImg;

		float** label2displacement;  // the Nx3 table for lookup
		float* labelDiffCost;          // the NxN table for lookup

		// image output
		//typename DfmfldImageType::Pointer deformationByGrid_x; // the deformation represented in grid
		//typename DfmfldImageType::Pointer deformationByGrid_y;

		typename DeformationFieldType::Pointer dfmfld_fix2mov; // the deformation field at image resolution
		typename DeformationFieldType::Pointer dfmfld_mov2fix; // the deformation field at image resolution
		typename InputImageType::Pointer warpedMovImage;
		typename InputImageType::Pointer warpedFixImage;

		// for histogram computing
		int m_numBinFix;
		int m_numBinMov;
		float m_binMinimumFix;
		float m_binMaximumFix;
		float m_binMinimumMov;
		float m_binMaximumMov;

		typename HistogramFilterType::Pointer histogramFilterJointFix;
		typename HistogramFilterType::Pointer histogramFilterJointMov;

		float* probabilityMov;
		float* probabilityFix;
		float* probabilityMovWarped; //probability for the warped image
		float* probabilityFixWarped;

		float** probabilityJointFix;  //the probability lookup
		float** probabilityJointMov;  //the probability lookup
                
                double m_similarity; //the current similarity between

	};

} //namespace ITK

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBlockMatchingImageRegistrationFilter.hxx"
#endif


#endif
