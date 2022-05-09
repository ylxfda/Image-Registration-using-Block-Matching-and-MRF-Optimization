#ifndef __itkBlockMatchingImageRegistrationFilter_h
#define __itkBlockMatchingImageRegistrationFilter_h
 
#include "itkImageToImageFilter.h"

template<typename T> T bilinear( const T &tx, const T &ty, const T &c00, const T &c10, const T &c01, const T &c11) 
{ 
#if 1 
    T a = c00 * (T(1) - tx) + c10 * tx; 
    T b = c01 * (T(1) - tx) + c11 * tx; 
    return a * (T(1) - ty) + b * ty; 
#else 
    return (T(1) - tx) * (T(1) - ty) * c00 + tx * (T(1) - ty) * c10 + (T(1) - tx) * ty * c01 + tx * ty * c11; 
#endif 
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
  typedef   itk::Image< VectorType,  ImageDimension >   DeformationFieldType;
  //typedef Image<float, ImageDimension>       DfmfldImageType;
  
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
  itkSetMacro(GaussianBlurFactor, int);
  itkGetConstMacro(GaussianBlurFactor, int);

  /** set/get # of levels */
  itkSetMacro(NumOfLevels, int);
  itkGetConstMacro(NumOfLevels, int);
  
  /** set/get num of steps */
  itkSetMacro(NumOfStep, int);
  itkGetConstMacro(NumOfStep, int);

   void outputResults(char* warpedFix, char* warpedMov, char* fld2Fix, char* Fld2Mov);  // output the results
 
protected:
  BlockMatchingImageRegistrationFilter();
  ~BlockMatchingImageRegistrationFilter(){}
  
  /** some internal functions */
  typename TImage::ConstPointer GetFixedImage();
  typename TImage::ConstPointer GetMovingImage();
 
  /** Does the real work. */
  virtual void GenerateData();
 
private:
  BlockMatchingImageRegistrationFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented
  
  // functions
  void GaussianSmoothing(const typename TImage::Pointer inputImg, typename InternalImageType::Pointer& smoothedImg, float sigma); //do gaussian smoothing of the image
  void PadImage(const TImage* inputImg, typename TImage::Pointer& paddedImg, int padSize);          //pad the image with grid size
  void CropImage(typename InternalImageType::Pointer& inputImg, typename InternalImageType::Pointer& croppedImg, int cropSize);          //pad the image with grid size
  //void BlockMatching();     //do block matching to build the graph model
  void CreateLabelCostTabel(int numOfSteps, float stepSize, float weight);  //create a table to map from labels to movements its cost
  void ComputeDfmfld( int level );      // caculate the deformation field at certain level (start from '1')     
  void PrepareImageForMatch( int level );   
  void RandomSampling(int radius_x, int radius_y, int** offsets, int numOfSample);     // generate a random sampling scheme 
  float NCC(typename InternalImageType::PixelType**& fixed, typename InternalImageType::PixelType**& moving, int center_x, int center_y, float translate_x, float translate_y, int** offsets, int numOfSample);
  void InterpImageBySpline( const typename InternalImageType::Pointer& InputImg, typename InternalImageType::Pointer& OutputImg, int in_x, int in_y, int out_x, int out_y);
  template<typename T> T LinearInterpImage(T** Input, float x, float y);  
  template<typename T> void Allocate2DArray(int x, int y, T**& array);   // return array[x][y]
  template<typename T> void Release2DArray(int x, int y, T**& array);
  template <typename T> void WriteImage( const char * filename, const T* image );
  void warpImage(void);  //warp the image using current dfmfld
  void symetricDfmfld(void);  // compute the symetric deformation field 
  
  // parameters
  float         m_GaussianBlurVariance;   //the size of gaussian blur
  float         m_SamplingRate;   //the sampling rate in the block
  float         m_Smoothness;   //the sampling rate in the block
  int            m_GridSize;       //the size of the grid
  int            m_NumOfStep;      //the number of steps when move the grid
  int            m_NumOfLevels;    //the number of levels of image downsampling
  int            m_GaussianBlurFactor;      //the ratio between gaussian variance and the step size
  typename InternalImageType::SizeType internalImgSize;
  typename InputImageType::SizeType inputImgSize;
  
  // images needed for computation
  typename InternalImageType::PixelType** fixedImg4Match;
  typename InternalImageType::PixelType** movingImg4Match;
  typename InternalImageType::PixelType** warpedMovImg4Match;
  typename InternalImageType::PixelType** warpedFixImg4Match;
  //typename InternalImageType::Pointer smoothedFixedImg;
  //typename InternalImageType::Pointer smoothedMovingImg;
  
  float** label2displacement;  // the Nx2 table for lookup
  float* label2cost;          // the NxN table for lookup 
  
  // image output
  //typename DfmfldImageType::Pointer deformationByGrid_x; // the deformation represented in grid
  //typename DfmfldImageType::Pointer deformationByGrid_y;
  
  typename DeformationFieldType::Pointer dfmfld_fix2mov; // the deformation field at image resolution
  typename DeformationFieldType::Pointer dfmfld_mov2fix; // the deformation field at image resolution
  typename InputImageType::Pointer warpedMovImage;
  typename InputImageType::Pointer warpedFixImage;
 
};

} //namespace ITK

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBlockMatchingImageRegistrationFilter.hxx"
#endif
 
 
#endif 
