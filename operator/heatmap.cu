/*!
 * 
 * \file heatmap.cu
 * \brief heatmap operator
*/
#include "./heatmap-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>

namespace mshadow {
namespace cuda {

template<typename Dtype>
__global__ void HEATMapForwardKernel(const int map_count,
										const float dsigma, const int num_parts,
										const int height, const int width, const int num_poses, const int pose_size,const int batch_size,
										const int map_size,const int hw,
										const Dtype* bottom_poses, Dtype* map_data) {
	for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		index < map_count;
		index += blockDim.x * gridDim.x * gridDim.y) {
		int w = index % width;
		int h = (index / width) % height;
		int no = (index / width / height) % num_parts;
		int ibatch = index / width / height / num_parts;

		float max_value = 0;

		for (int ipose = 0; ipose < num_poses; ++ipose){
			const int pose_index = ipose*pose_size;
			const int batch_ind = bottom_poses[pose_index];
			if (batch_ind < 0){
				continue;
			}
			if (batch_ind >= batch_size){
				continue;
			}

			if (batch_ind != ibatch){
				continue;
			}
			const int key_point_index = ipose*pose_size + 3 * no + 1;
			float key_point_x = bottom_poses[key_point_index];
			float key_point_y = bottom_poses[key_point_index + 1];
			float visible = bottom_poses[key_point_index + 2];
			if (visible < 0.5){
				continue;
			}
			float value = exp(-((w - key_point_x)*(w - key_point_x) + (h - key_point_y)*(h - key_point_y)) / dsigma);
			//pick up max index for the model
			max_value = max(max_value, value);
		}
		const int map_index = ibatch*map_size + no*hw + h*width+w;
		map_data[map_index] = max_value;
	}

}

template<typename Dtype>
inline void HEATMapForward(const Tensor<gpu, 2, Dtype> &pose,
                           const Tensor<gpu, 4, Dtype> &map,
                           const float sigma,
						   const int num_parts) {
	const Dtype *bottom_poses = pose.dptr_;
	Dtype *map_data = map.dptr_;
	const int num_poses = pose.size(0);
	const int pose_size = pose.size(1);
	const int batch_size = map.size(0);
	const int num_parts_map = map.size(1);
	CHECK_EQ(num_parts, num_parts_map);
	CHECK_EQ(num_parts * 3 + 1, pose_size);
	const int height = map.size(2);
	const int width = map.size(3);
	const int map_size = map.size(1) * map.size(2) * map.size(3);
	const int hw = map.size(2) * map.size(3);
	const float dsigma = 2 * sigma*sigma;
	const int map_count = map.shape_.Size();

    ///caculate the map_data:
	const int gridSizeMap = (map_count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
	dim3 dimGridMap(kMaxGridNum, (gridSizeMap + kMaxGridNum - 1) / kMaxGridNum);
	dim3 dimBlockMap(kMaxThreadsPerBlock);
	CheckLaunchParam(dimGridMap, dimBlockMap, "HEATMap Forward");
	cudaStream_t streamMap = Stream<gpu>::GetStream(map.stream_);
	HEATMapForwardKernel<Dtype> <<<dimGridMap, dimBlockMap, 0, streamMap >>>(
		map_count, dsigma, num_parts, height, width, num_poses, pose_size, batch_size,map_size,hw,bottom_poses, map_data);


}


}  // namespace cuda

template<typename Dtype>
inline void HEATMapForward(	const Tensor<gpu, 2, Dtype> &pose,
							const Tensor<gpu, 4, Dtype> &map,
							const float sigma,
							const int num_parts) {
	cuda::HEATMapForward(pose, map, sigma,num_parts);
}



}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(HEATMapParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new HEATMapOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
