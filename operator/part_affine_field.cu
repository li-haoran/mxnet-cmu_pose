/*!
 * 
 * \file part_affine_field.cu
 * \brief part_affine_field operator
 * \
*/
#include "./part_affine_field-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>

namespace mshadow {
namespace cuda {

template<typename Dtype>
__global__ void PARTAffineFieldForwardKernel(const int paf_count,
										const float beam_width, const int num_parts,const int num_pairs,const int* pair_config,
										const int height, const int width, const int num_poses, const int pose_size,const int batch_size,
										const int paf_size,const int hw,
										const Dtype* bottom_poses, Dtype* paf_data) {
	for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		index < paf_count;
		index += blockDim.x * gridDim.x * gridDim.y) {
		int w = index % width;
		int h = (index / width) % height;
		int no = (index / width / height) % num_pairs;
		no *= 2;
		int ibatch = index / width / height / num_pairs;

		float sum_value_x = 0;
		float sum_value_y = 0;
		int count = 0;
		for (int ipose = 0; ipose < num_poses; ++ipose){
			const int pose_index = ipose*pose_size;
			const int batch_ind = bottom_poses[pose_index];
			if (batch_ind < 0 || batch_ind >= batch_size){
				continue;
			}

			if (batch_ind != ibatch){
				continue;
			}
			if (pair_config[no] >= num_parts || pair_config[no + 1] >= num_parts){
				continue;
			}

			const int first_kp_index = pose_index + 3 * pair_config[no] + 1;
			float first_kp_x = bottom_poses[first_kp_index];
			float first_kp_y = bottom_poses[first_kp_index + 1];
			float visible = bottom_poses[first_kp_index + 2];
			if (visible < 0.5){
				continue;
			}
			const int last_kp_index = pose_index + 3 * pair_config[no + 1] + 1;
			float last_kp_x = bottom_poses[last_kp_index];
			float last_kp_y = bottom_poses[last_kp_index + 1];
			float visiblel = bottom_poses[last_kp_index + 2];
			if (visiblel < 0.5){
				continue;
			}

			float v_x = last_kp_x - first_kp_x;
			float v_y = last_kp_y - first_kp_y;

			float norm_v = sqrt(v_x*v_x + v_y*v_y) + 1e-10f;
			v_x /= norm_v;
			v_y /= norm_v;

			float v_t_x = v_y;
			float v_t_y = -v_x;

			float p_x = w - first_kp_x;
			float p_y = h - first_kp_y;

			float v_prop = v_x*p_x + v_y*p_y;
			float v_t_prop = abs(v_t_x*p_x + v_t_y*p_y);
			if (v_prop >= 0 && v_prop < norm_v && v_t_prop < beam_width*norm_v){
				count++;
				sum_value_x += v_x;
				sum_value_y += v_y;
			}

			//pick up max index for the model


		}
		const int paf_index_x = no*hw + h*width+w;
		paf_data[paf_index_x] = sum_value_x / (count + 1e-10f);
		const int paf_index_y = (no + 1)*hw + h*width+w;
		paf_data[paf_index_y] = sum_value_y / (count + 1e-10f);
	}

}

template<typename Dtype>
inline void PARTAffineFieldForward(	const Tensor<gpu, 2, Dtype> &pose,
									const Tensor<gpu, 4, Dtype> &paf,
									const int* pair_config,
									const int num_parts, const int num_pairs, 
									const float beam_width) {
	const Dtype *bottom_poses = pose.dptr_;
	Dtype *paf_data = paf.dptr_;
	const int num_poses = pose.size(0);
	const int pose_size = pose.size(1);
	const int batch_size = paf.size(0);
	const int num_paf = paf.size(1);
	CHECK_EQ(num_pairs*2, num_paf);
	CHECK_EQ(num_parts * 3 + 1, pose_size);
	const int height = paf.size(2);
	const int width = paf.size(3);
	const int paf_size = paf.size(1) * paf.size(2) * paf.size(3);
	const int hw = paf.size(2) * paf.size(3);
	const int paf_count = batch_size*num_pairs*hw;
	int* configs = NULL;
	cudaMalloc(&configs, sizeof(int) * num_pairs * 2);
	cudaMemcpy(configs, pair_config, sizeof(int) * num_pairs * 2, cudaMemcpyHostToDevice);
    ///caculate the map_data:
	const int gridSizeMap = (paf_count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
	dim3 dimGridMap(kMaxGridNum, (gridSizeMap + kMaxGridNum - 1) / kMaxGridNum);
	dim3 dimBlockMap(kMaxThreadsPerBlock);
	CheckLaunchParam(dimGridMap, dimBlockMap, "PARTAffineField Forward");
	cudaStream_t streamMap = Stream<gpu>::GetStream(paf.stream_);
	PARTAffineFieldForwardKernel<Dtype> << <dimGridMap, dimBlockMap, 0, streamMap >> >(
		paf_count, beam_width, num_parts, num_pairs,configs,height, width, num_poses, pose_size, batch_size,paf_size,hw,bottom_poses, paf_data);


}


}  // namespace cuda

template<typename Dtype>
inline void PARTAffineFieldForward(const Tensor<gpu, 2, Dtype> &pose,
							const Tensor<gpu, 4, Dtype> &paf,
							const int* pair_config,
							const int num_parts,const int num_pairs,const float beam_width) {
	cuda::PARTAffineFieldForward(pose, paf, pair_config, num_parts, num_pairs, beam_width);
}



}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(PARTAffineFieldParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
	  op = new PARTAffineFieldOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
