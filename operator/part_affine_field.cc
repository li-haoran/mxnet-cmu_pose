/*!
* Copyright (c) 2017 by Contributors
* \file part affine field-inl.h
* \brief part affine field operator and symbol
* \author haoran li
*/
#include "./part_affine_field-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

using std::max;
using std::sqrt;
using std::abs;

namespace mshadow {
	template<typename Dtype>
	inline void PARTAffineFieldForward(
		const Tensor<cpu, 2, Dtype> &pose,
		const Tensor<cpu, 4, Dtype> &paf,
		const int* pair_config,
		const int num_parts,
		const int num_pairs,
		const float beam_width) {
		const Dtype *bottom_poses = pose.dptr_;
		Dtype *paf_data = paf.dptr_;

		const int num_poses = pose.size(0);
		const int pose_size = pose.size(1);

		const int batch_size = paf.size(0);
		const int num_paf = paf.size(1);
		CHECK_EQ(num_pairs*2, num_paf);
		CHECK_EQ(num_parts*3+1, pose_size);
		const int height = paf.size(2);
		const int width = paf.size(3);

		const int paf_size = paf.size(1) * paf.size(2) * paf.size(3);
		const int hw = paf.size(2) * paf.size(3);
		
		// For each pose R = [batch_index x1 y1 x2 y2.....x14,y14]: weighted pool over R
		for (int ibatch = 0; ibatch < batch_size; ++ibatch){
			
			for (int no = 0; no < num_paf; no += 2){
				for (int h = 0; h < height; ++h){
					for (int w = 0; w < width; ++w){

						float sum_value_x = 0;
						float sum_value_y = 0;
						int count = 0;
						for (int ipose = 0; ipose < num_poses; ++ipose){
							const int pose_index = ipose*pose_size;
							const int batch_ind = bottom_poses[pose_index];
							assert(batch_ind >= 0);
							assert(batch_ind < batch_size);

							if (batch_ind != ibatch){
								continue;
							}
							assert(pair_config[no] < num_parts);
							assert(pair_config[no+1] < num_parts);

							const int first_kp_index = pose_index + 3 * pair_config[no] + 1;
							float first_kp_x = bottom_poses[first_kp_index];
							float first_kp_y = bottom_poses[first_kp_index + 1];
							float visible = bottom_poses[first_kp_index + 2];
							if (visible < 0.5){
								continue;
							}
							const int last_kp_index = pose_index + 3 * pair_config[no+1] + 1;
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
						const int paf_index_x =no*hw + h*width+w;
						paf_data[paf_index_x] = sum_value_x / (count + 1e-10f);
						const int paf_index_y = (no+1)*hw + h*width+w;
						paf_data[paf_index_y] = sum_value_y / (count + 1e-10f);

					}
				}
			}
			paf_data += paf_size;
		}
			
		return;
	}

	
}  // namespace mshadow

namespace mxnet {
	namespace op {

		template<>
		Operator *CreateOp<cpu>(PARTAffineFieldParam param, int dtype) {
			Operator* op = NULL;
			MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
				op = new PARTAffineFieldOp<cpu, DType>(param);
			});
			return op;
		}

		Operator *PARTAffineFieldProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
			std::vector<int> *in_type) const {
			std::vector<TShape> out_shape, aux_shape;
			std::vector<int> out_type, aux_type;
			CHECK(InferType(in_type, &out_type, &aux_type));
			CHECK(InferShape(in_shape, &out_shape, &aux_shape));
			DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
		}

		DMLC_REGISTER_PARAMETER(PARTAffineFieldParam);

		MXNET_REGISTER_OP_PROPERTY(PARTAffineField, PARTAffineFieldProp)
			.describe("generate part affine field for keypoint detection")
			.add_argument("poses", "Symbol", "Bounding box coordinates, a 2D array of "
			"[[batch_index, x1, y1.v1, x2, y2,v2,...x14,y14,vn]]. (xi, yi,vi) all are keypoints coordinates and visible "
			"batch_index indicates the index of corresponding image in the input data")
			.add_arguments(PARTAffineFieldParam::__FIELDS__());
	}  // namespace op
}  // namespace mxnet
