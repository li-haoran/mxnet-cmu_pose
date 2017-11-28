/*!
* Copyright (c) 2017 by Contributors
* \file heatmap-inl.h
* \brief heatmap operator and symbol
* \author haoran li
*/
#include "./heatmap-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

using std::max;
using std::exp;


namespace mshadow {
	template<typename Dtype>
	inline void HEATMapForward(
		const Tensor<cpu, 2, Dtype> &pose,
		const Tensor<cpu, 4, Dtype> &map,
		const float sigma,
		const int num_parts) {
		const Dtype *bottom_poses = pose.dptr_;
		Dtype *map_data = map.dptr_;

		const int num_poses = pose.size(0);
		const int pose_size = pose.size(1);
		const int batch_size = map.size(0);
		const int num_parts_map = map.size(1);
		CHECK_EQ(num_parts, num_parts_map);
		CHECK_EQ(num_parts*3+1, pose_size);
		const int height = map.size(2);
		const int width = map.size(3);

		const int map_size = map.size(1) * map.size(2) * map.size(3);
		const int hw = map.size(2) * map.size(3);
		const float dsigma = 2 * sigma*sigma;
		// For each pose R = [batch_index x1 y1 x2 y2.....x14,y14]: weighted pool over R
		for (int ibatch = 0; ibatch < batch_size; ++ibatch){
			
			for (int no = 0; no < num_parts_map; ++no){
				for (int h = 0; h < height; ++h){
					for (int w = 0; w < width; ++w){

						float max_value = 0;

						for (int ipose = 0; ipose < num_poses; ++ipose){
							const int pose_index = ipose*pose_size;
							int batch_ind = bottom_poses[pose_index];
							assert(batch_ind >= 0);
							assert(batch_ind < batch_size);

							if (batch_ind != ibatch){
								continue;
							}
							const int key_point_index = ipose*pose_size + 3*no + 1;
							float key_point_x = bottom_poses[key_point_index];
							float key_point_y = bottom_poses[key_point_index + 1];
							float visible = bottom_poses[key_point_index + 2];
							if (visible < 0.5){
								continue;
							}
							float value = exp(-((w - key_point_x)*(w - key_point_x) + (h - key_point_y)*(h - key_point_y))/dsigma);
							//pick up max index for the model
							max_value = max(max_value, value);

						}
						const int map_index = no*hw + h*width+w;
						map_data[map_index] = max_value;

					}
				}
			}
			map_data += map_size;
		}
			
		return;
	}

	
}  // namespace mshadow

namespace mxnet {
	namespace op {

		template<>
		Operator *CreateOp<cpu>(HEATMapParam param, int dtype) {
			Operator* op = NULL;
			MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
				op = new HEATMapOp<cpu, DType>(param);
			});
			return op;
		}

		Operator *HEATMapProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
			std::vector<int> *in_type) const {
			std::vector<TShape> out_shape, aux_shape;
			std::vector<int> out_type, aux_type;
			CHECK(InferType(in_type, &out_type, &aux_type));
			CHECK(InferShape(in_shape, &out_shape, &aux_shape));
			DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
		}

		DMLC_REGISTER_PARAMETER(HEATMapParam);

		MXNET_REGISTER_OP_PROPERTY(HEATMap, HEATMapProp)
			.describe("generate heatmap for keypoint detection")
			.add_argument("poses", "Symbol", "Bounding box coordinates, a 2D array of "
			"[[batch_index, x1, y1.v1, x2, y2,v2,...x14,y14,vn]]. (xi, yi,vi) all are keypoints coordinates and visible "
			"batch_index indicates the index of corresponding image in the input data")
			.add_arguments(HEATMapParam::__FIELDS__());
	}  // namespace op
}  // namespace mxnet
