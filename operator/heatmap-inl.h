/*!
* Copyright (c) 2017 by Contributors
* \file heatmap-inl.h
* \brief haetmap operator and symbol
* \author haoran li
*/
#ifndef MXNET_OPERATOR_HEATMAP_INL_H_
#define MXNET_OPERATOR_HEATMAP_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"


namespace mxnet {
	namespace op {

		// Declare enumeration of input order to make code more intuitive.
		// These enums are only visible within this header
		namespace heatmap {
			enum HEATMapOpInputs { kPose };
			enum HEATMapOpOutputs { kMap };
		}  // HEATMap
		
		struct HEATMapParam : public dmlc::Parameter<HEATMapParam> {
			TShape output_shape;
			float sigma;
			int num_parts;
			int batch_size;
			DMLC_DECLARE_PARAMETER(HEATMapParam) {
				DMLC_DECLARE_FIELD(output_shape)
					.set_expect_ndim(2).enforce_nonzero()
					.describe("fix output size: (h, w)");
				DMLC_DECLARE_FIELD(sigma).set_range(0.0, 32.0)
					.describe("The guassian kernal digma to contral the heatmap width");
				DMLC_DECLARE_FIELD(num_parts).set_range(5, 17).set_default(14)
					.describe("how many part to caculate");
				DMLC_DECLARE_FIELD(batch_size).set_range(1, 32).set_default(1)
					.describe("batch-size for training,default 1 for train/test");
			}
		};

		template<typename xpu, typename DType>
		class HEATMapOp : public Operator {
		public:
			explicit HEATMapOp(HEATMapParam p) {
				this->param_ = p;
			}

			virtual void Forward(const OpContext &ctx,
				const std::vector<TBlob> &in_data,
				const std::vector<OpReqType> &req,
				const std::vector<TBlob> &out_data,
				const std::vector<TBlob> &aux_args) {
				using namespace mshadow;
				size_t expected = 1;
				CHECK_EQ(in_data.size(), expected);
				CHECK_EQ(out_data.size(), expected);
				CHECK_EQ(out_data[heatmap::kMap].shape_[0], param_.batch_size);
				Stream<xpu> *s = ctx.get_stream<xpu>();

				Tensor<xpu, 2, DType> pose = in_data[heatmap::kPose].get<xpu, 2, DType>(s);
				Tensor<xpu, 4, DType> map = out_data[heatmap::kMap].get<xpu, 4, DType>(s);

				CHECK_EQ(pose.CheckContiguous(), true);
				CHECK_EQ(map.CheckContiguous(), true);
				map = 0.0f;
				HEATMapForward(pose, map, param_.sigma,param_.num_parts);
			}

			virtual void Backward(const OpContext &ctx,
				const std::vector<TBlob> &out_grad,
				const std::vector<TBlob> &in_data,
				const std::vector<TBlob> &out_data,
				const std::vector<OpReqType> &req,
				const std::vector<TBlob> &in_grad,
				const std::vector<TBlob> &aux_args) {
				
			}

		private:
			HEATMapParam param_;
		};  // class HEATMapOp

		// Decalre Factory function, used for dispatch specialization
		template<typename xpu>
		Operator* CreateOp(HEATMapParam param, int dtype);

#if DMLC_USE_CXX11
		class HEATMapProp : public OperatorProperty {
		public:
			std::vector<std::string> ListArguments() const override {
				return{ "poses" };
			}

			std::vector<std::string> ListOutputs() const override {
				return{"maps" };
			}

			int NumOutputs() const override {
				return 1;
			}

			int NumVisibleOutputs() const override {
				return 1;
			}

			void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
				param_.Init(kwargs);
			}

			std::map<std::string, std::string> GetParams() const override {
				return param_.__DICT__();
			}

			bool InferShape(std::vector<TShape> *in_shape,
				std::vector<TShape> *out_shape,
				std::vector<TShape> *aux_shape) const override {
				using namespace mshadow;
				CHECK_EQ(in_shape->size(), 1) << "Input:[ poses]";

				// pose: [num_poses, 29]
				TShape bshape = in_shape->at(heatmap::kPose);
				int pose_size = param_.num_parts * 3 + 1;
				CHECK_EQ(bshape.ndim(), 2) << "pose should be a 2D tensor of shape [batch, 29]";
				CHECK_EQ(bshape[1], pose_size) << "pose should be shape [batch_ind, (x,y,visible)^num_parts]";

				// out: [num_poses, c, pooled_h, pooled_w]
				// map: [num_poses, 1, indata_shape[2],indata_shape[3]]
				out_shape->clear();
				out_shape->push_back(
					Shape4(param_.batch_size, param_.num_parts, param_.output_shape[0], param_.output_shape[1]));
				
				return true;
			}

			bool InferType(std::vector<int> *in_type,
				std::vector<int> *out_type,
				std::vector<int> *aux_type) const override {
				CHECK_EQ(in_type->size(), 1);
				int dtype = (*in_type)[0];
				CHECK_NE(dtype, -1) << "Input must have specified type";

				out_type->clear();
				out_type->push_back(dtype);
				return true;
			}

			OperatorProperty* Copy() const override {
				HEATMapProp* heatmap_sym = new HEATMapProp();
				heatmap_sym->param_ = this->param_;
				return heatmap_sym;
			}

			std::string TypeString() const override {
				return "HEATMap";
			}

			// decalre dependency and inplace optimization options
			std::vector<int> DeclareBackwardDependency(
				const std::vector<int> &out_grad,
				const std::vector<int> &in_data,
				const std::vector<int> &out_data) const override {
				return{ in_data[heatmap::kPose], out_data[heatmap::kMap] };
			}

			Operator* CreateOperator(Context ctx) const override {
				LOG(FATAL) << "Not Implemented.";
				return NULL;
			}

			Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
				std::vector<int> *in_type) const override;

		private:
			HEATMapParam param_;
		};  // class HEATMapProp
#endif
	}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_HEATMAP_INL_H_
