/*!
* Copyright (c) 2017 by Contributors
* \filepart_affine field-inl.h
* \brief paf operator and symbol
* \author haoran li
*/
#ifndef MXNET_OPERATOR_PART_AFFINE_FIELD_INL_H_
#define MXNET_OPERATOR_PART_AFFINE_FIELD_INL_H_

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
		namespace partaffinefield {
			enum PARTAffineFieldOpInputs { kPose};
			enum PARTAffineFieldOpOutputs { kPaf };
		}  // HEATMap
		
		struct PARTAffineFieldParam : public dmlc::Parameter<PARTAffineFieldParam> {
			TShape output_shape;
			TShape pair_config;
			float beam_width;
			int num_parts;
			int num_pairs;
			int batch_size;
			DMLC_DECLARE_PARAMETER(PARTAffineFieldParam) {
				DMLC_DECLARE_FIELD(output_shape)
					.set_expect_ndim(2).enforce_nonzero()
					.describe("fix output size: (h, w)");
				DMLC_DECLARE_FIELD(pair_config).describe(
					"pair config for part affine field ");
				DMLC_DECLARE_FIELD(beam_width).set_range(0.0, 32.0)
					.describe("The part affine field beam width");
				DMLC_DECLARE_FIELD(num_parts).set_range(5, 17).set_default(14)
					.describe("how many part to caculate");
				DMLC_DECLARE_FIELD(num_pairs).set_range(5, 17).set_default(11)
					.describe("how many part to caculate");
				DMLC_DECLARE_FIELD(batch_size).set_range(1, 32).set_default(1)
					.describe("batch-size for training,default 1 for train/test");
			}
		};

		template<typename xpu, typename DType>
		class PARTAffineFieldOp : public Operator {
		public:
			explicit PARTAffineFieldOp(PARTAffineFieldParam p) {
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
				
				CHECK_EQ(out_data[partaffinefield::kPaf].shape_[0], param_.batch_size);
				CHECK_EQ(in_data[partaffinefield::kPose].shape_[1], param_.num_parts * 3 + 1);
				CHECK_EQ(out_data[partaffinefield::kPaf].shape_[1], param_.num_pairs * 2);
				Stream<xpu> *s = ctx.get_stream<xpu>();

				Tensor<xpu, 2, DType> pose = in_data[partaffinefield::kPose].get<xpu, 2, DType>(s);
				Tensor<xpu, 4, DType> paf = out_data[partaffinefield::kPaf].get<xpu, 4, DType>(s);

				CHECK_EQ(pose.CheckContiguous(), true);
				CHECK_EQ(paf.CheckContiguous(), true);
				paf = 0.0f;
				const int num_paf = param_.num_pairs * 2;
				int *pair_config= new int[num_paf];
				for (int i = 0; i < num_paf; ++i){
					pair_config[i] = static_cast<int>(param_.pair_config[i]);
				}
				
				PARTAffineFieldForward(pose, paf, pair_config, param_.num_parts, param_.num_pairs, param_.beam_width);
				delete pair_config;
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
			PARTAffineFieldParam param_;
		};  // class HEATMapOp

		// Decalre Factory function, used for dispatch specialization
		template<typename xpu>
		Operator* CreateOp(PARTAffineFieldParam param, int dtype);

#if DMLC_USE_CXX11
		class PARTAffineFieldProp : public OperatorProperty {
		public:
			std::vector<std::string> ListArguments() const override {
				return{ "poses" };
			}

			std::vector<std::string> ListOutputs() const override {
				return{"pafs" };
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
				TShape bshape = in_shape->at(partaffinefield::kPose);
				int pose_size = param_.num_parts * 3 + 1;
				CHECK_EQ(bshape.ndim(), 2) << "pose should be a 2D tensor of shape [batch, 3*num_parts+1]";
				CHECK_EQ(bshape[1], pose_size) << "pose should be shape [batch_ind, (x,y,visible)^num_parts]";

				// out: [num_poses, c, pooled_h, pooled_w]
				// map: [num_poses, 1, indata_shape[2],indata_shape[3]]
				out_shape->clear();
				out_shape->push_back(
					Shape4(param_.batch_size, param_.num_pairs*2, param_.output_shape[0], param_.output_shape[1]));
				
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
				PARTAffineFieldProp* PARTAffineField_sym = new PARTAffineFieldProp();
				PARTAffineField_sym->param_ = this->param_;
				return PARTAffineField_sym;
			}

			std::string TypeString() const override {
				return "PARTAffineField";
			}

			// decalre dependency and inplace optimization options
			std::vector<int> DeclareBackwardDependency(
				const std::vector<int> &out_grad,
				const std::vector<int> &in_data,
				const std::vector<int> &out_data) const override {
				return{ in_data[partaffinefield::kPose], out_data[partaffinefield::kPaf] };
			}

			Operator* CreateOperator(Context ctx) const override {
				LOG(FATAL) << "Not Implemented.";
				return NULL;
			}

			Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
				std::vector<int> *in_type) const override;

		private:
			PARTAffineFieldParam param_;
		};  // class PART_AFFINE_FIELDProp
#endif
	}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_PART_AFFINE_FIELDINL_H_
