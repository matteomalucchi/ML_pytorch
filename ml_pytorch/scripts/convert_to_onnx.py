import spox.opset.ai.onnx.v17 as op
from spox import argument, build, inline, Tensor
import os
import onnx
import numpy as np
import argparse
import onnxruntime as ort
import uproot

parser = argparse.ArgumentParser(description="Convert keras to onnx or average models")
parser.add_argument("-i", "--input", type=str, required=True, help="Input directory")
parser.add_argument(
    "-ar",
    "--average-ratio",
    action="store_true",
    default=False,
    help="Perform the average between the models in the directory of the ratios of the outputs",
)
parser.add_argument(
    "-mt",
    "--model_type",
    default="keras",
    help="Parameter to determine, what type of model is being converted (onnx or keras)",
)
args = parser.parse_args()

if args.model_type == "keras":
    import tensorflow as tf
    import tf2onnx

bkg_morphing_dnn_input_variables = [
    "era",
    "higgs1_reco_pt",
    "higgs1_reco_eta",
    "higgs1_reco_phi",
    "higgs1_reco_mass",
    "higgs2_reco_pt",
    "higgs2_reco_eta",
    "higgs2_reco_phi",
    "higgs2_reco_mass",
    "HT",
    "higgs1_DeltaRjj",
    "higgs2_DeltaRjj",
    "minDeltaR_Higgjj",
    "maxDeltaR_Higgjj",
    "higgs1_helicityCosTheta",
    "higgs2_helicityCosTheta",
    "hh_CosThetaStar_CS",
    "hh_vec_mass",
    "hh_vec_pt",
    "hh_vec_eta",
    "hh_vec_DeltaR",
    "hh_vec_DeltaPhi",
    "hh_vec_DeltaEta",
    "higgs1_reco_jet1_pt",
    "higgs1_reco_jet1_eta",
    "higgs1_reco_jet1_phi",
    "higgs1_reco_jet1_mass",
    "higgs1_reco_jet2_pt",
    "higgs1_reco_jet2_eta",
    "higgs1_reco_jet2_phi",
    "higgs1_reco_jet2_mass",
    "higgs2_reco_jet1_pt",
    "higgs2_reco_jet1_eta",
    "higgs2_reco_jet1_phi",
    "higgs2_reco_jet1_mass",
    "higgs2_reco_jet2_pt",
    "higgs2_reco_jet2_eta",
    "higgs2_reco_jet2_phi",
    "higgs2_reco_jet2_mass",
    "add_jet1pt_pt",
    "add_jet1pt_eta",
    "add_jet1pt_phi",
    "add_jet1pt_mass",
    "sigma_over_higgs1_reco_mass",
    "sigma_over_higgs2_reco_mass",
]

sig_bkg_dnn_input_variables = [
    "era",
    "HT",
    "hh_vec_mass",
    "hh_vec_pt",
    "hh_vec_eta",
    "hh_vec_phi",
    "hh_vec_DeltaPhi",
    "hh_vec_DeltaEta",
    "hh_vec_DeltaR",
    "hh_CosThetaStar_CS",
    "higgs1_reco_pt",
    "higgs1_reco_eta",
    "higgs1_reco_phi",
    "higgs1_reco_mass",
    "higgs2_reco_pt",
    "higgs2_reco_eta",
    "higgs2_reco_phi",
    "higgs2_reco_mass",
    "higgs1_DeltaPhijj",
    "higgs2_DeltaPhijj",
    "higgs1_DeltaEtajj",
    "higgs2_DeltaEtajj",
    "minDeltaR_Higgjj",
    "maxDeltaR_Higgjj",
    "higgs1_helicityCosTheta",
    "higgs2_helicityCosTheta",
    "higgs1_reco_jet1_pt",
    "higgs1_reco_jet1_eta",
    "higgs1_reco_jet1_phi",
    "higgs1_reco_jet1_mass",
    "higgs1_reco_jet2_pt",
    "higgs1_reco_jet2_eta",
    "higgs1_reco_jet2_phi",
    "higgs1_reco_jet2_mass",
    "higgs2_reco_jet1_pt",
    "higgs2_reco_jet1_eta",
    "higgs2_reco_jet1_phi",
    "higgs2_reco_jet1_mass",
    "higgs2_reco_jet2_pt",
    "higgs2_reco_jet2_eta",
    "higgs2_reco_jet2_phi",
    "higgs2_reco_jet2_mass",
    "add_jet1pt_pt",
    "add_jet1pt_eta",
    "add_jet1pt_phi",
    "add_jet1pt_mass",
    "add_jet1pt_Higgs1_deta",
    "add_jet1pt_Higgs1_dphi",
    "add_jet1pt_Higgs1_m",
    "add_jet1pt_Higgs2_deta",
    "add_jet1pt_Higgs2_dphi",
    "add_jet1pt_Higgs2_m",
    "sigma_over_higgs1_reco_mass",
    "sigma_over_higgs2_reco_mass",
]

columns=bkg_morphing_dnn_input_variables

def save_onnx_model(onnx_model_final, onnx_model_name):
    if os.path.exists(onnx_model_name):
        print(f"Removing {onnx_model_name}")
        os.remove(onnx_model_name)
    onnx.save(onnx_model_final, onnx_model_name)
    print(f"Model saved as {onnx_model_name}")


def compare_output_onnx_keras(onnx_model_name, keras_model):
    session = ort.InferenceSession(
        onnx_model_name, providers=ort.get_available_providers()
    )

    # print the input/putput name and shape
    input_name = [input.name for input in session.get_inputs()]
    output_name = [output.name for output in session.get_outputs()]
    print("Inputs name:", input_name)
    print("Outputs name:", output_name)

    input_shape = [input.shape for input in session.get_inputs()]
    output_shape = [output.shape for output in session.get_outputs()]
    print("Inputs shape:", input_shape)
    print("Outputs shape:", output_shape)
    # input_data = [[1] + [x*100] * (len(columns) - 1) for x in np.random.rand(100)]
    
    #load a root file
    file_name="/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/file_root/JetMET_2022EE_2b_signal_region_to_4b_soumya_january2025.root"
    tree=uproot.open(file_name)["tree"]
    input_data_dict = tree.arrays(columns, library="np")
    n_events = 10
    # get the input data as a numpy array
    input_data = np.array([input_data_dict[col][:n_events] for col in columns], dtype=np.float32).T

    input_example = {input_name[0]: input_data}
    output_onnx = session.run(output_name, input_example)

    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
    # input_tensor = tf.expand_dims(input_tensor, 0)
    output_keras = keras_model.predict(input_tensor)

    print(output_onnx)
    print(output_keras)
    
    assert np.allclose(output_onnx, output_keras, rtol=1e-03, atol=1e-05)

def main():
    main_dir = args.input
    # "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_morphing"
    
    if args.model_type == "keras":
        filending = ".keras"
    if args.model_type == "onnx":
        filending = ".onnx"
    
    model_files = [x for x in os.listdir(main_dir) if x.endswith(filending)]
    print(model_files)
    print("Lenght of input", len(columns))

    if args.average_ratio:
        print(f"Processing {model_files[0]}")

        tot_len = 1
        
        if args.model_type == "keras":
            model = tf.keras.models.load_model(os.path.join(main_dir, model_files[0]))
            model_ratio = tf.keras.models.Model(
                inputs=model.input, outputs=model.output[:, 1] / model.output[:, 0]
            )

            onnx_model_ratio_sum, _ = tf2onnx.convert.from_keras(
                model_ratio,
                input_signature=[
                    tf.TensorSpec(shape=(None, len(columns)), dtype=tf.float32)
                ],
            )
            b = argument(Tensor(np.float32, ("N", len(columns))))
            
        elif args.model_type == "onnx":
            input_shape = len(columns)
            onnx_model_ratio_sum = onnx.load(os.path.join(main_dir, model_files[0]))
            b = argument(Tensor(np.float32, ("N", input_shape)))
            
            # To take the ratio of the first model too.
            (r,) = inline(onnx_model_ratio_sum)(b).values()
            r = op.div(r,op.sub(op.const(1.0,dtype="float32"),r))

            onnx_model_ratio_sum = build({"args_0": b}, {"sum_w": r})

        for model_file in model_files[1:]:
            tot_len += 1
            print(f"\n\nAdding {model_file}")
            if args.model_type == "keras":
                model_add = tf.keras.models.load_model(os.path.join(main_dir, model_file))
                model_ratio_add = tf.keras.models.Model(
                    inputs=model_add.input,
                    outputs=model_add.output[:,0] / 1-model_add.output[:,1],
                )

                onnx_model_ratio_add, _ = tf2onnx.convert.from_keras(
                    model_ratio_add,
                    input_signature=[
                        tf.TensorSpec(shape=(None, len(columns)), dtype=tf.float32)
                    ],
                )
                
            elif args.model_type == "onnx":
                onnx_model_ratio_add = onnx.load(os.path.join(main_dir, model_file))
            
            print(b)
            (r,) = inline(onnx_model_ratio_sum)(b).values()
            (r1,) = inline(onnx_model_ratio_add)(b).values()
            if args.model_type == "onnx":
                r1 = op.div(r1,op.sub(op.const(1.0,dtype="float32"),r1))
            print(r)
            print(r1)

            s = op.add(r, r1)

            onnx_model_ratio_sum = build({"args_0": b}, {"sum_w": s})

        print(f"\ntotal length: {tot_len}")
        (r_sum,) = inline(onnx_model_ratio_sum)(b).values()
        a = op.div(r_sum, op.constant(value_float=tot_len))

        onnx_model_final = build({"args_0": b}, {"avg_w": a})
        onnx_model_name = f"{main_dir}/output/average_model_from_{args.model_type}.onnx"
        save_onnx_model(onnx_model_final, onnx_model_name)

    else:
        for model_file in model_files:
            print(f"Processing {model_file}")
            model = tf.keras.models.load_model(os.path.join(main_dir, model_file))
            onnx_model_name = f"{main_dir}/{model_file.replace('.keras', '.onnx')}"

            input_signature = tf.TensorSpec(
                shape=(None, len(columns)), dtype=tf.float32
            )

            if "2.16" in tf.__version__:
                output_name = model.layers[-1].name

                @tf.function(input_signature=[input_signature])
                def _wrapped_model(input_data):
                    return {output_name: model(input_data)}

                onnx_model, _ = tf2onnx.convert.from_function(
                    _wrapped_model,
                    input_signature=[input_signature],
                )
            else:
                onnx_model, _ = tf2onnx.convert.from_keras(
                    model,
                    input_signature=[input_signature],
                )
            save_onnx_model(onnx_model, onnx_model_name)

            compare_output_onnx_keras(onnx_model_name, model)

if __name__ == "__main__":
    main()
